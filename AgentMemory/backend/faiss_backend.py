from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from .base import MemoryBackend
from ..types import (
    BackendOpType,
    BackendRequest,
    CollectionSpec,
    Metric,
    RunResult,
    SearchHit,
)


def _import_faiss() -> Any:
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError as exc:
        raise RuntimeError(
            "faiss is required for the faiss backend. "
            "Install it with: pip install faiss-cpu  (or faiss-gpu)"
        ) from exc


class FaissBackend(MemoryBackend):
    """
    Pure-FAISS backend using IndexFlatL2 / IndexFlatIP wrapped in IndexIDMap2.

    - L2 metric  → IndexFlatL2
    - COSINE     → IndexFlatIP (vectors are L2-normalised on the way in/out)
    - IP         → IndexFlatIP

    nprobe is accepted but ignored (flat exhaustive search).
    Supports INSERT, UPDATE, DELETE_IDS, DELETE_KNN, SEARCH.
    """

    def __init__(self) -> None:
        super().__init__()
        self._faiss = _import_faiss()

        # Per-index state
        self._specs: Dict[int, CollectionSpec] = {}
        self._indices: Dict[int, Any] = {}           # faiss.IndexIDMap2 instances
        self._ext2int: Dict[int, Dict[str, int]] = {}
        self._int2ext: Dict[int, Dict[int, str]] = {}
        self._metas: Dict[int, Dict[str, Optional[dict]]] = {}
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._next_int_id: Dict[int, int] = {}

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        if index_id in self._indices:
            return

        faiss = self._faiss
        dim = int(spec.dim)

        if spec.metric == Metric.L2:
            base = faiss.IndexFlatL2(dim)
        else:
            # Both IP and COSINE use inner-product; cosine normalises vecs first
            base = faiss.IndexFlatIP(dim)

        # IndexIDMap2 lets us assign arbitrary int64 IDs and remove them later
        idx = faiss.IndexIDMap2(base)

        self._indices[index_id] = idx
        self._specs[index_id] = spec
        self._ext2int[index_id] = {}
        self._int2ext[index_id] = {}
        self._metas[index_id] = {}
        self._payloads[index_id] = {}
        self._next_int_id[index_id] = 0

    # ------------------------------------------------------------------ #
    #  Execute                                                             #
    # ------------------------------------------------------------------ #

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            if req.index_id not in self._indices:
                raise KeyError(
                    f"FaissBackend: index_id {req.index_id} not found. Call create_index() first."
                )

            if req.op == BackendOpType.INSERT:
                n = len(req.ext_ids or [])
                self._upsert(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    vectors=self._ensure_matrix(req.vectors),
                    metas=req.metas or [None] * n,
                    payloads=req.payloads or [None] * n,
                )
                ins_cnt += n

            elif req.op == BackendOpType.UPDATE:
                n = len(req.ext_ids or [])
                self._upsert(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    vectors=self._ensure_matrix(req.vectors),
                    metas=req.metas or [None] * n,
                    payloads=req.payloads or [None] * n,
                )
                upd_cnt += n

            elif req.op == BackendOpType.DELETE_IDS:
                removed = self._delete_ids(req.index_id, req.ext_ids or [])
                del_cnt += removed

            elif req.op == BackendOpType.DELETE_KNN:
                removed = self._delete_knn(
                    index_id=req.index_id,
                    queries=self._ensure_matrix(req.vectors),
                    k=int(req.k or 0),
                )
                del_cnt += removed

            elif req.op == BackendOpType.SEARCH:
                queries = self._ensure_matrix(req.vectors)
                k = int(req.k or 0)
                nprobe = int(req.nprobe or 1)
                rid = req.request_id or f"req-{len(search_payload)}"
                if k <= 0 or queries.shape[0] == 0:
                    search_payload[rid] = [[] for _ in range(queries.shape[0])]
                    continue
                hits = self._search(req.index_id, queries, k, nprobe)
                search_payload[rid] = hits

            elif req.op == BackendOpType.FLUSH:
                pass  # flat index — nothing to flush

            else:
                raise NotImplementedError(f"FaissBackend does not support op={req.op}")

        return RunResult(upserted=ins_cnt, updated=upd_cnt, deleted=del_cnt, searches=search_payload)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _prep_vectors(self, index_id: int, vectors: np.ndarray) -> np.ndarray:
        """Normalise to unit length for COSINE metric."""
        spec = self._specs[index_id]
        vecs = np.asarray(vectors, dtype=np.float32, order="C")
        if spec.metric == Metric.COSINE:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vecs = vecs / norms
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def _upsert(
        self,
        index_id: int,
        ext_ids: List[Any],
        vectors: np.ndarray,
        metas: List[Optional[dict]],
        payloads: List[Any],
    ) -> None:
        if not ext_ids:
            return
        if vectors.shape[0] != len(ext_ids):
            raise ValueError("FaissBackend: vectors/ext_ids length mismatch")

        idx = self._indices[index_id]
        ext2int = self._ext2int[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        vecs = self._prep_vectors(index_id, vectors)

        int_ids = np.empty(len(ext_ids), dtype=np.int64)
        for i, ext_id in enumerate(ext_ids):
            key = str(ext_id) if ext_id is not None else "None"
            old = ext2int.get(key)
            if old is not None:
                # Remove the old vector so the ID slot can be reused
                idx.remove_ids(np.array([old], dtype=np.int64))
                int2ext.pop(old, None)

            new_id = self._next_int_id[index_id]
            self._next_int_id[index_id] = new_id + 1
            ext2int[key] = new_id
            int2ext[new_id] = key
            meta_store[key] = metas[i]
            payload_store[key] = payloads[i]
            int_ids[i] = np.int64(new_id)

        idx.add_with_ids(vecs, int_ids)

    def _delete_ids(self, index_id: int, ext_ids: List[Any]) -> int:
        if not ext_ids:
            return 0
        idx = self._indices[index_id]
        ext2int = self._ext2int[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        to_remove: List[int] = []
        removed = 0
        for ext_id in ext_ids:
            key = str(ext_id) if ext_id is not None else "None"
            vec_id = ext2int.pop(key, None)
            if vec_id is None:
                continue
            to_remove.append(vec_id)
            int2ext.pop(vec_id, None)
            meta_store.pop(key, None)
            payload_store.pop(key, None)
            removed += 1

        if to_remove:
            idx.remove_ids(np.array(to_remove, dtype=np.int64))
        return removed

    def _delete_knn(self, index_id: int, queries: np.ndarray, k: int) -> int:
        if queries.shape[0] == 0 or k <= 0:
            return 0
        hits_per_query = self._search(index_id, queries, k)
        to_remove: List[str] = []
        for hits in hits_per_query:
            for h in hits:
                to_remove.append(h.id)
        return self._delete_ids(index_id, to_remove)

    def _search(self, index_id: int, queries: np.ndarray, k: int, nprobe: int = 1) -> List[List[SearchHit]]:
        if queries.shape[0] == 0:
            return []

        idx = self._indices[index_id]
        spec = self._specs[index_id]
        ntotal = idx.ntotal
        if ntotal == 0:
            return [[] for _ in range(queries.shape[0])]

        # Set nprobe on IVF indices; attribute is ignored on flat indices.
        if hasattr(idx, "nprobe"):
            idx.nprobe = nprobe

        effective_k = min(k, ntotal)
        qvecs = self._prep_vectors(index_id, queries)
        distances, ids = idx.search(qvecs, effective_k)

        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        results: List[List[SearchHit]] = []
        for qi in range(ids.shape[0]):
            hits: List[SearchHit] = []
            for vec_id, dist in zip(ids[qi].tolist(), distances[qi].tolist()):
                if vec_id < 0:
                    continue
                key = int2ext.get(int(vec_id), str(int(vec_id)))
                meta = dict(meta_store.get(key) or {})
                meta["_data"] = payload_store.get(key)
                score = self._score(spec.metric, dist)
                hits.append(SearchHit(id=key, score=score, metadata=meta))
            results.append(hits)
        return results

    # ------------------------------------------------------------------ #
    #  Direct Faiss index load (--faiss-index bootstrap)                  #
    # ------------------------------------------------------------------ #

    def rebuild_index_from_faiss(
        self,
        index_id: int,
        *,
        path: str,
        normalized: Optional[bool] = None,
    ) -> None:
        """
        Load a Faiss index file directly into this backend.

        Because the backend is already Faiss, we just call faiss.read_index()
        and slot it in — no vector-by-vector replay needed.

        The loaded index replaces whatever was previously in index_id.
        ID mappings are rebuilt from the IDs stored inside the index
        (IndexIDMap / IndexIDMap2) or from sequential integers for bare flat
        indices that carry no explicit IDs.
        """
        faiss = self._faiss
        from pathlib import Path as _Path

        p = _Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"FaissBackend: index file not found: {p}")

        loaded = faiss.read_index(str(p))

        # Unwrap to a searchable index; keep the outer shell for search so that
        # IDMap wrappers still translate IDs correctly.
        search_index = loaded

        # Recover the stored int64 IDs.
        # Priority 1: IndexIDMap / IndexIDMap2 carry an explicit id_map array.
        # Priority 2: IVF-family indices store per-list IDs in their inverted lists.
        # Priority 3: Fall back to sequential 0..N-1 for bare flat indices.
        id_map_arr = getattr(loaded, "id_map", None)
        if id_map_arr is not None:
            stored_ids = np.asarray(faiss.vector_to_array(id_map_arr), dtype=np.int64)
        else:
            # Try to extract IDs from IVF inverted lists.
            try:
                ivf = faiss.extract_index_ivf(loaded)
            except Exception:
                ivf = None
            if ivf is not None and ivf.ntotal > 0:
                invlists = ivf.invlists
                id_parts: List[np.ndarray] = []
                for list_no in range(ivf.nlist):
                    list_size = invlists.list_size(list_no)
                    if list_size == 0:
                        continue
                    ids_ptr = invlists.get_ids(list_no)
                    id_parts.append(
                        np.array(faiss.rev_swig_ptr(ids_ptr, list_size), dtype=np.int64)
                    )
                    invlists.release_ids(list_no, ids_ptr)
                stored_ids = np.concatenate(id_parts) if id_parts else np.empty(0, dtype=np.int64)
            else:
                ntotal = int(loaded.ntotal)
                stored_ids = np.arange(ntotal, dtype=np.int64)

        # Preserve the spec's name but update dim/metric from the loaded index
        old_spec = self._specs.get(index_id)
        name = old_spec.name if old_spec is not None else f"index-{index_id}"
        metric = old_spec.metric if old_spec is not None else Metric.COSINE
        spec = CollectionSpec(name=name, dim=int(loaded.d), metric=metric)

        # Do NOT build ext2int/int2ext for corpus vectors: for a loaded IVF index
        # the stored IDs are already the external IDs, so int2ext is a pure identity
        # map (str(x) == str(x)) that wastes O(N) memory for zero benefit. _search
        # falls back to str(vec_id) for any ID not in the map. Leave the maps empty
        # so only runtime-inserted vectors (inserts after load) are tracked.
        # _next_int_id is set above the max corpus ID to prevent collisions.
        next_id = int(stored_ids.max()) + 1 if stored_ids.size > 0 else 0

        # Slot everything in
        self._indices[index_id] = search_index
        self._specs[index_id] = spec
        self._ext2int[index_id] = {}
        self._int2ext[index_id] = {}
        self._metas[index_id] = {}
        self._payloads[index_id] = {}
        self._next_int_id[index_id] = next_id

        print(
            f"[FaissBackend] loaded {loaded.ntotal} vectors from {p.name} "
            f"(dim={loaded.d}, metric={metric.value})"
        )

    @staticmethod
    def _score(metric: Metric, raw_distance: float) -> float:
        if metric == Metric.L2:
            return -raw_distance   # smaller distance = higher score
        return raw_distance        # IP / COSINE: higher = better

    @staticmethod
    def _ensure_matrix(vectors: Optional[np.ndarray]) -> np.ndarray:
        if vectors is None:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr
