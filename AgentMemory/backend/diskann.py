from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

from AgentMemory.backend.DiskANN import require_native
from .base import MemoryBackend
from ..types import (
    BackendOpType,
    BackendRequest,
    CollectionSpec,
    Metric,
    RunResult,
    SearchHit,
)

_NATIVE_MODULE: Any | None = None
_NATIVE_ERROR: Exception | None = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _ensure_native() -> Any | None:
    global _NATIVE_MODULE, _NATIVE_ERROR
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE
    try:
        _NATIVE_MODULE = require_native()
        return _NATIVE_MODULE
    except Exception as exc:  # pragma: no cover - import guard mirrors other optional backends
        _NATIVE_ERROR = exc
        return None


class DiskANNCppBackend(MemoryBackend):
    """
    DiskANN backend powered by the local C++ bindings (no diskannpy dependency).

    It mirrors the python DiskANN backend but uses the compiled `_diskann_cpp` module for inserts/search.
    """

    def __init__(
        self,
        *,
        index_directory: Optional[str] = None,
        max_vectors: int = 1_000_000,
        graph_degree: int = 128,
        build_complexity: int = 256,
        search_complexity: int = 128,
        insert_threads: int = 0,
        search_threads: int = 0,
        auto_consolidate_every: int = 10_000,
    ) -> None:
        super().__init__()
        native = _ensure_native()
        if native is None:
            raise RuntimeError(
                "DiskANN C++ backend is unavailable. Make sure the C++ bindings are built "
                "(`-DAGENTMEMORY_BUILD_DISKANN=ON`) and DISKANN_HOME points to your DiskANN checkout. "
                f"Original import error: {repr(_NATIVE_ERROR)}"
            )
        self._native = native

        self._index_root = Path(index_directory or ".diskann_cpp")
        self._index_root.mkdir(parents=True, exist_ok=True)

        self._max_vectors = int(max_vectors)
        if self._max_vectors <= 1:
            raise ValueError("max_vectors must be > 1 because DiskANN reserves tag 0 internally")

        self._graph_degree = int(graph_degree)
        self._build_complexity = int(build_complexity)
        self._default_search_complexity = int(search_complexity)
        self._insert_threads = max(0, int(insert_threads))
        self._search_threads = max(0, int(search_threads))
        self._consolidate_every = max(0, int(auto_consolidate_every))

        # Per-index state
        self._specs: Dict[int, CollectionSpec] = {}
        self._indices: Dict[int, Any] = {}
        self._ext2int: Dict[int, Dict[str, int]] = {}
        self._int2ext: Dict[int, Dict[int, str]] = {}
        self._metas: Dict[int, Dict[str, Optional[dict]]] = {}
        self._payloads: Dict[int, Dict[str, Any]] = {}
        self._next_int_id: Dict[int, int] = {}
        self._pending_deletes: Dict[int, int] = {}

    # ---------- lifecycle ----------
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        if index_id in self._indices:
            return

        metric = self._metric_enum(spec.metric)
        dim = int(spec.dim)
        index_dir = self._index_root / f"index_{index_id}"
        index_dir.mkdir(parents=True, exist_ok=True)

        idx = self._native.DynamicMemoryIndex(
            distance_metric=metric,
            dimensions=dim,
            max_vectors=self._max_vectors,
            complexity=self._build_complexity,
            graph_degree=self._graph_degree,
            initial_search_complexity=self._default_search_complexity,
            search_threads=self._search_threads,
        )
        self._indices[index_id] = idx
        self._specs[index_id] = spec
        self._ext2int[index_id] = {}
        self._int2ext[index_id] = {}
        self._metas[index_id] = {}
        self._payloads[index_id] = {}
        self._next_int_id[index_id] = 1  # DiskANN reserves tag 0
        self._pending_deletes[index_id] = 0

    def save_index(self, index_id: int, *, path: str, prefix: str = "ann") -> None:
        """Persist a built index to disk for later reuse."""
        idx = self._indices.get(index_id)
        if idx is None:
            raise KeyError(f"DiskANNCppBackend: index_id {index_id} not initialized")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx.save(str(out_dir / prefix), compact_before_save=True)

    # ---------- execution ----------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            idx = self._indices.get(req.index_id)
            if idx is None:
                raise KeyError(f"DiskANNCppBackend: index_id {req.index_id} not found. Call create_index() first.")

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
                    complexity=int(
                        getattr(req, "complexity", None) or req.nprobe or self._default_search_complexity
                    ),
                )
                del_cnt += removed

            elif req.op == BackendOpType.SEARCH:
                queries = self._ensure_matrix(req.vectors)
                k = int(req.k or 0)
                if k <= 0 or queries.shape[0] == 0:
                    rid = req.request_id or f"req-{len(search_payload)}"
                    search_payload[rid] = [[] for _ in range(queries.shape[0])]
                    continue
                complexity = int(
                    getattr(req, "complexity", None) or req.nprobe or self._default_search_complexity
                )
                hits = self._search(req.index_id, queries, k, complexity)
                rid = req.request_id or f"req-{len(search_payload)}"
                search_payload[rid] = hits

            elif req.op == BackendOpType.FLUSH:
                self._maybe_consolidate(req.index_id, force=True)

            else:
                raise NotImplementedError(f"DiskANNCppBackend does not support op={req.op}")

        return RunResult(upserted=ins_cnt, updated=upd_cnt, deleted=del_cnt, searches=search_payload)

    # ---------- helpers ----------
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
            raise ValueError("DiskANNCppBackend: vectors/ext_ids length mismatch")

        idx = self._indices[index_id]
        ext2int = self._ext2int[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        ids = np.empty(len(ext_ids), dtype=np.uint32)
        for i, ext_id in enumerate(ext_ids):
            key = self._normalize_key(ext_id)
            old = ext2int.get(key)
            if old is not None:
                self._mark_deleted(index_id, old)
                int2ext.pop(old, None)

            next_id = self._next_vector_id(index_id)
            ext2int[key] = next_id
            int2ext[next_id] = key
            meta_store[key] = metas[i]
            payload_store[key] = payloads[i]
            ids[i] = np.uint32(next_id)

        idx.batch_insert(vectors, ids, self._insert_threads)

    def _delete_ids(self, index_id: int, ext_ids: List[Any]) -> int:
        if not ext_ids:
            return 0
        ext2int = self._ext2int[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        removed = 0
        for ext_id in ext_ids:
            key = self._normalize_key(ext_id)
            vec_id = ext2int.pop(key, None)
            if vec_id is None:
                continue
            self._mark_deleted(index_id, vec_id)
            int2ext.pop(vec_id, None)
            meta_store.pop(key, None)
            payload_store.pop(key, None)
            removed += 1
        return removed

    def _delete_knn(self, index_id: int, queries: np.ndarray, k: int, complexity: int) -> int:
        if queries.shape[0] == 0 or k <= 0:
            return 0
        complexity = max(k, complexity)
        idx = self._indices[index_id]
        resp_ids, _ = idx.batch_search(queries, k, complexity, self._search_threads)
        to_remove: List[str] = []
        for row in resp_ids:
            for vec_id in row.tolist():
                if vec_id == 0:
                    continue
                ext = self._int2ext[index_id].get(int(vec_id))
                if ext is None:
                    continue
                to_remove.append(ext)
        return self._delete_ids(index_id, to_remove)

    def _search(self, index_id: int, queries: np.ndarray, k: int, complexity: int) -> List[List[SearchHit]]:
        if queries.shape[0] == 0:
            return []
        idx = self._indices[index_id]
        complexity = max(k, complexity)
        resp_ids, resp_dists = idx.batch_search(queries, k, complexity, self._search_threads)
        spec = self._specs[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        results: List[List[SearchHit]] = []
        for qi in range(resp_ids.shape[0]):
            hits: List[SearchHit] = []
            ids_row = resp_ids[qi]
            dists_row = resp_dists[qi]
            for vid, dist in zip(ids_row, dists_row):
                vec_id = int(vid)
                if vec_id == 0:
                    continue
                ext_id = int2ext.get(vec_id)
                if not ext_id:
                    continue
                key = ext_id
                meta = dict(meta_store.get(key) or {})
                meta["_data"] = payload_store.get(key)
                score = self._score(spec.metric, float(dist))
                hits.append(SearchHit(id=key, score=score, metadata=meta))
            results.append(hits)
        return results

    def _mark_deleted(self, index_id: int, vec_id: int) -> None:
        idx = self._indices[index_id]
        idx.mark_deleted(np.uint32(vec_id))
        self._pending_deletes[index_id] += 1
        self._maybe_consolidate(index_id)

    def _maybe_consolidate(self, index_id: int, force: bool = False) -> None:
        if self._consolidate_every <= 0:
            return
        pending = self._pending_deletes.get(index_id, 0)
        if not force and pending < self._consolidate_every:
            return
        idx = self._indices[index_id]
        idx.consolidate_delete()
        self._pending_deletes[index_id] = 0

    def _next_vector_id(self, index_id: int) -> int:
        nxt = self._next_int_id[index_id]
        if nxt >= self._max_vectors:
            raise RuntimeError(
                f"DiskANNCppBackend: index_id {index_id} exhausted max_vectors={self._max_vectors}. "
                "Increase the capacity or rebuild the index."
            )
        self._next_int_id[index_id] = nxt + 1
        return nxt

    @staticmethod
    def _ensure_matrix(vectors: Optional[np.ndarray]) -> np.ndarray:
        if vectors is None:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.asarray(vectors, dtype=np.float32, order="C")
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    @staticmethod
    def _normalize_key(key: Any) -> str:
        if key is None:
            return "None"
        return str(key)

    def _metric_enum(self, metric: Metric) -> Any:
        if metric == Metric.L2:
            return self._native.Metric.L2
        if metric == Metric.IP:
            return self._native.Metric.INNER_PRODUCT
        return self._native.Metric.COSINE
    
    @staticmethod
    def _load_faiss_ivf(path: str | Path) -> Tuple[Metric, int, List[Tuple[np.ndarray, np.ndarray]]]:
        try:
            import faiss  # type: ignore
            from faiss.contrib.inspect_tools import get_invlist  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss is required to rebuild a DiskANN index from a Faiss file"
            ) from exc

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Faiss index file not found: {path}")

        index = faiss.read_index(str(path))

        def _unwrap_leaf(idx):
            """Drill through decorators (IDMap, PreTransform, etc.) to the concrete leaf."""
            current = faiss.downcast_index(idx)
            for _ in range(16):
                inner = None
                if hasattr(current, "index"):
                    inner = getattr(current, "index")
                elif hasattr(current, "base_index"):
                    inner = getattr(current, "base_index")
                elif hasattr(current, "sub_index"):
                    inner = getattr(current, "sub_index")
                if inner is None:
                    break
                current = faiss.downcast_index(inner)
            return current

        def _extract_ivf(idx):
            """
            Handle wrappers like IndexIDMap/IndexPreTransform that layer
            additional state on top of an IndexIVF core.
            """
            current = faiss.downcast_index(idx)
            for _ in range(16):
                try:
                    ivf_idx = faiss.extract_index_ivf(current)
                    if ivf_idx is None:
                        break
                    return faiss.downcast_index(ivf_idx)
                except RuntimeError:
                    inner = getattr(current, "index", None)
                    if inner is None:
                        break
                    current = faiss.downcast_index(inner)
            raise ValueError(
                "Provided Faiss index does not expose an IVF component "
                "(expected IndexIVFFlat or an IDMap around it)"
            )

        def _extract_flat(idx):
            """Fallback path for IndexFlat/IndexIDMap checkpoints."""
            leaf = _unwrap_leaf(idx)
            try:
                base_flat = faiss.downcast_index(leaf)
            except Exception as exc:  # pragma: no cover - unexpected leaf types
                raise ValueError("Unsupported Faiss index type") from exc
            if not isinstance(base_flat, faiss.IndexFlat):
                raise ValueError("Provided Faiss index is neither IVF nor Flat.")
            return base_flat

        def _collect_ids(idx, count: int) -> np.ndarray:
            """Try to recover user-provided IDs when wrapped in IndexIDMap."""
            current = faiss.downcast_index(idx)
            for _ in range(16):
                id_map = getattr(current, "id_map", None)
                if id_map is not None:
                    arr = faiss.vector_to_array(id_map)
                    return np.asarray(arr, dtype=np.int64)
                inner = getattr(current, "index", None)
                if inner is None:
                    break
                current = faiss.downcast_index(inner)
            return np.arange(count, dtype=np.int64)

        lists: List[Tuple[np.ndarray, np.ndarray]] = []
        ivf_metric_type: Optional[int] = None
        ivf_dim: Optional[int] = None

        try:
            ivf = _extract_ivf(index)
            ivf_metric_type = ivf.metric_type
            ivf_dim = int(ivf.d)
        except ValueError:
            ivf = None

        if ivf is not None:
            if ivf_dim is None or ivf_metric_type is None:
                raise RuntimeError("Failed to capture IVF metadata")
            invlists = faiss.downcast_InvertedLists(ivf.invlists)
            for list_id in range(ivf.nlist):
                list_ids, list_codes = get_invlist(invlists, list_id)
                if list_ids.size == 0:
                    continue
                if list_codes.dtype != np.uint8:
                    raise ValueError("Only IndexIVFFlat (float codes) is supported for now")
                vectors = list_codes.view(np.float32).reshape(list_ids.shape[0], ivf_dim)
                lists.append(
                    (
                        np.ascontiguousarray(list_ids, dtype=np.int64),
                        np.ascontiguousarray(vectors, dtype=np.float32),
                    )
                )
            metric_type = ivf_metric_type
            dim = ivf_dim
        else:
            flat = _extract_flat(index)
            metric_type = flat.metric_type
            dim = int(flat.d)
            ntotal = int(flat.ntotal)
            if ntotal == 0:
                lists = []
            else:
                if hasattr(flat, "xb"):
                    vectors = faiss.vector_to_array(flat.xb).astype(np.float32).reshape(ntotal, dim)
                else:
                    vectors = flat.reconstruct_n(0, ntotal).astype(np.float32)
                ids = _collect_ids(index, ntotal)
                lists = [
                    (
                        np.ascontiguousarray(ids, dtype=np.int64),
                        np.ascontiguousarray(vectors, dtype=np.float32),
                    )
                ]

        if metric_type == faiss.METRIC_L2:
            metric = Metric.L2
        elif metric_type == faiss.METRIC_INNER_PRODUCT:
            metric = Metric.IP
        else:
            metric = Metric.COSINE

        return metric, dim, lists

    @staticmethod
    def _score(metric: Metric, raw_distance: float) -> float:
        if metric == Metric.IP:
            return raw_distance
        return -raw_distance

    def rebuild_index_from_faiss(
        self,
        index_id: int,
        *,
        path: str,
        normalized: Optional[bool] = None,
    ) -> None:
        metric, dim, lists = self._load_faiss_ivf(path)
        total_vectors = sum(ids.shape[0] for ids, _ in lists)
        # DiskANN reserves tag 0, so capacity must be at least total_vectors + 1.
        # Add an extra buffer to accommodate forthcoming inserts after the rebuild.
        base_capacity = max(total_vectors + 1, 2)
        buffer = max(1_000_000, base_capacity // 10)  # >=1M or 10% slack
        desired_capacity = base_capacity + buffer
        if desired_capacity > self._max_vectors:
            warnings.warn(
                f"DiskANNCppBackend: increasing max_vectors from {self._max_vectors} to {desired_capacity} "
                f"(loaded {total_vectors} vectors from Faiss; reserving {buffer} extra for future inserts).",
                RuntimeWarning,
            )
            self._max_vectors = desired_capacity

        old_spec = self._specs.get(index_id)
        if old_spec is None:
            spec = CollectionSpec(name=f"index-{index_id}", dim=dim, metric=metric)
        else:
            spec = CollectionSpec(name=old_spec.name, dim=dim, metric=metric)

        # Clear any existing state for this index_id so create_index() can recreate it
        self._indices.pop(index_id, None)
        self._specs.pop(index_id, None)
        self._ext2int.pop(index_id, None)
        self._int2ext.pop(index_id, None)
        self._metas.pop(index_id, None)
        self._payloads.pop(index_id, None)
        self._next_int_id.pop(index_id, None)
        self._pending_deletes.pop(index_id, None)

        self.create_index(index_id, spec)

        all_ids: List[str] = []
        vec_chunks: List[np.ndarray] = []

        for ids_arr, vecs in lists:
            if ids_arr.size == 0:
                continue

            vecs = np.asarray(vecs, dtype="float32", order="C")

            if normalized or (normalized is None and metric == Metric.COSINE):
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                vecs = vecs / norms

            str_ids = [str(int(doc_id)) for doc_id in ids_arr.tolist()]
            all_ids.extend(str_ids)
            vec_chunks.append(vecs)

        if not all_ids:
            return

        total = len(all_ids)
        mat = np.vstack(vec_chunks).astype("float32", copy=False)

        chunk_size = 200000
        use_tqdm = tqdm is not None and total > 0

        if chunk_size <= 0 or total <= chunk_size:
            metas = [None] * total
            payloads = [None] * total
            self._upsert(
                index_id=index_id,
                ext_ids=all_ids,
                vectors=mat,
                metas=metas,
                payloads=payloads,
            )
            return

        iterator = range(0, total, chunk_size)
        if use_tqdm:
            iterator = tqdm(iterator, total=(total + chunk_size - 1) // chunk_size, desc="diskann_cpp rebuild")

        for start in iterator:
            end = min(start + chunk_size, total)
            ids_slice = all_ids[start:end]
            vec_slice = mat[start:end]
            metas = [None] * (end - start)
            payloads = [None] * (end - start)
            self._upsert(
                index_id=index_id,
                ext_ids=ids_slice,
                vectors=vec_slice,
                metas=metas,
                payloads=payloads,
            )
