from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
import hashlib
import numpy as np
import torch

from ..types import CollectionSpec, SearchHit, RunResult, Metric, BackendRequest, BackendOpType
from .base import MemoryBackend

# ---- optional import guard for quake ----
try:
    from quake import SearchParams
    from quake.index_wrappers.quake import QuakeWrapper
except Exception as e:
    _QUAKE_IMPORT_ERROR = e
    QuakeWrapper = None
    SearchParams = None


class QuakeBackend(MemoryBackend):
    """
    Quake-powered backend that executes a serialized list of BackendRequest.
    - Interface layer handles encoding and queuing; this backend only executes in order:
        INSERT, UPDATE, DELETE_IDS, DELETE_KNN, SEARCH
    - Maintains ext<->int mapping and metadata/original payload; search results return in metadata['_data'].
    - Lazy build: build on first insert/update, then incremental add/remove.
    """

    def __init__(self, *, default_cluster_size: int = 256, default_num_threads: int = 16) -> None:
        super().__init__()
        if QuakeWrapper is None:
            raise RuntimeError(
                "Quake is not available. Please install it (e.g., `pip install quake-vector`). "
                f"Original import error: {repr(_QUAKE_IMPORT_ERROR)}"
            )
        print("[backend] init quake")

        # Index registry
        self._specs: Dict[int, CollectionSpec] = {}
        self._indices: Dict[int, QuakeWrapper] = {}
        self._built: Dict[int, bool] = {}

        # External<->internal id mapping per index
        self._ext2int: Dict[int, Dict[str, int]] = {}
        self._int2ext: Dict[int, Dict[int, str]] = {}
        self._next_int_id: Dict[int, int] = {}
        # Metadata & payload (raw MemoryItem.data)
        self._meta: Dict[int, Dict[str, Optional[dict]]] = {}
        self._data: Dict[int, Dict[str, Any]] = {}

        # For initial build / rebuild: ext_id -> vector
        self._all_vectors: Dict[int, Dict[str, np.ndarray]] = {}

        # Defaults
        self._default_cluster_size = int(default_cluster_size)
        self._default_num_threads = int(default_num_threads)

    # ---------- index management ----------
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        self._specs[index_id] = spec
        self._indices[index_id] = QuakeWrapper()  # not built yet
        self._built[index_id] = False

        self._ext2int[index_id] = {}
        self._int2ext[index_id] = {}
        self._next_int_id[index_id] = 0
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._all_vectors[index_id] = {}

        print(f"  - created quake index_id={index_id} name='{spec.name}' (dim={spec.dim}, metric={spec.metric})")

    # ---------- execution ----------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_results: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            if req.op == BackendOpType.INSERT:
                n = len(req.ext_ids or [])
                self._insert(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    vectors=self._ensure_mat(req.vectors),
                    metas=req.metas or [None] * n,
                    payloads=req.payloads or [None] * n,
                )
                ins_cnt += n

            elif req.op == BackendOpType.UPDATE:
                n = len(req.ext_ids or [])
                self._update(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    vectors=self._ensure_mat(req.vectors),
                    metas=req.metas or [None] * n,
                    payloads=req.payloads or [None] * n,
                )
                upd_cnt += n

            elif req.op == BackendOpType.DELETE_IDS:
                ids = req.ext_ids or []
                self._delete_ids(req.index_id, ids)
                del_cnt += len(ids)

            elif req.op == BackendOpType.DELETE_KNN:
                Q = self._ensure_mat(req.vectors)
                k = int(req.k or 0)
                removed = self._delete_knn(req.index_id, Q, k)
                del_cnt += removed

            elif req.op == BackendOpType.SEARCH:
                Q = self._ensure_mat(req.vectors)
                k = int(req.k or 0)
                nprobe = int(req.nprobe or 32)
                rid = req.request_id or "req-unknown"
                hits = self._search(req.index_id, Q, k, nprobe)
                search_results[rid] = hits

            else:
                print(f"[WARN] unknown backend op skipped: {req.op}")

        return RunResult(
            upserted=ins_cnt,
            updated=upd_cnt,
            deleted=del_cnt,
            searches=search_results,
        )

    # ---------- helpers ----------
    @staticmethod
    def _ensure_mat(v: Optional[np.ndarray]) -> np.ndarray:
        if v is None:
            return np.zeros((0, 0), dtype="float32")
        if v.ndim == 1:
            return v.reshape(1, -1).astype("float32")
        return v.astype("float32")

    def _metric_str(self, spec: CollectionSpec) -> str:
        if spec.metric == Metric.L2:
            return "l2"
        if spec.metric == Metric.IP:
            return "ip"
        # cosine → usually encode as normalized vectors and use inner product
        return "ip"

    @staticmethod
    def _load_faiss_ivf(path: str | Path) -> Tuple[Metric, int, List[Tuple[np.ndarray, np.ndarray]]]:
        try:
            import faiss  # type: ignore
            from faiss.contrib.inspect_tools import get_invlist  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "faiss is required to rebuild a Quake index from a Faiss file"
            ) from exc

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Faiss index file not found: {path}")

        index = faiss.read_index(str(path))
        ivf = faiss.extract_index_ivf(index)
        if ivf is None:
            raise ValueError("Provided index does not contain an IVF component")
        ivf = faiss.downcast_index(ivf)

        if ivf.metric_type == faiss.METRIC_L2:
            metric = Metric.L2
        elif ivf.metric_type == faiss.METRIC_INNER_PRODUCT:
            metric = Metric.IP
        else:
            metric = Metric.COSINE

        dim = int(ivf.d)
        lists: List[Tuple[np.ndarray, np.ndarray]] = []
        invlists = faiss.downcast_InvertedLists(ivf.invlists)
        for list_id in range(ivf.nlist):
            list_ids, list_codes = get_invlist(invlists, list_id)
            if list_ids.size == 0:
                continue
            if list_codes.dtype != np.uint8:
                raise ValueError("Only IndexIVFFlat (float codes) is supported for now")
            vectors = list_codes.view(np.float32).reshape(list_ids.shape[0], dim)
            lists.append(
                (
                    np.ascontiguousarray(list_ids, dtype=np.int64),
                    np.ascontiguousarray(vectors, dtype=np.float32),
                )
            )
        return metric, dim, lists

    def _ensure_built(self, index_id: int) -> None:
        """Build underlying Quake index if not built yet (lazy build using all known vectors)."""
        if self._built.get(index_id, False):
            return

        spec = self._specs[index_id]
        all_items = self._all_vectors[index_id]  # ext_id -> np.ndarray
        index = self._indices[index_id]

        if not all_items:
            # Build empty index with 1 cluster
            index.build(
                vectors=torch.zeros((0, spec.dim), dtype=torch.float32),
                nc=1,
                metric=self._metric_str(spec),
                ids=torch.zeros((0,), dtype=torch.int64),
            )
            self._built[index_id] = True
            return

        ext_ids = list(all_items.keys())
        int_ids = []
        vecs = []
        for ext_id in ext_ids:
            if ext_id not in self._ext2int[index_id]:
                internal = self._next_int_id[index_id]
                self._next_int_id[index_id] += 1
                self._ext2int[index_id][ext_id] = internal
                self._int2ext[index_id][internal] = ext_id
            int_ids.append(self._ext2int[index_id][ext_id])
            vecs.append(all_items[ext_id])

        V = torch.from_numpy(np.stack(vecs).astype("float32"))
        I = torch.tensor(int_ids, dtype=torch.int64)

        n_clusters = max(1, int(max(1, V.shape[0] // self._default_cluster_size)))
        index.build(vectors=V, nc=n_clusters, metric=self._metric_str(spec), ids=I)
        self._built[index_id] = True

    # ---------- core ops ----------
    def _insert(
        self,
        index_id: int,
        ext_ids: List[str],
        vectors: np.ndarray,
        metas: List[Optional[dict]],
        payloads: List[Any],
    ) -> None:
        print(f"[quake] insert index_id={index_id} n={len(ext_ids)}")
        # Update local stores & id mappings
        for i, ext_id in enumerate(ext_ids):
            self._all_vectors[index_id][ext_id] = vectors[i]
            self._meta[index_id][ext_id] = metas[i]
            self._data[index_id][ext_id] = payloads[i]
            if ext_id not in self._ext2int[index_id]:
                internal = self._next_int_id[index_id]
                self._next_int_id[index_id] += 1
                self._ext2int[index_id][ext_id] = internal
                self._int2ext[index_id][internal] = ext_id

        # Ensure index exists, then incremental add
        self._ensure_built(index_id)
        V = torch.from_numpy(vectors.astype("float32"))
        I = torch.tensor([self._ext2int[index_id][e] for e in ext_ids], dtype=torch.int64)
        self._indices[index_id].add(V, ids=I, num_threads=self._default_num_threads)

    def _update(
        self,
        index_id: int,
        ext_ids: List[str],
        vectors: np.ndarray,
        metas: List[Optional[dict]],
        payloads: List[Any],
    ) -> None:
        print(f"[quake] update index_id={index_id} n={len(ext_ids)}")
        self._ensure_built(index_id)

        to_add_ids: List[str] = []
        to_add_vecs: List[np.ndarray] = []
        to_remove_internal: List[int] = []

        # Update local stores first
        for i, ext_id in enumerate(ext_ids):
            vec = vectors[i]
            self._all_vectors[index_id][ext_id] = vec
            self._meta[index_id][ext_id] = metas[i]
            self._data[index_id][ext_id] = payloads[i]

            if ext_id in self._ext2int[index_id]:
                internal = self._ext2int[index_id][ext_id]
                to_remove_internal.append(internal)
            else:
                internal = self._next_int_id[index_id]
                self._next_int_id[index_id] += 1
                self._ext2int[index_id][ext_id] = internal
                self._int2ext[index_id][internal] = ext_id
            to_add_ids.append(ext_id)
            to_add_vecs.append(vec)

        if to_remove_internal:
            self._indices[index_id].remove(torch.tensor(to_remove_internal, dtype=torch.int64))

        if to_add_ids:
            V = torch.from_numpy(np.stack(to_add_vecs).astype("float32"))
            I = torch.tensor([self._ext2int[index_id][e] for e in to_add_ids], dtype=torch.int64)
            self._indices[index_id].add(V, ids=I, num_threads=self._default_num_threads)

    def _delete_ids(self, index_id: int, ext_ids: List[str]) -> None:
        print(f"[quake] delete_ids index_id={index_id} n={len(ext_ids)}")
        self._ensure_built(index_id)

        to_remove_internal: List[int] = []
        for ext_id in ext_ids:
            if ext_id in self._ext2int[index_id]:
                internal = self._ext2int[index_id][ext_id]
                to_remove_internal.append(internal)
                # remove from local stores
                self._all_vectors[index_id].pop(ext_id, None)
                self._meta[index_id].pop(ext_id, None)
                self._data[index_id].pop(ext_id, None)
            else:
                print(f"  [WARN] delete skipped: id not found '{ext_id}'")

        if to_remove_internal:
            self._indices[index_id].remove(torch.tensor(to_remove_internal, dtype=torch.int64))

    def _delete_knn(self, index_id: int, Q: np.ndarray, k: int) -> int:
        """
        Delete the k nearest vectors for each query; duplicates are removed once.
        Returns the number of unique deletions performed.
        """
        print(f"[quake] delete_knn index_id={index_id} q={len(Q)} k={k}")
        if not self._built.get(index_id, False) or k <= 0 or Q.shape[0] == 0:
            return 0

        index = self._indices[index_id]
        to_remove_internal: Set[int] = set()

        # Quake supports batching; for simplicity, we do per-query search here (can be changed to batched if needed).
        for qi in range(Q.shape[0]):
            q = torch.from_numpy(Q[qi:qi+1].astype("float32"))
            sr = index.search(q, **{"k": int(k), "batched_scan": True})
            pred_ids = sr.ids.flatten().to(torch.int64).cpu().tolist()
            for pid in pred_ids:
                to_remove_internal.add(int(pid))

        if not to_remove_internal:
            return 0

        keep_ext: List[str] = []
        remove_internal_list = sorted(list(to_remove_internal))
        # count the number of unique deletions performed, and clean up local caches
        for pid in remove_internal_list:
            ext = self._int2ext[index_id].get(pid)
            if ext is not None:
                self._all_vectors[index_id].pop(ext, None)
                self._meta[index_id].pop(ext, None)
                self._data[index_id].pop(ext, None)

        self._indices[index_id].remove(torch.tensor(remove_internal_list, dtype=torch.int64))
        return len(remove_internal_list)

    def _search(self, index_id: int, Q: np.ndarray, k: int, nprobe: int) -> List[List[SearchHit]]:
        print(f"[quake] search index_id={index_id} q={len(Q)} k={k} nprobe={nprobe}")
        if not self._built.get(index_id, False) or Q.shape[0] == 0:
            return [[] for _ in range(Q.shape[0])]

        index = self._indices[index_id]
        results: List[List[SearchHit]] = []

        for qi in range(Q.shape[0]):
            q = torch.from_numpy(Q[qi:qi+1].astype("float32"))
            sr = index.search(q, **{"k": int(k), "nprobe": int(nprobe), "batched_scan": True})
            pred_ids = sr.ids.flatten().to(torch.int64).cpu().tolist()
            distances = sr.distances.flatten().cpu().tolist()

            hits: List[SearchHit] = []
            for pid, dist in zip(pred_ids, distances):
                ext_id = self._int2ext[index_id].get(int(pid))
                if ext_id is None:
                    continue
                base_meta = self._meta[index_id].get(ext_id) or {}
                meta = dict(base_meta)
                meta["_data"] = self._data[index_id].get(ext_id)
                # Quake distance is directly used as score; for "higher is better", L2 can be negated externally
                doc_id = self._doc_id_from_ext(ext_id)
                hits.append(SearchHit(id=doc_id, score=float(dist), metadata=meta))
            results.append(hits)

        return results

    # ---------- rebuild from faiss ----------

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        metric, dim, lists = self._load_faiss_ivf(path)

        spec = self._specs.get(index_id)
        if spec is None:
            spec = CollectionSpec(name=f"index-{index_id}", dim=dim, metric=metric)
        else:
            spec = CollectionSpec(name=spec.name, dim=dim, metric=metric)

        # reset state
        self.create_index(index_id, spec)

        all_ids: List[str] = []
        vec_chunks: List[np.ndarray] = []
        for ids_arr, vecs in lists:
            if ids_arr.size == 0:
                continue
            str_ids = [str(int(doc_id)) for doc_id in ids_arr.tolist()]
            all_ids.extend(str_ids)
            vec_chunks.append(vecs)

        if all_ids:
            mat = np.vstack(vec_chunks).astype("float32", copy=False)
            metas = [None] * len(all_ids)
            payloads = [None] * len(all_ids)
            self._insert(index_id, all_ids, mat, metas, payloads)
        else:
            self._ensure_built(index_id)
    @staticmethod
    def _doc_id_from_ext(ext_id: Optional[str]) -> str:
        if ext_id is None:
            ext_id = ""
        try:
            return str(int(ext_id))
        except (ValueError, TypeError):
            h = hashlib.blake2b(str(ext_id).encode("utf-8"), digest_size=8).digest()
            return str(int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff)
