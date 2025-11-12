from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from pathlib import Path

from ..types import CollectionSpec, SearchHit, RunResult, Metric, BackendOpType, BackendRequest
from .base import MemoryBackend


class PlaceholderBackend(MemoryBackend):
    """
    Minimal in-memory backend with serialized execution.
    - Receives already-encoded vectors via BackendRequest (no encoding here).
    - Executes requests strictly in-order.
    - Stores BOTH vectors and original payloads (MemoryItem.data) per ext_id.
      SearchHit.metadata includes merged metadata plus {"_data": original_payload}.
    - Supports: INSERT, UPDATE, DELETE_IDS, DELETE_KNN, SEARCH.
    """

    def __init__(self) -> None:
        super().__init__()
        print("[backend] init placeholder")

        # index_id -> CollectionSpec
        self._specs: Dict[int, CollectionSpec] = {}

        # index_id -> (ids, vectors, metas, payloads)
        #   ids:       List[str]
        #   vectors:   np.ndarray shape (N, dim)
        #   metas:     List[Optional[dict]]
        #   payloads:  List[Any]
        self._store: Dict[int, Tuple[List[str], np.ndarray, List[Optional[dict]], List[Any]]] = {}

    # ---------- index management ----------
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        """Create/prepare an empty index with the given spec."""
        self._specs[index_id] = spec
        if index_id not in self._store:
            self._store[index_id] = (
                [],  # ids
                np.zeros((0, spec.dim), dtype="float32"),  # vectors
                [],  # metas
                [],  # payloads
            )
        print(f"  - created index_id={index_id} name='{spec.name}' (dim={spec.dim}, metric={spec.metric})")

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        """
        Load all vectors from a Faiss flat or IVF-flat index and store them in-memory.
        """
        metric, dim, ids, vectors = self._load_faiss_vectors(path)

        spec = self._specs.get(index_id)
        if spec is None or spec.dim != dim:
            spec = CollectionSpec(name=f"index-{index_id}", dim=dim, metric=metric)
            self.create_index(index_id, spec)

        metas = [None] * len(ids)
        payloads = [None] * len(ids)
        self._store[index_id] = (ids, vectors.astype(np.float32, copy=False), metas, payloads)
        print(f"[backend] placeholder rebuilt index_id={index_id} from Faiss ({len(ids)} vectors)")

    @staticmethod
    def _load_faiss_vectors(path: str) -> Tuple[Metric, int, List[str], np.ndarray]:
        try:
            import faiss  # type: ignore
            from faiss.contrib.inspect_tools import get_invlist  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss is required to rebuild the placeholder backend from a Faiss index") from exc

        index = faiss.read_index(str(Path(path)))
        metric_type = getattr(index, "metric_type", faiss.METRIC_INNER_PRODUCT)
        if metric_type == faiss.METRIC_L2:
            metric = Metric.L2
        elif metric_type == faiss.METRIC_INNER_PRODUCT:
            metric = Metric.IP
        else:
            metric = Metric.COSINE

        dim = int(index.d)
        ids: List[str] = []
        vectors: List[np.ndarray] = []

        if hasattr(index, "xb"):
            xb = faiss.vector_float_to_array(index.xb)
            if xb.size:
                vectors.append(xb.reshape(-1, dim).astype(np.float32))
            try:
                id_arr = faiss.vector_idx_t_to_array(index.id_map)
                ids = [str(int(i)) for i in id_arr]
            except Exception:
                ids = [str(i) for i in range(vectors[0].shape[0] if vectors else 0)]
        else:
            ivf = faiss.extract_index_ivf(index)
            if ivf is None:
                raise ValueError("Unsupported Faiss index type for placeholder rebuild (expected flat or IVF-Flat)")
            invlists = faiss.downcast_InvertedLists(ivf.invlists)
            for list_id in range(ivf.nlist):
                list_ids, list_codes = get_invlist(invlists, list_id)
                if list_ids.size == 0:
                    continue
                if list_codes.dtype != np.uint8:
                    raise ValueError("Only IVF-Flat (float) codes are supported for placeholder rebuilds")
                vecs = list_codes.view(np.float32).reshape(list_ids.shape[0], dim)
                vectors.append(vecs.astype(np.float32))
                ids.extend([str(int(doc_id)) for doc_id in list_ids.tolist()])

        mat = np.vstack(vectors) if vectors else np.zeros((0, dim), dtype=np.float32)
        if not ids:
            ids = [str(i) for i in range(mat.shape[0])]
        return metric, dim, ids, mat

    # ---------- execution ----------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        """
        Execute all backend requests strictly in the given order.
        """
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_results: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            if req.op == BackendOpType.INSERT:
                n = 0 if req.ext_ids is None else len(req.ext_ids)
                self._insert(
                    req.index_id,
                    req.ext_ids or [],
                    self._ensure_mat(req.vectors),
                    req.metas or [None] * n,
                    req.payloads or [None] * n,
                )
                ins_cnt += n

            elif req.op == BackendOpType.UPDATE:
                n = 0 if req.ext_ids is None else len(req.ext_ids)
                self._update(
                    req.index_id,
                    req.ext_ids or [],
                    self._ensure_mat(req.vectors),
                    req.metas or [None] * n,
                    req.payloads or [None] * n,
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
                rid = req.request_id or "req-unknown"
                hits = self._search(req.index_id, Q, k)
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

    # ---------- internal ops (mutate in-memory store) ----------
    def _insert(
        self,
        index_id: int,
        ids: List[str],
        vectors: np.ndarray,
        metas: List[Optional[dict]],
        payloads: List[Any],
    ) -> None:
        print(f"[backend] insert index_id={index_id} n={len(ids)}")
        id_list, mat, meta_list, data_list = self._store[index_id]
        id_to_index = {i: idx for idx, i in enumerate(id_list)}

        for i, vid in enumerate(ids):
            if vid in id_to_index:
                j = id_to_index[vid]
                mat[j] = vectors[i]
                meta_list[j] = metas[i]
                data_list[j] = payloads[i]
            else:
                id_list.append(vid)
                mat = np.vstack([mat, vectors[i:i + 1]])
                meta_list.append(metas[i])
                data_list.append(payloads[i])

        self._store[index_id] = (id_list, mat, meta_list, data_list)

    def _update(
        self,
        index_id: int,
        ids: List[str],
        vectors: np.ndarray,
        metas: List[Optional[dict]],
        payloads: List[Any],
    ) -> None:
        print(f"[backend] update index_id={index_id} n={len(ids)}")
        id_list, mat, meta_list, data_list = self._store[index_id]
        id_to_index = {i: idx for idx, i in enumerate(id_list)}

        for i, vid in enumerate(ids):
            if vid not in id_to_index:
                print(f"  [WARN] update skipped: id not found '{vid}'")
                continue
            j = id_to_index[vid]
            mat[j] = vectors[i]
            meta_list[j] = metas[i]
            data_list[j] = payloads[i]

        self._store[index_id] = (id_list, mat, meta_list, data_list)

    def _delete_ids(self, index_id: int, ids: List[str]) -> None:
        print(f"[backend] delete_ids index_id={index_id} n={len(ids)}")
        id_list, mat, meta_list, data_list = self._store[index_id]
        to_remove: Set[str] = set(ids)
        keep_idx = [i for i, vid in enumerate(id_list) if vid not in to_remove]
        removed = len(id_list) - len(keep_idx)

        self._store[index_id] = (
            [id_list[i] for i in keep_idx],
            mat[keep_idx],
            [meta_list[i] for i in keep_idx],
            [data_list[i] for i in keep_idx],
        )
        print(f"  - removed {removed}")

    def _delete_knn(self, index_id: int, Q: np.ndarray, k: int) -> int:
        """
        Delete the k nearest vectors for each query; duplicates are removed once.
        Returns the number of unique deletions performed.
        """
        print(f"[backend] delete_knn index_id={index_id} q={len(Q)} k={k}")
        id_list, mat, meta_list, data_list = self._store[index_id]
        if len(id_list) == 0 or k <= 0 or Q.shape[0] == 0:
            return 0

        spec = self._specs[index_id]
        S = self._score(spec, Q, mat)
        k_eff = min(k, S.shape[1]) if S.shape[1] > 0 else 0
        if k_eff == 0:
            return 0

        # Collect unique indices to remove
        to_remove_idx: Set[int] = set()
        topk = np.argpartition(-S, kth=k_eff - 1, axis=1)[:, :k_eff]
        for qi in range(S.shape[0]):
            idxs = topk[qi]
            # Order by score (desc) to be deterministic
            idxs = idxs[np.argsort(-S[qi, idxs])]
            for j in idxs:
                to_remove_idx.add(int(j))

        keep_idx = [i for i in range(len(id_list)) if i not in to_remove_idx]
        removed = len(id_list) - len(keep_idx)

        self._store[index_id] = (
            [id_list[i] for i in keep_idx],
            mat[keep_idx],
            [meta_list[i] for i in keep_idx],
            [data_list[i] for i in keep_idx],
        )
        print(f"  - removed {removed} by knn")
        return removed

    def _search(self, index_id: int, Q: np.ndarray, k: int) -> List[List[SearchHit]]:
        print(f"[backend] search index_id={index_id} q={len(Q)} k={k}")
        id_list, mat, meta_list, data_list = self._store[index_id]
        if len(id_list) == 0 or Q.shape[0] == 0:
            return [[] for _ in range(Q.shape[0])]

        spec = self._specs[index_id]
        S = self._score(spec, Q, mat)

        k_eff = min(k, S.shape[1]) if S.shape[1] > 0 else 0
        if k_eff == 0:
            return [[] for _ in range(Q.shape[0])]

        topk = np.argpartition(-S, kth=k_eff - 1, axis=1)[:, :k_eff]
        results: List[List[SearchHit]] = []
        for qi in range(S.shape[0]):
            idxs = topk[qi]
            idxs = idxs[np.argsort(-S[qi, idxs])]
            hits: List[SearchHit] = []
            for j in idxs:
                base_meta = meta_list[j] or {}
                meta = dict(base_meta)
                meta["_data"] = data_list[j]
                hits.append(SearchHit(id=id_list[j], score=float(S[qi, j]), metadata=meta))
            results.append(hits)
        return results

    # ---------- scoring ----------
    @staticmethod
    def _score(spec: CollectionSpec, Q: np.ndarray, M: np.ndarray) -> np.ndarray:
        if M.shape[0] == 0:
            return np.zeros((Q.shape[0], 0), dtype=np.float32)
        if spec.metric == Metric.COSINE:
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            return Qn @ Mn.T
        if spec.metric == Metric.IP:
            return Q @ M.T
        # L2: negative distance as score
        D = np.linalg.norm(Q[:, None, :] - M[None, :, :], axis=-1)
        return -D
