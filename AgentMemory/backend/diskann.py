from __future__ import annotations

from pathlib import Path
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

try:  # pragma: no cover - import guard mirrors other optional backends
    import diskannpy as dap
except Exception as exc:  # pragma: no cover - surfaced during backend init
    dap = None
    _DISKANN_IMPORT_ERROR = exc


class DiskANNBackend(MemoryBackend):
    """
    diskannpy-powered backend that maintains a DynamicMemoryIndex per AgentMemory collection.

    The interface layer still performs single-pass encoding and request serialization; this backend
    focuses on replaying INSERT/UPDATE/DELETE/SEARCH operations in order against diskannpy.
    """

    def __init__(
        self,
        *,
        index_directory: Optional[str] = None,
        max_vectors: int = 1_000_000,
        graph_degree: int = 48,
        build_complexity: int = 128,
        search_complexity: int = 64,
        insert_threads: int = 0,
        search_threads: int = 0,
        auto_consolidate_every: int = 10_000,
    ) -> None:
        super().__init__()
        if dap is None:
            raise RuntimeError(
                "diskannpy is required for DiskANNBackend. Install it via `pip install diskannpy`. "
                f"Original import error: {repr(_DISKANN_IMPORT_ERROR)}"
            )

        self._index_root = Path(index_directory or ".diskann")
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
        self._indices: Dict[int, dap.DynamicMemoryIndex] = {}
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

        metric = self._metric_name(spec.metric)
        dim = int(spec.dim)
        index_dir = self._index_root / f"index_{index_id}"
        index_dir.mkdir(parents=True, exist_ok=True)

        idx = dap.DynamicMemoryIndex(
            distance_metric=metric,
            vector_dtype=np.float32,
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
            raise KeyError(f"DiskANNBackend: index_id {index_id} not initialized")
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        idx.save(str(out_dir), index_prefix=prefix)

    # ---------- execution ----------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            idx = self._indices.get(req.index_id)
            if idx is None:
                raise KeyError(f"DiskANNBackend: index_id {req.index_id} not found. Call create_index() first.")

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
                    complexity=int(req.nprobe or self._default_search_complexity),
                )
                del_cnt += removed

            elif req.op == BackendOpType.SEARCH:
                queries = self._ensure_matrix(req.vectors)
                k = int(req.k or 0)
                if k <= 0 or queries.shape[0] == 0:
                    rid = req.request_id or f"req-{len(search_payload)}"
                    search_payload[rid] = [[] for _ in range(queries.shape[0])]
                    continue
                complexity = int(req.nprobe or self._default_search_complexity)
                hits = self._search(req.index_id, queries, k, complexity)
                rid = req.request_id or f"req-{len(search_payload)}"
                search_payload[rid] = hits

            elif req.op == BackendOpType.FLUSH:
                self._maybe_consolidate(req.index_id, force=True)

            else:
                raise NotImplementedError(f"DiskANNBackend does not support op={req.op}")

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
            raise ValueError("DiskANNBackend: vectors/ext_ids length mismatch")

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

        idx.batch_insert(vectors, ids, num_threads=self._insert_threads)

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
        resp = idx.batch_search(queries, k_neighbors=k, complexity=complexity, num_threads=self._search_threads)
        to_remove: List[str] = []
        for ids in resp.identifiers:
            for vec_id in ids.tolist():
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
        resp = idx.batch_search(queries, k_neighbors=k, complexity=complexity, num_threads=self._search_threads)
        spec = self._specs[index_id]
        int2ext = self._int2ext[index_id]
        meta_store = self._metas[index_id]
        payload_store = self._payloads[index_id]

        results: List[List[SearchHit]] = []
        for qi in range(resp.identifiers.shape[0]):
            hits: List[SearchHit] = []
            for vid, dist in zip(resp.identifiers[qi], resp.distances[qi]):
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
                f"DiskANNBackend: index_id {index_id} exhausted max_vectors={self._max_vectors}. "
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

    @staticmethod
    def _metric_name(metric: Metric) -> str:
        if metric == Metric.L2:
            return "l2"
        if metric == Metric.IP:
            return "mips"
        return "cosine"

    @staticmethod
    def _score(metric: Metric, raw_distance: float) -> float:
        if metric == Metric.IP:
            return raw_distance
        return -raw_distance
