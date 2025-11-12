# AgentMemory/backend/m3.py
from __future__ import annotations
from typing import Any, Dict, List, Set, Optional
import hashlib
import numpy as np

from .base import MemoryBackend
from ..types import CollectionSpec, RunResult, BackendRequest, BackendOpType, SearchHit

from AgentMemory.M3 import _m3_async as m3, rebuild_from_faiss  # pybind module + loader


class M3Backend(MemoryBackend):
    """
    M3 backend that matches the current C++ async engine:

    - engine 支持多个 index_id
    - 每个 index 是一个 IVFIndex
    - 写操作需要 (index_id, cluster_id)，我们当前先全写 cluster 0
    - search 是按 index_id 查
    """

    def __init__(
        self,
        autostart: bool = True,
        writer_threads: int = 2,
        maintenance_threads: int = 1,
    ) -> None:
        super().__init__()
        self._eng = m3.AsyncEngine()
        self._autostart = autostart
        self._started = False

        self._writer_threads = writer_threads
        self._maintenance_threads = maintenance_threads

        # 记录已经创建过的 index_id
        self._indices: Set[int] = set()

    # ---------- lifecycle ----------

    def _ensure_started(self) -> None:
        if not self._started:
            self._eng.start(self._writer_threads, self._maintenance_threads)
            self._started = True

    def close(self) -> None:
        if self._started:
            self._eng.flush()
            self._eng.stop()
            self._started = False

    # ---------- create index (real IVF) ----------

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        """
        真正创建一个 IVF index:
        - 传下去 index_id
        - 必须有 dim / metric
        - centroids: 从 spec.params['centroids'] 取，如果没有，就建一个全 0 的 [1, dim]，表示只有 cluster 0
        """
        if self._autostart:
            self._ensure_started()

        if index_id in self._indices:
            return

        dim = int(spec.dim)
        metric = self._to_metric(getattr(spec, "metric", "l2"))
        normalized = (metric == m3.Metric.COSINE)

        centroids = None
        params = getattr(spec, "params", None)
        if params is not None:
            centroids = params.get("centroids")

        if centroids is None:
            # 默认 1 个 cluster，质心全 0
            centroids = np.zeros((1, dim), dtype=np.float32)
        else:
            centroids = np.ascontiguousarray(centroids, dtype=np.float32)
            if centroids.ndim != 2 or centroids.shape[1] != dim:
                raise ValueError("centroids must be [nlist, dim]")

        # 真正创建这个 index
        self._eng.create_ivf(index_id, dim, metric, normalized, centroids)
        self._indices.add(index_id)

    # ---------- main execute ----------

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        if self._autostart:
            self._ensure_started()

        insert_cnt = 0
        update_cnt = 0
        delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            idx = int(op.index_id)
            if idx not in self._indices:
                raise KeyError(f"M3Backend: index_id {idx} not found. Call create_index() first.")

            if op.op == BackendOpType.INSERT:
                ids = self._keys_to_int64(op.ext_ids, "INSERT requires 'ext_ids'")
                vecs = self._as_f32_2d(op.vectors, "INSERT requires 2D 'vectors'")
                self._eng.enqueue_insert_auto(idx, ids, vecs)
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = self._as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                # 没有自动分配的 update，先沿用 cluster 0
                self._eng.enqueue_update(idx, 0, ids, vecs, True)
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                self._eng.enqueue_delete(idx, 0, ids)
                delete_cnt += len(ids)

            elif op.op == BackendOpType.FLUSH:
                self._eng.flush()

            elif op.op == BackendOpType.SEARCH:
                # 默认还是先 flush，这样语义跟你前面说的一致
                flush = True
                if flush:
                    self._eng.flush()

                queries = self._as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
                k = int(op.k or 1)
                nprobe = int(op.nprobe or 32)

                out_ids, out_scores = self._eng.search(idx, queries, k, nprobe)

                rid = op.request_id or f"req-{len(search_payload)}"
                query_hits: List[List[SearchHit]] = []
                for ids_list, scores_list in zip(out_ids, out_scores):
                    hits = [
                        SearchHit(id=str(doc_id), score=float(score), metadata=None)
                        for doc_id, score in zip(ids_list, scores_list)
                    ]
                    query_hits.append(hits)
                search_payload[rid] = query_hits

            else:
                raise NotImplementedError(f"Unsupported op: {op.op}")

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    # ---------- rebuild from faiss ----------

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        if self._autostart:
            self._ensure_started()

        rebuild_from_faiss(
            self._eng,
            index_id=index_id,
            path=path,
            normalized=bool(normalized) if normalized is not None else True,
        )
        self._indices.add(index_id)

    # ---------- helpers ----------

    @staticmethod
    def _to_metric(name: str):
        s = (name or "").lower()
        if s in ("l2", "euclidean"):
            return m3.Metric.L2
        if s in ("ip", "inner_product", "dot"):
            return m3.Metric.IP
        if s in ("cos", "cosine"):
            return m3.Metric.COSINE
        raise ValueError(f"Unsupported metric: {name!r}")

    @staticmethod
    def _as_f32_2d(arr, err: str):
        if arr is None:
            raise ValueError(err)
        a = np.asarray(arr, dtype=np.float32, order="C")
        if a.ndim != 2:
            raise ValueError(err)
        return a

    @staticmethod
    def _as_int64_1d(arr, err: str):
        if arr is None:
            raise ValueError(err)
        a = np.asarray(arr, dtype=np.int64, order="C")
        if a.ndim != 1:
            raise ValueError(err)
        return a

    @staticmethod
    def _keys_to_int64(keys, err: str):
        if keys is None:
            raise ValueError(err)
        out = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            if key is None:
                h = hashlib.blake2b(str(i).encode("utf-8"), digest_size=8).digest()
                out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
            else:
                try:
                    out[i] = int(key)
                except (ValueError, TypeError):
                    h = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
                    out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
        return out
