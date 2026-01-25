# AgentMemory/backend/m3.py
from __future__ import annotations
from typing import Any, Dict, List, Set, Optional
import hashlib
import numpy as np

from .base import MemoryBackend
from ..types import CollectionSpec, RunResult, BackendRequest, BackendOpType, SearchHit

from AgentMemory.M3 import _m3_async as m3, rebuild_from_faiss, M3MultiLevelIndex  # pybind module + loader

def _metric_enum(name: str):
    s = (name or "").lower()
    if s in ("l2", "euclidean"):
        return m3.Metric.L2
    if s in ("ip", "inner_product", "dot"):
        return m3.Metric.IP
    if s in ("cos", "cosine"):
        return m3.Metric.COSINE
    raise ValueError(f"Unsupported metric: {name!r}")


def _as_f32_2d(arr, err: str):
    if arr is None:
        raise ValueError(err)
    a = np.asarray(arr, dtype=np.float32, order="C")
    if a.ndim != 2:
        raise ValueError(err)
    return a


def _as_int64_1d(arr, err: str):
    if arr is None:
        raise ValueError(err)
    a = np.asarray(arr, dtype=np.int64, order="C")
    if a.ndim != 1:
        raise ValueError(err)
    return a


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
        
        # Store metadata and data for each index: index_id -> {int64_id -> {metadata, data, ext_id}}
        self._meta: Dict[int, Dict[int, Optional[Dict[str, Any]]]] = {}
        self._data: Dict[int, Dict[int, Any]] = {}
        self._int2ext: Dict[int, Dict[int, Any]] = {}  # int64_id -> ext_id

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
        metric = _metric_enum(getattr(spec, "metric", "l2"))
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
        
        # Initialize storage for this index
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

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
                # Store metadata and data
                if idx not in self._int2ext:
                    self._int2ext[idx] = {}
                    self._meta[idx] = {}
                    self._data[idx] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx][int(int_id)] = ext_id
                    self._meta[idx][int(int_id)] = meta
                    self._data[idx][int(int_id)] = payload
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = self._as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                # 没有自动分配的 update，先沿用 cluster 0
                self._eng.enqueue_update(idx, 0, ids, vecs, True)
                # Update metadata and data
                if idx not in self._int2ext:
                    self._int2ext[idx] = {}
                    self._meta[idx] = {}
                    self._data[idx] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx][int(int_id)] = ext_id
                    if meta is not None:
                        self._meta[idx][int(int_id)] = meta
                    if payload is not None:
                        self._data[idx][int(int_id)] = payload
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                self._eng.enqueue_delete(idx, 0, ids)
                # Remove from storage
                for int_id in ids:
                    self._meta.get(idx, {}).pop(int(int_id), None)
                    self._data.get(idx, {}).pop(int(int_id), None)
                    self._int2ext.get(idx, {}).pop(int(int_id), None)
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
                    hits = []
                    for doc_id, score in zip(ids_list, scores_list):
                        int_id = int(doc_id)
                        # Get external ID (use int_id as fallback)
                        ext_id = self._int2ext.get(idx, {}).get(int_id, str(int_id))
                        # Build metadata dict
                        base_meta = self._meta.get(idx, {}).get(int_id) or {}
                        meta = dict(base_meta) if base_meta else {}
                        # Add original data to metadata
                        if idx in self._data and int_id in self._data[idx]:
                            meta["_data"] = self._data[idx][int_id]
                        # Use external ID for the hit ID
                        doc_id_str = str(ext_id) if ext_id is not None else str(int_id)
                        # Note: score is distance (smaller is better for cosine/L2, larger is better for IP)
                        hits.append(SearchHit(
                            id=doc_id_str,
                            score=float(score),
                            metadata=meta if meta else None
                        ))
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
    def _as_f32_2d(arr, err: str):
        return _as_f32_2d(arr, err)

    @staticmethod
    def _as_int64_1d(arr, err: str):
        return _as_int64_1d(arr, err)

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


class M3MultiLevelBackend(MemoryBackend):
    """
    Simple synchronous backend backed by MultiLevelIndex (no async writers).
    """

    def __init__(self) -> None:
        super().__init__()
        self._indices: Dict[int, M3MultiLevelIndex] = {}
        
        # Store metadata and data for each index: index_id -> {int64_id -> {metadata, data, ext_id}}
        self._meta: Dict[int, Dict[int, Optional[Dict[str, Any]]]] = {}
        self._data: Dict[int, Dict[int, Any]] = {}
        self._int2ext: Dict[int, Dict[int, Any]] = {}  # int64_id -> ext_id

    def close(self) -> None:
        self._indices.clear()

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        if index_id in self._indices:
            return
        dim = int(spec.dim)
        metric = _metric_enum(getattr(spec, "metric", "l2"))
        normalized = (metric == m3.Metric.COSINE)

        params = getattr(spec, "params", {}) or {}
        cfg_kwargs = {
            "l0_nlist": int(params.get("l0_nlist", 1)),
            "l1_nlist": int(params.get("l1_nlist", 1)),
            "l2_nlist": int(params.get("l2_nlist", 1)),
            "l0_new_cluster_threshold": float(params.get("l0_new_cluster_threshold", float("inf"))),
            "search_threshold": float(params.get("search_threshold", float("inf"))),
            "l0_merge_threshold": float(params.get("l0_merge_threshold", float("inf"))),
            "l0_max_nlist": int(params.get("l0_max_nlist", 0)),
        }
        idx = M3MultiLevelIndex(dim=dim, metric=metric, normalized=normalized, **cfg_kwargs)

        # seed L0 centroids if provided; else zero centroid
        centroids = params.get("centroids")
        if centroids is None:
            centroids = np.zeros((cfg_kwargs["l0_nlist"], dim), dtype=np.float32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[1] != dim:
            raise ValueError("centroids must be [nlist, dim]")
        idx.set_l0_centroids(centroids)

        self._indices[index_id] = idx
        
        # Initialize storage for this index
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        insert_cnt = 0
        update_cnt = 0
        delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            idx_id = int(op.index_id)
            if idx_id not in self._indices:
                raise KeyError(f"M3MultiLevelBackend: index_id {idx_id} not found. Call create_index() first.")
            idx = self._indices[idx_id]

            if op.op == BackendOpType.INSERT:
                ids = self._keys_to_int64(op.ext_ids, "INSERT requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "INSERT requires 2D 'vectors'")
                idx.insert(ids, vecs)
                # Store metadata and data
                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    self._meta[idx_id][int(int_id)] = meta
                    self._data[idx_id][int(int_id)] = payload
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                idx.update(ids, vecs, insert_if_absent=True)
                # Update metadata and data
                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    if meta is not None:
                        self._meta[idx_id][int(int_id)] = meta
                    if payload is not None:
                        self._data[idx_id][int(int_id)] = payload
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                idx.erase(ids)
                # Remove from storage
                for int_id in ids:
                    self._meta.get(idx_id, {}).pop(int(int_id), None)
                    self._data.get(idx_id, {}).pop(int(int_id), None)
                    self._int2ext.get(idx_id, {}).pop(int(int_id), None)
                delete_cnt += len(ids)

            elif op.op == BackendOpType.FLUSH:
                # synchronous: nothing to do
                continue

            elif op.op == BackendOpType.SEARCH:
                queries = _as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
                k = int(op.k or 1)
                nprobe = int(op.nprobe or 32)

                out_ids, out_scores = idx.search(queries, k, nprobe)
                rid = op.request_id or f"req-{len(search_payload)}"
                hits_per_query: List[List[SearchHit]] = []
                for ids_list, scores_list in zip(out_ids, out_scores):
                    hits = []
                    for doc_id, score in zip(ids_list, scores_list):
                        int_id = int(doc_id)
                        # Get external ID (use int_id as fallback)
                        ext_id = self._int2ext.get(idx_id, {}).get(int_id, str(int_id))
                        # Build metadata dict
                        base_meta = self._meta.get(idx_id, {}).get(int_id) or {}
                        meta = dict(base_meta) if base_meta else {}
                        # Add original data to metadata
                        if idx_id in self._data and int_id in self._data[idx_id]:
                            meta["_data"] = self._data[idx_id][int_id]
                        # Use external ID for the hit ID
                        doc_id_str = str(ext_id) if ext_id is not None else str(int_id)
                        # Note: score is distance (smaller is better for cosine/L2, larger is better for IP)
                        hits.append(SearchHit(
                            id=doc_id_str,
                            score=float(score),
                            metadata=meta if meta else None
                        ))
                    hits_per_query.append(hits)
                search_payload[rid] = hits_per_query

            else:
                raise NotImplementedError(f"Unsupported op: {op.op}")

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        raise NotImplementedError("M3MultiLevelBackend does not support rebuild_index_from_faiss yet")

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
