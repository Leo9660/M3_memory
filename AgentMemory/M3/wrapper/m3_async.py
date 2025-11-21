from __future__ import annotations
from typing import List, Tuple
import numpy as np

from AgentMemory.M3 import _m3_async  # compiled extension


Metric = _m3_async.Metric
MultiLevelConfig = _m3_async.MultiLevelConfig
_MultiLevelIndex = _m3_async.MultiLevelIndex


class M3AsyncEngine:
    """
    Thin Python wrapper around the C++ AsyncEngine (pthread).
    - IDs are int64 for speed (no string hashing/alloc).
    - Enqueue ops are NON-BLOCKING by default (block=False).
    """

    def __init__(self, *, queue_capacity: int = 1024, autostart: bool = True) -> None:
        self._eng = _m3_async.AsyncEngine()
        if queue_capacity:
            self._eng.set_queue_capacity(int(queue_capacity))
        if autostart:
            self._eng.start()

    # Lifecycle
    def start(self) -> None:
        self._eng.start()

    def stop(self) -> None:
        self._eng.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    # Index APIs
    def create_index(self, index_id: int, dim: int, metric: Metric, normalized: bool = True) -> None:
        self._eng.create_index(int(index_id), int(dim), metric, bool(normalized))

    # Write-side enqueue (NON-BLOCKING by default)
    def enqueue_insert(
        self,
        index_id: int,
        ids: np.ndarray,            # int64 [N]
        vectors: np.ndarray,        # float32 [N, D]
        *,
        block: bool = False,
    ) -> bool:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        return bool(self._eng.enqueue_insert(int(index_id), ids64, vecs, bool(block)))

    def enqueue_update(
        self,
        index_id: int,
        ids: np.ndarray,            # int64 [N]
        vectors: np.ndarray,        # float32 [N, D]
        *,
        block: bool = False,
    ) -> bool:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        return bool(self._eng.enqueue_update(int(index_id), ids64, vecs, bool(block)))

    def enqueue_delete(
        self,
        index_id: int,
        ids: np.ndarray,            # int64 [M]
        *,
        block: bool = False,
    ) -> bool:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        return bool(self._eng.enqueue_delete(int(index_id), ids64, bool(block)))

    def enqueue_insert_auto(
        self,
        index_id: int,
        ids: np.ndarray,
        vectors: np.ndarray,
    ) -> bool:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        return bool(self._eng.enqueue_insert_auto(int(index_id), ids64, vecs))

    def flush(self) -> None:
        self._eng.flush()

    # Search (immediate; runs under RW read lock)
    def search(
        self,
        index_id: int,
        queries: np.ndarray,        # float32 [Q, D]
        k: int
    ) -> Tuple[List[List[int]], List[List[float]]]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return self._eng.search(int(index_id), q, int(k))


class M3MultiLevelIndex:
    def __init__(
        self,
        dim: int,
        metric: Metric,
        *,
        normalized: bool = True,
        l0_nlist: int = 1,
        l1_nlist: int = 1,
        l2_nlist: int = 1,
        l0_new_cluster_threshold: float = float("inf"),
        search_threshold: float = float("inf"),
        l0_merge_threshold: float = float("inf"),
        l0_max_nlist: int | None = None,
    ) -> None:
        cfg = MultiLevelConfig()
        cfg.l0_nlist = int(l0_nlist)
        cfg.l1_nlist = int(l1_nlist)
        cfg.l2_nlist = int(l2_nlist)
        if hasattr(cfg, "l0_new_cluster_threshold"):
            cfg.l0_new_cluster_threshold = float(l0_new_cluster_threshold)
        if hasattr(cfg, "search_threshold"):
            cfg.search_threshold = float(search_threshold)
        if hasattr(cfg, "l0_merge_threshold"):
            cfg.l0_merge_threshold = float(l0_merge_threshold)
        if hasattr(cfg, "l0_max_nlist"):
            cfg.l0_max_nlist = int(l0_max_nlist) if l0_max_nlist is not None else 0
        self._idx = _MultiLevelIndex(int(dim), metric, bool(normalized), cfg)

    def set_l0_centroids(self, centroids: np.ndarray) -> None:
        c = np.ascontiguousarray(centroids, dtype=np.float32)
        self._idx.set_l0_centroids(c)

    def set_l1_centroids(self, centroids: np.ndarray) -> None:
        c = np.ascontiguousarray(centroids, dtype=np.float32)
        self._idx.set_l1_centroids(c)

    def set_l2_centroids(self, centroids: np.ndarray) -> None:
        c = np.ascontiguousarray(centroids, dtype=np.float32)
        self._idx.set_l2_centroids(c)

    def insert(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._idx.insert(ids64, vecs)

    def update(self, ids: np.ndarray, vectors: np.ndarray, *, insert_if_absent: bool = False) -> None:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._idx.update(ids64, vecs, bool(insert_if_absent))

    def erase(self, ids: np.ndarray) -> None:
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        self._idx.erase(ids64)

    def search(
        self,
        queries: np.ndarray,
        k: int,
        nprobe: int = -1,
    ) -> Tuple[List[List[int]], List[List[float]]]:
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return self._idx.search(q, int(k), int(nprobe))

    def maintenance_pass(self) -> None:
        self._idx.maintenance_pass()
