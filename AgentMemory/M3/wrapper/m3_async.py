from __future__ import annotations
from typing import List, Tuple
import numpy as np

from AgentMemory.M3 import _m3_async  # compiled extension


Metric = _m3_async.Metric
MultiLevelConfig = _m3_async.MultiLevelConfig
CacheConfig = getattr(_m3_async, "CacheConfig", None)  # optional
_MultiLevelIndex  = _m3_async.MultiLevelIndex
_GpuCoordinator   = _m3_async.GpuCoordinator


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

    def set_cache_config(self, cache_config: "CacheConfig") -> None:
        if CacheConfig is not None and hasattr(self._idx, "set_cache_config"):
            self._idx.set_cache_config(cache_config)

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

    def load_cluster(self, cluster_id: int, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Bulk-load directly into L2 cluster. Call set_l2_centroids first."""
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)
        self._idx.load_cluster(int(cluster_id), ids64, vecs)

    def set_gpu_coordinator(self, coordinator: "GpuCoordinator | None") -> None:
        """Wire (or unwire) a GpuCoordinator so GPU-resident clusters are used for search/insert."""
        if coordinator is None:
            self._idx.set_gpu_coordinator(None)
        else:
            self._idx.set_gpu_coordinator(coordinator._coord)


class GpuCoordinator:
    """
    Python wrapper for the C++ GpuCoordinator.

    Orchestrates GPU hotspot caching on top of a MultiLevelIndex:
      - Promotes hot clusters from L2 into VRAM (GPU-resident).
      - Routes inserts for GPU-resident clusters to a CPU insert buffer
        (async H2D flush to GPU + L2 durability write via background thread).
      - Routes searches for GPU-resident clusters to GPU distance kernels.
      - Rebalances: promotes hotter non-GPU clusters, evicts coldest GPU ones.

    Lifecycle:
      coord = GpuCoordinator(idx, gpu_budget_bytes=2 * 1024**3, dim=768, metric=Metric.L2)
      idx.set_gpu_coordinator(coord)
      coord.start_background()
      ...
      coord.stop_background()
      idx.set_gpu_coordinator(None)
    """

    def __init__(
        self,
        idx: M3MultiLevelIndex,
        gpu_budget_bytes: int,
        dim: int,
        metric: Metric,
        *,
        normalized: bool = False,
        insert_buf_cap: int = 128,
    ) -> None:
        # Keep a Python reference to idx so GC cannot collect it while we hold a C++ ref.
        self._idx_ref = idx
        self._coord = _GpuCoordinator(
            idx._idx,
            int(gpu_budget_bytes),
            int(dim),
            metric,
            bool(normalized),
            int(insert_buf_cap),
        )

    def promote_to_gpu(self, cid: int) -> bool:
        return self._coord.promote_to_gpu(int(cid))

    def enqueue_promote(self, cid: int) -> None:
        self._coord.enqueue_promote(int(cid))

    def enqueue_demote(self, cid: int) -> None:
        self._coord.enqueue_demote(int(cid))

    def drain_pending(self) -> None:
        self._coord.drain_pending()

    def flush_buffers(self) -> int:
        return self._coord.flush_buffers()

    def rebalance(self) -> int:
        return self._coord.rebalance()

    def start_background(
        self,
        flush_ms: int = 50,
        maintenance_ms: int = 5000,
        rebalance_ms: int = 500,
    ) -> None:
        self._coord.start_background(int(flush_ms), int(maintenance_ms), int(rebalance_ms))

    def stop_background(self) -> None:
        self._coord.stop_background()

    def is_gpu_resident(self, cid: int) -> bool:
        return self._coord.is_gpu_resident(int(cid))

    def gpu_bytes_used(self) -> int:
        return self._coord.gpu_bytes_used()

    def gpu_budget_bytes(self) -> int:
        return self._coord.gpu_budget_bytes()

    def gpu_resident_cids(self) -> list:
        return self._coord.gpu_resident_cids()

    def background_running(self) -> bool:
        return self._coord.background_running()

    def __del__(self) -> None:
        try:
            self._coord.stop_background()
        except Exception:
            pass
