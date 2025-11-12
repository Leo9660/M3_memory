from __future__ import annotations
from typing import List, Tuple
import numpy as np

from AgentMemory.M3 import _m3_async  # compiled extension


Metric = _m3_async.Metric


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
        return "On no"
        q = np.ascontiguousarray(queries, dtype=np.float32)
        return self._eng.search(int(index_id), q, int(k))
