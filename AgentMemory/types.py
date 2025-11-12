from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import numpy as np


class Metric(str, Enum):
    COSINE = "cosine"
    IP = "ip"
    L2 = "l2"


@dataclass
class CollectionSpec:
    name: str
    dim: int
    metric: Metric = Metric.COSINE


@dataclass
class MemoryItem:
    """user-side data container: can be vector/text/dict/other (will be encoded internally by backend)."""
    id: Optional[Any]
    data: Any
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchHit:
    id: str
    score: float
    metadata: Optional[Dict[str, Any]]


@dataclass
class RunResult:
    """Summary return of run(): counts of changes + search results (grouped by request_id)"""
    upserted: int
    updated: int
    deleted: int
    searches: Dict[str, List[List[SearchHit]]]

class BackendOpType(Enum):
    INSERT = auto()
    UPDATE = auto()
    DELETE_IDS = auto()
    DELETE_KNN = auto()
    FLUSH = auto()
    SEARCH = auto()

@dataclass
class BackendRequest:
    """
    A single backend operation to be executed in-order.
    Exactly one of the vector/id fields is used depending on op type.
    """
    op: BackendOpType
    index_id: int

    # Common payloads
    ext_ids: Optional[List[Any]] = None                 # for INSERT/UPDATE (keys) and DELETE_IDS
    vectors: Optional[np.ndarray] = None                # (N, D) for INSERT/UPDATE or (Q, D) for SEARCH/DELETE_KNN
    metas: Optional[List[Optional[dict]]] = None        # for INSERT/UPDATE
    payloads: Optional[List[Any]] = None                # original MemoryItem.data for INSERT/UPDATE

    # Search / Delete-KNN params
    k: Optional[int] = None
    request_id: Optional[str] = None                    # for SEARCH
    nprobe: Optional[int] = None                        # for SEARCH (backend-specific tuning)
