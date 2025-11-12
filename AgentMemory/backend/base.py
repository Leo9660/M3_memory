from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from ..types import CollectionSpec, RunResult, BackendRequest, BackendOpType
from ..encoder import MemoryEncoder

class MemoryBackend(ABC):
    """
    Minimal backend interface:
      - create_index(index_id, spec)
      - execute(ops): run a serialized list of BackendRequest in-order and return RunResult

    Encoding model:
      - Interface encodes all MemoryItems at run() time. Backend receives ready-to-use vectors/ids.
      - 'encoder' is kept only for legacy compatibility, can be ignored by new backends.
    """

    def __init__(self) -> None:
        self.encoder: Optional[MemoryEncoder] = None

    def set_encoder(self, encoder: MemoryEncoder) -> None:
        """Optional legacy injection; new backends can ignore."""
        self.encoder = encoder

    @abstractmethod
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        """Create/prepare a vector index identified by a numeric index_id."""

    @abstractmethod
    def execute(self, ops: list[BackendRequest]) -> RunResult:
        """
        Execute all backend requests strictly in the given order and return a RunResult.
        Backends should be deterministic with respect to the provided sequence.
        """
        ...
