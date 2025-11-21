"""
Common dataset helpers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Mapping, Sequence


class DatasetBase(ABC):
    """
    Minimal dataset interface that caches the processed samples and exposes a
    uniform ``get_data`` accessor for downstream evaluation code.
    """

    def __init__(self, split: str = "train", cache_dir: str | None = None):
        self.split = split
        self.cache_dir = cache_dir
        self._data: List[Mapping[str, Any]] | None = None

    @abstractmethod
    def _build(self) -> Iterable[Mapping[str, Any]]:
        """
        Implementations must yield processed rows that downstream consumers can
        iterate over.
        """

    def get_data(self) -> Sequence[Mapping[str, Any]]:
        """
        Returns the processed dataset. Results are cached after the first call.
        """

        if self._data is None:
            self._data = list(self._build())
        return self._data
