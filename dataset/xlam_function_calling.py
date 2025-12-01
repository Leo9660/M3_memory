"""
Loader for Salesforce/xlam-function-calling-60k entries.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping

from .base import DatasetBase

try:  # pragma: no cover - datasets only needed at runtime
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "XLAMFunctionCallingDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class XLAMFunctionCallingDataset(DatasetBase):
    """
    Treats each record from Salesforce/xlam-function-calling-60k as a distinct
    request and emits three memory items per entry: the agent role plus query,
    the tool specification, and the tool call answer.
    """

    DATASET_NAME = "Salesforce/xlam-function-calling-60k"
    AGENT_ROLE = (
        "You are a helpful agent that can use API calls to solve problems. You need to select the best tool, read the specification and generate the target function call."
    )

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(split=split, cache_dir=cache_dir)
        self.limit = limit

    def _load_raw_split(self):
        try:
            return load_dataset(
                self.DATASET_NAME,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:  # pragma: no cover - forwards HF errors
            raise RuntimeError(
                f"Failed to load {self.DATASET_NAME} split '{self.split}': {exc}"
            ) from exc

    def _build(self) -> Iterable[Mapping[str, Any]]:
        dataset = self._load_raw_split()
        requests: List[Mapping[str, Any]] = []

        for entry in dataset:
            query = self._stringify(entry.get("query"))
            tools = self._stringify(entry.get("tools"))
            answers = self._stringify(entry.get("answers"))

            if not query or not tools or not answers:
                continue

            trace = [
                {
                    "text": (
                        f"agent role: {self.AGENT_ROLE} "
                        f"question: {query}"
                    )
                },
                {"text": f"agent role: You are a useful agent to use different tools. Following is the specification to an agent tool you need to use: {tools}"},
                {"text": f"agent role: You are a useful agent to generate tool calls. Agent tool call. answer: {answers}"},
            ]

            requests.append(
                {
                    "request_index": len(requests),
                    "query": query,
                    "tools": tools,
                    "answers": answers,
                    "trace": trace,
                }
            )

            if self.limit is not None and len(requests) >= self.limit:
                break

        if not requests:
            return []

        return [
            {
                "agent": self.DATASET_NAME,
                "agent_role": self.AGENT_ROLE,
                "requests": requests,
            }
        ]

    @staticmethod
    def _stringify(value: Any) -> str:
        """
        Converts a dataset field to a compact string, tolerating lists/dicts.
        """

        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value).strip()
