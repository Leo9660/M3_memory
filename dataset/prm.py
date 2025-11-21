"""Loader for sl-alex/openai-prm800k-stepwise-best."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping

from .base import DatasetBase

try:  # pragma: no cover - datasets only needed at runtime
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "PRMStepwiseDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class PRMStepwiseDataset(DatasetBase):
    """Groups PRM800k stepwise records into Agent/Request traces."""

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
                "sl-alex/openai-prm800k-stepwise-best",
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:  # pragma: no cover - forwards HF errors
            raise RuntimeError(
                "Failed to load sl-alex/openai-prm800k-stepwise-best "
                f"split '{self.split}': {exc}"
            ) from exc

    def _build(self) -> Iterable[Mapping[str, Any]]:
        dataset = self._load_raw_split()
        requests_by_instruction: Dict[str, List[Mapping[str, str]]] = OrderedDict()
        request_order: List[str] = []

        for entry in dataset:
            instruction = (entry.get("instruction") or "").strip()
            response = (entry.get("next_response") or "").strip()
            if not instruction or not response:
                continue

            if instruction not in requests_by_instruction:
                if self.limit is not None and len(request_order) >= self.limit:
                    # Skip unseen instructions once limit reached but keep
                    # accepting additional steps for existing requests.
                    continue
                requests_by_instruction[instruction] = []
                request_order.append(instruction)

            lines = [
                "agent role: You are a mathematician who excels at advanced mathematical reasoning tasks.",
                f"question: {instruction}",
                f"response: {response}",
            ]
            answer = entry.get("answer")
            if isinstance(answer, str):
                answer = answer.strip()
            if answer and str(answer).lower() != "none":
                lines.append(f"answer: {answer}")

            requests_by_instruction[instruction].append({"text": "\n".join(lines)})

        requests: List[Mapping[str, Any]] = []
        for instruction in request_order:
            trace = requests_by_instruction.get(instruction, [])
            if not trace:
                continue
            requests.append(
                {
                    "request_index": len(requests),
                    "instruction": instruction,
                    "trace": trace,
                }
            )

        if not requests:
            return []

        return [
            {
                "agent": "sl-alex/openai-prm800k-stepwise-best",
                "requests": requests,
            }
        ]
