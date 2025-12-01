"""
Loader for openbmb/UltraFeedback entries.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping, Sequence

from .base import DatasetBase

try:  # pragma: no cover - datasets only needed at runtime
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "UltraFeedbackDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class UltraFeedbackDataset(DatasetBase):
    """
    Treats each openbmb/UltraFeedback record as a request where every model
    completion produces six memory items: an answer, a critique, and four
    evaluation scores.
    """

    DATASET_NAME = "openbmb/UltraFeedback"
    SOLVE_ROLE = "You are a useful agent to solve user questions."
    CRITIQUE_ROLE = "You are a agent to criticize the response to a given question."
    EVAL_ROLE_TEMPLATE = (
        "You are a agent to evaluate the {metric} of a given response."
    )
    EVAL_KEYS = (
        "helpfulness",
        "honesty",
        "instruction_following",
        "truthfulness",
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
            instruction = self._stringify(entry.get("instruction"))
            models = entry.get("models")
            completions = entry.get("completions")

            if not instruction or not isinstance(models, Sequence) or not isinstance(
                completions, Sequence
            ):
                continue

            items = self._build_items(instruction, models, completions)
            if not items:
                continue

            requests.append(
                {
                    "request_index": len(requests),
                    "instruction": instruction,
                    "trace": items,
                }
            )

            if self.limit is not None and len(requests) >= self.limit:
                break

        if not requests:
            return []

        return [
            {
                "agent": self.DATASET_NAME,
                "requests": requests,
            }
        ]

    def _build_items(
        self,
        instruction: str,
        models: Sequence[Any],
        completions: Sequence[Any],
    ) -> List[Mapping[str, str]]:
        items: List[Mapping[str, str]] = []
        count = min(len(models), len(completions))

        for idx in range(count):
            model_name = self._stringify(models[idx])
            completion = completions[idx] if isinstance(completions[idx], Mapping) else {}

            response = self._stringify(completion.get("response"))
            critique = self._stringify(completion.get("critique"))
            overall_score = self._stringify(completion.get("overall_score"))
            annotations = completion.get("annotations") if isinstance(completion.get("annotations"), Mapping) else {}

            if not model_name or not response:
                continue

            items.append(
                {
                    "text": (
                        f"agent role: {self.SOLVE_ROLE} "
                        f"question: {instruction} model: {model_name} "
                        f"answer: {response}"
                    )
                }
            )

            items.append(
                {
                    "text": (
                        f"agent role: {self.CRITIQUE_ROLE} "
                        f"critique: {critique}, "
                        f"overall_score: {overall_score}"
                    )
                }
            )

            for key in self.EVAL_KEYS:
                value = self._stringify(annotations.get(key))
                eval_role = self.EVAL_ROLE_TEMPLATE.format(metric=key)
                items.append(
                    {
                        "text": (
                            f"agent role: {eval_role} "
                            f"evaluation: {value}"
                        )
                    }
                )

        return items

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value).strip()
