"""
Loader for thesven/gsm8k-reasoning Chain-of-Thought traces.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping, Sequence

from .base import DatasetBase

try:  # pragma: no cover - datasets only needed at runtime
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "GSM8KReasoningDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class GSM8KReasoningDataset(DatasetBase):
    """
    Treats thesven/gsm8k-reasoning as a single agent where every record becomes a
    request comprised of multiple memory items extracted from the ``generation``
    field.
    """

    DATASET_NAME = "thesven/gsm8k-reasoning"
    AGENT_ROLE = (
        "You are an AI assistant that uses a Chain of Thought (CoT) approach "
        "with reflection to answer queries."
    )
    TAG_PATTERN = re.compile(
        r"<(?P<tag>[a-zA-Z0-9_]+)>(?P<body>.*?)<[/\\](?P=tag)>",
        re.DOTALL,
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
            question = self._extract_first_nonempty(
                entry,
                ("question", "prompt", "query", "problem"),
            )
            final_answer = self._extract_first_nonempty(entry, ("answer",))
            generation = self._extract_first_nonempty(entry, ("generation",))

            if not question or not generation:
                continue

            trace = self._generation_to_trace(
                generation,
                question,
                final_answer,
            )
            if not trace:
                continue

            requests.append(
                {
                    "request_index": len(requests),
                    "question": question,
                    "answer": final_answer,
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

    def _generation_to_trace(
        self,
        generation: str,
        question: str,
        final_answer: str | None,
    ) -> List[Mapping[str, str]]:
        trace: List[Mapping[str, str]] = []
        for tag, body in self._iter_tagged_sections(generation):
            body_text = body.strip()
            if not body_text:
                continue

            lines = [
                f"agent role: {self.AGENT_ROLE}",
                f"question: {question}",
                f"answer: [{tag}] {body_text}",
            ]
            if tag.lower() == "output" and final_answer:
                lines.append(f"ground truth answer: {final_answer}")

            trace.append(
                {
                    "tag": tag,
                    "text": "\n".join(lines),
                }
            )

        return trace

    @classmethod
    def _iter_tagged_sections(cls, generation: str) -> Sequence[tuple[str, str]]:
        sections: List[tuple[str, str]] = []
        for match in cls.TAG_PATTERN.finditer(generation):
            tag = (match.group("tag") or "").strip()
            body = match.group("body") or ""
            if not tag:
                continue
            sections.append((tag, body))
        return sections

    @staticmethod
    def _extract_first_nonempty(
        entry: Mapping[str, Any],
        keys: Sequence[str],
    ) -> str | None:
        for key in keys:
            value = entry.get(key)
            if isinstance(value, str):
                value = value.strip()
            elif value is not None:
                value = str(value).strip()
            if value:
                return value
        return None
