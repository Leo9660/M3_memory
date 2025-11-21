"""
Loader for HuggingFaceH4/ultrachat_200k conversations.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence

from .base import DatasetBase

try:  # pragma: no cover - datasets only needed at runtime
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "UltraChatDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class UltraChatDataset(DatasetBase):
    """
    Treats the entire UltraChat split as a single agent where each conversation
    entry becomes a request containing user/assistant turn pairs.
    """

    AGENT_NAME = "HuggingFaceH4/ultrachat_200k"
    AGENT_ROLE = (
        "You are a helpful chat robot, you need to follow the instructions "
        "and solve the problems."
    )

    def __init__(
        self,
        split: str = "train_sft",
        cache_dir: str | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(split=split, cache_dir=cache_dir)
        self.limit = limit

    def _load_raw_split(self):
        try:
            return load_dataset(
                self.AGENT_NAME,
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:  # pragma: no cover - forwards HF errors
            raise RuntimeError(
                f"Failed to load {self.AGENT_NAME} split '{self.split}': {exc}"
            ) from exc

    def _build(self) -> Iterable[Mapping[str, Any]]:
        dataset = self._load_raw_split()
        requests: List[Mapping[str, Any]] = []

        for entry in dataset:
            messages = entry.get("messages")
            initial_prompt = (entry.get("prompt") or "").strip()
            trace = self._messages_to_trace(messages, initial_prompt)
            if not trace:
                continue

            requests.append(
                {
                    "request_index": len(requests),
                    "trace": trace,
                }
            )

            if self.limit is not None and len(requests) >= self.limit:
                break

        if not requests:
            return []

        return [
            {
                "agent": self.AGENT_NAME,
                "agent_role": self.AGENT_ROLE,
                "requests": requests,
            }
        ]

    def _messages_to_trace(
        self,
        messages: Sequence[Mapping[str, Any]] | None,
        initial_prompt: str,
    ) -> List[Mapping[str, str]]:
        if not isinstance(messages, Sequence):
            return []

        trace: List[Mapping[str, str]] = []
        pending_user: str | None = None
        prompt_text = initial_prompt.strip()

        for turn in messages:
            role = (turn.get("role") or "").strip().lower()
            content = (turn.get("content") or "").strip()
            if not content:
                continue

            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                trace.append(
                    {
                        "user": pending_user,
                        "assistant": content,
                        "text": (
                            "agent role: "
                            f"{self.AGENT_ROLE}\n initial question: {prompt_text}\n "
                            f"user question: {pending_user}\n answer: {content}"
                        ),
                    }
                )
                pending_user = None

        return trace
