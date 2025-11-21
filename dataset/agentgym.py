"""
AgentGym dataset loader that pairs human and GPT turns.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .base import DatasetBase

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at runtime when missing dep
    raise ImportError(
        "AgentGymDataset depends on the `datasets` package. "
        "Install it with `pip install datasets`."
    ) from exc


class AgentGymDataset(DatasetBase):
    """
    Streams AgentGym/AgentTraj-L and groups conversations into agents that contain
    multiple requests. Each request stores the ordered trace of Human/GPT pairs
    formatted as ``human question: ...`` / ``gpt answer: ...`` items, plus an
    ``agent_role`` summary built from the first human turn.
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: str | None = None,
        limit: int | None = None,
    ):
        super().__init__(split=split, cache_dir=cache_dir)
        self.limit = limit

    def _build(self) -> Iterable[Mapping[str, Any]]:
        dataset = self._load_raw_split()
        agents: Dict[str, Dict[str, Any]] = {}

        for i, item in enumerate(dataset):
            # if i < 10:
            #     print("DEBUG sample", i)
            conversation = self._extract_conversation(item)
            if not conversation:
                continue

            trace = self._conversation_to_trace(conversation)
            agent_key = self._agent_key_from_conversation(conversation)

            if not trace or not agent_key:
                continue

            agent_bucket = agents.setdefault(
                agent_key,
                {"agent": agent_key, "requests": []},
            )

            if self.limit is not None and len(agent_bucket["requests"]) >= self.limit:
                continue

            agent_bucket["requests"].append(
                {
                    "request_index": len(agent_bucket["requests"]),
                    "trace": trace,
                }
            )

            # if i < 10:
            #     print("DEBUG processed sample", i, "agent_key:", agent_key, "trace len:", len(trace))

        for agent_key, agent_data in agents.items():
            if not agent_data["requests"]:
                continue
            first_request = agent_data["requests"][0]
            # print(f"Agent prefix: {agent_key!r}")
            # for idx, item in enumerate(first_request["trace"]):
            #     print(f"  Item {idx}: {item.get('text')}")

        return list(agents.values())

    def _load_raw_split(self):
        try:
            return load_dataset(
                "AgentGym/AgentTraj-L",
                split=self.split,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:  # pragma: no cover - forwards HF errors
            raise RuntimeError(
                f"Failed to load AgentGym/AgentTraj-L split '{self.split}': {exc}"
            ) from exc

    @staticmethod
    def _extract_conversation(item: Mapping[str, Sequence[Mapping[str, str]]]):
        conversation = (
            item.get("conversation")
            or item.get("conversations")
            or item.get("messages")
        )
        if isinstance(conversation, list):
            return conversation
        return None

    @staticmethod
    def _conversation_to_trace(
        conversation: Sequence[Mapping[str, str]],
    ) -> List[Mapping[str, str]]:
        samples: List[Mapping[str, str]] = []
        pending_human: str | None = None

        agent_role = ""
        for turn in conversation:
            speaker = (turn.get("from") or "").lower()
            text = (turn.get("value") or "").strip()

            if not text:
                continue

            if speaker == "human":
                pending_human = text
                if not agent_role:
                    agent_role = text
            elif speaker == "gpt" and pending_human:
                samples.append(
                    {
                        "human": pending_human,
                        "gpt": text,
                        "text": (
                            f"agent role: {AgentGymDataset._summarize_role(agent_role)}\n"
                            f"human question: {pending_human}\n"
                            f"gpt answer: {text}"
                        ),
                    }
                )
                pending_human = None

        return samples

    @staticmethod
    def _agent_key_from_conversation(
        conversation: Sequence[Mapping[str, str]],
        max_words: int = 20,
    ) -> str | None:
        for turn in conversation:
            speaker = (turn.get("from") or "").lower()
            if speaker != "human":
                continue

            text = (turn.get("value") or "").strip()
            if not text:
                continue

            words = text.split()
            if not words:
                continue

            return " ".join(words[:max_words])

        return None

    def _summarize_role(text: str, max_sentences: int = 1) -> str:
        if not text:
            return ""
        sentences = re.split(r"(?<=[.!?。！？])\s+", text)
        collected: List[str] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            collected.append(sentence)
            if len(collected) >= max_sentences:
                break
        if not collected:
            return text
        return " ".join(collected)
