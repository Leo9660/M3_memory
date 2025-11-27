from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from letta_client import Letta
from letta_client.types import EmbeddingConfig, LlmConfig

from .base import MemoryBackend
from ..types import BackendOpType, BackendRequest, CollectionSpec, RunResult, SearchHit


class LettaBackend(MemoryBackend):
    """
    Backend that writes/queries Letta archival memory via the public HTTP API.

    Each AgentMemory index maps to a Letta agent identified by agent_id. When no agent id is
    provided, the backend can optionally auto-create a lightweight agent through the API so
    users can smoke-test the integration without pre-seeded data.
    """

    def __init__(
        self,
        *,
        client: Optional[Letta] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        project: Optional[str] = None,
        default_agent_id: Optional[str] = None,
        agent_map: Optional[Dict[str, str]] = None,
        auto_create_agent: bool = True,
        embed_queries: bool = True,
    ) -> None:
        super().__init__()
        self._auto_create_agent = bool(auto_create_agent)
        self._embed_queries = bool(embed_queries)
        self._index_cfg: Dict[int, Dict[str, Any]] = {}

        base_url = base_url or os.getenv("LETTA_BASE_URL") or "http://localhost:8283"
        token = token or os.getenv("LETTA_API_KEY") or os.getenv("LETTA_TOKEN")
        timeout = timeout or float(os.getenv("LETTA_TIMEOUT", "60"))
        project = project or os.getenv("LETTA_PROJECT_ID")

        self._client = client or Letta(
            base_url=base_url,
            token=token,
            timeout=timeout,
            project=project,
        )

        if default_agent_id is None:
            default_agent_id = os.getenv("LETTA_DEFAULT_AGENT_ID")
        self._default_agent_id = default_agent_id
        self._agent_map = agent_map or {}

    # ---------- helpers ----------
    def _ensure_context(self, index_id: int, spec: CollectionSpec) -> Dict[str, Any]:
        cfg = self._index_cfg.setdefault(
            index_id,
            {
                "spec": spec,
                "ext_to_remote": {},
                "remote_to_ext": {},
                "doc_store": {},
            },
        )

        agent_id = cfg.get("agent_id") or self._agent_map.get(spec.name) or self._default_agent_id
        if not agent_id:
            if not self._auto_create_agent:
                raise ValueError(f"LettaBackend: agent_id not configured for index '{spec.name}'")
            agent_id = self._create_ephemeral_agent(spec.name)
            self._agent_map[spec.name] = agent_id
            self._default_agent_id = agent_id
        cfg["agent_id"] = agent_id

        if "agent_state" not in cfg:
            cfg["agent_state"] = self._client.agents.retrieve(agent_id=agent_id)

        return cfg

    def _create_ephemeral_agent(self, handle: str) -> str:
        """
        Create a minimal Letta agent via the HTTP API so the backend can run without pre-existing data.
        """

        name = f"agentmemory-{handle}-{uuid4().hex[:6]}"
        llm_cfg = LlmConfig(
            model="gpt-4o-mini",
            model_endpoint_type="openai",
            model_endpoint="https://api.openai.com/v1",
            context_window=128000,
        )
        emb_cfg = EmbeddingConfig(
            embedding_endpoint_type="openai",
            embedding_endpoint="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        )
        state = self._client.agents.create(
            name=name,
            llm_config=llm_cfg,
            embedding_config=emb_cfg,
            include_base_tools=False,
            include_multi_agent_tools=False,
            include_default_source=False,
            include_base_tool_rules=False,
        )
        return state.id

    def _doc_store(self, cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return cfg.setdefault("doc_store", {})

    def _ext_to_remote(self, cfg: Dict[str, Any]) -> Dict[str, str]:
        return cfg.setdefault("ext_to_remote", {})

    def _remote_to_ext(self, cfg: Dict[str, Any]) -> Dict[str, str]:
        return cfg.setdefault("remote_to_ext", {})

    def _tag_list(self, meta: Any) -> Optional[List[str]]:
        if isinstance(meta, dict):
            tags = meta.get("tags")
            if isinstance(tags, list):
                return [str(t) for t in tags]
        return None

    def _normalize_meta(self, meta: Any) -> Dict[str, Any]:
        if isinstance(meta, dict):
            return dict(meta)
        return {}

    def _record_document(self, cfg: Dict[str, Any], key: str, text: str, meta: Dict[str, Any], remote_id: Optional[str]) -> None:
        doc_store = self._doc_store(cfg)
        doc_store[key] = {"text": text, "meta": meta, "remote_id": remote_id}
        ext_to_remote = self._ext_to_remote(cfg)
        remote_to_ext = self._remote_to_ext(cfg)
        if remote_id:
            ext_to_remote[key] = remote_id
            remote_to_ext[remote_id] = key
        else:
            ext_to_remote.pop(key, None)

    def _remove_document(self, cfg: Dict[str, Any], key: str) -> None:
        doc_store = self._doc_store(cfg)
        remote_id = self._ext_to_remote(cfg).pop(key, None)
        if remote_id:
            self._remote_to_ext(cfg).pop(remote_id, None)
        doc_store.pop(key, None)

    def _match_ext_id_by_text(self, cfg: Dict[str, Any], text: str) -> Optional[str]:
        for ext_id, doc in self._doc_store(cfg).items():
            if doc.get("text") == text:
                return ext_id
        return None

    def _created_at_from_meta(self, meta: Dict[str, Any]) -> Optional[datetime]:
        value = meta.get("created_at") or meta.get("timestamp")
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            candidate = value.replace("Z", "+00:00") if value.endswith("Z") else value
            try:
                return datetime.fromisoformat(candidate)
            except ValueError:
                return None
        return None

    def _extract_remote_id(self, obj: Any) -> Optional[str]:
        for attr in ("id", "memory_id", "passage_id"):
            candidate = getattr(obj, attr, None)
            if candidate:
                return str(candidate)
        extra = getattr(obj, "model_extra", None)
        if isinstance(extra, dict):
            for key in ("memory_id", "passage_id", "id"):
                candidate = extra.get(key)
                if candidate:
                    return str(candidate)
        return None

    def _extract_score(self, result: Any) -> float:
        """Prefer the Letta RRF score but gracefully fall back to legacy values."""

        relevance = getattr(result, "relevance", None)
        if relevance is not None:
            # rrf_score is the canonical similarity metric returned by Letta search
            rrf_score = getattr(relevance, "rrf_score", None)
            if rrf_score is None and isinstance(relevance, dict):
                rrf_score = relevance.get("rrf_score")
            if rrf_score is not None:
                try:
                    return float(rrf_score)
                except (TypeError, ValueError):
                    pass

        score = getattr(result, "score", None)
        if score is not None:
            try:
                return float(score)
            except (TypeError, ValueError):
                pass

        extra = getattr(result, "model_extra", None)
        if isinstance(extra, dict):
            raw = extra.get("score")
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        return 0.0

    # ---------- interface ----------
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        self._index_cfg[index_id] = {
            "spec": spec,
            "ext_to_remote": {},
            "remote_to_ext": {},
            "doc_store": {},
        }

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        ins = upd = deleted = 0
        search_results: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            spec = self._index_cfg.get(req.index_id, {}).get("spec")
            if spec is None:
                raise KeyError(f"LettaBackend: unknown index_id {req.index_id}, call create_index first")
            if req.op == BackendOpType.INSERT:
                ins += self._handle_insert(req, spec)
            elif req.op == BackendOpType.UPDATE:
                upd += self._handle_update(req, spec)
            elif req.op == BackendOpType.DELETE_IDS:
                deleted += self._handle_delete(req, spec)
            elif req.op == BackendOpType.SEARCH:
                rid = req.request_id or f"req-{len(search_results)}"
                search_results[rid] = self._handle_search(req, spec)
            elif req.op == BackendOpType.DELETE_KNN:
                continue
            else:
                raise NotImplementedError(f"LettaBackend does not support op: {req.op}")

        return RunResult(upserted=ins, updated=upd, deleted=deleted, searches=search_results)

    # ---------- op handlers ----------
    def _handle_insert(self, req: BackendRequest, spec: CollectionSpec) -> int:
        cfg = self._ensure_context(req.index_id, spec)

        ext_ids = req.ext_ids or []
        payloads = req.payloads or []
        metas = req.metas or [{} for _ in payloads]

        created = 0
        for i, payload in enumerate(payloads):
            key = str(ext_ids[i]) if i < len(ext_ids) else str(i)
            meta = self._normalize_meta(metas[i] if i < len(metas) else {})
            text = "" if payload is None else str(payload)
            created_at = self._created_at_from_meta(meta)

            tags = self._tag_list(meta)
            passages = self._client.agents.passages.create(
                agent_id=cfg["agent_id"],
                text=text,
                tags=tags,
                created_at=created_at,
            )
            passage_id = passages[0].id if passages else None
            self._record_document(cfg, key, text, meta, passage_id)
            created += 1
        return created

    def _handle_update(self, req: BackendRequest, spec: CollectionSpec) -> int:
        cfg = self._ensure_context(req.index_id, spec)

        ext_ids = req.ext_ids or []
        payloads = req.payloads or []
        metas = req.metas or [{} for _ in payloads]

        updated = 0
        for i, payload in enumerate(payloads):
            key = str(ext_ids[i]) if i < len(ext_ids) else str(i)
            meta = self._normalize_meta(metas[i] if i < len(metas) else {})
            text = "" if payload is None else str(payload)
            created_at = self._created_at_from_meta(meta)

            remote_id = self._ext_to_remote(cfg).get(key)
            if remote_id:
                try:
                    self._client.agents.passages.delete(agent_id=cfg["agent_id"], memory_id=remote_id)
                except Exception:
                    pass
            passages = self._client.agents.passages.create(
                agent_id=cfg["agent_id"],
                text=text,
                tags=self._tag_list(meta),
                created_at=created_at,
            )
            passage_id = passages[0].id if passages else None
            self._record_document(cfg, key, text, meta, passage_id)
            updated += 1
        return updated

    def _handle_delete(self, req: BackendRequest, spec: CollectionSpec) -> int:
        cfg = self._ensure_context(req.index_id, spec)
        removed = 0
        for key in req.ext_ids or []:
            remote_id = self._ext_to_remote(cfg).get(str(key))
            if not remote_id:
                continue
            try:
                self._client.agents.passages.delete(agent_id=cfg["agent_id"], memory_id=remote_id)
                removed += 1
            except Exception:
                continue
            finally:
                self._remove_document(cfg, str(key))
        return removed

    def _build_metadata(
        self,
        cfg: Dict[str, Any],
        ext_id: Optional[str],
        tags: Optional[List[str]],
        timestamp: Any,
        remote_id: Optional[str],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"tags": tags or [], "timestamp": timestamp}
        if remote_id:
            metadata["remote_id"] = remote_id
        if ext_id:
            doc = self._doc_store(cfg).get(ext_id, {})
            stored_meta = doc.get("meta") or {}
            if isinstance(stored_meta, dict):
                metadata.update(stored_meta)
        return metadata

    def _handle_search(self, req: BackendRequest, spec: CollectionSpec) -> List[List[SearchHit]]:
        cfg = self._ensure_context(req.index_id, spec)
        queries = req.payloads or []

        all_hits: List[List[SearchHit]] = []
        for q in queries:
            if q is None:
                all_hits.append([])
                continue

            hits: List[SearchHit] = []
            limit = int(req.k or 0) or None
            query_text = str(q)
            if self._embed_queries:
                response = self._client.agents.passages.search(
                    agent_id=cfg["agent_id"],
                    query=query_text,
                    top_k=limit,
                )
                for result in response.results:
                    remote_id = self._extract_remote_id(result)
                    ext_id = self._remote_to_ext(cfg).get(remote_id or "") if remote_id else None
                    if not ext_id:
                        ext_id = self._match_ext_id_by_text(cfg, result.content)
                    hit_id = ext_id or remote_id or result.content
                    metadata = self._build_metadata(
                        cfg,
                        ext_id,
                        getattr(result, "tags", []),
                        getattr(result, "timestamp", None),
                        remote_id,
                    )
                    hits.append(
                        SearchHit(
                            id=str(hit_id),
                            score=self._extract_score(result),
                            metadata=metadata,
                        )
                    )
            else:
                passages = self._client.agents.passages.list(
                    agent_id=cfg["agent_id"],
                    search=query_text,
                    limit=limit,
                )
                for passage in passages:
                    remote_id = getattr(passage, "id", None)
                    ext_id = self._remote_to_ext(cfg).get(str(remote_id)) if remote_id else None
                    metadata = self._build_metadata(
                        cfg,
                        ext_id,
                        getattr(passage, "tags", []),
                        getattr(passage, "created_at", None),
                        remote_id,
                    )
                    hits.append(
                        SearchHit(
                            id=str(ext_id or passage.id or passage.text),
                            score=0.0,
                            metadata=metadata,
                        )
                    )
            all_hits.append(hits)
        return all_hits
