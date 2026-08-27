from __future__ import annotations
from typing import Any, Dict, List, Optional
import sys
from pathlib import Path
import numpy as np

from ..types import CollectionSpec, SearchHit, RunResult, BackendRequest, BackendOpType

from .base import MemoryBackend

# ---- optional import guard for a-mem ----
try:
    # Add A-mem to path if not already there
    a_mem_path = Path(__file__).parent.parent.parent.parent / "A-mem"
    if str(a_mem_path) not in sys.path:
        sys.path.insert(0, str(a_mem_path))
    
    from memory_layer import AgenticMemorySystem
except Exception as e:
    _AMEM_IMPORT_ERROR = e
    AgenticMemorySystem = None


class AMemBackend(MemoryBackend):
    """
    A-mem-powered backend that executes a serialized list of BackendRequest.
    - Interface layer handles encoding and queuing; this backend extracts text from payloads.
    - Uses A-mem's AgenticMemorySystem for storage, retrieval, and evolution.
    - Maintains ext_id <-> memory_id mapping per index.
    - Supports: INSERT, UPDATE, DELETE_IDS, DELETE_KNN, SEARCH.
    """

    def __init__(
        self,
        *,
        model_name: str = 'all-MiniLM-L6-v2',
        llm_backend: str = "sglang",
        llm_model: str = "gpt-4o-mini",
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ) -> None:
        super().__init__()
        if AgenticMemorySystem is None:
            raise RuntimeError(
                "A-mem is not available. Please ensure A-mem/memory_layer.py is accessible. "
                f"Original import error: {repr(_AMEM_IMPORT_ERROR)}"
            )
        print("[backend] init a-mem")

        # Store A-mem configuration
        self._model_name = model_name
        self._llm_backend = llm_backend
        self._llm_model = llm_model
        self._evo_threshold = evo_threshold
        self._api_key = api_key
        self._api_base = api_base
        self._sglang_host = sglang_host
        self._sglang_port = sglang_port

        # Index registry
        self._specs: Dict[int, CollectionSpec] = {}
        self._memory_systems: Dict[int, AgenticMemorySystem] = {}

        # External ID <-> A-mem memory_id mapping per index
        self._ext2mem_id: Dict[int, Dict[str, str]] = {}
        self._mem2ext_id: Dict[int, Dict[str, str]] = {}

    # ---------- index management ----------
    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        """Create/prepare an A-mem memory system for this index."""
        self._specs[index_id] = spec
        
        # Create new AgenticMemorySystem instance for this index
        memory_system = AgenticMemorySystem(
            model_name=self._model_name,
            llm_backend=self._llm_backend,
            llm_model=self._llm_model,
            evo_threshold=self._evo_threshold,
            api_key=self._api_key,
            api_base=self._api_base,
            sglang_host=self._sglang_host,
            sglang_port=self._sglang_port,
        )
        self._memory_systems[index_id] = memory_system
        
        # Initialize ID mappings
        self._ext2mem_id[index_id] = {}
        self._mem2ext_id[index_id] = {}
        
        print(f"  - created a-mem index_id={index_id} name='{spec.name}' (dim={spec.dim}, metric={spec.metric})")

    # ---------- execution ----------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        """Execute all backend requests strictly in the given order."""
        ins_cnt = 0
        upd_cnt = 0
        del_cnt = 0
        search_results: Dict[str, List[List[SearchHit]]] = {}

        for req in ops:
            if req.op == BackendOpType.INSERT:
                n = len(req.ext_ids or [])
                self._insert(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    payloads=req.payloads or [None] * n,
                    metas=req.metas or [None] * n,
                )
                ins_cnt += n

            elif req.op == BackendOpType.UPDATE:
                n = len(req.ext_ids or [])
                self._update(
                    index_id=req.index_id,
                    ext_ids=req.ext_ids or [],
                    payloads=req.payloads or [None] * n,
                    metas=req.metas or [None] * n,
                )
                upd_cnt += n

            elif req.op == BackendOpType.DELETE_IDS:
                ids = req.ext_ids or []
                self._delete_ids(req.index_id, ids)
                del_cnt += len(ids)

            elif req.op == BackendOpType.DELETE_KNN:
                payloads = req.payloads or []
                k = int(req.k or 0)
                removed = self._delete_knn(req.index_id, payloads, k)
                del_cnt += removed

            elif req.op == BackendOpType.SEARCH:
                payloads = req.payloads or []
                k = int(req.k or 0)
                rid = req.request_id or "req-unknown"
                hits = self._search(req.index_id, payloads, k)
                search_results[rid] = hits

            else:
                print(f"[WARN] unknown backend op skipped: {req.op}")

        return RunResult(
            upserted=ins_cnt,
            updated=upd_cnt,
            deleted=del_cnt,
            searches=search_results,
        )

    # ---------- helpers ----------
    @staticmethod
    def _extract_text(payload: Any) -> str:
        """Extract text from payload (MemoryItem.data)."""
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        # Try to convert other types to string
        return str(payload)
    
    @staticmethod
    def _delete_memory(memory_system, memory_id: str) -> bool:
        """
        Delete a memory from A-mem by removing from memories dict and rebuilding retriever.
        Returns True if deletion was successful, False if memory_id not found.
        """
        if memory_id not in memory_system.memories:
            return False
        
        # Remove from memories dict
        del memory_system.memories[memory_id]
        
        # Rebuild retriever to reflect the deletion
        memory_system.consolidate_memories()
        
        return True

    # ---------- core ops ----------
    def _insert(
        self,
        index_id: int,
        ext_ids: List[str],
        payloads: List[Any],
        metas: List[Optional[dict]],
    ) -> None:
        """Insert new memories into A-mem."""
        print(f"[a-mem] insert index_id={index_id} n={len(ext_ids)}")
        
        if index_id not in self._memory_systems:
            raise KeyError(f"Index {index_id} not found. Call create_index() first.")
        
        memory_system = self._memory_systems[index_id]
        ext2mem = self._ext2mem_id[index_id]
        mem2ext = self._mem2ext_id[index_id]

        for i, ext_id in enumerate(ext_ids):
            text = self._extract_text(payloads[i] if i < len(payloads) else None)
            meta = metas[i] if i < len(metas) else None
            
            # Prepare kwargs for add_note
            kwargs = {}
            if meta:
                # Extract common metadata fields that A-mem supports
                if "keywords" in meta:
                    kwargs["keywords"] = meta["keywords"]
                if "context" in meta:
                    kwargs["context"] = meta["context"]
                if "tags" in meta:
                    kwargs["tags"] = meta["tags"]
                if "timestamp" in meta:
                    kwargs["timestamp"] = meta["timestamp"]
            
            # Add note to A-mem
            memory_id = memory_system.add_note(content=text, **kwargs)
            
            # Verify retriever was updated (A-mem's add_note should update retriever automatically)
            print(f"  [DEBUG] After add_note: retriever has {len(memory_system.retriever.corpus)} documents")
            if memory_system.retriever.embeddings is not None:
                print(f"  [DEBUG] Retriever embeddings shape: {memory_system.retriever.embeddings.shape}")
            else:
                print(f"  [DEBUG] Retriever embeddings: None")
            
            # Store ID mapping
            ext2mem[ext_id] = memory_id
            mem2ext[memory_id] = ext_id

    def _update(
        self,
        index_id: int,
        ext_ids: List[str],
        payloads: List[Any],
        metas: List[Optional[dict]],
    ) -> None:
        """Update existing memories in A-mem."""
        print(f"[a-mem] update index_id={index_id} n={len(ext_ids)}")
        
        if index_id not in self._memory_systems:
            raise KeyError(f"Index {index_id} not found. Call create_index() first.")
        
        memory_system = self._memory_systems[index_id]
        ext2mem = self._ext2mem_id[index_id]
        mem2ext = self._mem2ext_id[index_id]

        for i, ext_id in enumerate(ext_ids):
            text = self._extract_text(payloads[i] if i < len(payloads) else None)
            meta = metas[i] if i < len(metas) else None
            
            # Find memory_id
            memory_id = ext2mem.get(ext_id)
            
            if memory_id is None:
                # Not found, treat as INSERT
                print(f"  [WARN] update: ext_id '{ext_id}' not found, treating as insert")
                kwargs = {}
                if meta:
                    if "keywords" in meta:
                        kwargs["keywords"] = meta["keywords"]
                    if "context" in meta:
                        kwargs["context"] = meta["context"]
                    if "tags" in meta:
                        kwargs["tags"] = meta["tags"]
                    if "timestamp" in meta:
                        kwargs["timestamp"] = meta["timestamp"]
                
                memory_id = memory_system.add_note(content=text, **kwargs)
                ext2mem[ext_id] = memory_id
                mem2ext[memory_id] = ext_id
            else:
                # Update existing memory: delete and re-add
                # This ensures the retriever is properly updated
                
                # Delete the old memory
                success = self._delete_memory(memory_system, memory_id)
                if not success:
                    print(f"  [WARN] update: failed to delete memory_id '{memory_id}', skipping update")
                    continue
                
                # Remove from mappings
                ext2mem.pop(ext_id, None)
                mem2ext.pop(memory_id, None)
                
                # Prepare kwargs for re-adding
                kwargs = {}
                if meta:
                    if "keywords" in meta:
                        kwargs["keywords"] = meta["keywords"]
                    if "context" in meta:
                        kwargs["context"] = meta["context"]
                    if "tags" in meta:
                        kwargs["tags"] = meta["tags"]
                    if "timestamp" in meta:
                        kwargs["timestamp"] = meta["timestamp"]
                    if "category" in meta:
                        kwargs["category"] = meta["category"]
                
                # Re-add with new content/metadata
                new_memory_id = memory_system.add_note(content=text, **kwargs)
                
                # Update mappings with new memory_id
                ext2mem[ext_id] = new_memory_id
                mem2ext[new_memory_id] = ext_id
                
                print(f"  [DEBUG] Updated: deleted memory_id '{memory_id}', added new memory_id '{new_memory_id}'")

    def _delete_ids(self, index_id: int, ext_ids: List[str]) -> None:
        """Delete memories by external IDs."""
        print(f"[a-mem] delete_ids index_id={index_id} n={len(ext_ids)}")
        
        if index_id not in self._memory_systems:
            raise KeyError(f"Index {index_id} not found. Call create_index() first.")
        
        memory_system = self._memory_systems[index_id]
        ext2mem = self._ext2mem_id[index_id]
        mem2ext = self._mem2ext_id[index_id]

        for ext_id in ext_ids:
            memory_id = ext2mem.get(ext_id)
            if memory_id is None:
                print(f"  [WARN] delete skipped: ext_id '{ext_id}' not found")
                continue
            
            # Delete from A-mem
            success = self._delete_memory(memory_system, memory_id)
            if success:
                # Remove from mappings
                ext2mem.pop(ext_id, None)
                mem2ext.pop(memory_id, None)
            else:
                print(f"  [WARN] delete failed for memory_id '{memory_id}'")

    def _delete_knn(self, index_id: int, payloads: List[Any], k: int) -> int:
        """
        Delete the k nearest memories for each query.
        Returns the number of unique deletions performed.
        """
        print(f"[a-mem] delete_knn index_id={index_id} q={len(payloads)} k={k}")
        
        if index_id not in self._memory_systems:
            raise KeyError(f"Index {index_id} not found. Call create_index() first.")
        
        if k <= 0 or len(payloads) == 0:
            return 0
        
        memory_system = self._memory_systems[index_id]
        ext2mem = self._ext2mem_id[index_id]
        mem2ext = self._mem2ext_id[index_id]
        
        to_remove_memory_ids: set = set()
        
        # For each query, find k nearest and collect memory_ids
        for payload in payloads:
            query_text = self._extract_text(payload)
            if not query_text:
                continue
            
            # Use find_related_memories which returns (memory_str, indices)
            # indices are into list(memory_system.memories.values())
            _, indices = memory_system.find_related_memories(query_text, k=k)
            
            # Convert indices to memory_ids
            all_memory_ids = list(memory_system.memories.keys())
            for idx in indices:
                if idx < len(all_memory_ids):
                    to_remove_memory_ids.add(all_memory_ids[idx])
        
        if not to_remove_memory_ids:
            return 0
        
        # Delete all found memories
        removed_count = 0
        for memory_id in list(to_remove_memory_ids):
            success = self._delete_memory(memory_system, memory_id)
            if success:
                removed_count += 1
                # Remove from mappings
                ext_id = mem2ext.pop(memory_id, None)
                if ext_id:
                    ext2mem.pop(ext_id, None)
        
        print(f"  - removed {removed_count} by knn")
        return removed_count

    def _search(self, index_id: int, payloads: List[Any], k: int) -> List[List[SearchHit]]:
        """Search for memories using query text from payloads."""
        print(f"[a-mem] search index_id={index_id} q={len(payloads)} k={k}")
        
        if index_id not in self._memory_systems:
            raise KeyError(f"Index {index_id} not found. Call create_index() first.")
        
        if len(payloads) == 0:
            return []
        
        memory_system = self._memory_systems[index_id]
        mem2ext = self._mem2ext_id[index_id]
        results: List[List[SearchHit]] = []

        for payload in payloads:
            query_text = self._extract_text(payload)
            print(f"  [DEBUG] Extracted query text: '{query_text}'")
            
            if not query_text:
                print(f"  [DEBUG] Empty query text, appending empty result")
                results.append([])
                continue
            
            # Debug: Check memory system state
            print(f"  [DEBUG] Memory system state:")
            print(f"    - memories count: {len(memory_system.memories)}")
            print(f"    - retriever corpus size: {len(memory_system.retriever.corpus)}")
            if memory_system.retriever.embeddings is not None:
                print(f"    - retriever embeddings shape: {memory_system.retriever.embeddings.shape}")
            else:
                print(f"    - retriever embeddings: None")
            
            # Use find_related_memories which returns (memory_str, indices)
            # indices are into list(memory_system.memories.values())
            _, indices = memory_system.find_related_memories(query_text, k=k)
            print(f"  [DEBUG] find_related_memories returned {len(indices)} indices: {indices}")
            
            # Convert indices to memory objects and then to SearchHit
            all_memories = list(memory_system.memories.values())
            all_memory_ids = list(memory_system.memories.keys())
            print(f"  [DEBUG] all_memories length: {len(all_memories)}, all_memory_ids length: {len(all_memory_ids)}")
            hits: List[SearchHit] = []
            
            # Calculate scores based on position (indices are already ranked by similarity)
            for rank, idx in enumerate(indices):
                if idx >= len(all_memories):
                    print(f"  [DEBUG] WARN: index {idx} >= {len(all_memories)}, skipping")
                    continue
                
                memory = all_memories[idx]
                memory_id = all_memory_ids[idx]
                
                # Map memory_id back to ext_id if possible
                ext_id = mem2ext.get(memory_id, memory_id)
                
                # Calculate score based on position (higher rank = higher score)
                # Normalize to 0-1 range, with best match = 1.0
                score = 1.0 - (rank / max(len(indices), 1)) * 0.5  # Range: 0.5 to 1.0
                
                # Build metadata
                metadata = {
                    "content": memory.content,
                    "context": memory.context,
                    "keywords": memory.keywords,
                    "tags": memory.tags,
                    "category": memory.category,
                    "timestamp": memory.timestamp,
                    "_data": memory.content,  # Include original content
                }
                
                hits.append(SearchHit(
                    id=str(ext_id),
                    score=float(score),
                    metadata=metadata
                ))
            
            print(f"  [DEBUG] Converted to {len(hits)} SearchHits")
            results.append(hits)

        return results

