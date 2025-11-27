from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import uuid
import hashlib
import json
from pathlib import Path
import numpy as np

from .types import CollectionSpec, MemoryItem, RunResult, Metric, BackendOpType, BackendRequest
from .backend.quake import QuakeBackend
from .backend.base import MemoryBackend
from .encoder import MemoryEncoder, TransformerEncoder

IndexHandle = Union[int, str]

class MemoryManagement:
    """
    Interface-level queue + single-pass encoding + serialized backend requests.

    Workflow:
      - User enqueues logical operations into self.queue (FIFO).
      - run():
          1) Collect all MemoryItems across INSERT/UPDATE/SEARCH/DELETE_KNN.
          2) Encode once to a big (M, D) matrix.
          3) Replay FIFO: slice vectors and build BackendRequest objects.
          4) (Optional) merge consecutive same-type ops on same index into batches.
          5) Call backend.execute(serialized_ops).
    """

    def __init__(
        self,
        *,
        backend: Union[str, MemoryBackend] = "placeholder",
        encoder: Optional[MemoryEncoder] = None,
        model_name: str = "intfloat/e5-large-v2",
        precision: str = "fp32",
        dim: Optional[int] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        default_nprobe: int = 32,
        # key hashing config
        hash_mode: str = "none",          # "none" | "blake2b64" | "sha1_64" | "python"
        hash_prefix: str = "",
        auto_id_strategy: str = "sequential",   # "uuid" | "sequential"
    ) -> None:
        # Backend (default to placeholder)
        if isinstance(backend, MemoryBackend):
            self.backend: MemoryBackend = backend
        else:
            name = backend.lower()
            if name == "placeholder":
                from .backend.placeholder import PlaceholderBackend
                self.backend = PlaceholderBackend()
            elif name == "quake":
                from .backend.quake import QuakeBackend
                self.backend = QuakeBackend()
            elif name == "m3":
                from .backend.m3 import M3Backend
                self.backend = M3Backend()
            elif name == "letta":
                from .backend.letta import LettaBackend
                self.backend = LettaBackend()
            else:
                raise ValueError(f"Unknown backend: {backend}")

        # Encoder
        self.encoder: MemoryEncoder = encoder or TransformerEncoder(
            model_name=model_name,
            precision=precision,
            dim=dim,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
        # Legacy injection for old backends (new backends can ignore)
        if hasattr(self.backend, "set_encoder"):
            self.backend.set_encoder(self.encoder)

        # Index registry
        self._next_index_id: int = 0
        self._name_to_id: Dict[str, int] = {}
        self._id_to_name: Dict[int, str] = {}

        # FIFO logical queue (unencoded)
        # Each entry: ("insert"/"update"/"delete_ids"/"delete_knn"/"search", index_id, payload...)
        self.queue: List[Tuple] = []

        # Hash config
        self.hash_mode = hash_mode.lower().strip()
        self.hash_prefix = hash_prefix
        if default_nprobe <= 0:
            raise ValueError("default_nprobe must be a positive integer")
        self._default_nprobe = int(default_nprobe)

        # Document store for id -> payload lookups
        strategy = (auto_id_strategy or "uuid").lower().strip()
        if strategy not in {"uuid", "sequential"}:
            raise ValueError("auto_id_strategy must be either 'uuid' or 'sequential'")
        self._auto_id_strategy = strategy
        self._auto_seq_counter = 0
        self._doc_store: Dict[str, Dict[str, Any]] = {}

    # ---------- index management ----------
    def create_index(self, handle: IndexHandle, metric: Metric = Metric.COSINE) -> int:
        index_id = self._next_index_id
        self._next_index_id += 1
        name = str(handle)
        self._name_to_id[name] = index_id
        self._id_to_name[index_id] = name
        dim = self.encoder.dim
        spec = CollectionSpec(name=name, dim=dim, metric=metric)
        self.backend.create_index(index_id=index_id, spec=spec)
        return index_id

    def _resolve_index_id(self, index: IndexHandle) -> int:
        if isinstance(index, int):
            if index not in self._id_to_name:
                raise KeyError(f"Index id {index} does not exist")
            return index
        if index not in self._name_to_id:
            raise KeyError(f"Index '{index}' does not exist")
        return self._name_to_id[index]

    # ---------- hashing ----------
    def _auto_id(self) -> Union[str, int]:
        if self._auto_id_strategy == "sequential":
            self._auto_seq_counter += 1
            return self._auto_seq_counter
        return f"auto-{uuid.uuid4().hex}"

    def _hash_key(self, key: Union[str, int]) -> Union[str, int]:
        mode = self.hash_mode
        if mode == "none":
            return key
        key_str = str(key)
        if mode == "python":
            return f"{self.hash_prefix}{hash(key_str) & 0xffffffffffffffff:016x}"
        if mode == "blake2b64":
            h = hashlib.blake2b(key_str.encode("utf-8"), digest_size=8).digest()
            return f"{self.hash_prefix}{int.from_bytes(h, 'big'):016x}"
        if mode == "sha1_64":
            h = hashlib.sha1(key_str.encode("utf-8")).digest()[:8]
            return f"{self.hash_prefix}{int.from_bytes(h, 'big'):016x}"
        raise ValueError(f"Unknown hash_mode: {self.hash_mode}")

    def _to_keys(self, ids: List[Optional[str]]) -> List[Union[str, int]]:
        out: List[Union[str, int]] = []
        for _id in ids:
            raw = _id or self._auto_id()
            out.append(self._hash_key(raw))
        return out

    # ---------- document store helpers ----------
    def _ext_id_to_doc_id(self, ext_id: str) -> str:
        if ext_id is None:
            raise ValueError("ext_id cannot be None when tracking documents")
        try:
            return str(int(ext_id))
        except (ValueError, TypeError):
            h = hashlib.blake2b(str(ext_id).encode("utf-8"), digest_size=8).digest()
            return str(int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff)

    def _update_seq_counter_from_doc_id(self, doc_id: str) -> None:
        if self._auto_id_strategy != "sequential":
            return
        try:
            numeric_id = int(doc_id)
        except (ValueError, TypeError):
            return
        if numeric_id > self._auto_seq_counter:
            self._auto_seq_counter = numeric_id

    def _record_documents(
        self,
        ext_ids: List[str],
        payloads: List[Any],
        metas: List[Optional[Dict[str, Any]]],
    ) -> None:
        for ext_id, payload, meta in zip(ext_ids, payloads, metas):
            doc_id = self._ext_id_to_doc_id(ext_id)
            entry = self._doc_store.get(doc_id, {})
            if payload is not None:
                entry["data"] = payload
            if meta is not None:
                entry["metadata"] = meta
            if "data" not in entry and payload is None:
                # Skip storing empty entries (e.g., updates with no payload/metadata)
                continue
            self._doc_store[doc_id] = entry
            self._update_seq_counter_from_doc_id(doc_id)

    def _remove_documents(self, ext_ids: List[str]) -> None:
        for ext_id in ext_ids:
            doc_id = self._ext_id_to_doc_id(ext_id)
            self._doc_store.pop(doc_id, None)

    def load_doc_store(self, path: Union[str, Path]) -> None:
        """
        Rebuild the in-memory doc store from a JSON file.
        JSON format: { "doc_id": {"data": ..., "metadata": ...}, ... }
        Bare values are treated as {"data": value, "metadata": null}.
        """
        p = Path(path).expanduser().resolve()
        text = p.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw_lines: List[Any] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw_lines.append(json.loads(line))
            raw = raw_lines
        if isinstance(raw, list):
            raw = {str(i + 1): value for i, value in enumerate(raw)}
        if not isinstance(raw, dict):
            raise ValueError("doc store JSON must be a list or mapping of doc_id -> entry")
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in raw.items():
            entry: Dict[str, Any]
            if isinstance(value, dict):
                if "data" in value or "metadata" in value:
                    entry = {"data": value.get("data"), "metadata": value.get("metadata")}
                else:
                    entry = {"data": value, "metadata": None}
            else:
                entry = {"data": value, "metadata": None}
            normalized[str(key)] = entry
            self._update_seq_counter_from_doc_id(str(key))
        self._doc_store = normalized

    def save_doc_store(self, path: Union[str, Path]) -> None:
        """Persist the tracked doc store to disk as JSON."""
        p = Path(path).expanduser().resolve()
        with p.open("w", encoding="utf-8") as f:
            json.dump(self._doc_store, f, ensure_ascii=False, indent=2)

    def get_doc(self, doc_id: Union[str, int], *, include_metadata: bool = False) -> Optional[Any]:
        """
        Retrieve the stored document (and optionally metadata) for a backend id.
        Returns None if the document is unknown (e.g., missing JSON rebuild or deleted).
        """
        entry = self._doc_store.get(str(doc_id))
        if entry is None:
            return None
        if include_metadata:
            return entry
        return entry.get("data")

    def doc_store_size(self) -> int:
        """Return the number of documents currently tracked in the doc store."""
        return len(self._doc_store)

    def load_doc_store_from_records(
        self,
        records: Iterable[Tuple[Union[str, int], Any, Optional[Dict[str, Any]]]],
    ) -> None:
        """Populate the doc store directly from an iterable of (doc_id, data, metadata)."""
        normalized: Dict[str, Dict[str, Any]] = {}
        for doc_id, data, metadata in records:
            if doc_id is None:
                continue
            key = str(doc_id)
            normalized[key] = {
                "data": data,
                "metadata": metadata,
            }
            self._update_seq_counter_from_doc_id(key)
        self._doc_store = normalized

    # ---------- enqueue logical ops (FIFO, unencoded) ----------
    def add_insert(self, index: IndexHandle, items: List[MemoryItem]) -> None:
        idx = self._resolve_index_id(index)
        self.queue.append(("insert", idx, items))

    def add_update(self, index: IndexHandle, items: List[MemoryItem]) -> None:
        idx = self._resolve_index_id(index)
        self.queue.append(("update", idx, items))

    def add_delete_ids(self, index: IndexHandle, ids: List[str]) -> None:
        idx = self._resolve_index_id(index)
        self.queue.append(("delete_ids", idx, ids))

    def add_delete_knn(self, index: IndexHandle, queries: List[MemoryItem], k: int) -> None:
        idx = self._resolve_index_id(index)
        self.queue.append(("delete_knn", idx, queries, k))

    def add_search(
        self,
        index: IndexHandle,
        queries: List[MemoryItem],
        k: int,
        request_id: Optional[str] = None,
        nprobe: Optional[int] = None,
    ) -> None:
        idx = self._resolve_index_id(index)
        rid = request_id or f"req-{uuid.uuid4().hex}"
        derived_nprobe: Optional[int] = None
        if nprobe is not None:
            if nprobe <= 0:
                raise ValueError("nprobe must be positive")
            derived_nprobe = int(nprobe)
        self.queue.append(("search", idx, queries, k, rid, derived_nprobe))

    # ---------- run: encode once, build BackendRequests, execute ----------
    def run(self) -> RunResult:
        if not self.queue:
            # No-ops: let backend flush if it has internal state (usually returns zeros)
            return self.backend.execute([])

        ops = self.queue
        self.queue = []

        # Phase 1: gather everything needing encoding
        all_items: List[MemoryItem] = []
        counts: List[int] = [0] * len(ops)

        for i, op in enumerate(ops):
            kind = op[0]
            if kind in ("insert", "update"):
                _, _, items = op
                all_items.extend(items)
                counts[i] = len(items)
            elif kind in ("search", "delete_knn"):
                if kind == "search":
                    _, _, queries, _, _, _ = op
                    all_items.extend(queries)
                    counts[i] = len(queries)
                else:
                    _, _, queries, _ = op
                    all_items.extend(queries)
                    counts[i] = len(queries)
            # delete_ids: no encoding

        # Phase 2: single-pass encode
        all_vecs: Optional[np.ndarray] = None
        if all_items:
            all_vecs = self.encoder.encode_items(all_items)

        # Phase 3: replay FIFO and build BackendRequest list
        backend_reqs: List[BackendRequest] = []
        ptr = 0

        def take_slice(n: int) -> Optional[np.ndarray]:
            nonlocal ptr, all_vecs
            if n <= 0 or all_vecs is None:
                return None
            s = all_vecs[ptr:ptr + n]
            ptr += n
            return s

        for i, op in enumerate(ops):
            kind = op[0]
            n = counts[i]
            V = take_slice(n)

            if kind in ("insert", "update"):
                _, idx, items = op
                ext_ids = self._to_keys([it.id for it in items])
                metas = [it.metadata for it in items]
                payloads = [it.data for it in items]
                self._record_documents(ext_ids, payloads, metas)
                backend_reqs.append(
                    BackendRequest(
                        op=BackendOpType.INSERT if kind == "insert" else BackendOpType.UPDATE,
                        index_id=idx,
                        ext_ids=ext_ids,
                        vectors=V,
                        metas=metas,
                        payloads=payloads,
                    )
                )

            elif kind == "delete_ids":
                _, idx, ids = op
                ext_ids = self._to_keys(ids)
                self._remove_documents(ext_ids)
                backend_reqs.append(
                    BackendRequest(
                        op=BackendOpType.DELETE_IDS,
                        index_id=idx,
                        ext_ids=ext_ids,
                    )
                )

            elif kind == "delete_knn":
                _, idx, queries, k = op
                backend_reqs.append(
                    BackendRequest(
                        op=BackendOpType.DELETE_KNN,
                        index_id=idx,
                        vectors=V,
                        k=k,
                    )
                )

            elif kind == "search":
                _, idx, queries, k, rid, nprobe = op
                backend_reqs.append(
                    BackendRequest(
                        op=BackendOpType.SEARCH,
                        index_id=idx,
                        vectors=V,
                        payloads=[q.data for q in queries],
                        k=k,
                        request_id=rid,
                        nprobe=nprobe or self._default_nprobe,
                    )
                )

            else:
                print(f"[WARN] unknown op skipped: {kind}")

        # (Optional) Phase 4: merge adjacent batches of the same type and index
        merged: List[BackendRequest] = []
        for req in backend_reqs:
            if merged and self._can_merge(merged[-1], req):
                merged[-1] = self._merge_two(merged[-1], req)
            else:
                merged.append(req)

        # Phase 5: execute serially in backend
        return self.backend.execute(merged)

    # ---------- external index rebuild helpers ----------
    def rebuild_index_from_faiss(self, handle: IndexHandle, *, path: str, normalized: bool = True) -> None:
        """
        Rebuild an existing backend index from a Faiss IVF dump.
        Currently only supported by the M3 backend.
        """
        if not hasattr(self.backend, "rebuild_index_from_faiss"):
            raise NotImplementedError("Backend does not support rebuild_index_from_faiss")
        idx = self._resolve_index_id(handle)
        self.backend.rebuild_index_from_faiss(idx, path=path, normalized=normalized)

    # ---------- simple merge logic (adjacent same op & same index) ----------
    def _can_merge(self, a: BackendRequest, b: BackendRequest) -> bool:
        if a.op != b.op or a.index_id != b.index_id:
            return False
        # Do not merge SEARCH with different request ids
        if a.op == BackendOpType.SEARCH and a.request_id != b.request_id:
            return False
        if a.op == BackendOpType.SEARCH and a.nprobe != b.nprobe:
            return False
        # For DELETE_KNN, only merge if k is the same
        if a.op == BackendOpType.DELETE_KNN and a.k != b.k:
            return False
        return True

    def _merge_two(self, a: BackendRequest, b: BackendRequest) -> BackendRequest:
        def cat(X, Y):
            if X is None: return Y
            if Y is None: return X
            return np.vstack([X, Y]) if isinstance(X, np.ndarray) else X + Y

        return BackendRequest(
            op=a.op,
            index_id=a.index_id,
            ext_ids=(a.ext_ids or []) + (b.ext_ids or []),
            vectors=cat(a.vectors, b.vectors),
            metas=(a.metas or []) + (b.metas or []),
            payloads=(a.payloads or []) + (b.payloads or []),
            k=a.k or b.k,
            request_id=a.request_id or b.request_id,
            nprobe=a.nprobe or b.nprobe,
        )
