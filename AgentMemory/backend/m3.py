# AgentMemory/backend/m3.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import hashlib
import numpy as np

from .base import MemoryBackend
from ..types import CollectionSpec, RunResult, BackendRequest, BackendOpType, SearchHit

from AgentMemory.M3 import _m3_async as m3, rebuild_from_faiss, M3MultiLevelIndex, GpuCoordinator  # pybind module + loader

def _apply_cache_config(idx: "M3MultiLevelIndex", p: Dict[str, Any]) -> None:
    """Build a CacheConfig from resolved params dict and push it to the index."""
    cfg = m3.CacheConfig()
    cfg.l0_max_clusters            = int(p["l0_max_clusters"])
    cfg.l0_max_vectors_per_cluster = int(p["l0_max_vectors_per_cluster"])
    cfg.l1_max_clusters            = int(p["l1_max_clusters"])
    cfg.l1_max_vectors_per_cluster = int(p["l1_max_vectors_per_cluster"])
    cfg.l0_eviction_ratio          = float(p["l0_eviction_ratio"])
    cfg.l1_eviction_ratio          = float(p["l1_eviction_ratio"])
    cfg.cold_time_ns               = int(p["cold_time_ns"])
    cfg.l1_neighborhood_k          = int(p["l1_neighborhood_k"])
    cfg.l0_neighborhood_k          = int(p["l0_neighborhood_k"])
    cfg.max_promote_per_query      = int(p["max_promote_per_query"])
    cfg.l0_nprobe                  = int(p["l0_nprobe"])
    cfg.l1_nprobe                  = int(p["l1_nprobe"])
    cfg.alpha_et                   = float(p["alpha_et"])
    cfg.dagent_window              = int(p["dagent_window"])
    cfg.calibration_interval       = int(p["calibration_interval"])
    cfg.alpha_et_adapt_rate        = float(p["alpha_et_adapt_rate"])
    mode = p.get("dagent_mode", "cache_level_k")
    cfg.dagent_mode = (m3.DagentUpdateMode.true_k
                       if str(mode) == "true_k"
                       else m3.DagentUpdateMode.cache_level_k)
    idx.set_cache_config(cfg)


def _metric_enum(name: str):
    s = (name or "").lower()
    if s in ("l2", "euclidean"):
        return m3.Metric.L2
    if s in ("ip", "inner_product", "dot"):
        return m3.Metric.IP
    if s in ("cos", "cosine"):
        return m3.Metric.COSINE
    raise ValueError(f"Unsupported metric: {name!r}")


def _as_f32_2d(arr, err: str):
    if arr is None:
        raise ValueError(err)
    a = np.asarray(arr, dtype=np.float32, order="C")
    if a.ndim != 2:
        raise ValueError(err)
    return a


def _as_int64_1d(arr, err: str):
    if arr is None:
        raise ValueError(err)
    a = np.asarray(arr, dtype=np.int64, order="C")
    if a.ndim != 1:
        raise ValueError(err)
    return a


class M3Backend(MemoryBackend):
    """
    M3 backend that matches the current C++ async engine:

    - engine 支持多个 index_id
    - 每个 index 是一个 IVFIndex
    - 写操作需要 (index_id, cluster_id)，我们当前先全写 cluster 0
    - search 是按 index_id 查
    """

    def __init__(
        self,
        autostart: bool = True,
        writer_threads: int = 2,
        maintenance_threads: int = 1,
    ) -> None:
        super().__init__()
        self._eng = m3.AsyncEngine()
        self._autostart = autostart
        self._started = False

        self._writer_threads = writer_threads
        self._maintenance_threads = maintenance_threads

        # 记录已经创建过的 index_id
        self._indices: Set[int] = set()
        
        # Store metadata and data for each index: index_id -> {int64_id -> {metadata, data, ext_id}}
        self._meta: Dict[int, Dict[int, Optional[Dict[str, Any]]]] = {}
        self._data: Dict[int, Dict[int, Any]] = {}
        self._int2ext: Dict[int, Dict[int, Any]] = {}  # int64_id -> ext_id

    # ---------- lifecycle ----------

    def _ensure_started(self) -> None:
        if not self._started:
            self._eng.start(self._writer_threads, self._maintenance_threads)
            self._started = True

    def close(self) -> None:
        if self._started:
            self._eng.flush()
            self._eng.stop()
            self._started = False

    # ---------- create index (real IVF) ----------

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        """
        真正创建一个 IVF index:
        - 传下去 index_id
        - 必须有 dim / metric
        - centroids: 从 spec.params['centroids'] 取，如果没有，就建一个全 0 的 [1, dim]，表示只有 cluster 0
        """
        if self._autostart:
            self._ensure_started()

        if index_id in self._indices:
            return

        dim = int(spec.dim)
        metric = _metric_enum(getattr(spec, "metric", "l2"))
        normalized = (metric == m3.Metric.COSINE)

        centroids = None
        params = getattr(spec, "params", None)
        if params is not None:
            centroids = params.get("centroids")

        if centroids is None:
            # 默认 1 个 cluster，质心全 0
            centroids = np.zeros((1, dim), dtype=np.float32)
        else:
            centroids = np.ascontiguousarray(centroids, dtype=np.float32)
            if centroids.ndim != 2 or centroids.shape[1] != dim:
                raise ValueError("centroids must be [nlist, dim]")

        # 真正创建这个 index
        self._eng.create_ivf(index_id, dim, metric, normalized, centroids)
        self._indices.add(index_id)
        
        # Initialize storage for this index
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

    # ---------- main execute ----------

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        if self._autostart:
            self._ensure_started()

        insert_cnt = 0
        update_cnt = 0
        delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            idx = int(op.index_id)
            if idx not in self._indices:
                raise KeyError(f"M3Backend: index_id {idx} not found. Call create_index() first.")

            if op.op == BackendOpType.INSERT:
                ids = self._keys_to_int64(op.ext_ids, "INSERT requires 'ext_ids'")
                vecs = self._as_f32_2d(op.vectors, "INSERT requires 2D 'vectors'")
                self._eng.enqueue_insert_auto(idx, ids, vecs)
                # Store metadata and data
                if idx not in self._int2ext:
                    self._int2ext[idx] = {}
                    self._meta[idx] = {}
                    self._data[idx] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx][int(int_id)] = ext_id
                    self._meta[idx][int(int_id)] = meta
                    self._data[idx][int(int_id)] = payload
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = self._as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                # 没有自动分配的 update，先沿用 cluster 0
                self._eng.enqueue_update(idx, 0, ids, vecs, True)
                # Update metadata and data
                if idx not in self._int2ext:
                    self._int2ext[idx] = {}
                    self._meta[idx] = {}
                    self._data[idx] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx][int(int_id)] = ext_id
                    if meta is not None:
                        self._meta[idx][int(int_id)] = meta
                    if payload is not None:
                        self._data[idx][int(int_id)] = payload
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                self._eng.enqueue_delete(idx, 0, ids)
                # Remove from storage
                for int_id in ids:
                    self._meta.get(idx, {}).pop(int(int_id), None)
                    self._data.get(idx, {}).pop(int(int_id), None)
                    self._int2ext.get(idx, {}).pop(int(int_id), None)
                delete_cnt += len(ids)

            elif op.op == BackendOpType.FLUSH:
                self._eng.flush()

            elif op.op == BackendOpType.SEARCH:
                queries = self._as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
                k = int(op.k or 1)
                nprobe = int(op.nprobe or 32)

                # Flush pending writes before search for read-your-writes semantics.
                self._eng.flush()

                out_ids, out_scores = self._eng.search(idx, queries, k, nprobe)

                rid = op.request_id or f"req-{len(search_payload)}"
                query_hits: List[List[SearchHit]] = []
                for ids_list, scores_list in zip(out_ids, out_scores):
                    hits = []
                    for doc_id, score in zip(ids_list, scores_list):
                        int_id = int(doc_id)
                        if int_id < 0:
                            break  # -1 padding sentinel from numpy return format
                        # Get external ID (use int_id as fallback)
                        ext_id = self._int2ext.get(idx, {}).get(int_id, str(int_id))
                        # Build metadata dict
                        base_meta = self._meta.get(idx, {}).get(int_id) or {}
                        meta = dict(base_meta) if base_meta else {}
                        # Add original data to metadata
                        if idx in self._data and int_id in self._data[idx]:
                            meta["_data"] = self._data[idx][int_id]
                        # Use external ID for the hit ID
                        doc_id_str = str(ext_id) if ext_id is not None else str(int_id)
                        # Note: score is distance (smaller is better for cosine/L2, larger is better for IP)
                        hits.append(SearchHit(
                            id=doc_id_str,
                            score=float(score),
                            metadata=meta if meta else None
                        ))
                    query_hits.append(hits)
                search_payload[rid] = query_hits

            else:
                raise NotImplementedError(f"Unsupported op: {op.op}")

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    # ---------- rebuild from faiss ----------

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        if self._autostart:
            self._ensure_started()

        rebuild_from_faiss(
            self._eng,
            index_id=index_id,
            path=path,
            normalized=bool(normalized) if normalized is not None else True,
        )
        self._indices.add(index_id)

    # ---------- helpers ----------

    @staticmethod
    def _as_f32_2d(arr, err: str):
        return _as_f32_2d(arr, err)

    @staticmethod
    def _as_int64_1d(arr, err: str):
        return _as_int64_1d(arr, err)

    @staticmethod
    def _keys_to_int64(keys, err: str):
        if keys is None:
            raise ValueError(err)
        out = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            if key is None:
                h = hashlib.blake2b(str(i).encode("utf-8"), digest_size=8).digest()
                out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
            else:
                try:
                    out[i] = int(key)
                except (ValueError, TypeError):
                    h = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
                    out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
        return out


class M3MultiLevelBackend(MemoryBackend):
    """
    Simple synchronous backend backed by MultiLevelIndex (no async writers).
    """

    DEFAULTS: Dict[str, Any] = {
        # --- MultiLevelConfig ---
        "l0_nlist":                    1,
        "l1_nlist":                    1,
        "l2_nlist":                    1,
        "l0_new_cluster_threshold":    float("inf"),
        "search_threshold":            float("inf"),
        "l0_merge_threshold":          float("inf"),
        "l0_max_nlist":                0,

        # --- CacheConfig ---
        "l0_max_clusters":             64,
        "l0_max_vectors_per_cluster":  1000,
        "l1_max_clusters":             128,
        "l1_max_vectors_per_cluster":  10000,
        "l0_eviction_ratio":           0.8,
        "l1_eviction_ratio":           0.9,
        "cold_time_ns":                60_000_000_000,
        "l1_neighborhood_k":           20,
        "l0_neighborhood_k":           5,
        "max_promote_per_query":       20,
        "l0_nprobe":                   32,
        "l1_nprobe":                   32,
        "alpha_et":                    0.6,
        "dagent_window":               20,
        "dagent_mode":                 "true_k",
        "calibration_interval":        10,
        "alpha_et_adapt_rate":         0.05,
    }

    def __init__(self) -> None:
        super().__init__()
        self._indices: Dict[int, M3MultiLevelIndex] = {}
        
        # Store metadata and data for each index: index_id -> {int64_id -> {metadata, data, ext_id}}
        self._meta: Dict[int, Dict[int, Optional[Dict[str, Any]]]] = {}
        self._data: Dict[int, Dict[int, Any]] = {}
        self._int2ext: Dict[int, Dict[int, Any]] = {}  # int64_id -> ext_id

    def close(self) -> None:
        self._indices.clear()

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        if index_id in self._indices:
            return
        dim = int(spec.dim)
        metric = _metric_enum(getattr(spec, "metric", "l2"))
        normalized = (metric == m3.Metric.COSINE)

        raw = getattr(spec, "params", {}) or {}
        p: Dict[str, Any] = {**self.DEFAULTS, **raw}
        cfg_kwargs = {
            "l0_nlist": int(p["l0_nlist"]),
            "l1_nlist": int(p["l1_nlist"]),
            "l2_nlist": int(p["l2_nlist"]),
            "l0_new_cluster_threshold": float(p["l0_new_cluster_threshold"]),
            "search_threshold": float(p["search_threshold"]),
            "l0_merge_threshold": float(p["l0_merge_threshold"]),
            "l0_max_nlist": int(p["l0_max_nlist"]),
        }
        idx = M3MultiLevelIndex(dim=dim, metric=metric, normalized=normalized, **cfg_kwargs)
        _apply_cache_config(idx, p)

        # seed L0 centroids if provided; else zero centroid
        centroids = p.get("centroids")
        if centroids is None:
            centroids = np.zeros((cfg_kwargs["l0_nlist"], dim), dtype=np.float32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[1] != dim:
            raise ValueError("centroids must be [nlist, dim]")
        idx.set_l0_centroids(centroids)

        self._indices[index_id] = idx
        
        # Initialize storage for this index
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        insert_cnt = 0
        update_cnt = 0
        delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            idx_id = int(op.index_id)
            if idx_id not in self._indices:
                raise KeyError(f"M3MultiLevelBackend: index_id {idx_id} not found. Call create_index() first.")
            idx = self._indices[idx_id]

            if op.op == BackendOpType.INSERT:
                ids = self._keys_to_int64(op.ext_ids, "INSERT requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "INSERT requires 2D 'vectors'")
                idx.insert(ids, vecs)

                # Store metadata and data
                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    self._meta[idx_id][int(int_id)] = meta
                    self._data[idx_id][int(int_id)] = payload
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                idx.update(ids, vecs, insert_if_absent=True)
                # Update metadata and data
                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    if meta is not None:
                        self._meta[idx_id][int(int_id)] = meta
                    if payload is not None:
                        self._data[idx_id][int(int_id)] = payload
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                idx.erase(ids)
                # Remove from storage
                for int_id in ids:
                    self._meta.get(idx_id, {}).pop(int(int_id), None)
                    self._data.get(idx_id, {}).pop(int(int_id), None)
                    self._int2ext.get(idx_id, {}).pop(int(int_id), None)
                delete_cnt += len(ids)

            elif op.op == BackendOpType.FLUSH:
                # synchronous: nothing to do
                continue

            elif op.op == BackendOpType.SEARCH:
                queries = _as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
                k = int(op.k or 1)
                nprobe = int(op.nprobe or 32)

                out_ids, out_scores = idx.search(queries, k, nprobe)

                rid = op.request_id or f"req-{len(search_payload)}"
                hits_per_query: List[List[SearchHit]] = []
                for ids_list, scores_list in zip(out_ids, out_scores):
                    hits = []
                    for doc_id, score in zip(ids_list, scores_list):
                        int_id = int(doc_id)
                        if int_id < 0:
                            break  # -1 padding sentinel from numpy return format
                        # Get external ID (use int_id as fallback)
                        ext_id = self._int2ext.get(idx_id, {}).get(int_id, str(int_id))
                        # Build metadata dict
                        base_meta = self._meta.get(idx_id, {}).get(int_id) or {}
                        meta = dict(base_meta) if base_meta else {}
                        # Add original data to metadata
                        if idx_id in self._data and int_id in self._data[idx_id]:
                            meta["_data"] = self._data[idx_id][int_id]
                        # Use external ID for the hit ID
                        doc_id_str = str(ext_id) if ext_id is not None else str(int_id)
                        # Note: score is distance (smaller is better for cosine/L2, larger is better for IP)
                        hits.append(SearchHit(
                            id=doc_id_str,
                            score=float(score),
                            metadata=meta if meta else None
                        ))
                    hits_per_query.append(hits)
                search_payload[rid] = hits_per_query

            else:
                raise NotImplementedError(f"Unsupported op: {op.op}")

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        try:
            import faiss  # type: ignore
            from faiss.contrib.inspect_tools import get_invlist  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faiss is required to rebuild an index from a Faiss file") from exc

        if index_id not in self._indices:
            raise KeyError(f"M3MultiLevelBackend: index_id {index_id} not found. Call create_index() first.")

        from pathlib import Path

        idx = self._indices[index_id]
        faiss_path = Path(path)
        if not faiss_path.is_file():
            raise FileNotFoundError(f"Faiss index file not found: {faiss_path}")

        index = faiss.read_index(str(faiss_path))
        ivf = faiss.extract_index_ivf(index)
        if ivf is None:
            raise ValueError("Provided index does not contain an IVF component")
        ivf = faiss.downcast_index(ivf)
        if ivf.ntotal == 0:
            return

        # Extract centroids from the Faiss IVF quantizer.
        quantizer = faiss.downcast_index(ivf.quantizer)
        if hasattr(quantizer, "xb") and quantizer.ntotal == ivf.nlist:
            centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(ivf.nlist, ivf.d)
        else:
            centroids = np.vstack(
                [quantizer.reconstruct(i) for i in range(ivf.nlist)]
            ).astype(np.float32)
        centroids = np.ascontiguousarray(centroids)

        # Enable cache mode: set L2 topology (also aligns L0/L1 and resizes metadata).
        # All subsequent runtime inserts will route to L0 (hot tier) automatically.
        idx.set_l2_centroids(centroids)

        # Load corpus vectors directly into L2 — this is a cold bulk bootstrap,
        # not agent activity, so vectors bypass L0/L1 and land in the canonical store.
        invlists = faiss.downcast_InvertedLists(ivf.invlists)
        for list_id in range(ivf.nlist):
            list_ids, list_codes = get_invlist(invlists, list_id)
            if list_ids.size == 0:
                continue
            if list_codes.dtype != np.uint8:
                raise ValueError("Only IndexIVFFlat (float codes) is supported for now")
            vectors = list_codes.view(np.float32).reshape(list_ids.shape[0], ivf.d)
            idx.load_cluster(
                int(list_id),
                np.ascontiguousarray(list_ids, dtype=np.int64),
                np.ascontiguousarray(vectors, dtype=np.float32),
            )

        # Rebuild path loads raw vectors/ids only; metadata/payload stores are reset.
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

    @staticmethod
    def _keys_to_int64(keys, err: str):
        if keys is None:
            raise ValueError(err)
        out = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            if key is None:
                h = hashlib.blake2b(str(i).encode("utf-8"), digest_size=8).digest()
                out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
            else:
                try:
                    out[i] = int(key)
                except (ValueError, TypeError):
                    h = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
                    out[i] = int.from_bytes(h, "big", signed=False) & 0x7fffffffffffffff
        return out


class M3MultiGpuBackend(MemoryBackend):
    """
    M3 MultiLevelIndex backend with GPU hotspot caching via GpuCoordinator.

    Hot clusters are automatically promoted to VRAM based on access frequency.
    Inserts for GPU-resident clusters are buffered on CPU and flushed to GPU
    asynchronously. Searches partition probe clusters between GPU kernels and
    CPU L2 scans, merging results before returning.

    Configurable via spec.params:
      gpu_budget_bytes  : VRAM cap in bytes       (default: 2 GB)
      insert_buf_cap    : per-cluster buffer size  (default: 128)
      flush_ms          : buffer flush interval            (default: 50 ms)
      maintenance_ms    : L0/L1 eviction interval          (default: 50 ms)
      rebalance_ms      : hotspot rebalance period         (default: 500 ms)
      split_every_ops   : run split sweep every N inserts  (default: 0 = disabled)
      split_threshold   : split clusters exceeding N vecs  (default: 200 000)
      + all M3MultiLevelBackend centroid/config params
    """

    # ------------------------------------------------------------------ #
    #  All tuneable hyperparameters in one place.                        #
    #  Override any of these via spec.params when calling create_index.  #
    # ------------------------------------------------------------------ #
    DEFAULTS: Dict[str, Any] = {
        # --- GpuCoordinator ---
        "gpu_budget_bytes":  10 * 1024 ** 3,  # VRAM cap (10 GB)
        "insert_buf_cap":    128,              # per-cluster insert buffer
        "flush_ms":          50,               # GPU flush interval
        "maintenance_ms":    50,               # L0/L1 eviction interval
        "rebalance_ms":      500,              # GPU hotspot rebalance period
        "split_every_ops":   0,                # cluster split sweep (0 = off)
        "split_threshold":   200_000,          # split if cluster exceeds N vecs

        # --- MultiLevelConfig ---
        "l0_nlist":                    1,
        "l1_nlist":                    1,
        "l2_nlist":                    1,
        "l0_new_cluster_threshold":    float("inf"),
        "search_threshold":            float("inf"),
        "l0_merge_threshold":          float("inf"),
        "l0_max_nlist":                0,

        # --- CacheConfig ---
        "l0_max_clusters":             64,
        "l0_max_vectors_per_cluster":  1000,
        "l1_max_clusters":             128,
        "l1_max_vectors_per_cluster":  10000,
        "l0_eviction_ratio":           0.8,
        "l1_eviction_ratio":           0.9,
        "cold_time_ns":                60_000_000_000,  # 60 s
        "l1_neighborhood_k":           20,
        "l0_neighborhood_k":           5,
        "max_promote_per_query":       20,
        "l0_nprobe":                   32,
        "l1_nprobe":                   32,
        "alpha_et":                    0.7, #reducing alpha et improves recall
        "dagent_window":               20,
        "dagent_mode":                 "true_k",  # "cache_level_k" or "true_k"
        "calibration_interval":        10,
        "alpha_et_adapt_rate":         0.2,
    }

    def __init__(self) -> None:
        super().__init__()
        self._indices: Dict[int, M3MultiLevelIndex] = {}
        self._coordinators: Dict[int, GpuCoordinator] = {}
        self._index_params: Dict[int, Dict[str, Any]] = {}  # resolved params per index

        self._meta: Dict[int, Dict[int, Optional[Dict[str, Any]]]] = {}
        self._data: Dict[int, Dict[int, Any]] = {}
        self._int2ext: Dict[int, Dict[int, Any]] = {}

    def close(self) -> None:
        # Stop coordinators before destroying indices (coordinator holds a C++ ref to the index).
        for coord in self._coordinators.values():
            try:
                coord.stop_background()
            except Exception:
                pass
        for idx in self._indices.values():
            try:
                idx.set_gpu_coordinator(None)
            except Exception:
                pass
        self._coordinators.clear()
        self._indices.clear()

    def create_index(self, index_id: int, spec: CollectionSpec) -> None:
        if index_id in self._indices:
            return
        dim = int(spec.dim)
        metric = _metric_enum(getattr(spec, "metric", "l2"))
        normalized = (metric == m3.Metric.COSINE)

        # Merge spec.params over DEFAULTS — single source of truth for all knobs.
        raw = getattr(spec, "params", {}) or {}
        p: Dict[str, Any] = {**self.DEFAULTS, **raw}
        self._index_params[index_id] = p

        idx = M3MultiLevelIndex(
            dim=dim, metric=metric, normalized=normalized,
            l0_nlist=int(p["l0_nlist"]),
            l1_nlist=int(p["l1_nlist"]),
            l2_nlist=int(p["l2_nlist"]),
            l0_new_cluster_threshold=float(p["l0_new_cluster_threshold"]),
            search_threshold=float(p["search_threshold"]),
            l0_merge_threshold=float(p["l0_merge_threshold"]),
            l0_max_nlist=int(p["l0_max_nlist"]),
        )

        centroids = raw.get("centroids")
        if centroids is None:
            centroids = np.zeros((int(p["l0_nlist"]), dim), dtype=np.float32)
        centroids = np.ascontiguousarray(centroids, dtype=np.float32)
        if centroids.ndim != 2 or centroids.shape[1] != dim:
            raise ValueError("centroids must be [nlist, dim]")
        idx.set_l0_centroids(centroids)

        _apply_cache_config(idx, p)

        coord = GpuCoordinator(
            idx,
            gpu_budget_bytes=int(p["gpu_budget_bytes"]),
            dim=dim,
            metric=metric,
            normalized=normalized,
            insert_buf_cap=int(p["insert_buf_cap"]),
        )
        idx.set_gpu_coordinator(coord)
        coord.start_background(
            flush_ms=int(p["flush_ms"]),
            maintenance_ms=int(p["maintenance_ms"]),
            rebalance_ms=int(p["rebalance_ms"]),
            split_every_ops=int(p["split_every_ops"]),
            split_threshold=int(p["split_threshold"]),
        )

        self._indices[index_id] = idx
        self._coordinators[index_id] = coord
        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

    def execute(self, ops: List[BackendRequest]) -> RunResult:
        insert_cnt = 0
        update_cnt = 0
        delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            idx_id = int(op.index_id)
            if idx_id not in self._indices:
                raise KeyError(f"M3MultiGpuBackend: index_id {idx_id} not found. Call create_index() first.")
            idx = self._indices[idx_id]

            if op.op == BackendOpType.INSERT:
                ids = self._keys_to_int64(op.ext_ids, "INSERT requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "INSERT requires 2D 'vectors'")
                idx.insert(ids, vecs)

                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    self._meta[idx_id][int(int_id)] = meta
                    self._data[idx_id][int(int_id)] = payload
                insert_cnt += len(ids)

            elif op.op == BackendOpType.UPDATE:
                ids = self._keys_to_int64(op.ext_ids, "UPDATE requires 'ext_ids'")
                vecs = _as_f32_2d(op.vectors, "UPDATE requires 2D 'vectors'")
                idx.update(ids, vecs, insert_if_absent=True)
                if idx_id not in self._int2ext:
                    self._int2ext[idx_id] = {}
                    self._meta[idx_id] = {}
                    self._data[idx_id] = {}
                for int_id, ext_id, meta, payload in zip(ids, op.ext_ids or [], op.metas or [], op.payloads or []):
                    self._int2ext[idx_id][int(int_id)] = ext_id
                    if meta is not None:
                        self._meta[idx_id][int(int_id)] = meta
                    if payload is not None:
                        self._data[idx_id][int(int_id)] = payload
                update_cnt += len(ids)

            elif op.op == BackendOpType.DELETE_IDS:
                ids = self._keys_to_int64(op.ext_ids, "DELETE_IDS requires 'ext_ids'")
                idx.erase(ids)
                for int_id in ids:
                    self._meta.get(idx_id, {}).pop(int(int_id), None)
                    self._data.get(idx_id, {}).pop(int(int_id), None)
                    self._int2ext.get(idx_id, {}).pop(int(int_id), None)
                delete_cnt += len(ids)

            elif op.op == BackendOpType.FLUSH:
                continue  # background thread handles flush

            elif op.op == BackendOpType.SEARCH:
                queries = _as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
                k = int(op.k or 1)
                nprobe = int(op.nprobe or 32)

                # GPU backend has no explicit flush (background thread handles it).
                out_ids, out_scores = idx.search(queries, k, nprobe)

                rid = op.request_id or f"req-{len(search_payload)}"
                hits_per_query: List[List[SearchHit]] = []
                for ids_list, scores_list in zip(out_ids, out_scores):
                    hits = []
                    for doc_id, score in zip(ids_list, scores_list):
                        int_id = int(doc_id)
                        if int_id < 0:
                            break  # -1 padding sentinel from numpy return format
                        ext_id = self._int2ext.get(idx_id, {}).get(int_id, str(int_id))
                        base_meta = self._meta.get(idx_id, {}).get(int_id) or {}
                        meta = dict(base_meta) if base_meta else {}
                        if idx_id in self._data and int_id in self._data[idx_id]:
                            meta["_data"] = self._data[idx_id][int_id]
                        doc_id_str = str(ext_id) if ext_id is not None else str(int_id)
                        hits.append(SearchHit(
                            id=doc_id_str,
                            score=float(score),
                            metadata=meta if meta else None,
                        ))
                    hits_per_query.append(hits)
                search_payload[rid] = hits_per_query

            else:
                raise NotImplementedError(f"Unsupported op: {op.op}")

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        try:
            import faiss
            from faiss.contrib.inspect_tools import get_invlist
        except ImportError as exc:
            raise RuntimeError("faiss is required to rebuild an index from a Faiss file") from exc

        if index_id not in self._indices:
            raise KeyError(f"M3MultiGpuBackend: index_id {index_id} not found. Call create_index() first.")

        from pathlib import Path
        idx = self._indices[index_id]
        faiss_path = Path(path)
        if not faiss_path.is_file():
            raise FileNotFoundError(f"Faiss index file not found: {faiss_path}")

        index = faiss.read_index(str(faiss_path))
        ivf = faiss.extract_index_ivf(index)
        if ivf is None:
            raise ValueError("Provided index does not contain an IVF component")
        ivf = faiss.downcast_index(ivf)
        if ivf.ntotal == 0:
            return

        quantizer = faiss.downcast_index(ivf.quantizer)
        if hasattr(quantizer, "xb") and quantizer.ntotal == ivf.nlist:
            centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(ivf.nlist, ivf.d)
        else:
            centroids = np.vstack(
                [quantizer.reconstruct(i) for i in range(ivf.nlist)]
            ).astype(np.float32)
        centroids = np.ascontiguousarray(centroids)
        idx.set_l2_centroids(centroids)

        invlists = faiss.downcast_InvertedLists(ivf.invlists)
        for list_id in range(ivf.nlist):
            list_ids, list_codes = get_invlist(invlists, list_id)
            if list_ids.size == 0:
                continue
            if list_codes.dtype != np.uint8:
                raise ValueError("Only IndexIVFFlat (float codes) is supported for now")
            vectors = list_codes.view(np.float32).reshape(list_ids.shape[0], ivf.d)
            idx.load_cluster(
                int(list_id),
                np.ascontiguousarray(list_ids, dtype=np.int64),
                np.ascontiguousarray(vectors, dtype=np.float32),
            )

        self._meta[index_id] = {}
        self._data[index_id] = {}
        self._int2ext[index_id] = {}

        # Re-apply cache config after rebuild (set_l2_centroids resets internal state).
        if index_id in self._index_params:
            _apply_cache_config(idx, self._index_params[index_id])

    @staticmethod
    def _keys_to_int64(keys, err: str):
        return M3MultiLevelBackend._keys_to_int64(keys, err)


# ---------------------------------------------------------------------------
# M3MultiGpuFSMBackend
# ---------------------------------------------------------------------------

class M3MultiGpuFSMBackend(M3MultiGpuBackend):
    """
    Drop-in extension of M3MultiGpuBackend with FSM-based trajectory learning.
    M3MultiGpuBackend is completely untouched.

    Requires _m3_async to be compiled with -DM3_WITH_FSM (cmake option M3_WITH_FSM=ON).
    Raises RuntimeError at construction if the C++ FSM types are not available.

    When fsm_enabled=True (default):
      - Each request (keyed by BackendRequest.request_id) accumulates a C++
        RequestTrajectory across search steps.
      - Before each SEARCH, fsm_table.match_and_predict(traj) returns predicted
        next-cluster IDs used for non-blocking GPU prefetch via enqueue_promote().
      - idx.search_fsm() runs the search and records the winning cluster into
        the trajectory in one C++ call.
      - Call commit_request(request_id) after the last step of a request to
        update the FSM table (reinforce or create new pattern, then prune).

    When fsm_enabled=False:
      - execute() delegates directly to super().execute() — zero overhead.

    FSM-specific spec.params keys (all optional):
      fsm_max_patterns        (int,   500)
      fsm_ns_max_states       (int,   8)
      fsm_d_merge             (float, 0.3)
      fsm_reinforce_threshold (float, 0.5)
      fsm_min_hits_to_predict (int,   2)
      fsm_min_traj_len        (int,   2)
    """

    _FSM_DEFAULTS: Dict[str, Any] = {
        "fsm_max_patterns":        500,
        "fsm_ns_max_states":       8,
        "fsm_d_merge":             0.3,
        "fsm_reinforce_threshold": 0.5,
        "fsm_min_hits_to_predict": 2,
        "fsm_min_traj_len":        2,
    }

    def __init__(self, fsm_enabled: bool = True) -> None:
        super().__init__()
        self.fsm_enabled: bool = fsm_enabled

        if fsm_enabled:
            if not hasattr(m3, "FSMTable") or not hasattr(m3, "FSMConfig") or not hasattr(m3, "RequestTrajectory"):
                raise RuntimeError(
                    "M3MultiGpuFSMBackend requires _m3_async built with M3_WITH_FSM=ON. "
                    "Rebuild with: cmake -DM3_WITH_FSM=ON ..."
                )

        self._fsm_tables: Dict[int, Any] = {}          # index_id → m3.FSMTable
        self._fsm_params: Dict[int, Dict[str, Any]] = {}
        self._centroids:  Dict[int, np.ndarray] = {}   # L2 centroids per index

        self._active_trajs: Dict[str, Any] = {}        # rid → m3.RequestTrajectory
        self._traj_index:   Dict[str, int]  = {}        # rid → index_id

    # ------------------------------------------------------------------
    def create_index(self, index_id: int, spec: "CollectionSpec") -> None:
        super().create_index(index_id, spec)
        if not self.fsm_enabled:
            return
        raw = getattr(spec, "params", {}) or {}
        p: Dict[str, Any] = {**self._FSM_DEFAULTS, **raw}
        self._fsm_params[index_id] = p
        self._fsm_tables[index_id] = self._make_fsm_table(p)
        centroids = raw.get("centroids")
        if centroids is not None:
            self._centroids[index_id] = np.ascontiguousarray(centroids, dtype=np.float32)

    # ------------------------------------------------------------------
    def rebuild_index_from_faiss(self, index_id: int, *, path: str, normalized: Optional[bool] = None) -> None:
        super().rebuild_index_from_faiss(index_id, path=path, normalized=normalized)
        if not self.fsm_enabled:
            return
        try:
            import faiss
        except ImportError:
            return
        from pathlib import Path as _Path
        fi  = faiss.read_index(str(_Path(path)))
        ivf = faiss.extract_index_ivf(fi)
        if ivf is None:
            return
        ivf = faiss.downcast_index(ivf)
        q   = faiss.downcast_index(ivf.quantizer)
        if hasattr(q, "xb") and q.ntotal == ivf.nlist:
            centroids = faiss.vector_to_array(q.xb).astype(np.float32).reshape(ivf.nlist, ivf.d)
        else:
            centroids = np.vstack([q.reconstruct(i) for i in range(ivf.nlist)]).astype(np.float32)
        self._centroids[index_id] = np.ascontiguousarray(centroids)
        if index_id not in self._fsm_tables:
            p = self._fsm_params.get(index_id, self._FSM_DEFAULTS)
            self._fsm_tables[index_id] = self._make_fsm_table(p)

    # ------------------------------------------------------------------
    def commit_request(self, request_id: str) -> None:
        """
        Finalise and learn from a completed request's trajectory.
        Call after the last search step for request_id.
        Safe to call even if request_id was never searched (no-op).
        """
        traj   = self._active_trajs.pop(request_id, None)
        idx_id = self._traj_index.pop(request_id, None)
        if traj is None or idx_id is None:
            return
        fsm = self._fsm_tables.get(idx_id)
        if fsm is None:
            return
        min_len = self._fsm_params.get(idx_id, self._FSM_DEFAULTS).get("fsm_min_traj_len", 2)
        if traj.length() < min_len:
            return
        centroids = self._centroids.get(idx_id)
        if centroids is not None:
            fsm.update_from_trajectory(traj, centroids)

    # ------------------------------------------------------------------
    def fsm_stats(self, index_id: int) -> Dict[str, Any]:
        fsm = self._fsm_tables.get(index_id)
        if fsm is None:
            return {}
        p = self._fsm_params.get(index_id, self._FSM_DEFAULTS)
        return {
            "num_patterns":  fsm.num_patterns(),
            "total_hits":    fsm.total_hits(),
            "max_patterns":  p.get("fsm_max_patterns", 500),
            "ns_max_states": p.get("fsm_ns_max_states", 8),
            "d_merge":       p.get("fsm_d_merge", 0.3),
        }

    # ------------------------------------------------------------------
    def execute(self, ops: List[BackendRequest]) -> RunResult:
        if not self.fsm_enabled:
            return super().execute(ops)

        insert_cnt = update_cnt = delete_cnt = 0
        search_payload: Dict[str, List[List[SearchHit]]] = {}

        for op in ops:
            if op.op != BackendOpType.SEARCH:
                sub = super().execute([op])
                insert_cnt += sub.upserted
                update_cnt += sub.updated
                delete_cnt += sub.deleted
                continue

            # ---- SEARCH with FSM ----
            idx_id = int(op.index_id)
            if idx_id not in self._indices:
                raise KeyError(f"M3MultiGpuFSMBackend: index_id {idx_id} not found.")

            queries = _as_f32_2d(op.vectors, "SEARCH requires 2D 'vectors'")
            k       = int(op.k or 1)
            nprobe  = int(op.nprobe or 32)
            rid     = op.request_id or f"req-{len(search_payload)}"

            # Get or create trajectory for this request.
            if rid not in self._active_trajs:
                self._active_trajs[rid] = m3.RequestTrajectory(rid)
                self._traj_index[rid]   = idx_id
            traj = self._active_trajs[rid]

            fsm   = self._fsm_tables.get(idx_id)
            coord = self._coordinators.get(idx_id)
            idx   = self._indices[idx_id]

            # FSM prediction before search → predictive GPU prefetch.
            predicted_cids: List[int] = []
            if fsm is not None and traj.length() > 0:
                predicted_cids = fsm.match_and_predict(traj)
            if coord is not None and predicted_cids:
                for cid in predicted_cids[:8]:
                    try:
                        coord.enqueue_promote(cid)
                    except Exception:
                        pass

            # C++ search_fsm: runs search and appends winning cluster to traj.
            out_ids, out_scores = idx.search_fsm(queries, k, nprobe, fsm, traj)

            # Build SearchHit results.
            hits_per_query: List[List[SearchHit]] = []
            for ids_list, scores_list in zip(out_ids, out_scores):
                hits: List[SearchHit] = []
                for doc_id, score in zip(ids_list, scores_list):
                    int_id = int(doc_id)
                    if int_id < 0:
                        break
                    ext_id    = self._int2ext.get(idx_id, {}).get(int_id, str(int_id))
                    base_meta = self._meta.get(idx_id, {}).get(int_id) or {}
                    meta      = dict(base_meta) if base_meta else {}
                    if idx_id in self._data and int_id in self._data[idx_id]:
                        meta["_data"] = self._data[idx_id][int_id]
                    if predicted_cids:
                        meta["_fsm_predicted_cids"] = predicted_cids
                    hits.append(SearchHit(
                        id=str(ext_id) if ext_id is not None else str(int_id),
                        score=float(score),
                        metadata=meta if meta else None,
                    ))
                hits_per_query.append(hits)
            search_payload[rid] = hits_per_query

        return RunResult(
            upserted=insert_cnt,
            updated=update_cnt,
            deleted=delete_cnt,
            searches=search_payload,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_fsm_table(p: Dict[str, Any]):
        cfg = m3.FSMConfig()
        cfg.max_patterns        = int(p["fsm_max_patterns"])
        cfg.ns_max_states       = int(p["fsm_ns_max_states"])
        cfg.d_merge             = float(p["fsm_d_merge"])
        cfg.reinforce_threshold = float(p["fsm_reinforce_threshold"])
        cfg.min_hits_to_predict = int(p["fsm_min_hits_to_predict"])
        cfg.min_traj_len        = int(p["fsm_min_traj_len"])
        return m3.FSMTable(cfg)
