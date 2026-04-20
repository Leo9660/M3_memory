#!/usr/bin/env python
"""
Recall@k benchmark identical to recall_vs_faiss.py but automatically tunes
M3MultiGpuBackend neighbourhood-k values based on --top-k.

K_NEIGHBOURHOOD maps each supported k to (l0_neighborhood_k, l1_neighborhood_k).
Only the M3 backend is affected; GT-faiss and test-faiss are unchanged.

Usage:
  python recall_vs_faiss_nb.py \\
      --faiss-index /data/IVF.index \\
      --top-k 10 \\
      [all other recall_vs_faiss.py flags]
"""

from __future__ import annotations

# import os
# os.environ.setdefault("OMP_NUM_THREADS",      "32")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Neighbourhood-k table: top-k → (l0_neighborhood_k, l1_neighborhood_k)
# ---------------------------------------------------------------------------
K_NEIGHBOURHOOD: Dict[int, Tuple[int, int]] = {
    1:  (1,  3),
    5:  (3,  8),
    10: (8,  12),
    20: (10, 25),
}

# ---------------------------------------------------------------------------
# Parse backend / mode / bench-dir from sys.argv early so that M3_PROFILE_DIR
# can be set before the C++ module loads (M3Profiler singleton reads it at
# import time).
# ---------------------------------------------------------------------------
_BACKEND_CHOICES = ["m3", "m3multi", "m3multigpu", "faiss"]
_default_backend = "m3multigpu"
_mode_early      = "item_search_insert"
_bench_dir_early = None
_aet_early       = None
_aer_early       = None

_argv = sys.argv[1:]
_i = 0
while _i < len(_argv):
    _a = _argv[_i]
    def _next(a=_a):
        return _argv[_i + 1] if _i + 1 < len(_argv) else None
    if _a in ("--m3-backend", "--m3_backend"):
        if _next(): _default_backend = _next(); _i += 1
    elif _a.startswith(("--m3-backend=", "--m3_backend=")):
        _default_backend = _a.split("=", 1)[1]
    elif _a == "--mode":
        if _next(): _mode_early = _next(); _i += 1
    elif _a.startswith("--mode="):
        _mode_early = _a.split("=", 1)[1]
    elif _a == "--bench-dir":
        if _next(): _bench_dir_early = _next(); _i += 1
    elif _a.startswith("--bench-dir="):
        _bench_dir_early = _a.split("=", 1)[1]
    elif _a == "--alpha-et":
        if _next(): _aet_early = _next(); _i += 1
    elif _a.startswith("--alpha-et="):
        _aet_early = _a.split("=", 1)[1]
    elif _a == "--alpha-et-adapt-rate":
        if _next(): _aer_early = _next(); _i += 1
    elif _a.startswith("--alpha-et-adapt-rate="):
        _aer_early = _a.split("=", 1)[1]
    _i += 1
del _argv, _i, _a

# Build run dir: <MM_DD>/recall_vs_faiss_out/<HH_MM_SS>
_now = datetime.now()
if _bench_dir_early:
    _run_dir = Path(_bench_dir_early).expanduser()
else:
    _run_dir = (Path(__file__).parent
                / _now.strftime("%m_%d")
                / "recall_vs_faiss_out"
                / _now.strftime("%H_%M_%S"))

_run_dir.mkdir(parents=True, exist_ok=True)

# Always enable M3 profiler; direct all its CSVs into the run dir.
os.environ["M3_PROFILE"]        = "1"
os.environ["M3_PROFILE_DIR"]    = str(_run_dir)
os.environ.setdefault("M3_PROFILE_PREFIX", _default_backend)

from AgentMemory.interface import MemoryManagement
from AgentMemory.types import MemoryItem, Metric

DATASET_LOADERS: Dict[str, Any] = {}
try:
    from dataset import (
        AgentGymDataset,
        GSM8KReasoningDataset,
        PRMStepwiseDataset,
        UltraChatDataset,
        UltraFeedbackDataset,
        XLAMFunctionCallingDataset,
    )
    DATASET_LOADERS = {
        "agentgym":              AgentGymDataset,
        "gsm8k":                 GSM8KReasoningDataset,
        "prm800k":               PRMStepwiseDataset,
        "ultrachat":             UltraChatDataset,
        "ultrafeedback":         UltraFeedbackDataset,
        "xlam_function_calling": XLAMFunctionCallingDataset,
    }
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bench logger
# ---------------------------------------------------------------------------

class BenchLogger:
    def __init__(self, run_dir: Path) -> None:
        pass

    def log(self, msg: str) -> None:
        print(msg)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class VectorPassthroughEncoder:
    def __init__(self, dim: int, normalize: bool = False) -> None:
        self.dim = int(dim)
        self.normalize = normalize

    def _encode(self, items: List[MemoryItem]) -> np.ndarray:
        mat = np.vstack([np.asarray(it.data, dtype=np.float32).reshape(-1) for it in items])
        if self.normalize and mat.size:
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
            mat = mat / norms
        return mat

    def encode_items(self, items: List[MemoryItem]) -> np.ndarray:
        return self._encode(items)

    def encode_queries(self, items: List[MemoryItem]) -> np.ndarray:
        return self._encode(items)


def chunk(lst: list, size: int) -> Iterable[list]:
    for i in range(0, len(lst), size):
        yield lst[i: i + size]


def metric_from_str(name: str) -> Metric:
    k = name.strip().lower()
    if k in ("cos", "cosine"):
        return Metric.COSINE
    if k in ("ip", "inner", "dot"):
        return Metric.IP
    if k in ("l2", "euclidean"):
        return Metric.L2
    raise ValueError(f"Unknown metric: {name!r}")


def _extract_text(entry: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(entry, Mapping):
        return None
    for key in ("text", "user", "assistant", "human", "gpt", "question", "answer"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def flatten_dataset(name: str, split: Optional[str], limit: Optional[int]) -> List[str]:
    cls = DATASET_LOADERS[name]
    kw: Dict[str, Any] = {}
    if split:
        kw["split"] = split
    if limit and limit > 0:
        kw["limit"] = limit
    loader = cls(**kw)
    texts: List[str] = []
    for agent in loader.get_data():
        for req in agent.get("requests", []):
            for entry in req.get("trace", []):
                t = _extract_text(entry)
                if t:
                    texts.append(t)
                    if limit and len(texts) >= limit:
                        return texts
    return texts


# ---------------------------------------------------------------------------
# Recall computation
# ---------------------------------------------------------------------------

def recall_at_k(
    hits: List[List[Any]],
    gt_hits: List[List[Any]],
    k: int,
) -> Tuple[float, List[float]]:
    """
    Return (mean_recall, per_query_recall) where recall for one query is
    |top-k(hits) ∩ top-k(gt)| / k.
    """
    per_query: List[float] = []
    for q, gt_q in zip(hits, gt_hits):
        gt_ids = {h.id for h in gt_q[:k]}
        if not gt_ids:
            per_query.append(1.0)
            continue
        hit_ids = {h.id for h in q[:k]}
        per_query.append(len(hit_ids & gt_ids) / len(gt_ids))
    return float(np.mean(per_query)) if per_query else 0.0, per_query


# ---------------------------------------------------------------------------
# TripleRunner: GT faiss + test faiss + test m3/other backend
# ---------------------------------------------------------------------------

class TripleRunner:
    """
    Keeps three MemoryManagement instances in lockstep.
      mm_gt    — FaissBackend @ gt_nprobe  (ground truth, latency not reported)
      mm_faiss — FaissBackend @ faiss_nprobe (test, recall vs GT reported)
      mm_m3    — M3/other     @ m3_nprobe   (test, recall vs GT reported)
    """

    def __init__(
        self,
        mm_gt:       MemoryManagement,
        mm_faiss:    MemoryManagement,
        mm_m3:       MemoryManagement,
        idx_gt:      int,
        idx_faiss:   int,
        idx_m3:      int,
        k:           int,
        gt_nprobe:   int,
        faiss_nprobe: int,
        m3_nprobe:   int,
        ops_per_run: int,
        logger:      Optional[BenchLogger] = None,
    ) -> None:
        self.mm_gt     = mm_gt
        self.mm_faiss  = mm_faiss
        self.mm_m3     = mm_m3
        self.idx_gt    = idx_gt
        self.idx_faiss = idx_faiss
        self.idx_m3    = idx_m3
        self.k            = k
        self.gt_nprobe    = gt_nprobe
        self.faiss_nprobe = faiss_nprobe
        self.m3_nprobe    = m3_nprobe
        self.ops_per_run  = ops_per_run
        self.logger       = logger

        self._queued    = 0
        self._batch_idx = 0

        # recall accumulators
        self.total_queries    = 0
        self.recall_faiss_sum = 0.0
        self.recall_m3_sum    = 0.0

        # timing accumulators (GT not tracked)
        self._faiss_time:   float = 0.0
        self._m3_time:      float = 0.0
        self._total_inserts: int  = 0

        # CSV writer (set by caller)
        self.csv_writer: Optional[csv.writer] = None  # type: ignore[type-arg]


    # --- public API --------------------------------------------------------

    def add_search(self, items: List[MemoryItem], rid_prefix: str = "s") -> None:
        rid = f"{rid_prefix}-{self._batch_idx}"
        self.mm_gt.add_search(   self.idx_gt,    items, self.k, nprobe=self.gt_nprobe,    request_id=rid)
        self.mm_faiss.add_search(self.idx_faiss,  items, self.k, nprobe=self.faiss_nprobe, request_id=rid)
        self.mm_m3.add_search(   self.idx_m3,     items, self.k, nprobe=self.m3_nprobe,    request_id=rid)
        self._queued += 1
        self._maybe_flush()

    def add_insert(self, items: List[MemoryItem]) -> None:
        self.mm_gt.add_insert(   self.idx_gt,    items)
        self.mm_faiss.add_insert(self.idx_faiss,  items)
        self.mm_m3.add_insert(   self.idx_m3,     items)
        self._total_inserts += len(items)
        self._queued += 1
        self._maybe_flush()

    def flush_remaining(self) -> None:
        if self.mm_m3.queue:
            self._flush()

    # --- internals ---------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)

    def _maybe_flush(self) -> None:
        if self.ops_per_run > 0 and self._queued >= self.ops_per_run:
            self._flush()

    def _flush(self) -> None:
        self._batch_idx += 1

        # GT — run first, not timed for reporting
        res_gt    = self.mm_gt.run()

        t0 = time.perf_counter()
        res_faiss = self.mm_faiss.run()
        t_faiss   = time.perf_counter() - t0
        self._faiss_time += t_faiss

        t0 = time.perf_counter()
        res_m3    = self.mm_m3.run()
        t_m3      = time.perf_counter() - t0
        self._m3_time += t_m3

        self._queued = 0

        for rid in res_gt.searches:
            gt = res_gt.searches[rid]
            if not gt:
                continue

            faiss_res = res_faiss.searches.get(rid)
            m3_res    = res_m3.searches.get(rid)

            if not faiss_res and not m3_res:
                continue

            n_q = len(gt)

            recall_faiss = 0.0
            if faiss_res:
                recall_faiss, _ = recall_at_k(faiss_res, gt, self.k)

            recall_m3 = 0.0
            if m3_res:
                recall_m3, _ = recall_at_k(m3_res, gt, self.k)

            self.total_queries    += n_q
            self.recall_faiss_sum += recall_faiss * n_q
            self.recall_m3_sum    += recall_m3    * n_q

            cum_faiss = self.recall_faiss_sum / max(self.total_queries, 1)
            cum_m3    = self.recall_m3_sum    / max(self.total_queries, 1)

            faiss_lat_ms = t_faiss / n_q * 1e3
            m3_lat_ms    = t_m3    / n_q * 1e3

            self._log(
                f"  [batch {self._batch_idx:04d}] rid={rid!r}  nq={n_q}  "
                f"recall_faiss@{self.k}={recall_faiss:.4f}(cum={cum_faiss:.4f})  "
                f"recall_m3@{self.k}={recall_m3:.4f}(cum={cum_m3:.4f})  "
                f"faiss={t_faiss*1e3:.1f}ms({faiss_lat_ms:.2f}ms/q)  "
                f"m3={t_m3*1e3:.1f}ms({m3_lat_ms:.2f}ms/q)"
            )

            if self.csv_writer is not None:
                self.csv_writer.writerow([
                    self._batch_idx, rid, n_q,
                    f"{recall_faiss:.6f}", f"{cum_faiss:.6f}",
                    f"{recall_m3:.6f}",    f"{cum_m3:.6f}",
                    f"{t_faiss*1e3:.3f}",  f"{faiss_lat_ms:.3f}",
                    f"{t_m3*1e3:.3f}",     f"{m3_lat_ms:.3f}",
                ])


    # --- aggregate properties ----------------------------------------------

    @property
    def cumulative_recall_faiss(self) -> float:
        return self.recall_faiss_sum / max(self.total_queries, 1)

    @property
    def cumulative_recall_m3(self) -> float:
        return self.recall_m3_sum / max(self.total_queries, 1)

    @property
    def faiss_search_throughput(self) -> float:
        return self.total_queries / self._faiss_time if self._faiss_time > 0 else 0.0

    @property
    def m3_search_throughput(self) -> float:
        return self.total_queries / self._m3_time if self._m3_time > 0 else 0.0

    @property
    def faiss_insert_throughput(self) -> float:
        return self._total_inserts / self._faiss_time if self._faiss_time > 0 else 0.0

    @property
    def m3_insert_throughput(self) -> float:
        return self._total_inserts / self._m3_time if self._m3_time > 0 else 0.0

    @property
    def faiss_search_latency_ms(self) -> float:
        return self._faiss_time / self.total_queries * 1e3 if self.total_queries > 0 else 0.0

    @property
    def m3_search_latency_ms(self) -> float:
        return self._m3_time / self.total_queries * 1e3 if self.total_queries > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@k benchmark with per-k neighbourhood tuning for M3MultiGpu."
    )
    parser.add_argument("--faiss-index", default="/data/IVF.index",
                        help="Path to Faiss IVF checkpoint (default: /data/IVF.index).")
    parser.add_argument("--faiss-normalized", action="store_true", default=True,
                        help="Vectors in the checkpoint are already L2-normalised (default: True).")
    parser.add_argument("--no-faiss-normalized", dest="faiss_normalized", action="store_false",
                        help="Disable --faiss-normalized.")
    parser.add_argument("--m3-backend", default="m3multigpu",
                        choices=["m3", "m3multi", "m3multigpu", "faiss"],
                        help="Which M3 backend variant to test.")
    parser.add_argument("--metric", default="l2",
                        help="cosine / ip / l2 (default: l2 to match /data/IVF.index)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k for search. Also drives neighbourhood-k lookup in K_NEIGHBOURHOOD.")
    parser.add_argument("--nprobe", type=int, default=64,
                        help="nprobe for the M3/test backend.")
    parser.add_argument("--faiss-nprobe", type=int, default=64,
                        help="nprobe for the test Faiss backend.")
    parser.add_argument("--gt-nprobe", type=int, default=512,
                        help="nprobe for the GT Faiss backend (ground truth).")
    parser.add_argument("--insert-batch", type=int, default=512)
    parser.add_argument("--search-batch", type=int, default=256)
    parser.add_argument("--ops-per-run", type=int, default=64,
                        help="Flush to all backends after this many queued ops.")
    parser.add_argument("--mode",
                        choices=["search_only", "item_search_insert", "step_search_then_insert"],
                        default="item_search_insert",
                        help="Search/insert replay pattern.")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None,
                        help="Text dataset to encode as query vectors.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of items to load from the dataset.")
    parser.add_argument("--dim", type=int, default=1024,
                        help="Dimension for synthetic query vectors (ignored when --dataset is set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench-dir", default=None,
                        help="Override bench output directory (default: bench/<tag>/ next to script).")
    parser.add_argument("--alpha-et", type=float, default=None,
                        help="Override M3MultiGpuBackend alpha_et.")
    parser.add_argument("--alpha-et-adapt-rate", type=float, default=None,
                        help="Override M3MultiGpuBackend alpha_et_adapt_rate.")
    args = parser.parse_args()

    # Patch M3MultiGpuBackend class defaults before the backend is instantiated.
    from AgentMemory.backend.m3 import M3MultiGpuBackend as _M3MG

    # Apply neighbourhood-k overrides from the lookup table.
    nb = K_NEIGHBOURHOOD.get(args.top_k)
    if nb is not None:
        l0_nb, l1_nb = nb
        _M3MG.DEFAULTS["l0_neighborhood_k"] = l0_nb
        _M3MG.DEFAULTS["l1_neighborhood_k"] = l1_nb
    else:
        l0_nb = _M3MG.DEFAULTS["l0_neighborhood_k"]
        l1_nb = _M3MG.DEFAULTS["l1_neighborhood_k"]

    if args.alpha_et is not None:
        _M3MG.DEFAULTS["alpha_et"] = args.alpha_et
    if args.alpha_et_adapt_rate is not None:
        _M3MG.DEFAULTS["alpha_et_adapt_rate"] = args.alpha_et_adapt_rate

    # Use the run dir computed at module load time (M3_PROFILE_DIR already set)
    run_dir = _run_dir
    logger  = BenchLogger(run_dir)

    logger.log(
        f"[nb] top_k={args.top_k}  "
        f"l0_neighborhood_k={l0_nb}  l1_neighborhood_k={l1_nb}"
        + ("" if nb is not None else "  (not in K_NEIGHBOURHOOD, using defaults)")
    )

    # per-batch CSV
    batches_path = run_dir / "recall_latency.csv"
    batches_f    = batches_path.open("w", newline="", encoding="utf-8")
    csv_w        = csv.writer(batches_f)
    csv_w.writerow([
        "batch", "request_id", "n_queries",
        f"recall_faiss@{args.top_k}", "cumul_recall_faiss",
        f"recall_m3@{args.top_k}",    "cumul_recall_m3",
        "faiss_ms", "faiss_lat_ms_per_q",
        "m3_ms",    "m3_lat_ms_per_q",
    ])

    logger.log(f"[init] bench run dir: {run_dir}")
    logger.log(f"[init] nprobe  gt={args.gt_nprobe}  faiss={args.faiss_nprobe}  m3={args.nprobe}")

    metric     = metric_from_str(args.metric)
    faiss_path = Path(args.faiss_index).expanduser()
    if not faiss_path.is_file():
        raise FileNotFoundError(f"Faiss index not found: {faiss_path}")

    # Sanity-check on-disk metric
    try:
        import faiss as _faiss
        _idx = _faiss.read_index(str(faiss_path), _faiss.IO_FLAG_MMAP)
        _ivf = _faiss.extract_index_ivf(_idx)
        _disk_metric = {0: "ip", 1: "l2"}.get(int(_ivf.metric_type), str(_ivf.metric_type))
        _disk_ntotal = int(_ivf.ntotal)
        _disk_nlist  = int(_ivf.nlist)
        _disk_d      = int(_ivf.d)
        del _idx, _ivf
    except Exception as _e:
        _disk_metric = "unknown"
        _disk_ntotal = _disk_nlist = _disk_d = -1
        logger.log(f"[warn] could not inspect index file: {_e}")

    logger.log(
        f"[sanity] index: ntotal={_disk_ntotal}, nlist={_disk_nlist}, d={_disk_d}, "
        f"on-disk metric={_disk_metric}"
    )
    logger.log(f"[sanity] using metric={args.metric}  faiss_normalized={args.faiss_normalized}")
    if _disk_metric != "unknown" and _disk_metric != args.metric:
        logger.log(
            f"[warn] on-disk metric ({_disk_metric}) != --metric ({args.metric}). "
            "Both backends will use --metric; scores may not be comparable to the "
            "original index training objective."
        )

    # Build query / insert items
    if args.dataset:
        if args.dataset not in DATASET_LOADERS:
            raise ValueError(f"Dataset {args.dataset!r} not available.")
        limit = args.limit if args.limit > 0 else None
        logger.log(f"[init] loading dataset={args.dataset} limit={limit} ...")
        texts = flatten_dataset(args.dataset, args.split, limit)
        if not texts:
            raise RuntimeError("No texts loaded from dataset.")
        logger.log(f"[init] encoding {len(texts)} texts ...")
        from AgentMemory.encoder import TransformerEncoder
        enc        = TransformerEncoder(normalize=args.faiss_normalized)
        raw_items  = [MemoryItem(id=f"q-{i}", data=t) for i, t in enumerate(texts)]
        vecs       = enc.encode_items(raw_items)
        dim        = vecs.shape[1]
        logger.log(f"[init] encoded dim={dim}")
        _mid = len(vecs) // 2
        insert_items = [MemoryItem(id=f"ins-{i}", data=vecs[i])        for i in range(_mid)]
        query_items  = [MemoryItem(id=f"q-{i}",   data=vecs[_mid + i]) for i in range(len(vecs) - _mid)]
        encoder      = VectorPassthroughEncoder(dim=dim, normalize=False)
        logger.log(f"[init] split: {len(insert_items)} insert items, {len(query_items)} query items (non-overlapping)")
    else:
        rng = np.random.default_rng(args.seed)
        dim = args.dim
        n   = args.limit if args.limit > 0 else 8192
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        if args.faiss_normalized:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs /= norms
        _mid = n // 2
        insert_items = [MemoryItem(id=f"ins-{i}", data=vecs[i])        for i in range(_mid)]
        query_items  = [MemoryItem(id=f"q-{i}",   data=vecs[_mid + i]) for i in range(n - _mid)]
        encoder      = VectorPassthroughEncoder(dim=dim, normalize=False)
        logger.log(f"[init] synthetic vectors: n={n}, dim={dim}, split at {_mid}")

    # Instantiate all three backends
    logger.log(f"[init] creating GT faiss backend (nprobe={args.gt_nprobe}) ...")
    mm_gt = MemoryManagement(backend="faiss", encoder=encoder,
                             default_nprobe=args.gt_nprobe)

    logger.log(f"[init] creating test faiss backend (nprobe={args.faiss_nprobe}) ...")
    mm_faiss = MemoryManagement(backend="faiss", encoder=encoder,
                                default_nprobe=args.faiss_nprobe)

    logger.log(f"[init] creating {args.m3_backend} backend (nprobe={args.nprobe}) ...")
    mm_m3 = MemoryManagement(backend=args.m3_backend, encoder=encoder,
                             default_nprobe=args.nprobe)

    idx_gt    = mm_gt.create_index("recall-bench-gt",    metric=metric)
    idx_faiss = mm_faiss.create_index("recall-bench-faiss", metric=metric)
    idx_m3    = mm_m3.create_index("recall-bench",          metric=metric)

    # Confirm what the M3 backend was actually constructed with.
    _l0_nb   = _M3MG.DEFAULTS["l0_neighborhood_k"]
    _l1_nb   = _M3MG.DEFAULTS["l1_neighborhood_k"]
    _k_promo = max(args.top_k, _l1_nb)
    _l0_cap  = min(_l0_nb, args.top_k)
    logger.log(
        f"[m3-config] backend constructed with:"
        f"  l0_neighborhood_k={_l0_nb}"
        f"  l1_neighborhood_k={_l1_nb}"
        f"  → k_promo=max(k={args.top_k}, l1_nb={_l1_nb})={_k_promo}"
        f"  → l0_cap=min(l0_nb={_l0_nb}, k={args.top_k})={_l0_cap}"
        + (f"  [WARN: l0_nb > k, {_l0_nb - args.top_k} l0 slots wasted]"
           if _l0_nb > args.top_k else "")
        + (f"  [WARN: k_promo={_k_promo} > k={args.top_k}, L2 fetches {_k_promo/args.top_k:.1f}x overhead]"
           if _k_promo > args.top_k else "")
    )

    # Hydrate all three from the same Faiss checkpoint
    for label, mm, idx in [
        ("GT faiss",       mm_gt,    idx_gt),
        ("test faiss",     mm_faiss, idx_faiss),
        (args.m3_backend,  mm_m3,    idx_m3),
    ]:
        logger.log(f"[init] loading Faiss checkpoint into {label} ...")
        t0 = time.perf_counter()
        mm.rebuild_index_from_faiss(idx, path=str(faiss_path),
                                    normalized=args.faiss_normalized)
        logger.log(f"[init]   {label} load: {time.perf_counter()-t0:.2f}s")

    # Triple runner
    runner = TripleRunner(
        mm_gt=mm_gt,       mm_faiss=mm_faiss,       mm_m3=mm_m3,
        idx_gt=idx_gt,     idx_faiss=idx_faiss,      idx_m3=idx_m3,
        k=args.top_k,
        gt_nprobe=args.gt_nprobe,
        faiss_nprobe=args.faiss_nprobe,
        m3_nprobe=args.nprobe,
        ops_per_run=args.ops_per_run,
        logger=logger,
    )
    runner.csv_writer = csv_w

    search_batches = list(chunk(query_items,  args.search_batch))
    insert_batches = list(chunk(insert_items, args.insert_batch))

    logger.log(f"[run] mode={args.mode}  search_batches={len(search_batches)}  "
               f"insert_batches={len(insert_batches)}")

    if args.mode == "search_only":
        for sb in search_batches:
            runner.add_search(sb)

    elif args.mode == "item_search_insert":
        ins_q = list(insert_batches)
        for i, sb in enumerate(search_batches):
            runner.add_search(sb, rid_prefix=f"s{i}")
            if ins_q:
                runner.add_insert(ins_q.pop(0))
        for ib in ins_q:
            runner.add_insert(ib)

    elif args.mode == "step_search_then_insert":
        for i, sb in enumerate(search_batches):
            runner.add_search(sb, rid_prefix=f"s{i}")
        for ib in insert_batches:
            runner.add_insert(ib)

    runner.flush_remaining()
    batches_f.close()

    # Summary
    logger.log(
        f"\n[summary] total_queries={runner.total_queries}  "
        f"total_inserts={runner._total_inserts}"
    )
    logger.log(
        f"[summary] final recall@{args.top_k}:  "
        f"faiss={runner.cumulative_recall_faiss:.4f}  "
        f"m3={runner.cumulative_recall_m3:.4f}"
    )

    col = 22
    logger.log(
        f"\n{'':>{col}}  {'recall@'+str(args.top_k):>10}  "
        f"{'throughput (search)':>22}  {'latency (search)':>18}  "
        f"{'throughput (insert)':>22}  {'wall time':>12}"
    )
    logger.log(f"  {'─'*106}")

    faiss_label = f"faiss(np={args.faiss_nprobe})"
    m3_label    = f"{args.m3_backend}(np={args.nprobe})"
    logger.log(
        f"  {faiss_label:>{col}}  "
        f"{runner.cumulative_recall_faiss:>10.4f}  "
        f"{runner.faiss_search_throughput:>19,.1f} q/s  "
        f"{runner.faiss_search_latency_ms:>14.3f} ms/q  "
        f"{runner.faiss_insert_throughput:>19,.1f} vec/s  "
        f"{runner._faiss_time*1e3:>10.1f} ms"
    )
    logger.log(
        f"  {m3_label:>{col}}  "
        f"{runner.cumulative_recall_m3:>10.4f}  "
        f"{runner.m3_search_throughput:>19,.1f} q/s  "
        f"{runner.m3_search_latency_ms:>14.3f} ms/q  "
        f"{runner.m3_insert_throughput:>19,.1f} vec/s  "
        f"{runner._m3_time*1e3:>10.1f} ms"
    )
    if runner.faiss_search_throughput > 0:
        ratio = runner.m3_search_throughput / runner.faiss_search_throughput
        logger.log(f"\n  [summary] m3 speedup vs test-faiss: {ratio:.2f}x  (search throughput)")

    logger.log(f"[summary] outputs written to {run_dir}")
    logger.close()


if __name__ == "__main__":
    main()
