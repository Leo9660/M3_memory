#!/usr/bin/env python
"""
Recall@k benchmark: M3MultiGpu vs FaissBackend ground truth.

Workflow:
  1. Load a Faiss IVF checkpoint into both backends (m3multigpu + faiss).
  2. Optionally encode dataset texts into query vectors (or use synthetic).
  3. Replay interleaved search/insert batches.
  4. For each search batch: compare M3 result IDs against Faiss result IDs
     and report recall@k per batch and cumulative.

Outputs (always written to bench/<run>/):
  console.csv  — every log line with elapsed time and tag
  batches.csv  — per-batch recall + latency for both backends

Usage:
  python recall_vs_faiss.py \
      --faiss-index /path/to/index.faiss \
      --dataset agentgym --limit 4096 \
      --top-k 10 --nprobe 64 \
      --search-batch 128 --insert-batch 512 \
      --mode item_search_insert
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS",      "32")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import argparse
import csv
import os
import re as _re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set M3_PROFILE_PREFIX before the C++ module loads (M3Profiler is a singleton
# that reads this env var in its constructor, called at import time).
_BACKEND_CHOICES = ["m3", "m3multi", "m3multigpu"]
_default_backend = "m3multigpu"
for _i, _arg in enumerate(sys.argv):
    if _arg in ("--m3-backend", "--m3_backend") and _i + 1 < len(sys.argv):
        _default_backend = sys.argv[_i + 1]
        break
    if _arg.startswith("--m3-backend=") or _arg.startswith("--m3_backend="):
        _default_backend = _arg.split("=", 1)[1]
        break
if "M3_PROFILE_PREFIX" not in os.environ:
    os.environ["M3_PROFILE_PREFIX"] = _default_backend

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
        "agentgym": AgentGymDataset,
        "gsm8k": GSM8KReasoningDataset,
        "prm800k": PRMStepwiseDataset,
        "ultrachat": UltraChatDataset,
        "ultrafeedback": UltraFeedbackDataset,
        "xlam_function_calling": XLAMFunctionCallingDataset,
    }
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Bench logger — tees every log line to stdout AND bench/<run>/console.csv
# ---------------------------------------------------------------------------

_TAG_RE = _re.compile(r"^\s*\[([^\]]+)\]")


class BenchLogger:
    """Writes every log line to stdout and to bench/<run>/console.csv."""

    def __init__(self, run_dir: Path) -> None:
        self._start = time.perf_counter()
        run_dir.mkdir(parents=True, exist_ok=True)
        self._f = (run_dir / "console.csv").open("w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["elapsed_s", "tag", "message"])
        self._f.flush()

    def log(self, msg: str) -> None:
        print(msg)
        elapsed = time.perf_counter() - self._start
        m = _TAG_RE.match(msg)
        tag = m.group(1) if m else ""
        self._w.writerow([f"{elapsed:.3f}", tag, msg.strip()])
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ---------------------------------------------------------------------------
# Helpers shared with ratio_throughput_index_only
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
    m3_hits: List[List[Any]],   # list[query] -> list[SearchHit]
    gt_hits: List[List[Any]],   # same shape, from Faiss ground truth
    k: int,
) -> Tuple[float, List[float]]:
    """
    Return (mean_recall, per_query_recall) where recall for one query is
    |top-k(m3) ∩ top-k(gt)| / k.
    """
    per_query: List[float] = []
    for m3_q, gt_q in zip(m3_hits, gt_hits):
        gt_ids = {h.id for h in gt_q[:k]}
        if not gt_ids:
            per_query.append(1.0)
            continue
        m3_ids = {h.id for h in m3_q[:k]}
        per_query.append(len(m3_ids & gt_ids) / len(gt_ids))
    return float(np.mean(per_query)) if per_query else 0.0, per_query


# ---------------------------------------------------------------------------
# Dual-backend flush: issues the same op to both backends, returns searches
# ---------------------------------------------------------------------------

class DualRunner:
    """
    Keeps two MemoryManagement instances in lockstep.
    The FaissBackend MM is the ground-truth source; m3 MM is under test.
    """

    def __init__(
        self,
        mm_m3: MemoryManagement,
        mm_faiss: MemoryManagement,
        index_id_m3: int,
        index_id_faiss: int,
        k: int,
        nprobe: int,
        ops_per_run: int,
        logger: Optional[BenchLogger] = None,
    ) -> None:
        self.mm_m3 = mm_m3
        self.mm_faiss = mm_faiss
        self.idx_m3 = index_id_m3
        self.idx_faiss = index_id_faiss
        self.k = k
        self.nprobe = nprobe
        self.ops_per_run = ops_per_run
        self.logger = logger
        self._queued = 0
        self._batch_idx = 0

        # running recall accumulators
        self.total_queries = 0
        self.recall_sum = 0.0

        # throughput accumulators
        self._m3_time: float = 0.0        # total seconds in mm_m3.run()
        self._faiss_time: float = 0.0     # total seconds in mm_faiss.run()
        self._total_inserts: int = 0      # total vectors inserted (both backends)

        # per-batch CSV writer (set by caller)
        self.csv_writer: Optional[csv.writer] = None  # type: ignore[type-arg]

        # recall_diag appender: writes RECALL_BATCH rows into the C++ profiler's
        # recall_diag CSV when M3_PROFILE=1.  Reuses existing columns:
        #   cid=batch, gpu_n=n_queries, buf_n=recall@k, l2_n=cumul_recall,
        #   invisible_n=m3_ms, cumul_overflows=faiss_ms
        self._recall_diag_f = None
        self._recall_diag_w = None
        if os.environ.get("M3_PROFILE"):
            prof_dir = os.environ.get("M3_PROFILE_DIR", "profile")
            import glob as _glob
            matches = sorted(_glob.glob(os.path.join(prof_dir, "*_recall_diag.csv")))
            if matches:
                self._recall_diag_f = open(matches[-1], "a", newline="", encoding="utf-8")
                self._recall_diag_w = csv.writer(self._recall_diag_f)

    # --- public API --------------------------------------------------------

    def add_search(self, items: List[MemoryItem], rid_prefix: str = "s") -> None:
        rid = f"{rid_prefix}-{self._batch_idx}"
        self.mm_m3.add_search(self.idx_m3, items, self.k, nprobe=self.nprobe, request_id=rid)
        self.mm_faiss.add_search(self.idx_faiss, items, self.k, nprobe=self.nprobe, request_id=rid)
        self._queued += 1
        self._maybe_flush()

    def add_insert(self, items: List[MemoryItem]) -> None:
        self.mm_m3.add_insert(self.idx_m3, items)
        self.mm_faiss.add_insert(self.idx_faiss, items)
        self._total_inserts += len(items)
        self._queued += 1
        self._maybe_flush()

    def flush_remaining(self) -> None:
        if self.mm_m3.queue:
            self._flush()
        if self._recall_diag_f is not None:
            self._recall_diag_f.close()
            self._recall_diag_f = None

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
        t0 = time.perf_counter()
        res_faiss = self.mm_faiss.run()
        t_faiss = time.perf_counter() - t0
        self._faiss_time += t_faiss

        t0 = time.perf_counter()
        res_m3 = self.mm_m3.run()
        t_m3 = time.perf_counter() - t0
        self._m3_time += t_m3

        self._queued = 0

        # Match search results by request_id
        for rid in res_faiss.searches:
            if rid not in res_m3.searches:
                self._log(f"  [warn] request_id {rid!r} missing from m3 results — skipping")
                continue
            gt = res_faiss.searches[rid]
            m3 = res_m3.searches[rid]
            if not gt:
                continue
            mean_rec, _ = recall_at_k(m3, gt, self.k)
            n_q = len(gt)
            self.total_queries += n_q
            self.recall_sum += mean_rec * n_q

            cum_recall = self.recall_sum / max(self.total_queries, 1)
            m3_lat_ms = t_m3 / n_q * 1e3
            faiss_lat_ms = t_faiss / n_q * 1e3
            self._log(
                f"  [batch {self._batch_idx:04d}] rid={rid!r}  "
                f"nq={n_q}  recall@{self.k}={mean_rec:.4f}  cum={cum_recall:.4f}  "
                f"m3={t_m3*1e3:.1f}ms({m3_lat_ms:.2f}ms/q)  "
                f"faiss={t_faiss*1e3:.1f}ms({faiss_lat_ms:.2f}ms/q)"
            )
            if self.csv_writer is not None:
                self.csv_writer.writerow([
                    self._batch_idx, rid, n_q,
                    f"{mean_rec:.6f}", f"{cum_recall:.6f}",
                    f"{t_faiss*1e3:.3f}", f"{t_faiss/n_q*1e3:.3f}",
                    f"{t_m3*1e3:.3f}", f"{t_m3/n_q*1e3:.3f}",
                ])
            if self._recall_diag_w is not None:
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self._recall_diag_w.writerow([
                    ts, "RECALL_BATCH",
                    self._batch_idx, n_q,
                    f"{mean_rec:.6f}", f"{cum_recall:.6f}",
                    f"{t_m3*1e3:.3f}", f"{t_faiss*1e3:.3f}",
                ])
                self._recall_diag_f.flush()

    @property
    def cumulative_recall(self) -> float:
        return self.recall_sum / max(self.total_queries, 1)

    @property
    def search_throughput(self) -> float:
        """Queries/sec for m3 (wall-clock time in mm_m3.run() calls)."""
        return self.total_queries / self._m3_time if self._m3_time > 0 else 0.0

    @property
    def insert_throughput(self) -> float:
        """Vectors/sec for m3 inserts (wall-clock time in mm_m3.run() calls)."""
        return self._total_inserts / self._m3_time if self._m3_time > 0 else 0.0

    @property
    def faiss_search_throughput(self) -> float:
        """Queries/sec for faiss (wall-clock time in mm_faiss.run() calls)."""
        return self.total_queries / self._faiss_time if self._faiss_time > 0 else 0.0

    @property
    def faiss_insert_throughput(self) -> float:
        """Vectors/sec for faiss inserts (wall-clock time in mm_faiss.run() calls)."""
        return self._total_inserts / self._faiss_time if self._faiss_time > 0 else 0.0

    @property
    def search_latency_ms(self) -> float:
        """Mean ms per query for m3."""
        return self._m3_time / self.total_queries * 1e3 if self.total_queries > 0 else 0.0

    @property
    def faiss_search_latency_ms(self) -> float:
        """Mean ms per query for faiss."""
        return self._faiss_time / self.total_queries * 1e3 if self.total_queries > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recall@k benchmark: M3MultiGpu vs FaissBackend ground truth."
    )
    parser.add_argument("--faiss-index", default="/data/IVF.index",
                        help="Path to Faiss IVF checkpoint (default: /data/IVF.index).")
    parser.add_argument("--faiss-normalized", action="store_true", default=True,
                        help="Vectors in the checkpoint are already L2-normalised (default: True).")
    parser.add_argument("--no-faiss-normalized", dest="faiss_normalized", action="store_false",
                        help="Disable --faiss-normalized.")
    parser.add_argument("--m3-backend", default="m3multigpu",
                        choices=["m3", "m3multi", "m3multigpu"],
                        help="Which M3 backend variant to test.")
    parser.add_argument("--metric", default="l2", help="cosine / ip / l2 (default: l2 to match /data/IVF.index)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=64)
    parser.add_argument("--insert-batch", type=int, default=512)
    parser.add_argument("--search-batch", type=int, default=256)
    parser.add_argument("--ops-per-run", type=int, default=64,
                        help="Flush to both backends after this many queued ops.")
    parser.add_argument("--mode",
                        choices=["search_only", "item_search_insert", "step_search_then_insert"],
                        default="item_search_insert",
                        help="Search/insert replay pattern.")
    # Dataset / synthetic query vectors
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None,
                        help="Text dataset to encode as query vectors.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max number of items to load from the dataset.")
    parser.add_argument("--dim", type=int, default=1024,
                        help="Dimension for synthetic query vectors (ignored when --dataset is set).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bench-dir", default=None,
                        help="Override the bench output directory (default: bench/<timestamp>/ "
                             "next to this script).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Set up bench output directory and logger
    # ------------------------------------------------------------------
    script_dir = Path(__file__).parent
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.m3_backend}_{args.mode}"
    run_dir = Path(args.bench_dir).expanduser() if args.bench_dir else script_dir / "bench" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = BenchLogger(run_dir)

    # per-batch CSV
    batches_path = run_dir / "batches.csv"
    batches_f = batches_path.open("w", newline="", encoding="utf-8")
    csv_w = csv.writer(batches_f)
    csv_w.writerow(["batch", "request_id", "n_queries",
                    f"recall@{args.top_k}", "cumulative_recall",
                    "faiss_ms", "faiss_lat_ms_per_q",
                    "m3_ms", "m3_lat_ms_per_q"])

    logger.log(f"[init] bench run dir: {run_dir}")

    metric = metric_from_str(args.metric)
    faiss_path = Path(args.faiss_index).expanduser()
    if not faiss_path.is_file():
        raise FileNotFoundError(f"Faiss index not found: {faiss_path}")

    # ------------------------------------------------------------------
    # Sanity-check: read the on-disk metric and warn if it mismatches
    # ------------------------------------------------------------------
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
    logger.log(
        f"[sanity] using metric={args.metric}  faiss_normalized={args.faiss_normalized}"
    )
    if _disk_metric != "unknown" and _disk_metric != args.metric:
        logger.log(
            f"[warn] on-disk metric ({_disk_metric}) != --metric ({args.metric}). "
            "Both backends will use --metric; scores may not be comparable to the "
            "original index training objective."
        )

    # ------------------------------------------------------------------
    # Build query items
    # ------------------------------------------------------------------
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
        enc = TransformerEncoder(normalize=args.faiss_normalized)
        raw_items = [MemoryItem(id=f"q-{i}", data=t) for i, t in enumerate(texts)]
        vecs = enc.encode_items(raw_items)
        dim = vecs.shape[1]
        logger.log(f"[init] encoded dim={dim}")
        query_items = [MemoryItem(id=f"q-{i}", data=vecs[i]) for i in range(len(vecs))]
        insert_items = query_items  # also use same items as inserts
        encoder = VectorPassthroughEncoder(dim=dim, normalize=False)
    else:
        rng = np.random.default_rng(args.seed)
        dim = args.dim
        n = args.limit if args.limit > 0 else 8192
        vecs = rng.standard_normal((n, dim)).astype(np.float32)
        if args.faiss_normalized:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
            vecs /= norms
        query_items = [MemoryItem(id=f"q-{i}", data=vecs[i]) for i in range(n)]
        insert_items = query_items
        encoder = VectorPassthroughEncoder(dim=dim, normalize=False)
        logger.log(f"[init] synthetic vectors: n={n}, dim={dim}")

    # ------------------------------------------------------------------
    # Instantiate backends
    # ------------------------------------------------------------------
    logger.log(f"[init] creating {args.m3_backend} backend ...")
    mm_m3 = MemoryManagement(backend=args.m3_backend, encoder=encoder,
                             default_nprobe=args.nprobe)
    logger.log("[init] creating faiss backend (ground truth) ...")
    mm_faiss = MemoryManagement(backend="faiss", encoder=encoder,
                                default_nprobe=args.nprobe)

    idx_m3 = mm_m3.create_index("recall-bench", metric=metric)
    idx_faiss = mm_faiss.create_index("recall-bench", metric=metric)

    # ------------------------------------------------------------------
    # Hydrate both backends from the Faiss checkpoint
    # ------------------------------------------------------------------
    logger.log(f"[init] loading Faiss checkpoint into {args.m3_backend} ...")
    t0 = time.perf_counter()
    mm_m3.rebuild_index_from_faiss(idx_m3, path=str(faiss_path),
                                   normalized=args.faiss_normalized)
    logger.log(f"[init]   m3 load: {time.perf_counter()-t0:.2f}s")

    logger.log("[init] loading Faiss checkpoint into faiss backend ...")
    t0 = time.perf_counter()
    mm_faiss.rebuild_index_from_faiss(idx_faiss, path=str(faiss_path),
                                      normalized=args.faiss_normalized)
    logger.log(f"[init]   faiss load: {time.perf_counter()-t0:.2f}s")

    # ------------------------------------------------------------------
    # Dual runner
    # ------------------------------------------------------------------
    runner = DualRunner(
        mm_m3=mm_m3,
        mm_faiss=mm_faiss,
        index_id_m3=idx_m3,
        index_id_faiss=idx_faiss,
        k=args.top_k,
        nprobe=args.nprobe,
        ops_per_run=args.ops_per_run,
        logger=logger,
    )
    runner.csv_writer = csv_w

    search_batches = list(chunk(query_items, args.search_batch))
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

    logger.log(
        f"\n[summary] total_queries={runner.total_queries}  "
        f"final_recall@{args.top_k}={runner.cumulative_recall:.4f}  "
        f"total_inserts={runner._total_inserts}"
    )
    col = 14
    logger.log(f"\n{'':>{col}}  {'throughput (search)':>22}  {'latency (search)':>18}  "
               f"{'throughput (insert)':>22}  {'wall time':>12}")
    logger.log(f"  {'─'*90}")
    logger.log(
        f"  {args.m3_backend:>{col}}  "
        f"{runner.search_throughput:>19,.1f} q/s  "
        f"{runner.search_latency_ms:>14.3f} ms/q  "
        f"{runner.insert_throughput:>19,.1f} vec/s  "
        f"{runner._m3_time*1e3:>10.1f} ms"
    )
    logger.log(
        f"  {'faiss':>{col}}  "
        f"{runner.faiss_search_throughput:>19,.1f} q/s  "
        f"{runner.faiss_search_latency_ms:>14.3f} ms/q  "
        f"{runner.faiss_insert_throughput:>19,.1f} vec/s  "
        f"{runner._faiss_time*1e3:>10.1f} ms"
    )
    if runner.faiss_search_throughput > 0:
        ratio = runner.search_throughput / runner.faiss_search_throughput
        logger.log(f"\n  [summary] speedup vs faiss: {ratio:.2f}x  (search throughput)")
    logger.log(f"[summary] outputs written to {run_dir}")

    logger.close()


if __name__ == "__main__":
    main()
