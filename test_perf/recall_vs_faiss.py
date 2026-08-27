#!/usr/bin/env python
"""
Recall@k benchmark: test backend (M3MultiGpu or other) + optional test Faiss vs GT Faiss ground truth.

Backends run in lockstep:
  GT    faiss  @ --gt-nprobe   (default 512) — ground truth, latency not reported
  test  faiss  @ --faiss-nprobe              — compared against GT  (skipped with --no-test-faiss)
  test  m3/etc @ --nprobe                    — compared against GT

GT results are cached per (dataset, limit, normalized, metric, mode, top_k) so subsequent runs
skip the GT faiss backend entirely and load results from disk.  Pass --no-gt-cache to disable.

Outputs (all written to bench/<backend>_<mode>_<DD>_<HHMMSS>/):
  console.csv                       — every log line with elapsed time and tag
  batches.csv                       — per-batch recall@k + latency for test-faiss and m3 backends
  <prefix>_<ts>_search_profile.csv  — M3 C++ profiler step timings (auto-enabled)
  <prefix>_<ts>_search_stats.csv    — M3 C++ profiler routing/exit stats (auto-enabled)
  <prefix>_<ts>_insert.csv          — M3 C++ profiler insert timings (auto-enabled)
  <prefix>_<ts>_recall_diag.csv     — M3 C++ profiler recall diagnostic rows

Usage:
  python recall_vs_faiss.py \\
      --faiss-index /path/to/index.faiss \\
      --dataset agentgym --limit 4096 \\
      --top-k 10 --nprobe 64 --faiss-nprobe 64 --gt-nprobe 512 \\
      --search-batch 128 --insert-batch 512 \\
      --mode item_search_insert

  # M3 vs GT only, no test-faiss, use cached GT if available:
  python recall_vs_faiss.py --no-test-faiss --dataset agentgym --limit 4096 ...
"""

from __future__ import annotations

# import os
# os.environ.setdefault("OMP_NUM_THREADS",      "32")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

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

# Build run dir: <backend>_<mode>_<DD>_<HHMMSS>
_now = datetime.now()
_run_ts = _now.strftime("%d_%H%M%S")
_aet_tag_e = f"_aet{_aet_early}" if _aet_early else ""
_aer_tag_e = f"_aer{_aer_early}" if _aer_early else ""
_run_tag = f"{_default_backend}_{_mode_early}_{_run_ts}{_aet_tag_e}{_aer_tag_e}"

if _bench_dir_early:
    _run_dir = Path(_bench_dir_early).expanduser()
else:
    _run_dir = Path(__file__).parent / "bench" / _run_tag

_run_dir.mkdir(parents=True, exist_ok=True)

_EMBED_CACHE_DIR = Path(__file__).parent / "embeddings_cache"

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


def encode_texts_cached(texts: List[str], dataset_name: str, limit, normalized: bool) -> np.ndarray:
    """Encode texts to float32 vectors, caching under test_perf/embeddings_cache/."""
    norm_tag = "_norm" if normalized else ""
    suffix   = f"_n{limit}" if limit else "_all"
    cache_path = _EMBED_CACHE_DIR / f"{dataset_name}{suffix}{norm_tag}.npy"
    if cache_path.exists():
        print(f"[encode] loading cached vectors from {cache_path}")
        return np.load(str(cache_path))
    from AgentMemory.encoder import TransformerEncoder
    from AgentMemory.types import MemoryItem as _MI
    enc   = TransformerEncoder(normalize=normalized)
    items = [_MI(id=str(i), data=t) for i, t in enumerate(texts)]
    vecs  = np.asarray(enc.encode_items(items), dtype=np.float32)
    _EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), vecs)
    print(f"[encode] saved {len(vecs)} vectors to {cache_path}")
    return vecs


_GT_CACHE_DIR = Path(__file__).parent / "gt_cache"


def _gt_cache_path(dataset: str, limit, normalized: bool, metric: str, mode: str, top_k: int) -> Path:
    norm_tag   = "_norm" if normalized else ""
    limit_tag  = f"_n{limit}" if limit else "_all"
    return _GT_CACHE_DIR / f"{dataset}{limit_tag}{norm_tag}_{metric}_{mode}_top{top_k}.npz"


def load_gt_cache(dataset: str, limit, normalized: bool, metric: str, mode: str, top_k: int):
    """
    Return dict[request_id -> list[list[str]]] (per-query ordered doc-id lists), or None if absent.
    """
    path = _gt_cache_path(dataset, limit, normalized, metric, mode, top_k)
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {k: data[k].tolist() for k in data.files}


def save_gt_cache(
    dataset: str, limit, normalized: bool, metric: str, mode: str, top_k: int,
    gt_results: Dict[str, List[List[str]]],
) -> None:
    """
    Persist GT results (dict[request_id -> list[list[str]]]) to disk as a .npz.
    Each value is a 2-D object array of shape (n_queries, top_k) holding doc-id strings.
    """
    path = _gt_cache_path(dataset, limit, normalized, metric, mode, top_k)
    _GT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {rid: np.array(rows, dtype=object) for rid, rows in gt_results.items()}
    np.savez(str(path), **arrays)


def rebuild_with_flat_quantizer(faiss_index):
    """
    Return a new IndexIVFFlat backed by IndexFlatL2, keeping the same centroids
    and inverted-list assignments as the source index.  Exact centroid L2 search
    (same as M3's exhaustive sgemm) — required for a fair recall comparison.
    """
    import faiss
    from faiss.contrib.inspect_tools import get_invlist

    ivf  = faiss.extract_index_ivf(faiss_index)
    ivf  = faiss.downcast_index(ivf)
    nlist, dim = ivf.nlist, ivf.d
    quantizer  = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "xb") and quantizer.ntotal == nlist:
        centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(nlist, dim)
    else:
        centroids = np.vstack([quantizer.reconstruct(i) for i in range(nlist)]).astype(np.float32)

    invlists = faiss.downcast_InvertedLists(ivf.invlists)
    flat_q   = faiss.IndexFlatL2(dim)
    flat_q.add(centroids)
    new_idx  = faiss.IndexIVFFlat(flat_q, dim, nlist, faiss.METRIC_L2)
    new_idx.is_trained = True
    for list_id in range(nlist):
        ids, codes = get_invlist(invlists, list_id)
        if ids.size == 0:
            continue
        ids64 = np.ascontiguousarray(ids, dtype=np.int64)
        new_idx.invlists.add_entries(list_id, len(ids64),
                                     faiss.swig_ptr(ids64),
                                     faiss.swig_ptr(codes))
    new_idx.ntotal = faiss_index.ntotal
    return new_idx


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
# TripleRunner: GT faiss (live or cached) + optional test faiss + test m3
# ---------------------------------------------------------------------------

class TripleRunner:
    """
    Keeps up to three MemoryManagement instances in lockstep.
      mm_gt    — FaissBackend @ gt_nprobe  (ground truth, latency not reported).
                 May be None when gt_cache is provided — GT results come from cache.
      mm_faiss — FaissBackend @ faiss_nprobe (optional, skipped when None).
      mm_m3    — M3/other     @ m3_nprobe   (always present).

    gt_cache: dict[request_id -> list[list[str]]] pre-loaded doc-id lists.
              When provided and a rid is present in the cache, the live GT backend
              is not queried for that rid.  New GT results are accumulated in
              _gt_new so the caller can persist them after the run.
    """

    def __init__(
        self,
        mm_gt:        Optional[MemoryManagement],
        mm_faiss:     Optional[MemoryManagement],
        mm_m3:        MemoryManagement,
        idx_gt:       Optional[int],
        idx_faiss:    Optional[int],
        idx_m3:       int,
        k:            int,
        gt_nprobe:    int,
        faiss_nprobe: int,
        m3_nprobe:    int,
        ops_per_run:  int,
        logger:       Optional[BenchLogger] = None,
        gt_cache:     Optional[Dict[str, List[List[str]]]] = None,
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
        self._gt_cache    = gt_cache or {}
        # GT results collected this run (rids not in cache); caller may persist these.
        self._gt_new: Dict[str, List[List[str]]] = {}

        self._queued    = 0
        self._batch_idx = 0

        # recall accumulators
        self.total_queries    = 0
        self.recall_faiss_sum = 0.0
        self.recall_m3_sum    = 0.0

        # timing accumulators (GT not tracked)
        self._faiss_time:    float = 0.0
        self._m3_time:       float = 0.0
        self._total_inserts: int   = 0

        # CSV writer (set by caller)
        self.csv_writer: Optional[csv.writer] = None  # type: ignore[type-arg]

        # recall_diag appender
        self._recall_diag_f = None
        self._recall_diag_w = None
        import glob as _glob
        matches = sorted(_glob.glob(os.path.join(str(_run_dir), "*_recall_diag.csv")))
        if matches:
            self._recall_diag_f = open(matches[-1], "a", newline="", encoding="utf-8")
            self._recall_diag_w = csv.writer(self._recall_diag_f)

    # --- public API --------------------------------------------------------

    def add_search(self, items: List[MemoryItem], rid_prefix: str = "s") -> None:
        rid = f"{rid_prefix}-{self._batch_idx}"
        if self.mm_gt is not None:
            self.mm_gt.add_search(self.idx_gt, items, self.k, nprobe=self.gt_nprobe, request_id=rid)
        if self.mm_faiss is not None:
            self.mm_faiss.add_search(self.idx_faiss, items, self.k, nprobe=self.faiss_nprobe, request_id=rid)
        self.mm_m3.add_search(self.idx_m3, items, self.k, nprobe=self.m3_nprobe, request_id=rid)
        self._queued += 1
        self._maybe_flush()

    def add_insert(self, items: List[MemoryItem]) -> None:
        if self.mm_gt is not None:
            self.mm_gt.add_insert(self.idx_gt, items)
        if self.mm_faiss is not None:
            self.mm_faiss.add_insert(self.idx_faiss, items)
        self.mm_m3.add_insert(self.idx_m3, items)
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

        # GT — run live if backend present, else rely purely on cache
        if self.mm_gt is not None:
            res_gt_run = self.mm_gt.run()
            # Merge live results into cache and _gt_new
            for rid, hits in res_gt_run.searches.items():
                id_lists = [[h.id for h in q_hits] for q_hits in hits]
                self._gt_cache[rid] = id_lists
                self._gt_new[rid]   = id_lists

        if self.mm_faiss is not None:
            t0 = time.perf_counter()
            res_faiss = self.mm_faiss.run()
            t_faiss   = time.perf_counter() - t0
            self._faiss_time += t_faiss
        else:
            res_faiss = None
            t_faiss   = 0.0

        t0 = time.perf_counter()
        res_m3 = self.mm_m3.run()
        t_m3   = time.perf_counter() - t0
        self._m3_time += t_m3

        self._queued = 0

        # Determine which rids to score — from live GT or from cache
        rids_to_score = set(self._gt_cache.keys())
        if res_faiss is not None:
            rids_to_score |= set(res_faiss.searches.keys())
        rids_to_score &= set(res_m3.searches.keys())

        for rid in rids_to_score:
            gt_id_lists = self._gt_cache.get(rid)
            if not gt_id_lists:
                continue

            faiss_res = res_faiss.searches.get(rid) if res_faiss else None
            m3_res    = res_m3.searches.get(rid)

            if not m3_res and not faiss_res:
                continue

            n_q = len(gt_id_lists)

            # Build SearchHit-compatible objects from cached id lists for recall_at_k
            class _FakeHit:
                __slots__ = ("id",)
                def __init__(self, i): self.id = i
            gt_hits = [[_FakeHit(i) for i in row[:self.k]] for row in gt_id_lists]

            recall_faiss = 0.0
            if faiss_res:
                recall_faiss, _ = recall_at_k(faiss_res, gt_hits, self.k)

            recall_m3 = 0.0
            if m3_res:
                recall_m3, _ = recall_at_k(m3_res, gt_hits, self.k)

            self.total_queries    += n_q
            self.recall_faiss_sum += recall_faiss * n_q
            self.recall_m3_sum    += recall_m3    * n_q

            cum_faiss = self.recall_faiss_sum / max(self.total_queries, 1)
            cum_m3    = self.recall_m3_sum    / max(self.total_queries, 1)

            faiss_lat_ms = t_faiss / n_q * 1e3
            m3_lat_ms    = t_m3    / n_q * 1e3

            if self.mm_faiss is not None:
                self._log(
                    f"  [batch {self._batch_idx:04d}] rid={rid!r}  nq={n_q}  "
                    f"recall_faiss@{self.k}={recall_faiss:.4f}(cum={cum_faiss:.4f})  "
                    f"recall_m3@{self.k}={recall_m3:.4f}(cum={cum_m3:.4f})  "
                    f"faiss={t_faiss*1e3:.1f}ms({faiss_lat_ms:.2f}ms/q)  "
                    f"m3={t_m3*1e3:.1f}ms({m3_lat_ms:.2f}ms/q)"
                )
            else:
                self._log(
                    f"  [batch {self._batch_idx:04d}] rid={rid!r}  nq={n_q}  "
                    f"recall_m3@{self.k}={recall_m3:.4f}(cum={cum_m3:.4f})  "
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

            if self._recall_diag_w is not None:
                from datetime import datetime as _dt
                ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self._recall_diag_w.writerow([
                    ts, "RECALL_BATCH",
                    self._batch_idx, n_q,
                    f"{recall_faiss:.6f}", f"{cum_faiss:.6f}",
                    f"{recall_m3:.6f}",    f"{cum_m3:.6f}",
                    f"{t_faiss*1e3:.3f}",  f"{t_m3*1e3:.3f}",
                ])
                self._recall_diag_f.flush()

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
        description="Recall@k benchmark: test-faiss + M3MultiGpu vs GT Faiss."
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
    parser.add_argument("--top-k", type=int, default=10)
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
    parser.add_argument("--no-test-faiss", action="store_true", default=False,
                        help="Skip the test-faiss backend; only run M3 vs GT.")
    parser.add_argument("--no-gt-cache", action="store_true", default=False,
                        help="Disable GT result caching (always run GT faiss live).")
    args = parser.parse_args()

    # Patch M3MultiGpuBackend class defaults before the backend is instantiated.
    if args.alpha_et is not None or args.alpha_et_adapt_rate is not None:
        from AgentMemory.backend.m3 import M3MultiGpuBackend as _M3MG
        if args.alpha_et is not None:
            _M3MG.DEFAULTS["alpha_et"] = args.alpha_et
        if args.alpha_et_adapt_rate is not None:
            _M3MG.DEFAULTS["alpha_et_adapt_rate"] = args.alpha_et_adapt_rate

    # Use the run dir computed at module load time (M3_PROFILE_DIR already set)
    run_dir = _run_dir
    logger  = BenchLogger(run_dir)

    # per-batch CSV
    batches_path = run_dir / "batches.csv"
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

    # GT must scan every cluster so it is truly exhaustive regardless of quantizer type.
    # With nprobe=nlist the IVF search enumerates all clusters; nprobe > nlist is clamped by FAISS.
    if _disk_nlist > 0 and args.gt_nprobe < _disk_nlist:
        logger.log(
            f"[init] overriding gt_nprobe {args.gt_nprobe} → {_disk_nlist} "
            f"(exhaustive scan over all {_disk_nlist} clusters)"
        )
        args.gt_nprobe = _disk_nlist

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
        logger.log(f"[init] encoding {len(texts)} texts (cache: {_EMBED_CACHE_DIR}) ...")
        vecs = encode_texts_cached(texts, args.dataset, limit, normalized=args.faiss_normalized)
        dim  = vecs.shape[1]
        logger.log(f"[init] encoded dim={dim}")
        insert_items = [MemoryItem(id=f"{args.dataset}-{i}", data=vecs[i]) for i in range(len(vecs))]
        query_items  = [MemoryItem(id=f"q-{i}", data=vecs[i])              for i in range(len(vecs))]
        encoder      = VectorPassthroughEncoder(dim=dim, normalize=False)
        logger.log(f"[init] {len(insert_items)} insert items, {len(query_items)} query items (same vectors)")
    else:
        rng = np.random.default_rng(args.seed)
        dim = args.dim
        n   = args.limit if args.limit > 0 else 8192
        insert_count = n
        search_count = n
        insert_mat = rng.standard_normal((insert_count, dim)).astype(np.float32)
        search_mat = rng.standard_normal((search_count, dim)).astype(np.float32)
        if args.faiss_normalized:
            insert_mat /= np.linalg.norm(insert_mat, axis=1, keepdims=True) + 1e-8
            search_mat /= np.linalg.norm(search_mat, axis=1, keepdims=True) + 1e-8
        insert_items = [MemoryItem(id=f"ins-{i}", data=insert_mat[i]) for i in range(insert_count)]
        query_items  = [MemoryItem(id=f"q-{i}",   data=search_mat[i]) for i in range(search_count)]
        encoder      = VectorPassthroughEncoder(dim=dim, normalize=False)
        logger.log(f"[init] synthetic vectors: insert_count={insert_count}, search_count={search_count}, dim={dim}")

    # --- GT cache -----------------------------------------------------------
    use_gt_cache = (not args.no_gt_cache) and bool(args.dataset)
    gt_cache: Optional[Dict[str, List[List[str]]]] = None
    if use_gt_cache:
        gt_cache = load_gt_cache(
            args.dataset, args.limit if args.limit > 0 else None,
            args.faiss_normalized, args.metric, args.mode, args.top_k,
        )
        if gt_cache is not None:
            logger.log(f"[gt-cache] loaded {len(gt_cache)} cached request-ids from disk")
        else:
            logger.log("[gt-cache] no cache found — GT faiss will run live and results will be saved")

    # Decide whether we need a live GT backend.
    # We need it if: cache is disabled, or cache is absent/incomplete (we treat absent as needing full run).
    need_live_gt = (not use_gt_cache) or (gt_cache is None)

    # --- Instantiate backends -----------------------------------------------
    mm_gt    = None
    idx_gt   = None
    mm_faiss = None
    idx_faiss = None

    if need_live_gt:
        logger.log(f"[init] creating GT faiss backend (nprobe={args.gt_nprobe}) ...")
        mm_gt  = MemoryManagement(backend="faiss", encoder=encoder, default_nprobe=args.gt_nprobe)
        idx_gt = mm_gt.create_index("recall-bench-gt", metric=metric)
        logger.log("[init] loading Faiss checkpoint into GT faiss ...")
        t0 = time.perf_counter()
        mm_gt.rebuild_index_from_faiss(idx_gt, path=str(faiss_path), normalized=args.faiss_normalized)
        logger.log(f"[init]   GT faiss load: {time.perf_counter()-t0:.2f}s")
    else:
        logger.log(f"[gt-cache] using cached GT — skipping live GT faiss backend")

    if not args.no_test_faiss:
        logger.log(f"[init] creating test faiss backend (nprobe={args.faiss_nprobe}) ...")
        mm_faiss  = MemoryManagement(backend="faiss", encoder=encoder, default_nprobe=args.faiss_nprobe)
        idx_faiss = mm_faiss.create_index("recall-bench-faiss", metric=metric)
        logger.log("[init] loading Faiss checkpoint into test faiss ...")
        t0 = time.perf_counter()
        mm_faiss.rebuild_index_from_faiss(idx_faiss, path=str(faiss_path), normalized=args.faiss_normalized)
        logger.log(f"[init]   test faiss load: {time.perf_counter()-t0:.2f}s")

        # If the on-disk index uses an approximate quantizer (e.g. HNSW), rebuild with
        # IndexFlatL2 so it uses exact centroid search matching M3.
        try:
            import faiss as _f_chk
            _loaded  = mm_faiss.backend._indices[idx_faiss]
            _ivf_chk = _f_chk.extract_index_ivf(_loaded)
            _q_chk   = _f_chk.downcast_index(_ivf_chk.quantizer)
            _q_exact = hasattr(_q_chk, "xb") and _q_chk.ntotal == _ivf_chk.nlist
        except Exception:
            _q_exact = True
        if not _q_exact:
            logger.log("[init] test-faiss quantizer is approximate (HNSW) — rebuilding with IndexFlatL2 ...")
            t0 = time.perf_counter()
            _flat_idx = rebuild_with_flat_quantizer(mm_faiss.backend._indices[idx_faiss])
            _flat_idx.nprobe = args.faiss_nprobe
            mm_faiss.backend._indices[idx_faiss] = _flat_idx
            logger.log(f"[init] test-faiss flat-quantizer index ready  ntotal={_flat_idx.ntotal}"
                       f"  ({time.perf_counter()-t0:.2f}s)")
        else:
            logger.log("[init] test-faiss quantizer is exact (IndexFlatL2) — no rebuild needed")
    else:
        logger.log("[init] --no-test-faiss: skipping test faiss backend")

    logger.log(f"[init] creating {args.m3_backend} backend (nprobe={args.nprobe}) ...")
    mm_m3  = MemoryManagement(backend=args.m3_backend, encoder=encoder, default_nprobe=args.nprobe)
    idx_m3 = mm_m3.create_index("recall-bench", metric=metric)
    logger.log("[init] loading Faiss checkpoint into M3 ...")
    t0 = time.perf_counter()
    mm_m3.rebuild_index_from_faiss(idx_m3, path=str(faiss_path), normalized=args.faiss_normalized)
    logger.log(f"[init]   {args.m3_backend} load: {time.perf_counter()-t0:.2f}s")

    # Runner
    runner = TripleRunner(
        mm_gt=mm_gt,       mm_faiss=mm_faiss,        mm_m3=mm_m3,
        idx_gt=idx_gt,     idx_faiss=idx_faiss,       idx_m3=idx_m3,
        k=args.top_k,
        gt_nprobe=args.gt_nprobe,
        faiss_nprobe=args.faiss_nprobe,
        m3_nprobe=args.nprobe,
        ops_per_run=args.ops_per_run,
        logger=logger,
        gt_cache=gt_cache,
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

    # Persist new GT results to cache
    if use_gt_cache and runner._gt_new and args.dataset:
        save_gt_cache(
            args.dataset, args.limit if args.limit > 0 else None,
            args.faiss_normalized, args.metric, args.mode, args.top_k,
            runner._gt_new,
        )
        logger.log(f"[gt-cache] saved {len(runner._gt_new)} new request-ids to cache")

    # Summary
    logger.log(
        f"\n[summary] total_queries={runner.total_queries}  "
        f"total_inserts={runner._total_inserts}"
    )
    if mm_faiss is not None:
        logger.log(
            f"[summary] final recall@{args.top_k}:  "
            f"faiss={runner.cumulative_recall_faiss:.4f}  "
            f"m3={runner.cumulative_recall_m3:.4f}"
        )
    else:
        logger.log(
            f"[summary] final recall@{args.top_k}:  "
            f"m3={runner.cumulative_recall_m3:.4f}"
        )

    col = 22
    logger.log(
        f"\n{'':>{col}}  {'recall@'+str(args.top_k):>10}  "
        f"{'throughput (search)':>22}  {'latency (search)':>18}  "
        f"{'throughput (insert)':>22}  {'wall time':>12}"
    )
    logger.log(f"  {'─'*106}")

    if mm_faiss is not None:
        faiss_label = f"faiss(np={args.faiss_nprobe})"
        logger.log(
            f"  {faiss_label:>{col}}  "
            f"{runner.cumulative_recall_faiss:>10.4f}  "
            f"{runner.faiss_search_throughput:>19,.1f} q/s  "
            f"{runner.faiss_search_latency_ms:>14.3f} ms/q  "
            f"{runner.faiss_insert_throughput:>19,.1f} vec/s  "
            f"{runner._faiss_time*1e3:>10.1f} ms"
        )

    m3_label = f"{args.m3_backend}(np={args.nprobe})"
    logger.log(
        f"  {m3_label:>{col}}  "
        f"{runner.cumulative_recall_m3:>10.4f}  "
        f"{runner.m3_search_throughput:>19,.1f} q/s  "
        f"{runner.m3_search_latency_ms:>14.3f} ms/q  "
        f"{runner.m3_insert_throughput:>19,.1f} vec/s  "
        f"{runner._m3_time*1e3:>10.1f} ms"
    )
    if mm_faiss is not None and runner.faiss_search_throughput > 0:
        ratio = runner.m3_search_throughput / runner.faiss_search_throughput
        logger.log(f"\n  [summary] m3 speedup vs test-faiss: {ratio:.2f}x  (search throughput)")

    logger.log(f"[summary] outputs written to {run_dir}")
    logger.close()


if __name__ == "__main__":
    main()
