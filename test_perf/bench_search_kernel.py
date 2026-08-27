from __future__ import annotations

# ---- Profiling env vars — set BEFORE any M3 C++ module is imported ----
# M3Profiler singleton reads these on first use (inside search()).
import os
_PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_20")
os.environ.setdefault("M3_PROFILE",     "1")
os.environ.setdefault("M3_PROFILE_DIR", _PROFILE_DIR)

"""
Isolated search-kernel benchmark: FAISS vs M3 backends on identical data.

Both engines share the same IVFFlat index (same centroids, same vectors).
All Python/queue/encoding overhead is stripped.

Backends:
  m3        — AsyncEngine (flat IVF, same as FAISS)
  m3multi   — MultiLevelIndex: L0/L1 cache + early-exit, no GPU
  m3multigpu— MultiLevelIndex + GpuCoordinator

Profiling (M3_PROFILE=1 by default):
  Output CSVs written to test_perf/04_20/
    *_search_profile.csv — per-batch L0/L1/L2 timing breakdown
    *_search_stats.csv   — per-batch L0/L1 exit counts, dagent, alpha_et
    *_insert.csv         — insert timing (if any inserts)
    *_recall_diag.csv    — GPU recall divergence events (m3multigpu only)

Usage:
    python bench_search_kernel.py --backend m3 --n 100000 --dim 128
    python bench_search_kernel.py --backend m3multi --nprobe 32 --alpha-et 0.7
    python bench_search_kernel.py --backend m3multigpu --dataset gsm8k --limit 5000
"""

import argparse
import sys
import time
from typing import Any, Dict, List, Mapping

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _extract_text(entry: Mapping[str, Any]) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    text = entry.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    for key in ("user", "assistant", "human", "gpt", "question", "answer"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def flatten_dataset(dataset_name: str, split: str | None, limit: int | None) -> List[str]:
    loader_cls = DATASET_LOADERS[dataset_name]
    kwargs: Dict[str, object] = {}
    if split:
        kwargs["split"] = split
    if limit and limit > 0:
        kwargs["limit"] = limit
    loader = loader_cls(**kwargs)
    texts: List[str] = []
    for agent in loader.get_data():
        for req in agent.get("requests", []):
            for entry in req.get("trace", []):
                text = _extract_text(entry)
                if not text:
                    continue
                texts.append(text)
                if limit and limit > 0 and len(texts) >= limit:
                    return texts
    return texts


_FAISS_INDEX_PATH = "/data/IVF.index"
_EMBED_CACHE_DIR  = os.path.join(os.path.dirname(__file__), "embeddings_cache")


def encode_texts(texts: List[str], dataset_name: str, limit: int | None) -> np.ndarray:
    """Encode texts to float32 vectors, caching under test_perf/embeddings_cache/."""
    suffix = f"_n{limit}" if limit else "_all"
    cache_path = os.path.join(_EMBED_CACHE_DIR, f"{dataset_name}{suffix}.npy")
    if os.path.exists(cache_path):
        print(f"[encode] loading cached vectors from {cache_path}")
        return np.load(cache_path)
    from AgentMemory.encoder import TransformerEncoder
    from AgentMemory.types import MemoryItem
    enc = TransformerEncoder(model_name="intfloat/e5-large-v2", batch_size=64)
    items = [MemoryItem(id=str(i), data=t) for i, t in enumerate(texts)]
    vecs = np.asarray(enc.encode_items(items), dtype=np.float32)
    os.makedirs(_EMBED_CACHE_DIR, exist_ok=True)
    np.save(cache_path, vecs)
    print(f"[encode] saved {len(vecs)} vectors to {cache_path}")
    return vecs


# ---------------------------------------------------------------------------
# FAISS helpers
# ---------------------------------------------------------------------------

def build_faiss_index(vecs: np.ndarray, nlist: int):
    import faiss
    dim = vecs.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(vecs)
    index.add(vecs)
    return index


def load_faiss_index(path: str):
    import faiss
    return faiss.read_index(path)


# ---------------------------------------------------------------------------
# M3 backend construction
# ---------------------------------------------------------------------------

def _extract_faiss_ivf(faiss_index):
    """Return (ivf, centroids_np) from an in-memory FAISS IVF index."""
    import faiss
    from faiss.contrib.inspect_tools import get_invlist  # noqa: F401 — imported here for availability
    ivf = faiss.extract_index_ivf(faiss_index)
    ivf = faiss.downcast_index(ivf)
    quantizer = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "xb") and quantizer.ntotal == ivf.nlist:
        centroids = faiss.vector_to_array(quantizer.xb).astype(np.float32).reshape(ivf.nlist, ivf.d)
    else:
        centroids = np.vstack([quantizer.reconstruct(i) for i in range(ivf.nlist)]).astype(np.float32)
    return ivf, centroids


def build_m3_index(backend_name: str, faiss_index, args):
    """
    Build the M3 search object from an in-memory FAISS index.

    Returns
    -------
    search_fn  : callable(qb) -> np.ndarray shape (nq, k) of int64 IDs
    stats_fn   : callable() -> CacheStats  (m3multi/m3multigpu) or None
    cleanup_fn : callable()  (m3multigpu stops background threads) or None
    """
    import faiss
    from faiss.contrib.inspect_tools import get_invlist
    from AgentMemory.M3 import _m3_async as m3

    ivf, centroids = _extract_faiss_ivf(faiss_index)
    dim   = ivf.d
    nlist = ivf.nlist
    invlists = faiss.downcast_InvertedLists(ivf.invlists)

    # ------------------------------------------------------------------
    # m3 — flat AsyncEngine (same path as before)
    # ------------------------------------------------------------------
    if backend_name == "m3":
        engine   = m3.AsyncEngine()
        index_id = 0
        engine.create_ivf(index_id, dim, m3.Metric.L2, False, centroids)
        for list_id in range(nlist):
            ids, codes = get_invlist(invlists, list_id)
            if ids.size == 0:
                continue
            vecs = codes.view(np.float32).reshape(ids.shape[0], dim)
            engine.load_cluster(index_id, list_id,
                                np.ascontiguousarray(ids,  dtype=np.int64),
                                np.ascontiguousarray(vecs, dtype=np.float32))

        def search_fn(qb):
            result = engine.search(index_id, qb, args.k, args.nprobe)
            ids = result[0] if isinstance(result, tuple) else result
            return np.asarray(ids, dtype=np.int64).reshape(len(qb), args.k)

        return search_fn, None, None

    # ------------------------------------------------------------------
    # m3multi / m3multigpu — MultiLevelIndex, L2 loaded, L0/L1 empty
    # ------------------------------------------------------------------
    from AgentMemory.M3 import M3MultiLevelIndex
    from AgentMemory.backend.m3 import _apply_cache_config, M3MultiLevelBackend

    # Start from backend defaults, then apply CLI overrides.
    p = dict(M3MultiLevelBackend.DEFAULTS)
    p.update({
        "l2_nlist":              nlist,
        "alpha_et":              args.alpha_et,
        "alpha_et_adapt_rate":   args.alpha_et_adapt_rate,
        "dagent_window":         args.dagent_window,
        "calibration_interval":  args.calibration_interval,
        "l0_nprobe":             args.l0_nprobe,
        "l1_nprobe":             args.l1_nprobe,
        "l0_max_clusters":       args.l0_max_clusters,
        "l1_max_clusters":       args.l1_max_clusters,
        "l1_neighborhood_k":     args.l1_neighborhood_k,
        "l0_neighborhood_k":     args.l0_neighborhood_k,
    })

    ml_idx = M3MultiLevelIndex(
        dim=dim, metric=m3.Metric.L2, normalized=False,
        l0_nlist=int(p["l0_nlist"]),
        l1_nlist=int(p["l1_nlist"]),
        l2_nlist=nlist,
        l0_new_cluster_threshold=float(p["l0_new_cluster_threshold"]),
        search_threshold=float(p["search_threshold"]),
        l0_merge_threshold=float(p["l0_merge_threshold"]),
        l0_max_nlist=int(p["l0_max_nlist"]),
    )
    _apply_cache_config(ml_idx, p)

    # Load corpus into L2 only — L0/L1 start empty and warm up during search.
    ml_idx.set_l2_centroids(np.ascontiguousarray(centroids))
    for list_id in range(nlist):
        ids, codes = get_invlist(invlists, list_id)
        if ids.size == 0:
            continue
        vecs = codes.view(np.float32).reshape(ids.shape[0], dim)
        ml_idx.load_cluster(list_id,
                            np.ascontiguousarray(ids,  dtype=np.int64),
                            np.ascontiguousarray(vecs, dtype=np.float32))

    cleanup_fn = None
    if backend_name == "m3multigpu":
        from AgentMemory.M3 import GpuCoordinator
        coord = GpuCoordinator(ml_idx,
                               gpu_budget_bytes=10 * 1024 ** 3,
                               dim=dim, metric=m3.Metric.L2, normalized=False,
                               insert_buf_cap=128)
        ml_idx.set_gpu_coordinator(coord)
        coord.start_background(flush_ms=50, maintenance_ms=50,
                               rebalance_ms=500, split_every_ops=0,
                               split_threshold=200_000)

        def cleanup_fn():
            coord.stop_background()
            ml_idx.set_gpu_coordinator(None)

    def search_fn(qb):
        out_ids, _ = ml_idx.search(qb, args.k, args.nprobe)
        # out_ids: list[list[int64]] — pack into (nq, k) numpy array
        nq = len(qb)
        arr = np.full((nq, args.k), -1, dtype=np.int64)
        for i, row in enumerate(out_ids):
            n = min(len(row), args.k)
            arr[i, :n] = row[:n]
        return arr

    def stats_fn():
        return ml_idx.get_cache_stats()

    return search_fn, stats_fn, cleanup_fn


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench(label: str, fn, queries: np.ndarray, batch: int,
          rounds: int, trials: int = 5, warmup_rounds: int = 3,
          stats_fn=None) -> float:
    """
    Run fn(query_batch) across multiple independent trials.
    Prints per-trial QPS (with live cache stats for m3multi/m3multigpu).
    Returns median QPS.
    """
    n = len(queries)
    batches = [queries[i:i + batch] for i in range(0, n, batch)]
    n_per_round = sum(len(b) for b in batches)

    qps_list: list[float] = []

    for t in range(trials):
        for _ in range(warmup_rounds):
            for qb in batches:
                fn(qb)

        t0 = time.perf_counter()
        for _ in range(rounds):
            for qb in batches:
                fn(qb)
        elapsed = time.perf_counter() - t0
        qps = rounds * n_per_round / elapsed
        qps_list.append(qps)

        line = f"    trial {t + 1:2d}: {qps:>10,.0f} qps"
        if stats_fn:
            s = stats_fn()
            line += (f"  | L0={s.l0_clusters}cl/{s.l0_total_vecs}v"
                     f"  L1={s.l1_clusters}cl/{s.l1_total_vecs}v"
                     f"  dagent={s.dagent:.4f}  thr={s.dynamic_threshold:.4f}")
        print(line)

    arr  = np.array(qps_list)
    med  = float(np.median(arr))
    mean = float(arr.mean())
    std  = float(arr.std())
    p95  = float(np.percentile(arr, 95))
    lo   = float(arr.min())
    hi   = float(arr.max())
    cv   = 100 * std / mean

    print(f"  {label:12s}  median {med:>10,.0f}  mean {mean:>10,.0f} ± {std:>8,.0f}"
          f"  p95 {p95:>10,.0f}  [{lo:>10,.0f} – {hi:>10,.0f}]  CV={cv:.1f}%")
    return med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FAISS vs M3 search kernel benchmark")

    # ---- backend ----
    parser.add_argument("--backend", choices=["m3", "m3multi", "m3multigpu"], default="m3",
                        help="M3 backend to bench against FAISS (default: m3).")

    # ---- query source (index always loaded from _FAISS_INDEX_PATH) ----
    parser.add_argument("--dataset",  choices=list(DATASET_LOADERS.keys()), default=None,
                        help="Encode dataset texts and use as queries (default: random).")
    parser.add_argument("--limit",    type=int, default=0, help="Max dataset items to encode (0=all).")

    # ---- search args ----
    parser.add_argument("--nprobe",   type=int, default=32,  help="nprobe at search time.")
    parser.add_argument("--k",        type=int, default=10,  help="Top-k results.")
    parser.add_argument("--nq",       type=int, default=1000, help="Number of query vectors.")
    parser.add_argument("--batch",    type=int, default=256,  help="Query batch size per call.")
    parser.add_argument("--rounds",   type=int, default=10,   help="Timed rounds per trial.")
    parser.add_argument("--trials",   type=int, default=5,    help="Independent trials.")
    parser.add_argument("--warmup",   type=int, default=3,    help="Warmup rounds before each trial.")

    # ---- cache tuning (m3multi / m3multigpu only) ----
    parser.add_argument("--alpha-et",             type=float, default=0.7,
                        help="Initial early-exit threshold α_et.")
    parser.add_argument("--alpha-et-adapt-rate",  type=float, default=0.05,
                        help="EMA learning rate for α_et adaptation.")
    parser.add_argument("--dagent-window",        type=int,   default=20,
                        help="Rolling window size for d_agent mean.")
    parser.add_argument("--calibration-interval", type=int,   default=10,
                        help="Background full-search calibration frequency (0=off).")
    parser.add_argument("--l0-nprobe",            type=int,   default=0,
                        help="Fixed nprobe for L0 (0=inherit --nprobe).")
    parser.add_argument("--l1-nprobe",            type=int,   default=0,
                        help="Fixed nprobe for L1 (0=inherit --nprobe).")
    parser.add_argument("--l0-max-clusters",      type=int,   default=64,
                        help="Max clusters in L0 cache.")
    parser.add_argument("--l1-max-clusters",      type=int,   default=128,
                        help="Max clusters in L1 cache.")
    parser.add_argument("--l1-neighborhood-k",    type=int,   default=20,
                        help="Top-k' results promoted to L1 after each L2 search.")
    parser.add_argument("--l0-neighborhood-k",    type=int,   default=5,
                        help="Top-k'' subset of L1 promotion also written to L0.")

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(_PROFILE_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Load FAISS index --------------------------------------------------
    print(f"[init] loading FAISS index from {_FAISS_INDEX_PATH}")
    faiss_index = load_faiss_index(_FAISS_INDEX_PATH)
    dim = faiss_index.d
    print(f"[init] loaded  ntotal={faiss_index.ntotal}  dim={dim}")
    faiss_index.nprobe = args.nprobe

    # ---- Build M3 index ----------------------------------------------------
    print(f"[init] building M3 backend={args.backend}...")
    m3_search_fn, m3_stats_fn, m3_cleanup_fn = build_m3_index(args.backend, faiss_index, args)
    print(f"[init] {args.backend} ready")

    # ---- Query set ---------------------------------------------------------
    if args.dataset:
        limit = args.limit if args.limit > 0 else None
        print(f"[init] loading dataset={args.dataset}  limit={limit or 'all'}")
        texts = flatten_dataset(args.dataset, None, limit)
        if not texts:
            raise RuntimeError(f"No texts loaded from dataset '{args.dataset}'")
        print(f"[init] loaded {len(texts)} texts, encoding...")
        all_vecs = encode_texts(texts, args.dataset, limit)
        queries  = all_vecs[:args.nq].astype(np.float32)
    else:
        queries = rng.standard_normal((args.nq, dim)).astype(np.float32)
    nq_actual = len(queries)

    print(f"\n[bench] backend={args.backend}  nq={nq_actual}"
          f"  dataset={args.dataset or 'synthetic'}  nprobe={args.nprobe}"
          f"  k={args.k}  batch={args.batch}  rounds={args.rounds}"
          f"  trials={args.trials}  warmup={args.warmup}")

    if args.backend in ("m3multi", "m3multigpu"):
        print(f"[cache] alpha_et={args.alpha_et}  adapt_rate={args.alpha_et_adapt_rate}"
              f"  dagent_window={args.dagent_window}  calib_interval={args.calibration_interval}"
              f"  l0_nprobe={args.l0_nprobe}  l1_nprobe={args.l1_nprobe}"
              f"  l0_max_clusters={args.l0_max_clusters}  l1_max_clusters={args.l1_max_clusters}"
              f"  l1_neighborhood_k={args.l1_neighborhood_k}"
              f"  l0_neighborhood_k={args.l0_neighborhood_k}")

    print(f"[profiling] M3_PROFILE=1  output → {_PROFILE_DIR}/")
    print()

    # ---- FAISS bench -------------------------------------------------------
    print("  faiss:")
    faiss_qps = bench("faiss", lambda qb: faiss_index.search(qb, args.k),
                      queries, args.batch, args.rounds, args.trials, args.warmup)

    # ---- M3 bench ----------------------------------------------------------
    print(f"\n  {args.backend}:")
    m3_qps = bench(args.backend, m3_search_fn,
                   queries, args.batch, args.rounds, args.trials, args.warmup,
                   stats_fn=m3_stats_fn)

    ratio = faiss_qps / m3_qps if m3_qps > 0 else float("inf")
    print(f"\n  ratio  faiss/{args.backend} = {ratio:.2f}x  (median/median)")

    # ---- Recall: FAISS GT vs M3 --------------------------------------------
    print("\n[recall] comparing FAISS vs M3 result IDs on all queries...")
    faiss_ids_all = []
    m3_ids_all    = []
    for i in range(0, len(queries), args.batch):
        qb = queries[i:i + args.batch]
        _, fi = faiss_index.search(qb, args.k)
        mi    = m3_search_fn(qb)
        faiss_ids_all.append(fi)
        m3_ids_all.append(np.asarray(mi, dtype=np.int64).reshape(len(qb), args.k))

    faiss_ids = np.vstack(faiss_ids_all)
    m3_ids    = np.vstack(m3_ids_all)

    per_query_recall = np.array([
        len(np.intersect1d(faiss_ids[q], m3_ids[q])) / args.k
        for q in range(len(queries))
    ])
    exact_match_pct = 100.0 * np.mean(faiss_ids == m3_ids)

    print(f"  recall@{args.k}  mean={per_query_recall.mean():.4f}"
          f"  min={per_query_recall.min():.4f}"
          f"  perfect={100 * np.mean(per_query_recall == 1.0):.1f}% of queries")
    print(f"  exact order match: {exact_match_pct:.2f}% of (query, rank) pairs identical")

    if m3_cleanup_fn:
        m3_cleanup_fn()


if __name__ == "__main__":
    main()
