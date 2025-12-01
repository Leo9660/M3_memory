"""
Measure time split between HNSW coarse search and IVF fine search on an
IVF-HNSW-Flat Faiss index (e.g., factory "IVF4096_HNSW32,Flat").

The script runs:
1) quantizer.search (HNSW coarse) with nprobe
2) ivf.search_preassigned (fine scan using the coarse assignments)
3) ivf.search (end-to-end) for reference

Example:
    python pic/hnsw_ivf_timebreakdown.py \
        --faiss-index /path/to/ivf4096_hnsw32.faiss \
        --num-queries 2000 --topk 10 --nprobe 8 --ef-search 64
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss is required to run this script.") from exc


def _load_queries(args, dim: int) -> np.ndarray:
    if args.queries:
        arr = np.load(args.queries)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"Query shape mismatch: expected (*, {dim}), got {arr.shape}")
        queries = arr.astype(np.float32, copy=False)
    else:
        rng = np.random.default_rng(args.seed)
        queries = rng.standard_normal((args.num_queries, dim), dtype=np.float32)

    if args.normalize_input:
        norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12
        queries = queries / norms
    return queries.astype(np.float32, copy=False)


def _timeit(fn, *args, **kwargs) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def _median_time(fn, repeats: int, warmup: int, *args, **kwargs) -> Tuple[float, Tuple[np.ndarray, np.ndarray]]:
    """
    Run fn multiple times, discard warmup, return median duration + last result.
    """
    durations = []
    last_result: Tuple[np.ndarray, np.ndarray] | None = None
    total_runs = warmup + repeats
    for i in range(total_runs):
        dur, res = _timeit(fn, *args, **kwargs)
        if i >= warmup:
            durations.append(dur)
        last_result = res
    median = statistics.median(durations) if durations else 0.0
    return median, last_result  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down HNSW coarse vs IVF fine search time on an IVF-HNSW-Flat index.")
    parser.add_argument("--faiss-index", required=True, help="Path to IVF-HNSW-Flat Faiss index.")
    parser.add_argument("--queries", help="Optional .npy of queries shaped (n, dim); defaults to random.")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of random queries if --queries is not provided. Default: 1000.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k to search. Default: 10.")
    parser.add_argument("--nprobe", type=int, default=8, help="nprobe for IVF search (and coarse timing). Default: 8.")
    parser.add_argument("--ef-search", type=int, default=64, help="HNSW efSearch for the coarse quantizer. Default: 64.")
    parser.add_argument("--normalize-input", action="store_true", help="L2-normalize queries before search (useful for IP/Cosine).")
    parser.add_argument("--threads", type=int, default=None, help="Set faiss.omp_set_num_threads; default: keep library default.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed repeats per stage (median reported). Default: 3.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per stage (not timed). Default: 1.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for random queries.")
    args = parser.parse_args()

    index_path = Path(args.faiss_index).expanduser().resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Faiss index not found: {index_path}")

    if args.threads is not None:
        faiss.omp_set_num_threads(int(args.threads))

    index = faiss.read_index(str(index_path))
    ivf = faiss.extract_index_ivf(index)
    if ivf is None:
        raise ValueError("Index does not contain an IVF component.")
    ivf = faiss.downcast_index(ivf)

    quantizer = faiss.downcast_index(ivf.quantizer)
    if hasattr(quantizer, "hnsw"):
        quantizer.hnsw.efSearch = int(args.ef_search)

    ivf.nprobe = int(args.nprobe)
    dim = int(ivf.d)

    queries = _load_queries(args, dim)

    # 1) coarse (HNSW) timing
    t_coarse, (D_coarse, I_coarse) = _median_time(
        quantizer.search, args.repeats, args.warmup, queries, args.nprobe
    )

    # 2) fine search only (reuse coarse assignments)
    t_fine, (D_fine, I_fine) = _median_time(
        ivf.search_preassigned,
        args.repeats,
        args.warmup,
        queries,
        args.topk,
        I_coarse,
        D_coarse,
    )

    # print(D_fine, I_fine)  # prevent unused variable warning

    # 3) full search for reference
    faiss.cvar.indexIVF_stats.reset()
    faiss.cvar.indexIVF_stats.enable = True

    t_total, (D_total, I_total) = _median_time(
        ivf.search, args.repeats, args.warmup, queries, args.topk
    )

    stats = faiss.cvar.indexIVF_stats
    # print(f"IVF stats: nq={stats.nq}, "
    #     f"nlist_scanned={stats.nlist}, "
    #     f"nscan={stats.nscan}, "
    #     f"quantization_time={stats.quantization_time:.3f} s, "
    #     f"search_time={stats.search_time:.3f} s")
    # print(D_total, I_total)  # prevent unused variable warning

    q = queries.shape[0]
    print(f"Index: {index_path}")
    print(f"Queries: {q}, dim={dim}, topk={args.topk}, nprobe={args.nprobe}, efSearch={args.ef_search}")
    print(f"Coarse (HNSW) time: {t_coarse:.4f}s ({t_coarse/q*1000:.4f} ms/q, ~{q/t_coarse:.2f} QPS)")
    print(f"Fine   (IVF)  time: {t_fine:.4f}s ({t_fine/q*1000:.4f} ms/q, ~{q/t_fine:.2f} QPS)")
    print(f"Total  search time: {t_total:.4f}s ({t_total/q*1000:.4f} ms/q, ~{q/t_total:.2f} QPS)")

    sum_cf = t_coarse + t_fine
    if sum_cf > 0:
        print(f"Coarse share (coarse/(coarse+fine)): {t_coarse / sum_cf * 100:.2f}%")
        print(f"Fine   share (fine/(coarse+fine)):  {t_fine / sum_cf * 100:.2f}%")
    if t_total > 0:
        print(f"Total vs coarse+fine ratio: {t_total / sum_cf * 100:.2f}%  (caching/threading can make these differ)")


if __name__ == "__main__":  # pragma: no cover
    main()
