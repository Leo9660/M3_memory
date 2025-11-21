#!/usr/bin/env python
"""
Minimal multi-level index smoke/perf harness.

Usage:
    python test/perf.py --n 10000 --dim 128 --k 10 --threshold 0.5
"""

from __future__ import annotations
import argparse
import time
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from AgentMemory.M3 import M3MultiLevelIndex, Metric


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000, help="number of rows to insert")
    p.add_argument("--dim", type=int, default=128, help="vector dimension")
    p.add_argument("--k", type=int, default=10, help="top-k for search")
    p.add_argument("--nprobe", type=int, default=32, help="nprobe")
    p.add_argument("--l0-nlist", type=int, default=1, help="initial L0 clusters")
    p.add_argument("--threshold", type=float, default=float("inf"),
                   help="new-cluster threshold for L0 (distance)")
    p.add_argument("--search-threshold", type=float, default=float("inf"),
                   help="early-stop search threshold")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg_kwargs = dict(
        l0_nlist=args.l0_nlist,
        l0_new_cluster_threshold=args.threshold,
        search_threshold=args.search_threshold,
    )
    idx = M3MultiLevelIndex(
        dim=args.dim,
        metric=Metric.COSINE,
        normalized=True,
        **cfg_kwargs,
    )

    # optional explicit centroids for L0 (all zeros); keeps routing cheap
    idx.set_l0_centroids(np.zeros((args.l0_nlist, args.dim), dtype=np.float32))

    ids = np.arange(args.n, dtype=np.int64)
    vecs = np.random.randn(args.n, args.dim).astype(np.float32)
    t0 = time.perf_counter()
    idx.insert(ids, vecs)
    t1 = time.perf_counter()

    q = vecs[: min(1024, args.n)]
    out_ids, out_scores = idx.search(q, args.k, nprobe=args.nprobe)
    t2 = time.perf_counter()

    total_rows = sum(len(row) for row in out_ids)
    print(f"inserted {args.n} rows in {(t1 - t0)*1000:.1f} ms")
    print(f"searched {len(q)} queries (return {total_rows} hits) in {(t2 - t1)*1000:.1f} ms")
    print("example first query ids:", out_ids[0][: min(5, len(out_ids[0]))])
    print("example first query scores:", out_scores[0][: min(5, len(out_scores[0]))])


if __name__ == "__main__":
    main()
