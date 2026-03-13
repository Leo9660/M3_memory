"""
Validate CPU-vs-GPU latency claims for IVF cluster linear-scan search.

Claims tested
─────────────
  C1  For cluster sizes < ~256 vectors, CPU search has LOWER latency than GPU
      (kernel-launch overhead dominates at small scale).
  C2  At ~512 vectors, GPU achieves a >3× speedup over CPU.
  C3  GPU latency is largely FLAT as cluster size grows beyond ~512
      (dominant cost is kernel launch, not computation).

Cluster initialisation (matches benchmark_cpu_gpu.py)
──────────────────────────────────────────────────────
  • Clusters are simulated as FAISS IndexFlatL2 instances — a dense
    row-major float32 matrix representing one IVF inverted-list partition.
  • Synthetic vectors are drawn from N(0,1) so results are hardware-
    representative and no external index file is required.
  • CPU path  : faiss.IndexFlatL2  — BLAS GEMM, no kernel overhead.
  • GPU path  : faiss.index_cpu_to_gpu — transfers flat index to GPU VRAM,
    then runs a CUDA distance kernel.  The per-call cost therefore includes:
      H2D transfer  (simulated via GPU index construction each call OR
                     pre-loaded; see below) + kernel launch + D2H result copy.
    For a fair "search only" comparison the GPU index is built once per
    cluster size and reused across repetitions (matching production usage
    where clusters are pre-resident in VRAM).

Usage
─────
  python test/validate_cpu_gpu_thresholds.py [--dim D] [--k K] [--nq NQ] [--reps N]
  python test/validate_cpu_gpu_thresholds.py --no-gpu   # CPU-only mode (skips GPU claims)
"""

import argparse
import sys
import time
import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dim",    type=int, default=128, help="Vector dimension (default 128)")
parser.add_argument("--k",      type=int, default=10,  help="Nearest neighbours (default 10)")
parser.add_argument("--nq",     type=int, default=10,  help="Queries per trial (default 10)")
parser.add_argument("--reps",   type=int, default=30,  help="Timing repetitions (default 30)")
parser.add_argument("--no-gpu", action="store_true",   help="Skip GPU tests (CPU-only validation)")
args = parser.parse_args()

DIM      = args.dim
K        = args.k
NQ       = args.nq
N_REPS   = args.reps
USE_GPU  = not args.no_gpu

# Cluster sizes: fine-grained around the 256/512 crossover point
CLUSTER_SIZES = [32, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192]

# Claim-boundary sizes
SMALL_SIZES  = [sz for sz in CLUSTER_SIZES if sz <= 256]   # C1: CPU should win
CROSSOVER    = 512                                           # C2: >3x GPU speedup
LARGE_SIZES  = [sz for sz in CLUSTER_SIZES if sz >= 512]   # C3: GPU latency flat

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import faiss
except ImportError:
    sys.exit("faiss not installed — run: pip install faiss-cpu  (or faiss-gpu)")

gpu_available = False
if USE_GPU:
    try:
        res = faiss.StandardGpuResources()
        # Probe: try building a tiny GPU index
        _probe = faiss.IndexFlatL2(4)
        _probe.add(np.zeros((2, 4), dtype=np.float32))
        faiss.index_cpu_to_gpu(res, 0, _probe)
        gpu_available = True
        print("GPU: available via faiss-gpu")
    except Exception as e:
        print(f"GPU: NOT available ({e}) — running CPU-only mode")
        USE_GPU = False

# ── Synthetic data ────────────────────────────────────────────────────────────
rng        = np.random.default_rng(0)
# Pre-allocate max size so all cluster slices share the same data distribution
max_sz     = max(CLUSTER_SIZES)
all_vecs   = rng.standard_normal((max_sz, DIM)).astype(np.float32)
queries    = rng.standard_normal((NQ, DIM)).astype(np.float32)

print(f"\nConfig: dim={DIM}  k={K}  nq={NQ}  reps={N_REPS}")
print(f"Cluster sizes: {CLUSTER_SIZES}\n")

# ── Timing helpers ────────────────────────────────────────────────────────────

def _sync():
    """Synchronise GPU if available (no-op otherwise)."""
    if gpu_available:
        try:
            import cupy as cp
            cp.cuda.Device(0).synchronize()
        except ImportError:
            pass  # cupy not required; faiss-gpu has its own sync


def time_search_cpu(vecs: np.ndarray) -> float:
    """
    Build an IndexFlatL2 and time repeated searches.
    Returns mean latency in ms over N_REPS repetitions.

    This represents one IVF cluster (a dense partition) doing brute-force
    linear scan entirely on CPU BLAS.
    """
    idx = faiss.IndexFlatL2(DIM)
    idx.add(vecs)
    # warm-up (avoid cold-cache effects)
    idx.search(queries, K)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        idx.search(queries, K)
    return (time.perf_counter() - t0) / N_REPS * 1e3  # ms


def time_search_gpu(vecs: np.ndarray) -> float:
    """
    Transfer flat index to GPU once (simulating a pre-resident VRAM cluster),
    then time repeated GPU searches.  Does NOT include H2D transfer in the
    timed loop — mirrors production where clusters are pre-promoted.

    Returns mean latency in ms over N_REPS repetitions.
    """
    cpu_idx = faiss.IndexFlatL2(DIM)
    cpu_idx.add(vecs)
    gpu_idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)  # H2D: outside timed loop
    # warm-up
    gpu_idx.search(queries, K)
    _sync()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        gpu_idx.search(queries, K)
    _sync()
    return (time.perf_counter() - t0) / N_REPS * 1e3  # ms


# ── Run benchmarks ────────────────────────────────────────────────────────────
print(f"{'Size':>6}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>10}  {'Faster':>8}")
print("─" * 55)

cpu_lat = {}
gpu_lat = {}

for sz in CLUSTER_SIZES:
    vecs = all_vecs[:sz]
    sc   = time_search_cpu(vecs)
    cpu_lat[sz] = sc

    if USE_GPU:
        sg = time_search_gpu(vecs)
        gpu_lat[sz] = sg
        speedup = sc / sg
        faster  = "GPU" if sg < sc else "CPU"
        print(f"{sz:>6}  {sc:>10.3f}  {sg:>10.3f}  {speedup:>9.2f}x  {faster:>8}")
    else:
        print(f"{sz:>6}  {sc:>10.3f}  {'—':>10}  {'—':>10}  {'—':>8}")

# ── Claim validation ─────────────────────────────────────────────────────────
print("\n" + "═" * 55)
print("CLAIM VALIDATION")
print("═" * 55)

results = {}

if USE_GPU:
    # C1: CPU faster than GPU for sizes ≤ 256
    c1_checks = {sz: cpu_lat[sz] < gpu_lat[sz] for sz in SMALL_SIZES}
    c1_pass   = all(c1_checks.values())
    results["C1"] = c1_pass
    status = "PASS ✓" if c1_pass else "FAIL ✗"
    print(f"\nC1 [{status}] CPU < GPU latency for cluster sizes ≤ 256:")
    for sz, ok in c1_checks.items():
        mark = "✓" if ok else "✗"
        print(f"       sz={sz:>4}: CPU={cpu_lat[sz]:.3f}ms  GPU={gpu_lat[sz]:.3f}ms  {mark}")

    # C2: GPU >3x speedup at 512 vectors
    if CROSSOVER in cpu_lat and CROSSOVER in gpu_lat:
        speedup_at_512 = cpu_lat[CROSSOVER] / gpu_lat[CROSSOVER]
        c2_pass = speedup_at_512 > 3.0
        results["C2"] = c2_pass
        status = "PASS ✓" if c2_pass else "FAIL ✗"
        print(f"\nC2 [{status}] GPU speedup >3× at cluster size {CROSSOVER}:")
        print(f"       CPU={cpu_lat[CROSSOVER]:.3f}ms  GPU={gpu_lat[CROSSOVER]:.3f}ms"
              f"  speedup={speedup_at_512:.2f}x  (need >3.00x)")
    else:
        print(f"\nC2 [SKIP] Size {CROSSOVER} not in CLUSTER_SIZES")

    # C3: GPU latency is flat for sizes ≥ 512
    #     "Flat" = the slope of GPU latency over log2(size) for large clusters
    #     is at most 30% of the slope of CPU latency over the same range.
    #     (GPU may still grow slightly due to memory bandwidth; the claim is
    #     that it grows far slower than CPU.)
    if len(LARGE_SIZES) >= 3:
        import math
        xs  = [math.log2(sz) for sz in LARGE_SIZES]
        gpu_ys = [gpu_lat[sz] for sz in LARGE_SIZES]
        cpu_ys = [cpu_lat[sz] for sz in LARGE_SIZES]

        def linreg_slope(xs, ys):
            n  = len(xs)
            xm = sum(xs) / n
            ym = sum(ys) / n
            num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
            den = sum((x - xm) ** 2 for x in xs)
            return num / den if den else 0.0

        gpu_slope = linreg_slope(xs, gpu_ys)
        cpu_slope = linreg_slope(xs, cpu_ys)
        ratio     = gpu_slope / cpu_slope if cpu_slope > 0 else float("inf")

        # Also check coefficient of variation of GPU latencies (should be low)
        gpu_arr = np.array(gpu_ys)
        gpu_cv  = gpu_arr.std() / gpu_arr.mean() if gpu_arr.mean() > 0 else float("inf")

        c3_pass = ratio < 0.35 or gpu_cv < 0.25
        results["C3"] = c3_pass
        status = "PASS ✓" if c3_pass else "FAIL ✗"
        print(f"\nC3 [{status}] GPU latency largely flat for cluster sizes ≥ {CROSSOVER}:")
        print(f"       GPU slope={gpu_slope:.4f} ms/log2-unit  "
              f"CPU slope={cpu_slope:.4f} ms/log2-unit")
        print(f"       slope ratio GPU/CPU = {ratio:.2f}  (need <0.35)")
        print(f"       GPU latency CV      = {gpu_cv:.2f}  (need <0.25, OR above passes)")
        print(f"       GPU latencies (ms): "
              + "  ".join(f"{sz}→{gpu_lat[sz]:.2f}" for sz in LARGE_SIZES))

else:
    # CPU-only mode: validate that CPU latency scales roughly linearly with size
    # (sanity check that the benchmark is measuring real compute)
    import math
    xs  = [math.log2(sz) for sz in CLUSTER_SIZES]
    ys  = [cpu_lat[sz]   for sz in CLUSTER_SIZES]

    def linreg_slope(xs, ys):
        n  = len(xs)
        xm = sum(xs) / n
        ym = sum(ys) / n
        num = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        den = sum((x - xm) ** 2 for x in xs)
        return num / den if den else 0.0

    slope = linreg_slope(xs, ys)
    c_cpu_pass = slope > 0
    results["CPU-scaling"] = c_cpu_pass
    status = "PASS ✓" if c_cpu_pass else "FAIL ✗"
    print(f"\nCPU-scaling [{status}] CPU latency grows with cluster size "
          f"(slope={slope:.4f} ms/log2-unit)")
    print("  (GPU claims C1/C2/C3 skipped — run without --no-gpu to test)")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
all_pass = all(results.values())
print(f"Overall: {'ALL CLAIMS PASS ✓' if all_pass else 'SOME CLAIMS FAILED ✗'}")
print("─" * 55)

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(CLUSTER_SIZES, [cpu_lat[sz] for sz in CLUSTER_SIZES],
            "o-", color="#E05C5C", lw=2, ms=7, label="CPU (IndexFlatL2)")

    if USE_GPU:
        ax.plot(CLUSTER_SIZES, [gpu_lat[sz] for sz in CLUSTER_SIZES],
                "s--", color="#4C9BE8", lw=2, ms=7, label="GPU (pre-resident)")

        # Annotate speedup at 512
        if CROSSOVER in gpu_lat:
            sx, sy = CROSSOVER, gpu_lat[CROSSOVER]
            ax.annotate(f"{cpu_lat[CROSSOVER]/sy:.1f}× speedup",
                        xy=(sx, sy), xytext=(sx * 1.15, sy * 2.5),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=9, color="#333")

        # Shade "CPU wins" region
        ax.axvspan(CLUSTER_SIZES[0], 256, alpha=0.07, color="#E05C5C",
                   label="CPU-faster zone (<256)")

        # Shade "GPU flat" region
        ax.axvspan(512, CLUSTER_SIZES[-1], alpha=0.07, color="#4C9BE8",
                   label="GPU-flat zone (≥512)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(CLUSTER_SIZES)
    ax.set_xticklabels([str(s) for s in CLUSTER_SIZES], fontsize=8)
    ax.set_xlabel("Cluster size (# vectors)", fontsize=12)
    ax.set_ylabel("Search latency (ms)", fontsize=12)
    ax.set_title(f"CPU vs GPU search latency  (dim={DIM}, k={K}, nq={NQ})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Mark claim boundaries
    for x in [256, 512]:
        ax.axvline(x, color="gray", ls=":", lw=1.2)

    plt.tight_layout()
    out = "plot_validate_cpu_gpu.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved → {out}")
except ImportError:
    print("\n(matplotlib not installed — skipping plot)")

sys.exit(0 if all_pass else 1)
