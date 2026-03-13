"""
Benchmark: CPU vs GPU operation costs using vectors from /data/IVF.index
  Plot 1: Search latency (CPU vs GPU) as cluster size grows
  Plot 2: CPU->GPU transfer, GPU->GPU transfer, and GPU allocation costs
"""

import numpy as np
import faiss
import time
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
INDEX_PATH  = "/data/IVF.index"
CLUSTER_SIZES = [32, 128, 512, 2048, 8192]
N_QUERIES   = 10          # queries per search trial
K           = 10          # nearest neighbours
N_REPS      = 20          # repetitions for stable timing
D           = 1024        # vector dimension (confirmed above)

# ── Load index & extract vectors ─────────────────────────────────────────────
print("Loading index …")
idx      = faiss.read_index(INDEX_PATH)
invlists = idx.invlists
d        = idx.d
assert d == D

# Collect enough vectors by concatenating small clusters
print("Extracting vectors from inverted lists …")
all_vecs = []
for c in range(idx.nlist):
    sz = invlists.list_size(c)
    if sz == 0:
        continue
    codes_ptr = invlists.get_codes(c)
    codes = faiss.rev_swig_ptr(codes_ptr, sz * d).reshape(sz, d).copy()
    all_vecs.append(codes.astype(np.float32))
    if sum(v.shape[0] for v in all_vecs) >= max(CLUSTER_SIZES) * 2:
        break

all_vecs = np.vstack(all_vecs)
print(f"  Collected {all_vecs.shape[0]} vectors of dim {d}")

# Fixed query vectors
rng     = np.random.default_rng(42)
queries = all_vecs[rng.choice(all_vecs.shape[0], N_QUERIES, replace=False)].copy()

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_flat_cpu(vecs):
    fi = faiss.IndexFlatL2(D)
    fi.add(vecs)
    return fi

def make_flat_gpu(vecs, res):
    fi = make_flat_cpu(vecs)
    return faiss.index_cpu_to_gpu(res, 0, fi)

def time_search_cpu(vecs):
    fi = make_flat_cpu(vecs)
    # warm-up
    fi.search(queries, K)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        fi.search(queries, K)
    return (time.perf_counter() - t0) / N_REPS * 1e3   # ms

def time_search_gpu(vecs, res):
    gi = make_flat_gpu(vecs, res)
    # warm-up
    gi.search(queries, K)
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        gi.search(queries, K)
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / N_REPS * 1e3   # ms

def time_cpu_to_gpu(vecs):
    """Host->Device transfer using cupy."""
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        gpu_arr = cp.asarray(vecs)
        cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / N_REPS * 1e3   # ms

def time_gpu_to_gpu(vecs):
    """Device->Device copy (simulate inter-buffer move on GPU)."""
    src = cp.asarray(vecs)
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        dst = src.copy()
        cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / N_REPS * 1e3   # ms

def time_gpu_alloc(vecs):
    """GPU memory allocation (malloc + free) for cluster-sized buffer."""
    n_bytes = vecs.nbytes
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        buf = cp.empty(n_bytes // 4, dtype=cp.float32)
        cp.cuda.Device(0).synchronize()
        del buf
    return (time.perf_counter() - t0) / N_REPS * 1e3   # ms

# ── Run benchmarks ────────────────────────────────────────────────────────────
print("Initialising FAISS GPU resource …")
res = faiss.StandardGpuResources()

search_cpu  = []
search_gpu  = []
cpu_to_gpu  = []
gpu_to_gpu  = []
gpu_alloc   = []

for sz in CLUSTER_SIZES:
    vecs = all_vecs[:sz].copy()
    mb   = vecs.nbytes / 1024**2
    print(f"\nCluster size {sz:>5}  ({mb:.2f} MB)")

    sc = time_search_cpu(vecs)
    print(f"  search  CPU : {sc:.3f} ms")
    search_cpu.append(sc)

    sg = time_search_gpu(vecs, res)
    print(f"  search  GPU : {sg:.3f} ms")
    search_gpu.append(sg)

    c2g = time_cpu_to_gpu(vecs)
    print(f"  CPU->GPU    : {c2g:.3f} ms")
    cpu_to_gpu.append(c2g)

    g2g = time_gpu_to_gpu(vecs)
    print(f"  GPU->GPU    : {g2g:.3f} ms")
    gpu_to_gpu.append(g2g)

    ga = time_gpu_alloc(vecs)
    print(f"  GPU alloc   : {ga:.3f} ms")
    gpu_alloc.append(ga)

# ── Plot 1: Search Latency ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(CLUSTER_SIZES, search_cpu, "o-",  color="#E05C5C", label="CPU search",  linewidth=2, markersize=7)
ax.plot(CLUSTER_SIZES, search_gpu, "s--", color="#4C9BE8", label="GPU search",  linewidth=2, markersize=7)
ax.set_xscale("log", base=2)
ax.set_xticks(CLUSTER_SIZES)
ax.set_xticklabels([str(s) for s in CLUSTER_SIZES])
ax.set_xlabel("Cluster size (# vectors)", fontsize=12)
ax.set_ylabel("Search latency (ms)", fontsize=12)
ax.set_title(f"Search latency: CPU vs GPU  (k={K}, nq={N_QUERIES})", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot1_search_latency.png", dpi=150)
print("\nSaved plot1_search_latency.png")

# ── Plot 2: Data Transfer & Allocation Costs ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(CLUSTER_SIZES, cpu_to_gpu, "o-",  color="#F4A136", label="CPU → GPU transfer", linewidth=2, markersize=7)
ax.plot(CLUSTER_SIZES, gpu_to_gpu, "s--", color="#6BCB77", label="GPU → GPU copy",     linewidth=2, markersize=7)
ax.plot(CLUSTER_SIZES, gpu_alloc,  "^:",  color="#9B59B6", label="GPU allocation",      linewidth=2, markersize=7)
ax.set_xscale("log", base=2)
ax.set_xticks(CLUSTER_SIZES)
ax.set_xticklabels([str(s) for s in CLUSTER_SIZES])
ax.set_xlabel("Cluster size (# vectors)", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title("Memory operation costs vs cluster size", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot2_transfer_alloc.png", dpi=150)
print("Saved plot2_transfer_alloc.png")
