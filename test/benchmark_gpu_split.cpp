// =============================================================================
// benchmark_gpu_split.cpp
//
// Measures the time saved by reading cluster data from GPU-resident storage
// (GpuClusterIndex::export_cluster) versus exporting from the L2 IVF index
// (MultiLevelIndex::export_l2_cluster) before running the k-means split kernel.
//
// In a real CUDA build:
//   GPU path  — data already in VRAM, k-means kernel runs with zero H2D cost.
//   L2 path   — requires D2H export from L2, then H2D re-upload for k-means.
//
// In this CPU simulation both "memory regions" are host RAM, so the difference
// reflects the overhead of traversing the IVF data structure (L2 path) versus
// reading from a flat std::vector (GPU path). The ratio scales to real GPU
// memory transfer costs on actual hardware.
//
// Usage:
//   ./build/benchmark_gpu_split [n_vectors] [dim] [n_trials] [max_iters]
//
// Defaults: n_vectors=8000  dim=64  n_trials=20  max_iters=20
//
// Output:
//   A table showing per-trial and aggregate timing for each path, plus the
//   speedup ratio.
// =============================================================================
#ifdef HAVE_CUDA
#  include <cuda_runtime.h>
#else
// ── CPU-build stubs for CUDA runtime symbols used in this file ──────────────
// The benchmark logic (IVF traversal vs flat-array export timing) is valid
// without a GPU; the stream/sync calls just become no-ops.
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess      = 0;
static constexpr cudaError_t cudaErrorNotReady = 600;
inline cudaError_t cudaStreamQuery(int)    { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
// ────────────────────────────────────────────────────────────────────────────
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>


#include "base.h"
#include "gpu_cluster_index.h"
#include "gpu_coordinator.h"
#include "split_kernel_v2.h"
#include "m3_multi_level.h"


using Clock = std::chrono::steady_clock;

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::atomic<int> g_trial{0};

// Build a MultiLevelIndex in cache mode with `nlist` L2 clusters.
static std::shared_ptr<m3::MultiLevelIndex>
make_index(int dim, int nlist = 1) {
    // MultiLevelConfig controls the nlist for each layer.
    m3::MultiLevelConfig ml_cfg;
    ml_cfg.l0_nlist = nlist;
    auto idx = std::make_shared<m3::MultiLevelIndex>(dim, m3::Metric::L2, false, ml_cfg);

    // Generate nlist centroids spread along the first axis.
    std::vector<float> centroids(static_cast<size_t>(nlist * dim), 0.f);
    for (int i = 0; i < nlist; ++i)
        centroids[static_cast<size_t>(i * dim)] = static_cast<float>(i);
    idx->set_l2_centroids(centroids);

    // Activate cache mode (required for load_cluster / export_l2_cluster).
    m3::CacheConfig cc;
    cc.l0_max_clusters            = nlist;
    cc.l1_max_clusters            = nlist;
    cc.l0_max_vectors_per_cluster = 0;  // unlimited
    cc.l1_max_vectors_per_cluster = 0;
    cc.cold_time_ns               = 600'000'000'000ULL;
    cc.alpha_et                   = 0.f;
    idx->set_cache_config(cc);
    return idx;
}

// Generate n random unit vectors of dimension dim.
static void gen_vectors(int n, int dim, std::mt19937& rng,
                        std::vector<m3::DocId>& ids,
                        std::vector<float>&     vecs) {
    std::normal_distribution<float> nd(0.f, 1.f);
    ids.resize(static_cast<size_t>(n));
    vecs.resize(static_cast<size_t>(n * dim));
    for (int i = 0; i < n; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<m3::DocId>(i + 1);
        float len = 0.f;
        for (int d = 0; d < dim; ++d) {
            const float v = nd(rng);
            vecs[static_cast<size_t>(i * dim + d)] = v;
            len += v * v;
        }
        len = std::sqrt(len);
        if (len > 0.f)
            for (int d = 0; d < dim; ++d)
                vecs[static_cast<size_t>(i * dim + d)] /= len;
    }
}

// ---- Path A: GPU-resident export ----
// Reads data from GpuClusterIndex (flat contiguous array, simulates VRAM).
static double bench_gpu_path(m3::GpuClusterIndex& gpu_idx, int cid,
                              int n, int dim, int max_iters) {

    int trial = g_trial.fetch_add(1);
    
    // Check queue depth BEFORE we start
    cudaError_t err;
    auto t_queue_check = Clock::now();
    err = cudaStreamQuery(0); // returns cudaSuccess if queue empty, cudaErrorNotReady if busy
    double queue_check_ms = elapsed_ms(t_queue_check);
    
    bool gpu_was_busy = (err == cudaErrorNotReady);
    
    // Now drain and measure drain time
    auto t_drain_start = Clock::now();
    cudaDeviceSynchronize();
    double drain_ms = elapsed_ms(t_drain_start);
    
    // Actual timed work
    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    const auto t0 = Clock::now();
    gpu_idx.export_cluster(cid, ids, vecs);
    m3::gpu_split_kmeans(vecs.data(), n, dim, max_iters);
    cudaDeviceSynchronize();
    double work_ms = elapsed_ms(t0);
    
    printf("  [GPU  t=%02d] busy_on_entry=%-5s  drain=%.3fms  work=%.3fms\n",
           trial, gpu_was_busy ? "YES" : "no", drain_ms, work_ms);

    return work_ms;
}

// ---- Path B: L2 export ----
// Exports from MultiLevelIndex IVF structure (simulates D2H + IVF traversal).
static double bench_l2_path(m3::MultiLevelIndex& idx, int cid,
                              int n, int dim, int max_iters) {
    int trial = g_trial.fetch_add(1);

    auto t_drain_start = Clock::now();
    bool gpu_was_busy = (cudaStreamQuery(0) == cudaErrorNotReady);
    cudaDeviceSynchronize();
    double drain_ms = elapsed_ms(t_drain_start);

    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    const auto t0 = Clock::now();
    idx.export_l2_cluster(cid, ids, vecs);
    m3::gpu_split_kmeans(vecs.data(), n, dim, max_iters);
    cudaDeviceSynchronize();
    double work_ms = elapsed_ms(t0);
    
    printf("  [L2   t=%02d] busy_on_entry=%-5s  drain=%.3fms  work=%.3fms\n",
           trial, gpu_was_busy ? "YES" : "no", drain_ms, work_ms);
    
    return work_ms;
}

static double mean(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(v.size()));
    return v[std::min(idx, v.size() - 1)];
}

int main(int argc, char** argv) {
    const int n_vectors  = (argc > 1) ? std::atoi(argv[1]) : 8000;
    const int dim        = (argc > 2) ? std::atoi(argv[2]) : 64;
    const int n_trials   = (argc > 3) ? std::atoi(argv[3]) : 20;
    const int max_iters  = (argc > 4) ? std::atoi(argv[4]) : 20;

    printf("=============================================================\n");
    printf("  M3 GPU Split Benchmark\n");
    printf("  n_vectors=%d  dim=%d  n_trials=%d  max_iters=%d\n",
           n_vectors, dim, n_trials, max_iters);
#ifdef HAVE_CUDA
    printf("  Build: CUDA enabled\n");
#else
    printf("  Build: CPU simulation (no CUDA)\n");
#endif
    printf("=============================================================\n\n");

    // Setup.
    std::mt19937 rng(42);
    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    gen_vectors(n_vectors, dim, rng, ids, vecs);

    // Build L2 index with data.
    auto idx = make_index(dim);
    idx->load_cluster(0, ids.data(), vecs.data(), static_cast<size_t>(n_vectors));

    // Build GPU index with same data (simulates post-promotion VRAM state).
    m3::GpuClusterIndex gpu_idx(dim, m3::Metric::L2, false);
    gpu_idx.store_cluster(0, ids.data(), vecs.data(), static_cast<size_t>(n_vectors));

    printf("  Cluster cid=0: %d vectors × dim=%d = %.1f KB\n\n",
           n_vectors, dim,
           static_cast<double>(n_vectors) * dim * sizeof(float) / 1024.0);

    // Warmup (not counted).
    bench_gpu_path(gpu_idx, 0, n_vectors, dim, max_iters);
    bench_l2_path(*idx,    0, n_vectors, dim, max_iters);

    // Timed trials — alternate A/B to avoid cache effects favouring one path.
    std::vector<double> times_gpu, times_l2;
    times_gpu.reserve(static_cast<size_t>(n_trials));
    times_l2.reserve(static_cast<size_t>(n_trials));

    printf("  %-6s  %12s  %12s\n", "Trial", "GPU path (ms)", "L2 path (ms)");
    printf("  %-6s  %12s  %12s\n", "-----", "-------------", "------------");

    for (int t = 0; t < n_trials; ++t) {
        const double tg = bench_gpu_path(gpu_idx, 0, n_vectors, dim, max_iters);
        const double tl = bench_l2_path(*idx,     0, n_vectors, dim, max_iters);
        times_gpu.push_back(tg);
        times_l2.push_back(tl);
        printf("  %-6d  %12.3f  %12.3f\n", t + 1, tg, tl);
    }

    const double mean_gpu   = mean(times_gpu);
    const double mean_l2    = mean(times_l2);
    const double med_gpu    = median(times_gpu);
    const double med_l2     = median(times_l2);
    const double p95_gpu    = pct(times_gpu, 95.0);
    const double p95_l2     = pct(times_l2,  95.0);
    const double speedup    = mean_l2 / mean_gpu;

    printf("\n  %-20s  %12s  %12s  %10s\n",
           "Statistic", "GPU path", "L2 path", "Speedup");
    printf("  %-20s  %12s  %12s  %10s\n",
           "--------------------", "--------", "-------", "-------");
    printf("  %-20s  %11.3f ms  %11.3f ms  %9.2fx\n",
           "Mean",   mean_gpu, mean_l2, speedup);
    printf("  %-20s  %11.3f ms  %11.3f ms  %9.2fx\n",
           "Median", med_gpu, med_l2, med_l2 / med_gpu);
    printf("  %-20s  %11.3f ms  %11.3f ms  %9.2fx\n",
           "P95",    p95_gpu, p95_l2, p95_l2 / p95_gpu);

    printf("\n  SUMMARY: GPU-resident path is %.2fx %s than L2-export path\n",
           speedup > 1.0 ? speedup : 1.0 / speedup,
           speedup > 1.0 ? "FASTER" : "slower");

#ifndef HAVE_CUDA
    printf("\n  NOTE: In CPU simulation both paths use host RAM; the speedup\n");
    printf("        reflects IVF traversal overhead vs flat array read.\n");
    printf("        On real GPU hardware the speedup is larger because the\n");
    printf("        L2 path requires an additional H2D memcpy before k-means.\n");
#endif
    printf("\n");
    return 0;
}
