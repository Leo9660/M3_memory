// gpu_split_kernel_cpu.cpp
//
// CPU fallback for Block 9 — 2-centroid k-means split.
// Compiled in place of gpu_split_kernel.cu when M3_USE_CUDA=OFF (default).
// The function signature is identical to the CUDA version so all callers
// compile without changes.

#include "split_kernel_v3.h"
#include "kmeans_gpu_v4.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace m3 {

GpuSplitResult gpu_split_kmeans(const float* h_vecs, int n, int dim,
                                 int max_iters)
{
    GpuSplitResult result;
    result.centroid_a.resize(dim, 0.f);
    result.centroid_b.resize(dim, 0.f);
    result.partition.resize(n, 0);

    if (n <= 0 || dim <= 0) return result;

    if (n == 1) {
        std::copy(h_vecs, h_vecs + dim, result.centroid_a.data());
        result.centroid_b = result.centroid_a;
        return result;
    }

    // Seed centroids: first and last vector.
    std::copy(h_vecs,             h_vecs + dim,      result.centroid_a.data());
    std::copy(h_vecs + (n-1)*dim, h_vecs + n * dim,  result.centroid_b.data());

    for (int iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // ---- Assign step ----
        for (int i = 0; i < n; ++i) {
            float dA = 0.f, dB = 0.f;
            const float* v = h_vecs + static_cast<ptrdiff_t>(i) * dim;
            for (int d = 0; d < dim; ++d) {
                float da = v[d] - result.centroid_a[d];
                float db = v[d] - result.centroid_b[d];
                dA += da * da;
                dB += db * db;
            }
            const int label = (dB < dA) ? 1 : 0;
            if (label != result.partition[i]) {
                result.partition[i] = label;
                changed = true;
            }
        }

        ++result.iters_run;
        if (!changed) break;

        // ---- Update step ----
        std::vector<float> sumA(dim, 0.f), sumB(dim, 0.f);
        int cntA = 0, cntB = 0;
        for (int i = 0; i < n; ++i) {
            const float* v = h_vecs + static_cast<ptrdiff_t>(i) * dim;
            if (result.partition[i] == 0) {
                for (int d = 0; d < dim; ++d) sumA[d] += v[d];
                ++cntA;
            } else {
                for (int d = 0; d < dim; ++d) sumB[d] += v[d];
                ++cntB;
            }
        }

        if (cntA == 0 || cntB == 0) break;  // degenerate: all on one side

        for (int d = 0; d < dim; ++d) {
            result.centroid_a[d] = sumA[d] / static_cast<float>(cntA);
            result.centroid_b[d] = sumB[d] / static_cast<float>(cntB);
        }
    }

    return result;
}

// ── v2 CPU stub ──────────────────────────────────────────────────────────────
// The shared-memory optimisation in accumulate_kernel_v2 only reduces global
// atomicAdd contention on the GPU.  On CPU there is no shared memory, so v2
// and v1 are algorithmically identical.  This stub lets CPU builds link against
// any caller that references gpu_split_kmeans_v2() (e.g. profiling code) and
// lets profile/bench_kmeans compare v1 vs v2 timing to confirm they match on CPU
// while diverging on real CUDA hardware.
GpuSplitResult gpu_split_kmeans_v2(const float* h_vecs, int n, int dim,
                                     int max_iters)
{
    return gpu_split_kmeans(h_vecs, n, dim, max_iters);
}

// ── v3 CPU stub ──────────────────────────────────────────────────────────────
// The warp-shuffle reduction in accumulate_kernel_v3 is a GPU-only optimisation
// (no equivalent on CPU).  On CPU this is algorithmically identical to v1.
GpuSplitResult gpu_split_kmeans_v3(const float* h_vecs, int n, int dim,
                                     int max_iters)
{
    return gpu_split_kmeans(h_vecs, n, dim, max_iters);
}

// ── v4 CPU stubs ──────────────────────────────────────────────────────────────
// v4 improvements (1-warp/vector coalescing, float4 loads, on-GPU centroid
// update, GPU-side convergence flag, device-pointer variant) are CUDA-only.
// On CPU both variants fall through to the v1 implementation.
GpuSplitResult gpu_split_kmeans_v4(const float* h_vecs, int n, int dim,
                                     int max_iters)
{
    return gpu_split_kmeans(h_vecs, n, dim, max_iters);
}

GpuSplitResult gpu_split_kmeans_v4_device(const float* d_vecs, int n, int dim,
                                            int max_iters)
{
    // No GPU available — d_vecs is treated as a host pointer in CPU builds.
    // In practice this stub is never called in production CPU-only deployments.
    return gpu_split_kmeans(d_vecs, n, dim, max_iters);
}

} // namespace m3
