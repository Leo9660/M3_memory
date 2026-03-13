// gpu_split_kernel.cu
//
// GPU-side 2-centroid k-means split (Block 9).
//
// Compiled by nvcc when M3_USE_CUDA=ON (defines HAVE_CUDA).
// When HAVE_CUDA is not defined (e.g. nvcc without the define, which
// shouldn't happen in practice), this file falls through to a
// compile-error guard — use gpu_split_kernel_cpu.cpp for CPU builds.

#ifdef HAVE_CUDA

#include "gpu_split_kernel.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace m3 {

// ======================================================================
// Device kernels
// ======================================================================

// assign_kernel — one thread per vector.
// Computes squared L2 distance to centroid A and B and writes 0 or 1 to
// labels[i]. Ties broken in favour of A (label = 0).
__global__ static void assign_kernel(const float* __restrict__ vecs,
                                      int n, int dim,
                                      const float* __restrict__ cA,
                                      const float* __restrict__ cB,
                                      int* __restrict__ labels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float dA = 0.f, dB = 0.f;
    const float* v = vecs + static_cast<ptrdiff_t>(i) * dim;
    for (int d = 0; d < dim; ++d) {
        float da = v[d] - cA[d];
        float db = v[d] - cB[d];
        dA += da * da;
        dB += db * db;
    }
    labels[i] = (dB < dA) ? 1 : 0;
}

// accumulate_kernel — one thread per vector.
// Atomically accumulates partial sums into sumA/sumB and increments
// countA/countB so the host can divide to get new centroid means.
// Uses float atomicAdd (available on sm_20+ for global memory).
__global__ static void accumulate_kernel(const float* __restrict__ vecs,
                                          int n, int dim,
                                          const int* __restrict__ labels,
                                          float* __restrict__ sumA,
                                          float* __restrict__ sumB,
                                          int*   __restrict__ countA,
                                          int*   __restrict__ countB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = vecs + static_cast<ptrdiff_t>(i) * dim;
    if (labels[i] == 0) {
        for (int d = 0; d < dim; ++d)
            atomicAdd(&sumA[d], v[d]);
        atomicAdd(countA, 1);
    } else {
        for (int d = 0; d < dim; ++d)
            atomicAdd(&sumB[d], v[d]);
        atomicAdd(countB, 1);
    }
}

// ======================================================================
// Host wrapper
// ======================================================================

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

    // ---- Allocate device memory ----
    const size_t vec_bytes = static_cast<size_t>(n) * dim * sizeof(float);
    float* d_vecs  = nullptr;
    float* d_sumA  = nullptr;
    float* d_sumB  = nullptr;
    int*   d_cntA  = nullptr;
    int*   d_cntB  = nullptr;
    float* d_cA    = nullptr;
    float* d_cB    = nullptr;
    int*   d_labels = nullptr;

    const size_t cen_bytes = static_cast<size_t>(dim) * sizeof(float);

    cudaMalloc(&d_vecs,   vec_bytes);
    cudaMalloc(&d_sumA,   cen_bytes);
    cudaMalloc(&d_sumB,   cen_bytes);
    cudaMalloc(&d_cntA,   sizeof(int));
    cudaMalloc(&d_cntB,   sizeof(int));
    cudaMalloc(&d_cA,     cen_bytes);
    cudaMalloc(&d_cB,     cen_bytes);
    cudaMalloc(&d_labels, static_cast<size_t>(n) * sizeof(int));

    // Upload vectors (H2D — done once, data is read-only during k-means).
    cudaMemcpy(d_vecs, h_vecs, vec_bytes, cudaMemcpyHostToDevice);

    // Seed centroids: first and last vector.
    cudaMemcpy(d_cA, h_vecs,                      cen_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cB, h_vecs + (n - 1) * dim,      cen_bytes, cudaMemcpyHostToDevice);

    std::copy(h_vecs,              h_vecs + dim,       result.centroid_a.data());
    std::copy(h_vecs + (n-1)*dim,  h_vecs + n * dim,   result.centroid_b.data());

    constexpr int BLOCK = 256;
    const int grid = (n + BLOCK - 1) / BLOCK;

    // Host-side label buffer to detect convergence.
    std::vector<int> prev_labels(n, -1);
    std::vector<int> cur_labels(n, 0);

    for (int iter = 0; iter < max_iters; ++iter) {
        // --- Assign step ---
        assign_kernel<<<grid, BLOCK>>>(d_vecs, n, dim, d_cA, d_cB, d_labels);

        // Download labels to check convergence.
        cudaMemcpy(cur_labels.data(), d_labels,
                   static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);
        ++result.iters_run;

        bool changed = (iter == 0) || (cur_labels != prev_labels);
        prev_labels = cur_labels;

        if (!changed) break;

        // --- Update step: zero accumulators, run accumulate_kernel ---
        cudaMemset(d_sumA, 0, cen_bytes);
        cudaMemset(d_sumB, 0, cen_bytes);
        int zero = 0;
        cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice);

        accumulate_kernel<<<grid, BLOCK>>>(d_vecs, n, dim, d_labels,
                                           d_sumA, d_sumB, d_cntA, d_cntB);

        // Download sums and counts; compute new centroids on host.
        int cntA = 0, cntB = 0;
        std::vector<float> sumA(dim), sumB(dim);
        cudaMemcpy(&cntA, d_cntA, sizeof(int),  cudaMemcpyDeviceToHost);
        cudaMemcpy(&cntB, d_cntB, sizeof(int),  cudaMemcpyDeviceToHost);
        cudaMemcpy(sumA.data(), d_sumA, cen_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(sumB.data(), d_sumB, cen_bytes, cudaMemcpyDeviceToHost);

        if (cntA == 0 || cntB == 0) break;  // degenerate split

        for (int d = 0; d < dim; ++d) {
            result.centroid_a[d] = sumA[d] / static_cast<float>(cntA);
            result.centroid_b[d] = sumB[d] / static_cast<float>(cntB);
        }

        // Upload updated centroids for next iteration.
        cudaMemcpy(d_cA, result.centroid_a.data(), cen_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_cB, result.centroid_b.data(), cen_bytes, cudaMemcpyHostToDevice);
    }

    result.partition = cur_labels;

    // ---- Free device memory ----
    cudaFree(d_vecs);
    cudaFree(d_sumA);
    cudaFree(d_sumB);
    cudaFree(d_cntA);
    cudaFree(d_cntB);
    cudaFree(d_cA);
    cudaFree(d_cB);
    cudaFree(d_labels);

    return result;
}

} // namespace m3

#else // !HAVE_CUDA

// This translation unit should only be compiled with HAVE_CUDA defined.
// For CPU-only builds, link gpu_split_kernel_cpu.cpp instead.
#error "gpu_split_kernel.cu compiled without HAVE_CUDA — use gpu_split_kernel_cpu.cpp"

#endif // HAVE_CUDA
