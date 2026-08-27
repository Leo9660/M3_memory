#pragma once

#include <vector>

namespace m3 {

// ======================================================================
// Block 9 — GPU-side 2-centroid k-means split
//
// Splits a cluster's vector data into two partitions by running k-means
// with k=2 on the cluster data. In a CUDA-enabled build (HAVE_CUDA
// defined) the assignment and accumulation steps run as device kernels,
// avoiding a full H2D→split→D2H→re-upload round-trip.
//
// When HAVE_CUDA is not defined, the same interface is backed by a CPU
// implementation so the caller compiles identically in both modes.
//
// Interface contract
// ──────────────────
// • Both the CUDA and CPU implementations take a HOST pointer (h_vecs).
//   The CUDA version manages its own H2D upload internally — the caller
//   never needs to know about device memory.
// • Centroids are seeded from the first and last vector in h_vecs.
// • Iteration stops when assignments are stable or max_iters reached.
// • If n <= 1, partition[0] = 0 and both centroids equal the single vector
//   (or zero if n == 0).
// ======================================================================

// Result of a 2-centroid k-means split.
//   partition[i] == 0  →  vector i assigned to centroid A (original cluster)
//   partition[i] == 1  →  vector i assigned to centroid B (new cluster)
struct GpuSplitResult {
    std::vector<int>   partition;   // length n: 0 = A side, 1 = B side
    std::vector<float> centroid_a;  // final centroid A (dim floats)
    std::vector<float> centroid_b;  // final centroid B (dim floats)
    int                iters_run = 0;
};

// Run 2-centroid k-means on h_vecs (host pointer, n × dim, row-major).
// With HAVE_CUDA: uploads to device, runs kernels, returns host result.
// Without HAVE_CUDA: runs entirely on the host (CPU fallback).
GpuSplitResult gpu_split_kmeans(const float* h_vecs, int n, int dim,
                                 int max_iters = 20);

} // namespace m3
