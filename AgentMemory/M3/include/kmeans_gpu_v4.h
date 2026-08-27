#pragma once

// kmeans_gpu_v4.h
//
// Warp-coalesced, on-GPU-centroid-update 2-centroid k-means split (v4).
//
// ══════════════════════════════════════════════════════════════════════
// Improvements over v3 (split_kernel_v3.h / split_kernel_v3.cu)
// ══════════════════════════════════════════════════════════════════════
//
//   assign: 1 warp/vector (vs 1 thread/vector in v3)
//   ──────
//   • 32 threads cooperate on each vector → coalesced 128-byte cache-line
//     reads instead of v3's stride-dim non-coalesced access pattern.
//   • Centroids loaded once into shared memory (8KB for dim=1024),
//     broadcast to all 32 lanes via smem rather than 32 independent
//     global reads.
//   • float4 vectorised loads: 4 floats per transaction → 4× throughput
//     on the vector data when dim % 128 == 0.
//   • dA, dB partial sums accumulated in registers and reduced across the
//     warp with 5 __shfl_down_sync ops (register-to-register, 0 memory).
//   • GPU-side convergence detection via atomicOr on a single int flag:
//     only 1 int D2H per iteration instead of n ints (e.g. 4 bytes vs
//     16 KB for n=4000). Lane 0 of each warp compares new label to
//     prev_label and calls atomicOr(d_changed, 1) on mismatch.
//
//   accumulate: 1 warp/vector + float4 coalesced reads
//   ─────────────────────────
//   • Same 1-warp-per-vector decomposition as v4 assign: all 32 lanes
//     read float4 chunks at stride-32 positions → 32× better memory
//     efficiency vs v3's 1-thread-per-vector scalar reads.
//   • Two-level smem reduction structure identical to v3
//     (warp-shuffle fold → warp leader → smem atomic → thread 0 global
//     flush), preserving the 32× shared-atomic reduction of v3.
//
//   centroid_update: on-GPU kernel (eliminates per-iteration D2H+H2D)
//   ─────────────────────────────────────────────────────────────────
//   • v3 transfers 2 × dim floats D2H (sums), divides on CPU, then
//     uploads 2 × dim floats H2D (new centroids) every iteration.
//     For dim=1024: ~8KB D2H + CPU sync + ~8KB H2D ≈ 3 PCIe round-trips.
//   • v4 centroid_update_kernel runs dim threads on GPU: each thread
//     computes cA[d] = sumA[d]/countA and cB[d] = sumB[d]/countB
//     entirely on-device. Zero host-device centroid traffic per iteration.
//
//   Device-pointer variant: gpu_split_kmeans_v4_device
//   ─────────────────────────────────────────────────
//   • Accepts a device pointer d_vecs directly (GPU-resident clusters).
//   • For a 4096-cluster IVF index with 4000 × 1024 float vectors,
//     the H2D upload is ~16MB. gpu_split_kmeans_v4_device skips this
//     entirely when the cluster data is already on the device (e.g.
//     produced by a prior GPU kernel), saving one full 16MB PCIe transfer
//     per split call.
//
// ══════════════════════════════════════════════════════════════════════
// Summary of per-iteration PCIe traffic
// ══════════════════════════════════════════════════════════════════════
//
//   v3  (n=4000, dim=1024):
//     D2H n labels          = 4000 × 4 =  ~16 KB  (convergence check)
//     D2H sumA+sumB+cntA+cntB = 2×4KB + 8 =  ~8 KB  (centroid compute)
//     H2D cA+cB             = 2 × 4KB   =  ~8 KB  (updated centroids)
//     Total per iteration   ≈ 32 KB     + 2 PCIe syncs
//
//   v4:
//     D2H d_changed         = 4 bytes            (convergence flag)
//     D2H cntA+cntB         = 8 bytes            (degenerate check only)
//     Total per iteration   = 12 bytes   + 1 PCIe sync
//
// ══════════════════════════════════════════════════════════════════════

#include "gpu_split_kernel.h"   // GpuSplitResult

namespace m3 {

// Warp-coalesced float4 2-centroid k-means with on-GPU centroid update.
// Drop-in replacement for gpu_split_kmeans(), v2, and v3.
//   h_vecs    — host pointer, n × dim row-major float
//   n, dim    — cluster size and vector dimensionality
//   max_iters — k-means iteration cap
GpuSplitResult gpu_split_kmeans_v4(const float* h_vecs, int n, int dim,
                                     int max_iters = 20);

// Device-pointer variant: d_vecs must be a device pointer (GPU-resident).
// Caller retains ownership of d_vecs — this function does NOT cudaFree it.
// All other device allocations (labels, centroids, accumulators) are
// managed internally and freed before returning.
// Use this when the cluster data is already on the GPU (e.g. from a prior
// kernel) to skip the ~16MB H2D upload entirely.
GpuSplitResult gpu_split_kmeans_v4_device(const float* d_vecs, int n, int dim,
                                            int max_iters = 20);

} // namespace m3
