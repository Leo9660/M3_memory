// kmeans_gpu_v4.cu
//
// Warp-coalesced, on-GPU-centroid-update 2-centroid k-means split (v4).
//
// ══════════════════════════════════════════════════════════════════════
// Key improvements over v3
// ══════════════════════════════════════════════════════════════════════
//
//   assign_kernel_v4
//   ─────────────────
//   v3: 1 thread/vector → non-coalesced reads (stride = dim floats between
//       consecutive threads in a warp → each warp transaction fetches 1
//       useful float out of 32 in a 128-byte cache line → 1/32 efficiency).
//       Convergence: D2H n labels per iteration (4000 × 4 = 16KB).
//
//   v4: 1 warp (32 threads)/vector → coalesced reads.
//       Consecutive threads access consecutive floats within the same
//       vector → 128-byte aligned transactions at full efficiency.
//       float4 loads: 4 floats per instruction → 4× fetch throughput.
//       Centroids cA, cB cached in shared memory (8KB for dim=1024),
//       broadcast to 32 lanes for free (smem bank width = 4 bytes).
//       Warp-shuffle reduction: 5 __shfl_down_sync ops reduce dA, dB
//       in registers (register-to-register, 0 memory traffic).
//       Convergence: GPU-side atomicOr on d_changed → only 1 int D2H
//       per iteration (4 bytes vs 16KB for n=4000).
//
//   accumulate_kernel_v4
//   ─────────────────────
//   v3: 1 thread/vector → scalar reads (same non-coalesced pattern).
//       Warp-shuffle fold reduces shared atomics by 32× vs v2, but the
//       read pattern is still 1/32 coalescing efficiency.
//
//   v4: 1 warp/vector → float4 coalesced reads (same as assign_v4).
//       32 lanes read a float4 each → 128 floats per warp per step,
//       all from a contiguous 512-byte region (4 full cache lines) →
//       full coalescing efficiency (32× better memory utilisation).
//       Two-level reduction structure identical to v3 otherwise.
//
//   centroid_update_kernel_v4
//   ──────────────────────────
//   v3: D2H sumA + sumB (8KB), CPU division, H2D cA + cB (8KB) per iter.
//       2 PCIe round-trips and a host synchronisation every iteration.
//   v4: dim-thread GPU kernel: cA[d] = sumA[d]/cntA, cB[d] = sumB[d]/cntB.
//       Zero host-device traffic for centroid update.
//
// ══════════════════════════════════════════════════════════════════════

#ifdef HAVE_CUDA

#include "kmeans_gpu_v4.h"
#include "gpu_split_kernel.h"   // GpuSplitResult

#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace m3 {

// ──────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────
// 4 warps × 32 = 128 threads/block; each warp handles one vector.
// Chosen so 4 vectors share the same centroid smem load phase,
// amortising the __syncthreads() barrier cost.
constexpr int ASSIGN_WARPS = 4;   // 128 threads/block

// 8 warps × 32 = 256 threads/block; each warp handles one vector.
// Higher occupancy for the memory-bound accumulate phase.
constexpr int ACCUM_WARPS  = 8;   // 256 threads/block

// ──────────────────────────────────────────────────────────────────────
// assign_kernel_v4
// ──────────────────────────────────────────────────────────────────────
// Block config : ASSIGN_WARPS × 32 = 128 threads/block
// Grid         : ceil(n / ASSIGN_WARPS) blocks
// Shared memory: sh_cA[dim] + sh_cB[dim]  = 2 × dim × sizeof(float)
//                (8KB for dim=1024, well within the 48KB per-block limit)
//
// Each warp handles exactly one vector:
//   1. All ASSIGN_WARPS × 32 threads in the block cooperatively load
//      centroids cA and cB into shared memory (coalesced global reads,
//      stride = blockDim.x = 128 between consecutive thread accesses).
//   2. Each lane t reads float4 chunks at positions [k*32 + t] within
//      the vector for k = 0..dim/128-1.  Consecutive lanes read
//      consecutive float4s → 128-byte cache-line aligned, full efficiency.
//      For dim % 128 != 0: scalar fallback with the same warp decomposition.
//   3. dA, dB accumulated in registers, then reduced across the 32-lane
//      warp with 5 __shfl_down_sync ops (register-to-register, ~1-2 cycles
//      each, zero memory traffic).
//   4. Lane 0 writes the label, compares to prev_label via register read,
//      and calls atomicOr(d_changed, 1) on mismatch → only 1 int D2H/iter.
//
// Parameters:
//   vecs        — device, n × dim row-major float (read-only)
//   n, dim      — cluster size and vector dimensionality
//   cA, cB      — device, dim floats each (current centroids)
//   prev_labels — device, n ints, initialised to -1 before first iter
//   labels      — device, n ints, output (current iteration labels)
//   d_changed   — device, 1 int, atomicOr'd to 1 if any label changed
// ──────────────────────────────────────────────────────────────────────
__global__ static void assign_kernel_v4(
        const float* __restrict__ vecs,  int n, int dim,
        const float* __restrict__ cA,    const float* __restrict__ cB,
        const int*   __restrict__ prev_labels,
        int*         __restrict__ labels,
        int*         __restrict__ d_changed)
{
    // Dynamic shared memory: [sh_cA | sh_cB]
    extern __shared__ float sh[];
    float* sh_cA = sh;
    float* sh_cB = sh + dim;

    const int tid        = (int)threadIdx.x;
    const int bdim       = (int)blockDim.x;          // = ASSIGN_WARPS * 32
    const int lane       = tid & 31;
    const int warp_in_blk = tid >> 5;

    // ── Cooperatively load centroids into shared memory ──────────────
    // All bdim threads stride across dim positions.  Consecutive threads
    // read consecutive floats → fully coalesced.
    for (int d = tid; d < dim; d += bdim) {
        sh_cA[d] = cA[d];
        sh_cB[d] = cB[d];
    }
    __syncthreads();

    // ── Warp decomposition ───────────────────────────────────────────
    const int vec_idx = blockIdx.x * ASSIGN_WARPS + warp_in_blk;
    if (vec_idx >= n) return;

    float dA = 0.f, dB = 0.f;

    if (dim % 128 == 0) {
        // ── float4 path (4 floats per load, 128-byte aligned per warp) ──
        // Each lane reads float4s at [k*32 + lane] within the vector.
        // Consecutive lanes read consecutive float4s → coalesced.
        const float4* v4   = reinterpret_cast<const float4*>(
                                 vecs + (ptrdiff_t)vec_idx * dim);
        const float4* cA4  = reinterpret_cast<const float4*>(sh_cA);
        const float4* cB4  = reinterpret_cast<const float4*>(sh_cB);

        // dim / 4 float4s per vector; each warp covers dim/4 in steps of 32
        // groups of 128 floats: dim/128 groups, each group = 32 float4s
        const int ngroups = dim / 128;
        for (int k = 0; k < ngroups; ++k) {
            const int idx4 = k * 32 + lane;    // float4 index in vector

            float4 v    = v4[idx4];
            float4 ca   = cA4[idx4];
            float4 cb   = cB4[idx4];

            float ex, ey;
            ex = v.x - ca.x; dA += ex * ex;
            ex = v.y - ca.y; dA += ex * ex;
            ex = v.z - ca.z; dA += ex * ex;
            ex = v.w - ca.w; dA += ex * ex;

            ey = v.x - cb.x; dB += ey * ey;
            ey = v.y - cb.y; dB += ey * ey;
            ey = v.z - cb.z; dB += ey * ey;
            ey = v.w - cb.w; dB += ey * ey;
        }
    } else {
        // ── Scalar fallback for dim not divisible by 128 ─────────────
        // Each lane handles elements at [lane, lane+32, lane+64, ...]
        const float* v = vecs + (ptrdiff_t)vec_idx * dim;
        for (int d = lane; d < dim; d += 32) {
            const float vd  = v[d];
            const float da  = vd - sh_cA[d];
            const float db  = vd - sh_cB[d];
            dA += da * da;
            dB += db * db;
        }
    }

    // ── Warp-reduce dA, dB (register-to-register, 5 shfl ops each) ──
    const unsigned FULL = 0xffffffffu;
    dA += __shfl_down_sync(FULL, dA, 16);
    dA += __shfl_down_sync(FULL, dA,  8);
    dA += __shfl_down_sync(FULL, dA,  4);
    dA += __shfl_down_sync(FULL, dA,  2);
    dA += __shfl_down_sync(FULL, dA,  1);

    dB += __shfl_down_sync(FULL, dB, 16);
    dB += __shfl_down_sync(FULL, dB,  8);
    dB += __shfl_down_sync(FULL, dB,  4);
    dB += __shfl_down_sync(FULL, dB,  2);
    dB += __shfl_down_sync(FULL, dB,  1);

    // ── Lane 0: write label + GPU-side convergence check ────────────
    if (lane == 0) {
        const int lbl = (dB < dA) ? 1 : 0;
        labels[vec_idx] = lbl;
        if (prev_labels[vec_idx] != lbl) {
            atomicOr(d_changed, 1);
        }
    }
}

// ──────────────────────────────────────────────────────────────────────
// accumulate_kernel_v4
// ──────────────────────────────────────────────────────────────────────
// Block config : ACCUM_WARPS × 32 = 256 threads/block
// Grid         : ceil(n / ACCUM_WARPS) blocks
// Shared memory: sh_sumA[dim] + sh_sumB[dim] + sh_cnt[2]
//                = 2 × dim × sizeof(float) + 2 × sizeof(int)
//                = 8200 bytes for dim=1024 (same layout as v3)
//
// Improvement over v3:
//   v3: 1 thread/vector → scalar reads (1 float per thread per dim iter).
//       Non-coalesced: consecutive threads in a warp belong to different
//       vectors separated by `dim` floats → ~1/32 memory efficiency.
//   v4: 1 warp/vector → float4 loads (32× better memory efficiency).
//       32 lanes read 32 consecutive float4s = 128 floats from one vector
//       in a single step → 128-byte cache-line aligned, full efficiency.
//
// Two-level reduction (identical hierarchy to v3):
//   Level 1 (registers): for each float4 chunk, smem atomicAdd per lane.
//   Level 2 (shared): warp leaders accumulate to smem.
//   Level 3 (global): thread 0 flushes block sums to global.
//
//   (Note: unlike assign_kernel_v4 where warp-shuffle reduces a scalar
//    dA/dB, here we accumulate to smem directly via atomicAdd because
//    each lane has a unique output dimension range — there's no further
//    intra-warp reduction needed for the sum.)
//
// Phase 0: zero sh_sumA, sh_sumB, sh_cnt cooperatively.
// Phase 1: each warp reads its vector with float4 loads, all 32 lanes
//          atomicAdd their 4 floats to the appropriate sh_sumA or sh_sumB
//          positions.  Shared atomic contention: 32 lanes × (dim/128)
//          float4 groups × 4 atomicAdds each = 32 × dim/128 × 4 × warps
//          per block. Lane 0 atomicAdds the count.
// Phase 2: thread 0 per block flushes sh_sumA, sh_sumB, sh_cnt to global.
// ──────────────────────────────────────────────────────────────────────
__global__ static void accumulate_kernel_v4(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA,   float* __restrict__ sumB,
        int*   __restrict__ cntA,   int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;                                  // [0,   dim)
    float* sh_sumB = sh + dim;                            // [dim, 2*dim)
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim); // sh_cnt[0]=A, [1]=B

    const int tid        = (int)threadIdx.x;
    const int bdim       = (int)blockDim.x;              // ACCUM_WARPS * 32
    const int lane       = tid & 31;
    const int warp_in_blk = tid >> 5;

    // ── Phase 0: zero shared memory cooperatively ───────────────────
    for (int d = tid; d < dim; d += bdim) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (tid == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // ── Phase 1: accumulate into shared ─────────────────────────────
    const int vec_idx = blockIdx.x * ACCUM_WARPS + warp_in_blk;
    if (vec_idx < n) {
        const int lbl   = labels[vec_idx];
        float* sh_target = (lbl == 0) ? sh_sumA : sh_sumB;

        if (dim % 128 == 0) {
            // ── float4 path ──────────────────────────────────────────
            // 32 lanes × (dim/128) groups × 32 float4s per group.
            // Each lane reads float4 at [k*32 + lane] within the vector.
            const float4* v4 = reinterpret_cast<const float4*>(
                                    vecs + (ptrdiff_t)vec_idx * dim);
            const int ngroups = dim / 128;
            for (int k = 0; k < ngroups; ++k) {
                const int idx4 = k * 32 + lane;          // float4 index
                float4 chunk   = v4[idx4];
                const int base = k * 128 + lane * 4;     // scalar index in dim
                atomicAdd(&sh_target[base + 0], chunk.x);
                atomicAdd(&sh_target[base + 1], chunk.y);
                atomicAdd(&sh_target[base + 2], chunk.z);
                atomicAdd(&sh_target[base + 3], chunk.w);
            }
        } else {
            // ── Scalar fallback ──────────────────────────────────────
            const float* v = vecs + (ptrdiff_t)vec_idx * dim;
            for (int d = lane; d < dim; d += 32) {
                atomicAdd(&sh_target[d], v[d]);
            }
        }

        // Count: only lane 0 to avoid 32× over-counting
        if (lane == 0) {
            atomicAdd(&sh_cnt[lbl], 1);
        }
    }

    // ── Phase 2: thread 0 flushes block partial sums to global ──────
    __syncthreads();
    if (tid == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

// ──────────────────────────────────────────────────────────────────────
// centroid_update_kernel_v4
// ──────────────────────────────────────────────────────────────────────
// Grid : ceil(dim / 256) blocks
// Block: 256 threads
//
// Each thread d computes:
//   cA[d] = sumA[d] / (float)(*d_cntA)
//   cB[d] = sumB[d] / (float)(*d_cntB)
//
// Eliminates per-iteration D2H (sums: 2×dim×4 bytes) + CPU division +
// H2D (updated centroids: 2×dim×4 bytes) of v3.
// For dim=1024: saves ~16KB PCIe per iteration and one host sync.
// ──────────────────────────────────────────────────────────────────────
__global__ static void centroid_update_kernel_v4(
        const float* __restrict__ sumA,  const float* __restrict__ sumB,
        const int*   __restrict__ d_cntA, const int*  __restrict__ d_cntB,
        float* __restrict__ cA, float* __restrict__ cB,
        int dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    cA[d] = sumA[d] / (float)(*d_cntA);
    cB[d] = sumB[d] / (float)(*d_cntB);
}

// ──────────────────────────────────────────────────────────────────────
// gpu_split_kmeans_v4_device  (core implementation)
// ──────────────────────────────────────────────────────────────────────
// Accepts a device pointer d_vecs.  Caller retains ownership — this
// function does NOT cudaFree d_vecs.
//
// All other device memory (centroids, labels, accumulators, prev_labels,
// d_changed) is allocated internally and freed before returning.
// ──────────────────────────────────────────────────────────────────────
GpuSplitResult gpu_split_kmeans_v4_device(const float* d_vecs,
                                            int n, int dim, int max_iters)
{
    GpuSplitResult result;
    result.centroid_a.resize(dim, 0.f);
    result.centroid_b.resize(dim, 0.f);
    result.partition.resize(n, 0);

    if (n <= 0 || dim <= 0) return result;

    if (n == 1) {
        // Single vector: both centroids equal that vector, partition stays 0
        cudaMemcpy(result.centroid_a.data(), d_vecs,
                   (size_t)dim * sizeof(float), cudaMemcpyDeviceToHost);
        result.centroid_b = result.centroid_a;
        return result;
    }

    // ── Sizes ─────────────────────────────────────────────────────────
    const size_t cen_bytes = (size_t)dim * sizeof(float);
    const size_t lbl_bytes = (size_t)n   * sizeof(int);

    // ── Device allocations (we do NOT free d_vecs) ────────────────────
    float *d_sumA, *d_sumB, *d_cA, *d_cB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    cudaMalloc(&d_sumA,       cen_bytes);
    cudaMalloc(&d_sumB,       cen_bytes);
    cudaMalloc(&d_cntA,       sizeof(int));
    cudaMalloc(&d_cntB,       sizeof(int));
    cudaMalloc(&d_cA,         cen_bytes);
    cudaMalloc(&d_cB,         cen_bytes);
    cudaMalloc(&d_labels,     lbl_bytes);
    cudaMalloc(&d_prev_labels, lbl_bytes);
    cudaMalloc(&d_changed,    sizeof(int));

    // ── Seed centroids from first and last vector (device-to-device) ──
    cudaMemcpy(d_cA, d_vecs,
               cen_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cB, d_vecs + (ptrdiff_t)(n - 1) * dim,
               cen_bytes, cudaMemcpyDeviceToDevice);

    // ── Initialise prev_labels to -1 (all-bits-1 → -1 for int) ──────
    // cudaMemset sets bytes; 0xFF bytes → 0xFFFFFFFF = -1 as int32.
    cudaMemset(d_prev_labels, 0xff, lbl_bytes);

    // ── Kernel launch parameters ──────────────────────────────────────
    const int assign_block  = ASSIGN_WARPS * 32;
    const int assign_grid   = (n + ASSIGN_WARPS - 1) / ASSIGN_WARPS;
    const size_t smem_assign = (size_t)2 * dim * sizeof(float);

    const int accum_block   = ACCUM_WARPS * 32;
    const int accum_grid    = (n + ACCUM_WARPS - 1) / ACCUM_WARPS;
    const size_t smem_accum = (size_t)2 * dim * sizeof(float)
                            + (size_t)2 * sizeof(int);

    const int update_block  = 256;
    const int update_grid   = (dim + update_block - 1) / update_block;

    // ── Main k-means loop ─────────────────────────────────────────────
    for (int iter = 0; iter < max_iters; ++iter) {

        // Reset convergence flag to 0
        int zero = 0;
        cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Assign step ───────────────────────────────────────────────
        assign_kernel_v4<<<assign_grid, assign_block, smem_assign>>>(
            d_vecs, n, dim,
            d_cA, d_cB,
            d_prev_labels, d_labels,
            d_changed);

        // D2H convergence check: 4 bytes (vs n×4 bytes in v3)
        int changed = 0;
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        ++result.iters_run;

        // Update prev_labels for next iteration (D2D copy)
        cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice);

        // Converged (and not the very first iteration — need at least one
        // accumulate so final centroids are computed)
        if (!changed && iter > 0) break;

        // ── Reset global accumulators ─────────────────────────────────
        cudaMemset(d_sumA, 0, cen_bytes);
        cudaMemset(d_sumB, 0, cen_bytes);
        cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Accumulate step ───────────────────────────────────────────
        accumulate_kernel_v4<<<accum_grid, accum_block, smem_accum>>>(
            d_vecs, n, dim,
            d_labels,
            d_sumA, d_sumB, d_cntA, d_cntB);

        // D2H counts only (8 bytes) for degenerate-split check
        int cntA_h = 0, cntB_h = 0;
        cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost);

        if (cntA_h == 0 || cntB_h == 0) break;   // degenerate split

        // ── On-GPU centroid update (zero PCIe traffic) ────────────────
        centroid_update_kernel_v4<<<update_grid, update_block>>>(
            d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
    }

    // ── D2H final results ─────────────────────────────────────────────
    cudaMemcpy(result.partition.data(),  d_labels, lbl_bytes,    cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_a.data(), d_cA,     cen_bytes,    cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_b.data(), d_cB,     cen_bytes,    cudaMemcpyDeviceToHost);

    // ── Free all internally-allocated device memory (NOT d_vecs) ─────
    cudaFree(d_sumA);       cudaFree(d_sumB);
    cudaFree(d_cntA);       cudaFree(d_cntB);
    cudaFree(d_cA);         cudaFree(d_cB);
    cudaFree(d_labels);     cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return result;
}

// ──────────────────────────────────────────────────────────────────────
// gpu_split_kmeans_v4  (host-pointer wrapper)
// ──────────────────────────────────────────────────────────────────────
// Accepts a host pointer h_vecs.  Uploads to device, runs the full
// v4 k-means loop, returns results, frees device memory.
// ──────────────────────────────────────────────────────────────────────
GpuSplitResult gpu_split_kmeans_v4(const float* h_vecs,
                                     int n, int dim, int max_iters)
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

    // ── Sizes ─────────────────────────────────────────────────────────
    const size_t vec_bytes = (size_t)n   * dim * sizeof(float);
    const size_t cen_bytes = (size_t)dim * sizeof(float);
    const size_t lbl_bytes = (size_t)n   * sizeof(int);

    // ── Device allocations ────────────────────────────────────────────
    float *d_vecs, *d_sumA, *d_sumB, *d_cA, *d_cB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    cudaMalloc(&d_vecs,       vec_bytes);
    cudaMalloc(&d_sumA,       cen_bytes);
    cudaMalloc(&d_sumB,       cen_bytes);
    cudaMalloc(&d_cntA,       sizeof(int));
    cudaMalloc(&d_cntB,       sizeof(int));
    cudaMalloc(&d_cA,         cen_bytes);
    cudaMalloc(&d_cB,         cen_bytes);
    cudaMalloc(&d_labels,     lbl_bytes);
    cudaMalloc(&d_prev_labels, lbl_bytes);
    cudaMalloc(&d_changed,    sizeof(int));

    // ── H2D: vectors ──────────────────────────────────────────────────
    cudaMemcpy(d_vecs, h_vecs, vec_bytes, cudaMemcpyHostToDevice);

    // ── Seed centroids from first and last vector ─────────────────────
    cudaMemcpy(d_cA, d_vecs,
               cen_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_cB, d_vecs + (ptrdiff_t)(n - 1) * dim,
               cen_bytes, cudaMemcpyDeviceToDevice);

    // ── Initialise prev_labels to -1 ──────────────────────────────────
    cudaMemset(d_prev_labels, 0xff, lbl_bytes);

    // ── Kernel launch parameters ──────────────────────────────────────
    const int assign_block   = ASSIGN_WARPS * 32;
    const int assign_grid    = (n + ASSIGN_WARPS - 1) / ASSIGN_WARPS;
    const size_t smem_assign = (size_t)2 * dim * sizeof(float);

    const int accum_block    = ACCUM_WARPS * 32;
    const int accum_grid     = (n + ACCUM_WARPS - 1) / ACCUM_WARPS;
    const size_t smem_accum  = (size_t)2 * dim * sizeof(float)
                             + (size_t)2 * sizeof(int);

    const int update_block   = 256;
    const int update_grid    = (dim + update_block - 1) / update_block;

    // ── Main k-means loop ─────────────────────────────────────────────
    for (int iter = 0; iter < max_iters; ++iter) {

        // Reset convergence flag
        int zero = 0;
        cudaMemcpy(d_changed, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Assign step ───────────────────────────────────────────────
        assign_kernel_v4<<<assign_grid, assign_block, smem_assign>>>(
            d_vecs, n, dim,
            d_cA, d_cB,
            d_prev_labels, d_labels,
            d_changed);

        // D2H convergence check: 4 bytes
        int changed = 0;
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        ++result.iters_run;

        // Update prev_labels for next iter
        cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice);

        if (!changed && iter > 0) break;

        // ── Reset global accumulators ─────────────────────────────────
        cudaMemset(d_sumA, 0, cen_bytes);
        cudaMemset(d_sumB, 0, cen_bytes);
        cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // ── Accumulate step ───────────────────────────────────────────
        accumulate_kernel_v4<<<accum_grid, accum_block, smem_accum>>>(
            d_vecs, n, dim,
            d_labels,
            d_sumA, d_sumB, d_cntA, d_cntB);

        // D2H counts (8 bytes) for degenerate-split check
        int cntA_h = 0, cntB_h = 0;
        cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost);

        if (cntA_h == 0 || cntB_h == 0) break;

        // ── On-GPU centroid update ────────────────────────────────────
        centroid_update_kernel_v4<<<update_grid, update_block>>>(
            d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, dim);
    }

    // ── D2H final results ─────────────────────────────────────────────
    cudaMemcpy(result.partition.data(),  d_labels, lbl_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_a.data(), d_cA,     cen_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(result.centroid_b.data(), d_cB,     cen_bytes, cudaMemcpyDeviceToHost);

    // ── Free all device memory (including d_vecs) ─────────────────────
    cudaFree(d_vecs);
    cudaFree(d_sumA);       cudaFree(d_sumB);
    cudaFree(d_cntA);       cudaFree(d_cntB);
    cudaFree(d_cA);         cudaFree(d_cB);
    cudaFree(d_labels);     cudaFree(d_prev_labels);
    cudaFree(d_changed);

    return result;
}

} // namespace m3

#else
#error "kmeans_gpu_v4.cu must be compiled with HAVE_CUDA defined (M3_USE_CUDA=ON)"
#endif
