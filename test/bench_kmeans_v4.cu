// bench_kmeans_v4.cu
//
// Per-step CUDA event benchmark: v3 (reference) vs v4 (warp-coalesced,
// on-GPU centroid update) 2-centroid k-means split kernels.
//
// Measures and reports timing for every sub-step of one k-means iteration:
//   assign kernel, D2H transfer, accumulate kernel, centroid update.
//
// Also runs complete k-means loops for end-to-end speedup comparison.
//
// ══════════════════════════════════════════════════════════════════════
// Build:
//   nvcc -O3 -DHAVE_CUDA --gpu-architecture=sm_86 \
//        -I AgentMemory/M3/include \
//        test/bench_kmeans_v4.cu -o bench_kmeans_v4
// Run:
//   ./bench_kmeans_v4
// Profile with nvprof:
//   nvprof --print-gpu-trace ./bench_kmeans_v4
// Profile with Nsight:
//   ncu --set full --target-processes all -o kmeans_v4_profile ./bench_kmeans_v4
// ══════════════════════════════════════════════════════════════════════

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>

// ─────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────
static constexpr int N         = 4000;   // IVF4096 cluster size
static constexpr int DIM       = 1024;   // embedding dimension
static constexpr int MAX_ITERS = 20;
static constexpr int N_WARMUP  = 3;
static constexpr int N_TRIALS  = 10;

// ─────────────────────────────────────────────────────────────────────
// Error checking
// ─────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(c) do {                                                   \
    cudaError_t _e = (c);                                                    \
    if (_e != cudaSuccess) {                                                 \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                  \
        exit(1);                                                             \
    }                                                                        \
} while(0)

// ─────────────────────────────────────────────────────────────────────
// GPU timer using CUDA events
// ─────────────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e)); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void  start()  { CUDA_CHECK(cudaEventRecord(s)); }
    float stop()   {                                // returns milliseconds
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────
static void print_sep(char c = '-', int w = 76) {
    for (int i = 0; i < w; ++i) putchar(c);
    putchar('\n');
}
static void phdr(const char* s) {
    print_sep('=');
    printf("  %s\n", s);
    print_sep('=');
}

// ══════════════════════════════════════════════════════════════════════
// KERNEL DEFINITIONS
// ══════════════════════════════════════════════════════════════════════

// Kernel configuration constants
constexpr int ASSIGN_WARPS = 4;   // 128 threads/block for assign_kernel_v4
constexpr int ACCUM_WARPS  = 8;   // 256 threads/block for accumulate_kernel_v4

// ─────────────────────────────────────────────────────────────────────
// flush_l2_kernel
// ─────────────────────────────────────────────────────────────────────
// Reads a buffer larger than the GPU L2 cache to evict all prior
// resident data.  Without this, v4 trials would see v3's warm cache
// and each trial would see the previous trial's warm cache, making
// measurements cache-state-dependent rather than independent.
//
// H100 L2 = 50 MB, A100 L2 = 40 MB → use 60 MB to cover both.
// Grid-stride loop ensures every byte is touched regardless of grid size.
// The atomicAdd to d_out is the only write; it prevents the compiler
// from optimizing away the reads as dead code.
// ─────────────────────────────────────────────────────────────────────
static constexpr size_t L2_FLUSH_BYTES   = 60ULL * 1024 * 1024;   // 60 MB
static constexpr int    L2_FLUSH_FLOATS  = (int)(L2_FLUSH_BYTES / sizeof(float));
static constexpr int    L2_FLUSH_BLOCKS  = 480;
static constexpr int    L2_FLUSH_THREADS = 256;

__global__ static void flush_l2_kernel(const float* __restrict__ buf,
                                        int n, float* __restrict__ out)
{
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x) {
        acc += buf[i];
    }
    // Write per-warp to avoid 256-way atomic contention while still
    // forcing the reads to be materialised.
    if (threadIdx.x == 0) atomicAdd(out, acc);
}

// ─────────────────────────────────────────────────────────────────────
// assign_kernel_ref  (v3's assign: 1 thread/vector, non-coalesced reads)
// ─────────────────────────────────────────────────────────────────────
// Convergence in v3: D2H all n labels after each call.
// Memory pattern: consecutive threads access vectors separated by `dim`
// floats → stride=dim non-coalesced reads (~1/32 cache-line efficiency).
// ─────────────────────────────────────────────────────────────────────
__global__ static void assign_kernel_ref(
        const float* __restrict__ vecs, int n, int dim,
        const float* __restrict__ cA,   const float* __restrict__ cB,
        int* __restrict__ labels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dA = 0.f, dB = 0.f;
    const float* v = vecs + (ptrdiff_t)i * dim;
    for (int d = 0; d < dim; ++d) {
        const float da = v[d] - cA[d];
        const float db = v[d] - cB[d];
        dA += da * da;
        dB += db * db;
    }
    labels[i] = (dB < dA) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────
// accumulate_kernel_ref  (v3's accumulate: warp-shuffle two-level reduction)
// ─────────────────────────────────────────────────────────────────────
// 1 thread/vector: scalar reads, non-coalesced (same stride=dim issue).
// Shared memory layout: sh_sumA[dim] + sh_sumB[dim] + sh_cnt[2]
// Two-level: warp-shuffle (5 shfl ops, register-only) → warp leaders
// atomicAdd to smem → thread 0 flushes smem to global.
// ─────────────────────────────────────────────────────────────────────
__global__ static void accumulate_kernel_ref(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA,  float* __restrict__ sumB,
        int*   __restrict__ cntA,  int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);

    const unsigned FULL = 0xffffffffu;
    const int lane = (int)threadIdx.x & 31;

    // Phase 0: zero shared memory cooperatively
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int lbl = (i < n) ? labels[i] : -1;

    // Phase 1: warp-shuffle fold → warp leaders atomicAdd to shared
    for (int d = 0; d < dim; ++d) {
        const float raw = (i < n) ? vecs[(ptrdiff_t)i * dim + d] : 0.f;

        float valA = (lbl == 0) ? raw : 0.f;
        valA += __shfl_down_sync(FULL, valA, 16);
        valA += __shfl_down_sync(FULL, valA,  8);
        valA += __shfl_down_sync(FULL, valA,  4);
        valA += __shfl_down_sync(FULL, valA,  2);
        valA += __shfl_down_sync(FULL, valA,  1);

        float valB = (lbl == 1) ? raw : 0.f;
        valB += __shfl_down_sync(FULL, valB, 16);
        valB += __shfl_down_sync(FULL, valB,  8);
        valB += __shfl_down_sync(FULL, valB,  4);
        valB += __shfl_down_sync(FULL, valB,  2);
        valB += __shfl_down_sync(FULL, valB,  1);

        if (lane == 0) {
            if (valA != 0.f) atomicAdd(&sh_sumA[d], valA);
            if (valB != 0.f) atomicAdd(&sh_sumB[d], valB);
        }
    }

    // Count reduction via warp shuffle
    int cA_loc = (lbl == 0 && i < n) ? 1 : 0;
    int cB_loc = (lbl == 1 && i < n) ? 1 : 0;
    cA_loc += __shfl_down_sync(FULL, cA_loc, 16);
    cA_loc += __shfl_down_sync(FULL, cA_loc,  8);
    cA_loc += __shfl_down_sync(FULL, cA_loc,  4);
    cA_loc += __shfl_down_sync(FULL, cA_loc,  2);
    cA_loc += __shfl_down_sync(FULL, cA_loc,  1);
    cB_loc += __shfl_down_sync(FULL, cB_loc, 16);
    cB_loc += __shfl_down_sync(FULL, cB_loc,  8);
    cB_loc += __shfl_down_sync(FULL, cB_loc,  4);
    cB_loc += __shfl_down_sync(FULL, cB_loc,  2);
    cB_loc += __shfl_down_sync(FULL, cB_loc,  1);
    if (lane == 0) {
        if (cA_loc > 0) atomicAdd(&sh_cnt[0], cA_loc);
        if (cB_loc > 0) atomicAdd(&sh_cnt[1], cB_loc);
    }

    // Phase 2: thread 0 flushes block totals to global
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

// ─────────────────────────────────────────────────────────────────────
// assign_kernel_v4
// ─────────────────────────────────────────────────────────────────────
// 1 warp (32 threads)/vector. Centroids in shared memory.
// float4 coalesced reads when dim % 128 == 0.
// Warp-shuffle reduction of dA, dB (5 shfl ops, register-only).
// GPU-side convergence: atomicOr(d_changed, 1) on lane 0 if label changed.
// ─────────────────────────────────────────────────────────────────────
__global__ static void assign_kernel_v4(
        const float* __restrict__ vecs,  int n, int dim,
        const float* __restrict__ cA,    const float* __restrict__ cB,
        const int*   __restrict__ prev_labels,
        int*         __restrict__ labels,
        int*         __restrict__ d_changed)
{
    extern __shared__ float sh[];
    float* sh_cA = sh;
    float* sh_cB = sh + dim;

    const int tid         = (int)threadIdx.x;
    const int bdim        = (int)blockDim.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    // Cooperatively load centroids into shared memory
    for (int d = tid; d < dim; d += bdim) {
        sh_cA[d] = cA[d];
        sh_cB[d] = cB[d];
    }
    __syncthreads();

    const int vec_idx = blockIdx.x * ASSIGN_WARPS + warp_in_blk;
    if (vec_idx >= n) return;

    float dA = 0.f, dB = 0.f;

    if (dim % 128 == 0) {
        // float4 path: 4 floats per load, 128-byte aligned per warp step
        const float4* v4  = reinterpret_cast<const float4*>(
                                vecs + (ptrdiff_t)vec_idx * dim);
        const float4* cA4 = reinterpret_cast<const float4*>(sh_cA);
        const float4* cB4 = reinterpret_cast<const float4*>(sh_cB);

        const int ngroups = dim / 128;
        for (int k = 0; k < ngroups; ++k) {
            const int idx4 = k * 32 + lane;
            float4 v   = v4[idx4];
            float4 ca  = cA4[idx4];
            float4 cb  = cB4[idx4];

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
        // Scalar fallback
        const float* v = vecs + (ptrdiff_t)vec_idx * dim;
        for (int d = lane; d < dim; d += 32) {
            const float vd = v[d];
            const float da = vd - sh_cA[d];
            const float db = vd - sh_cB[d];
            dA += da * da;
            dB += db * db;
        }
    }

    // Warp-reduce dA, dB
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

    if (lane == 0) {
        const int lbl = (dB < dA) ? 1 : 0;
        labels[vec_idx] = lbl;
        if (prev_labels[vec_idx] != lbl) {
            atomicOr(d_changed, 1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// accumulate_kernel_v4
// ─────────────────────────────────────────────────────────────────────
// 1 warp (32 threads)/vector. float4 coalesced reads.
// Shared memory layout: sh_sumA[dim] + sh_sumB[dim] + sh_cnt[2]
// All 32 lanes atomicAdd their float4 components to smem.
// Thread 0 flushes block sums to global.
// ─────────────────────────────────────────────────────────────────────
__global__ static void accumulate_kernel_v4(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA,  float* __restrict__ sumB,
        int*   __restrict__ cntA,  int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);

    const int tid         = (int)threadIdx.x;
    const int bdim        = (int)blockDim.x;
    const int lane        = tid & 31;
    const int warp_in_blk = tid >> 5;

    // Phase 0: zero shared memory cooperatively
    for (int d = tid; d < dim; d += bdim) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (tid == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // Phase 1: accumulate into shared
    const int vec_idx = blockIdx.x * ACCUM_WARPS + warp_in_blk;
    if (vec_idx < n) {
        const int lbl    = labels[vec_idx];
        float* sh_target = (lbl == 0) ? sh_sumA : sh_sumB;

        if (dim % 128 == 0) {
            const float4* v4 = reinterpret_cast<const float4*>(
                                    vecs + (ptrdiff_t)vec_idx * dim);
            const int ngroups = dim / 128;
            for (int k = 0; k < ngroups; ++k) {
                const int idx4 = k * 32 + lane;
                float4 chunk   = v4[idx4];
                const int base = k * 128 + lane * 4;
                atomicAdd(&sh_target[base + 0], chunk.x);
                atomicAdd(&sh_target[base + 1], chunk.y);
                atomicAdd(&sh_target[base + 2], chunk.z);
                atomicAdd(&sh_target[base + 3], chunk.w);
            }
        } else {
            const float* v = vecs + (ptrdiff_t)vec_idx * dim;
            for (int d = lane; d < dim; d += 32) {
                atomicAdd(&sh_target[d], v[d]);
            }
        }

        if (lane == 0) {
            atomicAdd(&sh_cnt[lbl], 1);
        }
    }

    // Phase 2: thread 0 flushes block partial sums to global
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

// ─────────────────────────────────────────────────────────────────────
// centroid_update_kernel_v4
// ─────────────────────────────────────────────────────────────────────
// dim threads; each computes cA[d] = sumA[d]/cntA and cB[d] = sumB[d]/cntB.
// Eliminates per-iteration D2H of 2×dim floats and H2D of updated centroids.
// ─────────────────────────────────────────────────────────────────────
__global__ static void centroid_update_kernel_v4(
        const float* __restrict__ sumA,   const float* __restrict__ sumB,
        const int*   __restrict__ d_cntA, const int*   __restrict__ d_cntB,
        float* __restrict__ cA,           float* __restrict__ cB,
        int dim)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    cA[d] = sumA[d] / (float)(*d_cntA);
    cB[d] = sumB[d] / (float)(*d_cntB);
}

// ══════════════════════════════════════════════════════════════════════
// main
// ══════════════════════════════════════════════════════════════════════
int main()
{
    // ── Device info ───────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    // Peak memory bandwidth (GB/s): 2 × memClockRate(Hz) × busWidth(bytes)
    const double peak_bw = 2.0 * (double)prop.memoryClockRate * 1e3
                         * ((double)prop.memoryBusWidth / 8.0) / 1e9;

    phdr("bench_kmeans_v4: v3 (reference) vs v4 (warp-coalesced + on-GPU centroid update)");
    printf("  Device      : %s\n", prop.name);
    printf("  SM count    : %d\n", prop.multiProcessorCount);
    printf("  Peak BW     : %.0f GB/s\n", peak_bw);
    printf("  N vectors   : %d\n", N);
    printf("  DIM         : %d\n", DIM);
    printf("  MAX_ITERS   : %d\n", MAX_ITERS);
    printf("  Warm-up     : %d    Trials: %d\n\n", N_WARMUP, N_TRIALS);

    // ── Host data ─────────────────────────────────────────────────────
    const size_t vec_bytes = (size_t)N   * DIM * sizeof(float);
    const size_t cen_bytes = (size_t)DIM * sizeof(float);
    const size_t lbl_bytes = (size_t)N   * sizeof(int);

    std::vector<float> h_vecs(N * DIM);
    std::vector<float> h_cA(DIM), h_cB(DIM);
    std::vector<int>   h_labels(N, 0);

    srand(0xdeadbeef);
    for (auto& x : h_vecs) x = (float)rand() / (float)RAND_MAX;
    for (auto& x : h_cA)   x = (float)rand() / (float)RAND_MAX;
    for (auto& x : h_cB)   x = (float)rand() / (float)RAND_MAX;

    // ── Device allocations (shared across v3 and v4) ──────────────────
    float *d_vecs, *d_cA, *d_cB, *d_sumA, *d_sumB;
    int   *d_cntA, *d_cntB, *d_labels, *d_prev_labels, *d_changed;

    CUDA_CHECK(cudaMalloc(&d_vecs,       vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_cA,         cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cB,         cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumA,       cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumB,       cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cntA,       sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cntB,       sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_labels,     lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_prev_labels, lbl_bytes));
    CUDA_CHECK(cudaMalloc(&d_changed,    sizeof(int)));

    // ── L2 cache flush infrastructure ─────────────────────────────────
    // Allocated once, reused for every inter-trial flush.
    // Without this, v3 trials warm the cache for v4, and each trial
    // within a section benefits from the previous trial's cache state.
    // d_flush_buf : read-only source; d_flush_out : write sink (prevents DCE)
    float *d_flush_buf, *d_flush_out;
    CUDA_CHECK(cudaMalloc(&d_flush_buf, L2_FLUSH_BYTES));
    CUDA_CHECK(cudaMalloc(&d_flush_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_flush_buf, 0, L2_FLUSH_BYTES));
    CUDA_CHECK(cudaMemset(d_flush_out, 0, sizeof(float)));

    // flush_l2(): evicts all L2 contents, synchronises the device.
    // Call before every individual trial and between sections.
    auto flush_l2 = [&]() {
        CUDA_CHECK(cudaMemset(d_flush_out, 0, sizeof(float)));
        flush_l2_kernel<<<L2_FLUSH_BLOCKS, L2_FLUSH_THREADS>>>(
            d_flush_buf, L2_FLUSH_FLOATS, d_flush_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Upload vector data
    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(), vec_bytes, cudaMemcpyHostToDevice));

    // ── Theoretical memory traffic analysis ───────────────────────────
    phdr("Theoretical memory traffic analysis");

    // Per-iteration kernel reads/writes
    const double assign_vec_bytes  = (double)N * DIM * sizeof(float);   // 16MB
    const double assign_lbl_write  = (double)N * sizeof(int);            // 16KB
    const double accum_vec_bytes   = (double)N * DIM * sizeof(float);   // 16MB
    const double accum_lbl_read    = (double)N * sizeof(int);            // 16KB
    const double accum_sum_write   = 2.0 * DIM * sizeof(float);          // 8KB
    const double cen_bytes_d       = (double)DIM * sizeof(float);        // 4KB

    printf("  Per-iteration kernel memory traffic:\n");
    printf("    assign reads  : %.1f MB (vectors) + 2×dim×4 = %.1f KB (centroids via smem, free)\n",
           assign_vec_bytes / 1e6, 2.0 * DIM * 4.0 / 1024.0);
    printf("    assign writes : %.1f KB (labels)\n",
           assign_lbl_write / 1024.0);
    printf("    accum reads   : %.1f MB (vectors) + %.1f KB (labels)\n",
           accum_vec_bytes / 1e6, accum_lbl_read / 1024.0);
    printf("    accum writes  : %.1f KB (sums to global)\n\n",
           accum_sum_write / 1024.0);

    printf("  Per-iteration PCIe (host-device) transfers:\n");
    printf("    v3: D2H %d labels  = %.1f KB  (convergence check)\n",
           N, (double)N * 4.0 / 1024.0);
    printf("    v3: D2H sumA+sumB+cntA+cntB = %.1f KB\n",
           (2.0 * DIM * 4.0 + 8.0) / 1024.0);
    printf("    v3: H2D cA+cB (updated cens) = %.1f KB\n",
           (2.0 * cen_bytes_d) / 1024.0);
    printf("    v3 total per iteration        ≈ %.1f KB + 2 PCIe syncs\n\n",
           ((double)N * 4.0 + 2.0 * DIM * 4.0 + 8.0 + 2.0 * cen_bytes_d) / 1024.0);

    printf("    v4: D2H d_changed             = 4 bytes (convergence flag)\n");
    printf("    v4: D2H cntA+cntB             = 8 bytes (degenerate check)\n");
    printf("    v4 total per iteration        = 12 bytes + 1 PCIe sync\n\n");

    printf("  Coalescing efficiency:\n");
    printf("    v3 assign: 1 thread/vector → stride=%d between warp threads\n", DIM);
    printf("               → 1 useful float per 32-float cache-line transaction (~1/32 efficiency)\n");
    printf("    v4 assign: 1 warp/vector → 32 consecutive floats per warp step\n");
    printf("               → 128-byte aligned transactions at full efficiency\n");
    printf("    v4 float4: 4 floats per load → 4× instruction throughput\n\n");

    // ── Kernel launch parameters ──────────────────────────────────────
    // v3 (reference): 1 thread/vector, BLOCK=256
    const int v3_block      = 256;
    const int v3_grid       = (N + v3_block - 1) / v3_block;
    const size_t smem_v3    = (size_t)2 * DIM * sizeof(float) + 2 * sizeof(int);

    // v4: 1 warp/vector
    const int v4_assign_block  = ASSIGN_WARPS * 32;               // 128
    const int v4_assign_grid   = (N + ASSIGN_WARPS - 1) / ASSIGN_WARPS;
    const size_t smem_v4_assign = (size_t)2 * DIM * sizeof(float); // centroids only

    const int v4_accum_block   = ACCUM_WARPS * 32;                // 256
    const int v4_accum_grid    = (N + ACCUM_WARPS - 1) / ACCUM_WARPS;
    const size_t smem_v4_accum = (size_t)2 * DIM * sizeof(float)
                               + (size_t)2 * sizeof(int);

    const int v4_update_block  = 256;
    const int v4_update_grid   = (DIM + v4_update_block - 1) / v4_update_block;

    printf("  Launch parameters:\n");
    printf("    v3 assign/accum : grid=%d  block=%d\n", v3_grid, v3_block);
    printf("    v4 assign       : grid=%d  block=%d  (ASSIGN_WARPS=%d)\n",
           v4_assign_grid, v4_assign_block, ASSIGN_WARPS);
    printf("    v4 accum        : grid=%d  block=%d  (ACCUM_WARPS=%d)\n",
           v4_accum_grid, v4_accum_block, ACCUM_WARPS);
    printf("    v4 cen_update   : grid=%d  block=%d\n\n",
           v4_update_grid, v4_update_block);

    // ── Section A: warm-up ────────────────────────────────────────────
    phdr("Section A — Warm-up + stable label generation");

    // Upload initial centroids
    CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));

    // Warm up the assign_kernel_ref (v3) to generate stable labels
    for (int w = 0; w < N_WARMUP; ++w)
        assign_kernel_ref<<<v3_grid, v3_block>>>(
            d_vecs, N, DIM, d_cA, d_cB, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Generate stable labels for use throughout the benchmark
    assign_kernel_ref<<<v3_grid, v3_block>>>(
        d_vecs, N, DIM, d_cA, d_cB, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels, lbl_bytes, cudaMemcpyDeviceToHost));

    // Init v4 prev_labels to -1 (all bytes 0xFF → int value -1)
    CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));

    printf("  Warm-up complete. Labels generated.\n");
    {
        int cnt1 = 0;
        for (int x : h_labels) cnt1 += x;
        printf("  Label distribution: side A=%d  side B=%d\n\n", N - cnt1, cnt1);
    }

    // ── CSV file ──────────────────────────────────────────────────────
    std::ofstream csv("bench_v4_results.csv");
    csv << "step,version,trial,ms,eff_bw_gbs\n";

    // Helper to compute effective BW from bytes and ms
    auto eff_bw = [](double bytes, float ms) -> double {
        return bytes / (ms * 1e-3) / 1e9;
    };

    // Timing accumulator storage
    std::vector<float> t_assign_v3(N_TRIALS), t_d2h_labels_v3(N_TRIALS);
    std::vector<float> t_accum_v3(N_TRIALS),  t_d2h_sums_v3(N_TRIALS);
    std::vector<float> t_h2d_cen_v3(N_TRIALS);

    std::vector<float> t_assign_v4(N_TRIALS),   t_d2h_changed_v4(N_TRIALS);
    std::vector<float> t_accum_v4(N_TRIALS),     t_d2h_counts_v4(N_TRIALS);
    std::vector<float> t_cen_update_v4(N_TRIALS);

    // ── Section B: Profile v3 iteration sub-steps ─────────────────────
    phdr("Section B — Profiling v3 iteration sub-steps");

    CUDA_CHECK(cudaDeviceSynchronize());

    {
        GpuTimer ta, td2h_l, tac, td2h_s, th2d_c;
        std::vector<float> h_sumA(DIM), h_sumB(DIM);
        int h_cntA_v3 = 0, h_cntB_v3 = 0;

        // Warm-up
        for (int w = 0; w < N_WARMUP; ++w) {
            assign_kernel_ref<<<v3_grid, v3_block>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_labels);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                                  lbl_bytes, cudaMemcpyDeviceToHost));
        }

        for (int r = 0; r < N_TRIALS; ++r) {
            // Flush L2 before each trial so measurements reflect a cold
            // cache.  Without this, trial r benefits from trial r-1's
            // warm L2, making early and late trials incomparable.
            flush_l2();

            // ── t_assign_v3 ───────────────────────────────────────────
            ta.start();
            assign_kernel_ref<<<v3_grid, v3_block>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_labels);
            t_assign_v3[r] = ta.stop();

            // ── t_d2h_labels_v3 ───────────────────────────────────────
            td2h_l.start();
            CUDA_CHECK(cudaMemcpy(h_labels.data(), d_labels,
                                  lbl_bytes, cudaMemcpyDeviceToHost));
            t_d2h_labels_v3[r] = td2h_l.stop();

            // Reset accumulators
            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            {   int z = 0;
                CUDA_CHECK(cudaMemcpy(d_cntA, &z, sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_cntB, &z, sizeof(int), cudaMemcpyHostToDevice)); }

            // ── t_accum_v3 ────────────────────────────────────────────
            tac.start();
            accumulate_kernel_ref<<<v3_grid, v3_block, smem_v3>>>(
                d_vecs, N, DIM, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);
            t_accum_v3[r] = tac.stop();

            // ── t_d2h_sums_v3 ─────────────────────────────────────────
            td2h_s.start();
            CUDA_CHECK(cudaMemcpy(h_sumA.data(), d_sumA, cen_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_sumB.data(), d_sumB, cen_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_cntA_v3, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_cntB_v3, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            t_d2h_sums_v3[r] = td2h_s.stop();

            // CPU centroid update (v3 does this on host)
            if (h_cntA_v3 > 0 && h_cntB_v3 > 0) {
                for (int d = 0; d < DIM; ++d) {
                    h_cA[d] = h_sumA[d] / (float)h_cntA_v3;
                    h_cB[d] = h_sumB[d] / (float)h_cntB_v3;
                }
            }

            // ── t_h2d_cen_v3 ──────────────────────────────────────────
            th2d_c.start();
            CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));
            t_h2d_cen_v3[r] = th2d_c.stop();

            // CSV
            csv << "assign_v3,v3," << r << "," << t_assign_v3[r] << ","
                << eff_bw(assign_vec_bytes + assign_lbl_write, t_assign_v3[r]) << "\n";
            csv << "d2h_labels_v3,v3," << r << "," << t_d2h_labels_v3[r] << ","
                << eff_bw(lbl_bytes, t_d2h_labels_v3[r]) << "\n";
            csv << "accum_v3,v3," << r << "," << t_accum_v3[r] << ","
                << eff_bw(accum_vec_bytes + accum_lbl_read, t_accum_v3[r]) << "\n";
            csv << "d2h_sums_v3,v3," << r << "," << t_d2h_sums_v3[r] << ","
                << eff_bw(2.0 * cen_bytes + 2.0 * sizeof(int), t_d2h_sums_v3[r]) << "\n";
            csv << "h2d_cen_v3,v3," << r << "," << t_h2d_cen_v3[r] << ","
                << eff_bw(2.0 * cen_bytes, t_h2d_cen_v3[r]) << "\n";
        }
    }

    // Print v3 sub-step summary
    // Statistical helpers — median is the headline (outlier-resistant without
    // needing more trials); stddev lets you judge if 10 trials are sufficient.
    auto avg_vec = [](const std::vector<float>& v) {
        return std::accumulate(v.begin(), v.end(), 0.f) / (float)v.size();
    };
    auto min_vec = [](const std::vector<float>& v) {
        return *std::min_element(v.begin(), v.end());
    };
    auto median_vec = [](std::vector<float> v) {            // copy, then sort
        std::sort(v.begin(), v.end());
        const int n = (int)v.size();
        return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
    };
    auto stddev_vec = [&avg_vec](const std::vector<float>& v) {
        float m = avg_vec(v);
        float s = 0.f;
        for (float x : v) s += (x - m) * (x - m);
        return std::sqrt(s / (float)v.size());
    };

    float v3_assign_avg    = avg_vec(t_assign_v3);
    float v3_d2h_lbl_avg   = avg_vec(t_d2h_labels_v3);
    float v3_accum_avg     = avg_vec(t_accum_v3);
    float v3_d2h_sum_avg   = avg_vec(t_d2h_sums_v3);
    float v3_h2d_cen_avg   = avg_vec(t_h2d_cen_v3);
    float v3_total_avg     = v3_assign_avg + v3_d2h_lbl_avg + v3_accum_avg
                           + v3_d2h_sum_avg + v3_h2d_cen_avg;
    float v3_assign_med    = median_vec(t_assign_v3);
    float v3_d2h_lbl_med   = median_vec(t_d2h_labels_v3);
    float v3_accum_med     = median_vec(t_accum_v3);
    float v3_d2h_sum_med   = median_vec(t_d2h_sums_v3);
    float v3_h2d_cen_med   = median_vec(t_h2d_cen_v3);
    float v3_total_med     = v3_assign_med + v3_d2h_lbl_med + v3_accum_med
                           + v3_d2h_sum_med + v3_h2d_cen_med;

    // Column header: median is the headline number, ±stddev shows stability,
    // min exposes the true GPU floor (no OS jitter).
    printf("  v3 sub-step timing (%d trials) — headline = median:\n\n", N_TRIALS);
    printf("  %-22s %8s %8s %7s %7s %7s %12s\n",
           "step", "median", "avg", "±sd", "min", "pct", "eff_BW_GBs");
    print_sep('-', 82);
    auto pct = [&](float ms) { return 100.f * ms / v3_total_med; };
#define V3ROW(label, vec, bw_bytes, extra) \
    printf("  %-22s %8.3f %8.3f %7.3f %7.3f %6.1f%% %12.2f" extra "\n", \
           (label),                                                        \
           median_vec(vec), avg_vec(vec), stddev_vec(vec), min_vec(vec),  \
           pct(median_vec(vec)),                                           \
           eff_bw((bw_bytes), median_vec(vec)))
    V3ROW("assign_kernel_ref",  t_assign_v3,     assign_vec_bytes + assign_lbl_write, "");
    printf("  %-22s %8.3f %8.3f %7.3f %7.3f %6.1f%% %12.2f"
           "   D2H %d labels = %.1f KB\n",
           "d2h_labels",
           v3_d2h_lbl_med, v3_d2h_lbl_avg, stddev_vec(t_d2h_labels_v3),
           min_vec(t_d2h_labels_v3), pct(v3_d2h_lbl_med),
           eff_bw(lbl_bytes, v3_d2h_lbl_med), N, lbl_bytes / 1024.0);
    V3ROW("accumulate_kernel_ref", t_accum_v3,   accum_vec_bytes + accum_lbl_read, "");
    printf("  %-22s %8.3f %8.3f %7.3f %7.3f %6.1f%% %12.2f"
           "   D2H sums+counts = %.1f KB\n",
           "d2h_sums+counts",
           v3_d2h_sum_med, v3_d2h_sum_avg, stddev_vec(t_d2h_sums_v3),
           min_vec(t_d2h_sums_v3), pct(v3_d2h_sum_med),
           eff_bw(2.0 * cen_bytes + 2.0 * sizeof(int), v3_d2h_sum_med),
           (2.0 * cen_bytes + 2.0 * sizeof(int)) / 1024.0);
    printf("  %-22s %8.3f %8.3f %7.3f %7.3f %6.1f%% %12.2f"
           "   H2D 2×centroids = %.1f KB\n",
           "h2d_centroids",
           v3_h2d_cen_med, v3_h2d_cen_avg, stddev_vec(t_h2d_cen_v3),
           min_vec(t_h2d_cen_v3), pct(v3_h2d_cen_med),
           eff_bw(2.0 * cen_bytes, v3_h2d_cen_med),
           2.0 * cen_bytes / 1024.0);
#undef V3ROW
    print_sep('-', 82);
    printf("  %-22s %8.3f %8.3f %7s %7s %6.1f%%\n\n",
           "TOTAL", v3_total_med, v3_total_avg, "-", "-", 100.f);

    // ── Inter-section L2 flush ────────────────────────────────────────
    // Evict all of v3's footprint before starting v4 measurements.
    // Without this, v4's first trial would see v3's warm L2 and appear
    // faster than it really is for a cold-cache workload.
    flush_l2();

    // ── Section C: Profile v4 iteration sub-steps ─────────────────────
    phdr("Section C — Profiling v4 iteration sub-steps");

    // Re-upload centroids to ensure clean state
    CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));

    CUDA_CHECK(cudaDeviceSynchronize());

    {
        GpuTimer ta, td2h_c, tac, td2h_cnt, tupd;
        int h_changed = 0;
        int h_cntA_v4 = 0, h_cntB_v4 = 0;

        // Warm-up
        for (int w = 0; w < N_WARMUP; ++w) {
            int z = 0;
            CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));
            assign_kernel_v4<<<v4_assign_grid, v4_assign_block, smem_v4_assign>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_prev_labels, d_labels, d_changed);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice));
        }

        for (int r = 0; r < N_TRIALS; ++r) {
            flush_l2();

            // Reset d_changed
            {   int z = 0;
                CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice)); }

            // ── t_assign_v4 ───────────────────────────────────────────
            ta.start();
            assign_kernel_v4<<<v4_assign_grid, v4_assign_block, smem_v4_assign>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_prev_labels, d_labels, d_changed);
            t_assign_v4[r] = ta.stop();

            // ── t_d2h_changed_v4 ──────────────────────────────────────
            td2h_c.start();
            CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            t_d2h_changed_v4[r] = td2h_c.stop();

            // Update prev_labels (D2D, not timed separately — overhead negligible)
            CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes, cudaMemcpyDeviceToDevice));

            // Reset accumulators
            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            {   int z = 0;
                CUDA_CHECK(cudaMemcpy(d_cntA, &z, sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_cntB, &z, sizeof(int), cudaMemcpyHostToDevice)); }

            // ── t_accum_v4 ────────────────────────────────────────────
            tac.start();
            accumulate_kernel_v4<<<v4_accum_grid, v4_accum_block, smem_v4_accum>>>(
                d_vecs, N, DIM, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);
            t_accum_v4[r] = tac.stop();

            // ── t_d2h_counts_v4 ───────────────────────────────────────
            td2h_cnt.start();
            CUDA_CHECK(cudaMemcpy(&h_cntA_v4, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&h_cntB_v4, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            t_d2h_counts_v4[r] = td2h_cnt.stop();

            // ── t_centroid_update_v4 ──────────────────────────────────
            tupd.start();
            centroid_update_kernel_v4<<<v4_update_grid, v4_update_block>>>(
                d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, DIM);
            t_cen_update_v4[r] = tupd.stop();

            // CSV
            csv << "assign_v4,v4," << r << "," << t_assign_v4[r] << ","
                << eff_bw(assign_vec_bytes + assign_lbl_write, t_assign_v4[r]) << "\n";
            csv << "d2h_changed_v4,v4," << r << "," << t_d2h_changed_v4[r] << ","
                << eff_bw(sizeof(int), t_d2h_changed_v4[r]) << "\n";
            csv << "accum_v4,v4," << r << "," << t_accum_v4[r] << ","
                << eff_bw(accum_vec_bytes + accum_lbl_read, t_accum_v4[r]) << "\n";
            csv << "d2h_counts_v4,v4," << r << "," << t_d2h_counts_v4[r] << ","
                << eff_bw(2.0 * sizeof(int), t_d2h_counts_v4[r]) << "\n";
            csv << "cen_update_v4,v4," << r << "," << t_cen_update_v4[r] << ","
                << eff_bw(2.0 * cen_bytes + 2.0 * cen_bytes, t_cen_update_v4[r]) << "\n";
        }
    }

    // Print v4 sub-step summary
    float v4_assign_avg    = avg_vec(t_assign_v4);
    float v4_d2h_chg_avg   = avg_vec(t_d2h_changed_v4);
    float v4_accum_avg     = avg_vec(t_accum_v4);
    float v4_d2h_cnt_avg   = avg_vec(t_d2h_counts_v4);
    float v4_upd_avg       = avg_vec(t_cen_update_v4);
    float v4_assign_med    = median_vec(t_assign_v4);
    float v4_d2h_chg_med   = median_vec(t_d2h_changed_v4);
    float v4_accum_med     = median_vec(t_accum_v4);
    float v4_d2h_cnt_med   = median_vec(t_d2h_counts_v4);
    float v4_upd_med       = median_vec(t_cen_update_v4);
    float v4_total_avg     = v4_assign_avg + v4_d2h_chg_avg + v4_accum_avg
                           + v4_d2h_cnt_avg + v4_upd_avg;
    float v4_total_med     = v4_assign_med + v4_d2h_chg_med + v4_accum_med
                           + v4_d2h_cnt_med + v4_upd_med;

    printf("  v4 sub-step timing (%d trials) — headline = median:\n\n", N_TRIALS);
    printf("  %-22s %8s %8s %7s %7s %7s %12s\n",
           "step", "median", "avg", "±sd", "min", "pct", "eff_BW_GBs");
    print_sep('-', 82);
    auto pct4 = [&](float ms) { return 100.f * ms / v4_total_med; };
#define V4ROW(label, vec, bw_bytes, extra) \
    printf("  %-22s %8.3f %8.3f %7.3f %7.3f %6.1f%% %12.2f" extra "\n", \
           (label),                                                        \
           median_vec(vec), avg_vec(vec), stddev_vec(vec), min_vec(vec),  \
           pct4(median_vec(vec)),                                          \
           eff_bw((bw_bytes), median_vec(vec)))
    V4ROW("assign_kernel_v4",    t_assign_v4,      assign_vec_bytes + assign_lbl_write, "");
    V4ROW("d2h_changed",         t_d2h_changed_v4, (double)sizeof(int),
          "   D2H 4 bytes = changed flag");
    V4ROW("accumulate_kernel_v4",t_accum_v4,       accum_vec_bytes + accum_lbl_read, "");
    V4ROW("d2h_counts",          t_d2h_counts_v4,  2.0 * sizeof(int),
          "   D2H 8 bytes = cntA+cntB");
    V4ROW("centroid_update_v4",  t_cen_update_v4,  2.0 * cen_bytes + 2.0 * cen_bytes,
          "   on-GPU: sum/cnt per dim");
#undef V4ROW
    print_sep('-', 82);
    printf("  %-22s %8.3f %8.3f %7s %7s %6.1f%%\n\n",
           "TOTAL", v4_total_med, v4_total_avg, "-", "-", 100.f);

    // ── Section D: Full k-means run comparison ────────────────────────
    // Run each version N_FULL_TRIALS times; report median total time.
    // A single trial is too noisy for a credible speedup claim.
    static constexpr int N_FULL_TRIALS = 5;
    phdr("Section D — Full k-means loop comparison (median of 5 full runs)");

    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Full v3 loop ──────────────────────────────────────────────────
    float v3_full_ms = 0.f;
    int   v3_iters   = 0;
    {
        std::vector<float> trial_ms(N_FULL_TRIALS);

        for (int t = 0; t < N_FULL_TRIALS; ++t) {
        flush_l2();
        CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));

        std::vector<float> h_sumA(DIM), h_sumB(DIM);
        std::vector<int>   prev_lbl(N, -1), cur_lbl(N, 0);
        v3_iters = 0;   // reset per trial (same data → same count each run)

        GpuTimer tloop;
        tloop.start();

        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            assign_kernel_ref<<<v3_grid, v3_block>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_labels);
            CUDA_CHECK(cudaMemcpy(cur_lbl.data(), d_labels, lbl_bytes,
                                  cudaMemcpyDeviceToHost));
            ++v3_iters;

            bool changed = (iter == 0) || (cur_lbl != prev_lbl);
            prev_lbl = cur_lbl;
            if (!changed) break;

            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            {   int z = 0;
                CUDA_CHECK(cudaMemcpy(d_cntA, &z, sizeof(int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_cntB, &z, sizeof(int), cudaMemcpyHostToDevice)); }

            accumulate_kernel_ref<<<v3_grid, v3_block, smem_v3>>>(
                d_vecs, N, DIM, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);

            int cntA_h = 0, cntB_h = 0;
            CUDA_CHECK(cudaMemcpy(h_sumA.data(), d_sumA, cen_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_sumB.data(), d_sumB, cen_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            if (cntA_h == 0 || cntB_h == 0) break;

            for (int d = 0; d < DIM; ++d) {
                h_cA[d] = h_sumA[d] / (float)cntA_h;
                h_cB[d] = h_sumB[d] / (float)cntB_h;
            }
            CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));
        }

        trial_ms[t] = tloop.stop();
        } // end N_FULL_TRIALS

        v3_full_ms = median_vec(trial_ms);
        // v3_iters holds the last trial's count (deterministic: same data each run)
    }

    // ── Full v4 loop ──────────────────────────────────────────────────
    float v4_full_ms = 0.f;
    int   v4_iters   = 0;
    {
        std::vector<float> trial_ms(N_FULL_TRIALS);

        for (int t = 0; t < N_FULL_TRIALS; ++t) {
        flush_l2();
        // Reset centroids and prev_labels
        CUDA_CHECK(cudaMemcpy(d_cA, h_cA.data(), cen_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cB, h_cB.data(), cen_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_prev_labels, 0xff, lbl_bytes));
        v4_iters = 0;

        GpuTimer tloop;
        tloop.start();

        for (int iter = 0; iter < MAX_ITERS; ++iter) {
            int z = 0;
            CUDA_CHECK(cudaMemcpy(d_changed, &z, sizeof(int), cudaMemcpyHostToDevice));

            assign_kernel_v4<<<v4_assign_grid, v4_assign_block, smem_v4_assign>>>(
                d_vecs, N, DIM, d_cA, d_cB, d_prev_labels, d_labels, d_changed);

            int changed = 0;
            CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
            ++v4_iters;

            CUDA_CHECK(cudaMemcpy(d_prev_labels, d_labels, lbl_bytes,
                                  cudaMemcpyDeviceToDevice));

            if (!changed && iter > 0) break;

            CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
            CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
            CUDA_CHECK(cudaMemcpy(d_cntA, &z, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_cntB, &z, sizeof(int), cudaMemcpyHostToDevice));

            accumulate_kernel_v4<<<v4_accum_grid, v4_accum_block, smem_v4_accum>>>(
                d_vecs, N, DIM, d_labels,
                d_sumA, d_sumB, d_cntA, d_cntB);

            int cntA_h = 0, cntB_h = 0;
            CUDA_CHECK(cudaMemcpy(&cntA_h, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&cntB_h, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
            if (cntA_h == 0 || cntB_h == 0) break;

            centroid_update_kernel_v4<<<v4_update_grid, v4_update_block>>>(
                d_sumA, d_sumB, d_cntA, d_cntB, d_cA, d_cB, DIM);
        }

        trial_ms[t] = tloop.stop();
        } // end N_FULL_TRIALS

        v4_full_ms = median_vec(trial_ms);
    }

    printf("  Full k-means loop results (median of %d runs):\n\n", N_FULL_TRIALS);
    printf("  %-12s %10s %8s %12s\n", "version", "median_ms", "iters", "ms/iter");
    print_sep('-', 50);
    printf("  %-12s %10.3f %8d %12.3f\n",
           "v3 (ref)", v3_full_ms, v3_iters, v3_full_ms / (float)v3_iters);
    printf("  %-12s %10.3f %8d %12.3f\n",
           "v4",       v4_full_ms, v4_iters, v4_full_ms / (float)v4_iters);
    print_sep('-', 50);
    printf("  Speedup (v4 vs v3): %.2fx\n\n",
           v3_full_ms / v4_full_ms);

    csv << "full_loop,v3,0," << v3_full_ms << ",0\n";
    csv << "full_loop,v4,0," << v4_full_ms << ",0\n";

    // ── Section E: Memory bandwidth analysis ──────────────────────────
    phdr("Section E — Memory bandwidth analysis");

    // Bytes read by assign kernel (vectors only; centroids via smem)
    const double assign_read_bytes = assign_vec_bytes;  // n × dim × 4
    // Bytes read by accumulate kernel (vectors + labels)
    const double accum_read_bytes  = accum_vec_bytes + accum_lbl_read;

    printf("  Memory traffic per kernel call:\n");
    printf("    assign: %.1f MB vector reads  + %.1f KB label writes\n",
           assign_read_bytes / 1e6, assign_lbl_write / 1024.0);
    printf("    accum : %.1f MB vector reads  + %.1f KB label reads\n",
           accum_vec_bytes / 1e6, accum_lbl_read / 1024.0);
    printf("            + %.1f KB sum writes (smem → global)\n\n",
           accum_sum_write / 1024.0);

    printf("  Effective bandwidth (median of %d trials):\n\n", N_TRIALS);
    printf("  %-28s %10s %7s %12s %10s\n",
           "kernel", "median_ms", "±sd_ms", "eff_BW_GBs", "pct_peak");
    print_sep('-', 76);

    auto print_bw_row = [&](const char* name, float ms, float sd, double bytes) {
        double bw  = eff_bw(bytes, ms);
        double pct = 100.0 * bw / peak_bw;
        printf("  %-28s %10.3f %7.3f %12.2f %9.1f%%\n", name, ms, sd, bw, pct);
    };

    print_bw_row("assign_kernel_ref (v3)",  v3_assign_med,  stddev_vec(t_assign_v3),
                 assign_read_bytes + assign_lbl_write);
    print_bw_row("assign_kernel_v4 (v4)",   v4_assign_med,  stddev_vec(t_assign_v4),
                 assign_read_bytes + assign_lbl_write);
    printf("\n");
    print_bw_row("accumulate_kernel_ref (v3)", v3_accum_med, stddev_vec(t_accum_v3), accum_read_bytes);
    print_bw_row("accumulate_kernel_v4 (v4)",  v4_accum_med, stddev_vec(t_accum_v4), accum_read_bytes);
    printf("\n");
    printf("  Peak device memory bandwidth: %.0f GB/s\n\n", peak_bw);

    // ── Per-step speedup summary ──────────────────────────────────────
    phdr("Per-step speedup summary: v4 vs v3");

    printf("  %-26s %10s %10s %10s\n", "step", "v3_ms", "v4_ms", "speedup");
    print_sep('-', 62);

    auto speedup_row = [&](const char* name, float ms3, float ms4) {
        printf("  %-26s %10.3f %10.3f %9.2fx\n",
               name, ms3, ms4, ms3 / ms4);
    };

    speedup_row("assign kernel",          v3_assign_med,  v4_assign_med);
    speedup_row("D2H labels/changed",     v3_d2h_lbl_med, v4_d2h_chg_med);
    speedup_row("accumulate kernel",      v3_accum_med,   v4_accum_med);
    speedup_row("D2H sums/counts",        v3_d2h_sum_med, v4_d2h_cnt_med);
    speedup_row("centroid update (CPU/GPU)", v3_h2d_cen_med, v4_upd_med);
    print_sep('-', 62);
    speedup_row("TOTAL per iteration",    v3_total_med,   v4_total_med);
    printf("\n");

    csv.close();
    printf("  Results saved to: bench_v4_results.csv\n\n");

    // ── Build / run / profile instructions ───────────────────────────
    print_sep('=');
    printf("  Build:\n");
    printf("    nvcc -O3 -DHAVE_CUDA --gpu-architecture=sm_86 \\\n");
    printf("         -I AgentMemory/M3/include \\\n");
    printf("         test/bench_kmeans_v4.cu -o bench_kmeans_v4\n");
    printf("  Run:\n");
    printf("    ./bench_kmeans_v4\n");
    printf("  Profile with nvprof:\n");
    printf("    nvprof --print-gpu-trace ./bench_kmeans_v4\n");
    printf("  Profile with Nsight:\n");
    printf("    ncu --set full --target-processes all -o kmeans_v4_profile ./bench_kmeans_v4\n");
    print_sep('=');

    // ── Cleanup ───────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_vecs));
    CUDA_CHECK(cudaFree(d_flush_buf));  CUDA_CHECK(cudaFree(d_flush_out));
    CUDA_CHECK(cudaFree(d_cA));         CUDA_CHECK(cudaFree(d_cB));
    CUDA_CHECK(cudaFree(d_sumA));       CUDA_CHECK(cudaFree(d_sumB));
    CUDA_CHECK(cudaFree(d_cntA));       CUDA_CHECK(cudaFree(d_cntB));
    CUDA_CHECK(cudaFree(d_labels));     CUDA_CHECK(cudaFree(d_prev_labels));
    CUDA_CHECK(cudaFree(d_changed));

    return 0;
}

#else
#error "Compile with nvcc -DHAVE_CUDA"
#endif
