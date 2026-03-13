// bench_kernels_full.cu
//
// Comprehensive comparison: v1 / v2 (block-size sweep) / v3 (warp-shuffle) / v4 (scatter+cuBLAS)
//
// All four implementations replace only the ACCUMULATE phase of 2-centroid k-means.
// The assign_kernel is identical in every version and timed separately as a baseline.
//
// Compile (from repo root):
//   nvcc -O3 -DHAVE_CUDA --gpu-architecture=sm_86                           \
//        -I AgentMemory/M3/include                                          \
//        test/bench_kernels_full.cu -lcublas -o bench_kernels_full
//   ./bench_kernels_full
//   python3 test/plot_bench_results.py   # produces bench_plots/*.png
//
// ═══════════════════════════════════════════════════════════════════════════
// Kernel taxonomy
// ───────────────
//  v1  accumulate_v1  — one global atomicAdd per (thread × dim).
//                       n × dim global atomics, all hitting same 2*dim addrs.
//                       L2 serialises → throughput collapses at large n.
//
//  v2  accumulate_v2  — phase 1: BLOCK threads → shared atomicAdd (~5-20 cyc).
//                       phase 2: thread 0 per block → global atomicAdd.
//                       Global atomics: ceil(n/BLOCK) × dim  (BLOCK× fewer).
//                       Swept over BLOCK = 128, 256, 512, 1024.
//
//  v3  accumulate_v3  — phase 1: warp-level __shfl_down_sync reduction (reg→reg,
//                       0 memory). One lane-0 shared atomicAdd per (warp × dim).
//                       32× fewer shared atomics than v2 for same BLOCK.
//                       phase 2: thread 0 per block → global atomicAdd (same
//                       global traffic as v2).
//
//  v4  scatter+gemv   — avoids atomics entirely.
//                       step 1: scatter_positions (atomic grab of output slot)
//                       step 2: scatter_data (copy vector to partitioned buffer)
//                       step 3: cuBLAS sgemv (column-sum via dense matrix-vector)
//                       Higher memory traffic (~3×) but zero atomic contention.
// ═══════════════════════════════════════════════════════════════════════════

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>
#include <string>
#include <fstream>

// ─────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────
static constexpr int   N_VECS    = 65536;   // enough to saturate global atomics
static constexpr int   DIM       = 1024;    // matches IVF.index dimension
static constexpr int   N_WARMUP  = 5;
static constexpr int   N_TRIALS  = 15;
static const int V2_BLOCKS[]   = {128, 256, 512, 1024};
static constexpr int V3_BLOCK  = 256;
static constexpr int V4_BLOCK  = 256;

// ─────────────────────────────────────────────────────────────────────────
// Error checking
// ─────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(c) do {                                                  \
    cudaError_t _e = (c);                                                   \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d — %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1); }                                                          \
} while(0)

#define CUBLAS_CHECK(c) do {                                                \
    cublasStatus_t _s = (c);                                                \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
        fprintf(stderr, "cuBLAS error %s:%d — status %d\n",                \
                __FILE__, __LINE__, (int)_s);                               \
        exit(1); }                                                          \
} while(0)

// ─────────────────────────────────────────────────────────────────────────
// GPU timer using CUDA events
// ─────────────────────────────────────────────────────────────────────────
struct GpuTimer {
    cudaEvent_t s, e;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&s)); CUDA_CHECK(cudaEventCreate(&e)); }
    ~GpuTimer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start() { CUDA_CHECK(cudaEventRecord(s)); }
    float stop()  {                          // returns milliseconds
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Shared helper: reset accumulator arrays
// ─────────────────────────────────────────────────────────────────────────
static void reset_accumulators(float* d_sumA, float* d_sumB,
                                int* d_cntA, int* d_cntB, size_t cen_bytes)
{
    CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
    CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_cntA, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cntB, &zero, sizeof(int), cudaMemcpyHostToDevice));
}

// ═════════════════════════════════════════════════════════════════════════
// KERNEL DEFINITIONS
// ═════════════════════════════════════════════════════════════════════════

// ─── assign_kernel (identical across all versions) ───────────────────────
// Each thread computes L2 distance to two centroids and writes a 0/1 label.
// No atomics; every write is to a distinct address → no contention.
// Global reads: n × dim (vectors) + 2 × dim (centroids)
// Global writes: n (labels)
__global__ static void assign_kernel(
        const float* __restrict__ vecs, int n, int dim,
        const float* __restrict__ cA,   const float* __restrict__ cB,
        int* __restrict__ labels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dA = 0.f, dB = 0.f;
    const float* v = vecs + (ptrdiff_t)i * dim;
    for (int d = 0; d < dim; ++d) {
        float da = v[d] - cA[d], db = v[d] - cB[d];
        dA += da * da;  dB += db * db;
    }
    labels[i] = (dB < dA) ? 1 : 0;
}

// ─── v1: every thread → global atomicAdd per dim ─────────────────────────
// BOTTLENECK: n × dim global atomics all targeting the same 2×dim addresses.
// At n=65536, dim=1024: 67M global atomicAdds per call.
// Global L2 serialises them → throughput ≪ peak.
// Global reads for accum: n × dim (vecs) + n (labels)
__global__ static void accumulate_v1(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA, float* __restrict__ sumB,
        int*   __restrict__ cntA, int*   __restrict__ cntB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* v = vecs + (ptrdiff_t)i * dim;
    if (labels[i] == 0) {
        for (int d = 0; d < dim; ++d) atomicAdd(&sumA[d], v[d]);  // global atomic
        atomicAdd(cntA, 1);
    } else {
        for (int d = 0; d < dim; ++d) atomicAdd(&sumB[d], v[d]);  // global atomic
        atomicAdd(cntB, 1);
    }
}

// ─── v2: shared-mem accumulate, thread-0 global flush ────────────────────
// Phase 0: cooperatively zero sh_sumA[dim], sh_sumB[dim], sh_cnt[2]
// Phase 1: every thread → shared atomicAdd  (~5-20 cycles, on-chip SRAM)
//          Contention is block-local only (different blocks own separate sh[]).
// Phase 2: thread 0 only → global atomicAdd (ceil(n/BLOCK) × dim total).
// Dynamic smem: 2×dim floats + 2 ints = 8200 bytes for dim=1024 (block-size-independent)
__global__ static void accumulate_v2(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA, float* __restrict__ sumB,
        int*   __restrict__ cntA, int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);

    // Phase 0: zero shared (cooperative, strided)
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        sh_sumA[d] = 0.f;  sh_sumB[d] = 0.f;
    }
    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // Phase 1: shared atomicAdd (block-local contention, cheap)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* v = vecs + (ptrdiff_t)i * dim;
        if (labels[i] == 0) {
            for (int d = 0; d < dim; ++d) atomicAdd(&sh_sumA[d], v[d]);
            atomicAdd(&sh_cnt[0], 1);
        } else {
            for (int d = 0; d < dim; ++d) atomicAdd(&sh_sumB[d], v[d]);
            atomicAdd(&sh_cnt[1], 1);
        }
    }
    __syncthreads();

    // Phase 2: thread 0 flushes block partial sums to global
    if (threadIdx.x == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

// ─── v3: warp-shuffle → warp leaders → shared → thread-0 global ─────────
//
// Phase 1: for each dim d, the warp performs a register-only tree reduction:
//            val += __shfl_down_sync(mask, val, 16);  // thread 0 absorbs thread 16
//            val += __shfl_down_sync(mask, val,  8);
//            val += __shfl_down_sync(mask, val,  4);
//            val += __shfl_down_sync(mask, val,  2);
//            val += __shfl_down_sync(mask, val,  1);
//          After 5 ops (purely register-to-register, ~1-2 cycles each),
//          lane 0 of each warp holds the partial sum for dim d over 32 vectors.
//          Only lane 0 (warp leader) then atomicAdds to shared memory.
//
// Shared atomics per block:  (BLOCK/32) × dim   (32× fewer than v2)
// Global atomics per call:   ceil(n/BLOCK) × dim (same as v2)
//
// Dynamic smem: same layout as v2 (2×dim floats + 2 ints)
__global__ static void accumulate_v3(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        float* __restrict__ sumA, float* __restrict__ sumB,
        int*   __restrict__ cntA, int*   __restrict__ cntB)
{
    extern __shared__ float sh[];
    float* sh_sumA = sh;
    float* sh_sumB = sh + dim;
    int*   sh_cnt  = reinterpret_cast<int*>(sh + 2 * dim);

    const int lane    = (int)threadIdx.x & 31;
    const unsigned FULL_MASK = 0xffffffff;

    // Phase 0: zero shared (cooperative)
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        sh_sumA[d] = 0.f;  sh_sumB[d] = 0.f;
    }
    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const int lbl = (i < n) ? labels[i] : -1;

    // Phase 1: warp-shuffle reduce each dim, warp leader writes to shared
    for (int d = 0; d < dim; ++d) {
        const float raw = (i < n) ? vecs[(ptrdiff_t)i * dim + d] : 0.f;

        // Fold valA = raw if label==0 else 0, across the 32-thread warp
        float valA = (lbl == 0) ? raw : 0.f;
        valA += __shfl_down_sync(FULL_MASK, valA, 16);  // register-to-register
        valA += __shfl_down_sync(FULL_MASK, valA,  8);
        valA += __shfl_down_sync(FULL_MASK, valA,  4);
        valA += __shfl_down_sync(FULL_MASK, valA,  2);
        valA += __shfl_down_sync(FULL_MASK, valA,  1);
        // lane 0 now holds the warp's partial sum for dim d on side A

        // Same for side B
        float valB = (lbl == 1) ? raw : 0.f;
        valB += __shfl_down_sync(FULL_MASK, valB, 16);
        valB += __shfl_down_sync(FULL_MASK, valB,  8);
        valB += __shfl_down_sync(FULL_MASK, valB,  4);
        valB += __shfl_down_sync(FULL_MASK, valB,  2);
        valB += __shfl_down_sync(FULL_MASK, valB,  1);

        // Only warp leaders write to shared memory (BLOCK/32 threads write vs BLOCK in v2)
        if (lane == 0) {
            if (valA != 0.f) atomicAdd(&sh_sumA[d], valA);  // shared atomic
            if (valB != 0.f) atomicAdd(&sh_sumB[d], valB);
        }
    }

    // Count reduction using warp shuffle
    int cA_local = (lbl == 0 && i < n) ? 1 : 0;
    int cB_local = (lbl == 1 && i < n) ? 1 : 0;
    cA_local += __shfl_down_sync(FULL_MASK, cA_local, 16);
    cA_local += __shfl_down_sync(FULL_MASK, cA_local,  8);
    cA_local += __shfl_down_sync(FULL_MASK, cA_local,  4);
    cA_local += __shfl_down_sync(FULL_MASK, cA_local,  2);
    cA_local += __shfl_down_sync(FULL_MASK, cA_local,  1);
    cB_local += __shfl_down_sync(FULL_MASK, cB_local, 16);
    cB_local += __shfl_down_sync(FULL_MASK, cB_local,  8);
    cB_local += __shfl_down_sync(FULL_MASK, cB_local,  4);
    cB_local += __shfl_down_sync(FULL_MASK, cB_local,  2);
    cB_local += __shfl_down_sync(FULL_MASK, cB_local,  1);
    if (lane == 0) {
        if (cA_local > 0) atomicAdd(&sh_cnt[0], cA_local);
        if (cB_local > 0) atomicAdd(&sh_cnt[1], cB_local);
    }
    __syncthreads();

    // Phase 2: thread 0 flushes block totals to global (same as v2)
    if (threadIdx.x == 0) {
        for (int d = 0; d < dim; ++d) {
            if (sh_sumA[d] != 0.f) atomicAdd(&sumA[d], sh_sumA[d]);
            if (sh_sumB[d] != 0.f) atomicAdd(&sumB[d], sh_sumB[d]);
        }
        if (sh_cnt[0] > 0) atomicAdd(cntA, sh_cnt[0]);
        if (sh_cnt[1] > 0) atomicAdd(cntB, sh_cnt[1]);
    }
}

// ─── v4 step 1a: assign scatter positions atomically ──────────────────────
// Each thread atomically grabs a slot in partA or partB.
// Returns position in pos_a[i] or pos_b[i].
__global__ static void scatter_positions(
        const int* __restrict__ labels, int n,
        int* __restrict__ pos_a, int* __restrict__ pos_b,
        int* __restrict__ d_cntA, int* __restrict__ d_cntB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (labels[i] == 0) pos_a[i] = atomicAdd(d_cntA, 1);
    else                 pos_b[i] = atomicAdd(d_cntB, 1);
}

// ─── v4 step 1b: copy vectors into contiguous partitioned buffers ──────────
// Each thread copies its vector (n × dim) to the slot assigned above.
// partA and partB are both row-major (count × dim).
// Write pattern: each thread writes `dim` consecutive floats to a computed row.
__global__ static void scatter_data(
        const float* __restrict__ vecs, int n, int dim,
        const int*   __restrict__ labels,
        const int*   __restrict__ pos_a, const int* __restrict__ pos_b,
        float* __restrict__ partA,       float* __restrict__ partB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* src = vecs + (ptrdiff_t)i * dim;
    if (labels[i] == 0) {
        float* dst = partA + (ptrdiff_t)pos_a[i] * dim;
        for (int d = 0; d < dim; ++d) dst[d] = src[d];
    } else {
        float* dst = partB + (ptrdiff_t)pos_b[i] * dim;
        for (int d = 0; d < dim; ++d) dst[d] = src[d];
    }
}
// After scatter, partA is (countA × dim) row-major.
// Viewed in CUBLAS column-major convention: a (dim × countA) matrix.
// cublasSgemv(handle, CUBLAS_OP_N, dim, countA, &1, partA, dim, ones, 1, &0, sumA, 1)
// → sumA[d] = sum over k of partA[k][d]  ✓   (zero atomics)

// ═════════════════════════════════════════════════════════════════════════
// Utility: print separator + section header
// ═════════════════════════════════════════════════════════════════════════
static void print_sep(char c = '-', int w = 74) {
    for (int i = 0; i < w; ++i) putchar(c);
    putchar('\n');
}
static void phdr(const char* s) { print_sep('='); printf("  %s\n", s); print_sep('='); }

// ═════════════════════════════════════════════════════════════════════════
// Main
// ═════════════════════════════════════════════════════════════════════════
int main()
{
    // ── Device info ──────────────────────────────────────────────────────
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const double peak_bw = 2.0 * prop.memoryClockRate * 1e3
                         * (prop.memoryBusWidth / 8.0) / 1e9;  // GB/s

    phdr("GPU Kernel Benchmark: v1 / v2(block sweep) / v3(warp-shfl) / v4(scatter+gemv)");
    printf("  Device   : %s\n", prop.name);
    printf("  Peak BW  : %.0f GB/s\n", peak_bw);
    printf("  SM count : %d\n", prop.multiProcessorCount);
    printf("  N vectors: %d    DIM: %d\n", N_VECS, DIM);
    printf("  Warm-up  : %d    Trials: %d\n\n", N_WARMUP, N_TRIALS);

    // ── Theoretical counts (reported per accumulate call) ─────────────────
    const long grid256 = (N_VECS + 255) / 256;
    printf("  Global atomics (v1)        : %ld   (n × dim)\n",
           (long)N_VECS * DIM);
    printf("  Global atomics (v2/v3@256) : %ld   (ceil(n/256) × dim)\n",
           grid256 * DIM);
    printf("  Shared atomics (v2@256)    : %ld   (BLOCK × dim per block)\n",
           256L * DIM);
    printf("  Shared atomics (v3@256)    : %ld   (8 warp-leaders × dim per block)\n",
           8L * DIM);
    printf("  Shared atomics (v2)→global reduction factor vs v1: %.0f×\n",
           (double)((long)N_VECS * DIM) / (grid256 * DIM));
    printf("  Warp-shfl (v3) further reduces shared atomics by 32×\n\n");

    // ── Host data ─────────────────────────────────────────────────────────
    const size_t vec_bytes = (size_t)N_VECS * DIM * sizeof(float);
    const size_t cen_bytes = (size_t)DIM         * sizeof(float);

    std::vector<float> h_vecs(N_VECS * DIM);
    std::vector<float> h_cA(DIM), h_cB(DIM);
    srand(0xc0ffee);
    for (auto& x : h_vecs) x = (float)rand() / RAND_MAX;
    for (auto& x : h_cA)   x = (float)rand() / RAND_MAX;
    for (auto& x : h_cB)   x = (float)rand() / RAND_MAX;

    // ── Device allocations (shared across versions) ───────────────────────
    float *d_vecs, *d_cA, *d_cB, *d_sumA, *d_sumB;
    int   *d_labels, *d_cntA, *d_cntB;
    CUDA_CHECK(cudaMalloc(&d_vecs,   vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_cA,     cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_cB,     cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumA,   cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_sumB,   cen_bytes));
    CUDA_CHECK(cudaMalloc(&d_labels, (size_t)N_VECS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cntA,   sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cntB,   sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cA,   h_cA.data(),   cen_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cB,   h_cB.data(),   cen_bytes, cudaMemcpyHostToDevice));

    // v4-specific allocations
    float *d_partA, *d_partB, *d_ones;
    int   *d_pos_a, *d_pos_b;
    CUDA_CHECK(cudaMalloc(&d_partA, vec_bytes));         // worst-case full n
    CUDA_CHECK(cudaMalloc(&d_partB, vec_bytes));
    CUDA_CHECK(cudaMalloc(&d_pos_a, (size_t)N_VECS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pos_b, (size_t)N_VECS * sizeof(int)));
    // ones vector (max size = N_VECS)
    CUDA_CHECK(cudaMalloc(&d_ones,  (size_t)N_VECS * sizeof(float)));
    {   std::vector<float> h_ones(N_VECS, 1.f);
        CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(),
                   (size_t)N_VECS * sizeof(float), cudaMemcpyHostToDevice)); }

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Shared memory for v2/v3 (independent of block size)
    const size_t smem = 2 * (size_t)DIM * sizeof(float) + 2 * sizeof(int);
    if (smem > prop.sharedMemPerBlock) {
        fprintf(stderr, "ERROR: smem %zu > device limit %zu\n",
                smem, prop.sharedMemPerBlock);
        return 1;
    }

    // ── Warm-up assign_kernel & produce stable labels ─────────────────────
    const int ga256 = (N_VECS + 255) / 256;
    for (int w = 0; w < N_WARMUP; ++w)
        assign_kernel<<<ga256, 256>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── Precompute countA/countB for v4 (from the stable label array) ─────
    int h_cntA_v4 = 0, h_cntB_v4 = 0;
    {
        CUDA_CHECK(cudaMemset(d_cntA, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_cntB, 0, sizeof(int)));
        scatter_positions<<<ga256, V4_BLOCK>>>(
            d_labels, N_VECS, d_pos_a, d_pos_b, d_cntA, d_cntB);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_cntA_v4, d_cntA, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_cntB_v4, d_cntB, sizeof(int), cudaMemcpyDeviceToHost));
        printf("  v4 calibration: countA=%d  countB=%d\n\n", h_cntA_v4, h_cntB_v4);
    }

    // ── Memory traffic constants ──────────────────────────────────────────
    const double accum_read_bytes =   // same for v1/v2/v3
          (double)N_VECS * DIM * sizeof(float)    // vectors
        + (double)N_VECS       * sizeof(int);      // labels
    const double assign_read_bytes =
          (double)N_VECS * DIM * sizeof(float)    // vectors
        + 2.0 * DIM * sizeof(float);               // centroids

    // ── CSV file setup ────────────────────────────────────────────────────
    std::ofstream csv("bench_results.csv");
    csv << "version,block_size,trial,accum_ms,assign_ms,"
        << "accum_read_bw_gbs,global_atomics,shared_atomics,notes\n";

    // ═════════════════════════════════════════════════════════════════════
    // SECTION 0: assign_kernel baseline
    // ═════════════════════════════════════════════════════════════════════
    phdr("SECTION 0 — assign_kernel baseline (same for all versions)");
    GpuTimer ta;
    float assign_avg = 0.f;
    for (int w = 0; w < N_WARMUP; ++w)
        assign_kernel<<<ga256, 256>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
    for (int r = 0; r < N_TRIALS; ++r) {
        ta.start();
        assign_kernel<<<ga256, 256>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
        float ms = ta.stop();
        assign_avg += ms;
        csv << "assign,256," << r << "," << 0 << "," << ms << ","
            << (assign_read_bytes / (ms*1e-3) / 1e9) << ",0,0,assign_baseline\n";
    }
    assign_avg /= N_TRIALS;
    printf("  avg assign time : %.3f ms   read BW: %.1f GB/s\n\n",
           assign_avg, assign_read_bytes / (assign_avg * 1e-3) / 1e9);

    // ═════════════════════════════════════════════════════════════════════
    // SECTION 1: v1 — global atomicAdd (BLOCK=256)
    // ═════════════════════════════════════════════════════════════════════
    phdr("SECTION 1 — v1: n × dim global atomicAdds  (BLOCK=256)");
    printf("  %-8s %-6s %-12s %-14s\n", "trial", "ms", "eff_BW_GBs", "global_atomics");
    print_sep();

    // re-run assign to get labels
    assign_kernel<<<ga256,256>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int w = 0; w < N_WARMUP; ++w) {
        reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
        accumulate_v1<<<ga256,256>>>(d_vecs,N_VECS,DIM,d_labels,d_sumA,d_sumB,d_cntA,d_cntB);
    }
    GpuTimer t1;
    float v1_avg = 0.f;
    for (int r = 0; r < N_TRIALS; ++r) {
        reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
        t1.start();
        accumulate_v1<<<ga256,256>>>(d_vecs,N_VECS,DIM,d_labels,d_sumA,d_sumB,d_cntA,d_cntB);
        float ms = t1.stop();
        v1_avg += ms;
        double bw = accum_read_bytes / (ms * 1e-3) / 1e9;
        long gatoms = (long)N_VECS * DIM;
        printf("  %-8d %-6.3f %-12.1f %-14ld\n", r, ms, bw, gatoms);
        csv << "v1,256," << r << "," << ms << "," << assign_avg << ","
            << bw << "," << gatoms << "," << ((long)N_VECS*DIM) << ",global_atomics\n";
    }
    v1_avg /= N_TRIALS;
    printf("  ─── avg: %.3f ms   (note: low BW = L2 serialising %ld atomics)\n\n",
           v1_avg, (long)N_VECS * DIM);

    // ═════════════════════════════════════════════════════════════════════
    // SECTION 2: v2 — block size sweep  (BLOCK = 128, 256, 512, 1024)
    // ═════════════════════════════════════════════════════════════════════
    phdr("SECTION 2 — v2: shared-mem reduction, block-size sweep");
    float v2_best_avg = 1e9f;  int v2_best_block = 256;

    for (int B : V2_BLOCKS) {
        int   grid    = (N_VECS + B - 1) / B;
        long  g_atoms = (long)grid * DIM;
        long  s_atoms = (long)B * DIM;   // shared atomics per block × grid ≈ n × dim

        printf("\n  [v2  BLOCK=%4d]  grid=%d  global_atomics=%ld  shared_atomics/block=%d\n",
               B, grid, g_atoms, B * DIM);
        printf("  %-8s %-6s %-12s\n", "trial", "ms", "eff_BW_GBs");
        print_sep('-', 40);

        for (int w = 0; w < N_WARMUP; ++w) {
            reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
            accumulate_v2<<<grid,B,smem>>>(d_vecs,N_VECS,DIM,d_labels,
                                           d_sumA,d_sumB,d_cntA,d_cntB);
        }
        GpuTimer t2;
        float avg = 0.f;
        for (int r = 0; r < N_TRIALS; ++r) {
            reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
            t2.start();
            accumulate_v2<<<grid,B,smem>>>(d_vecs,N_VECS,DIM,d_labels,
                                           d_sumA,d_sumB,d_cntA,d_cntB);
            float ms = t2.stop();
            avg += ms;
            double bw = accum_read_bytes / (ms * 1e-3) / 1e9;
            printf("  %-8d %-6.3f %-12.1f\n", r, ms, bw);
            csv << "v2," << B << "," << r << "," << ms << "," << assign_avg << ","
                << bw << "," << g_atoms << "," << s_atoms << ",shared_mem_reduction\n";
        }
        avg /= N_TRIALS;
        printf("  ─── avg: %.3f ms\n", avg);
        if (avg < v2_best_avg) { v2_best_avg = avg; v2_best_block = B; }
    }
    printf("\n  v2 best block size: %d  (avg %.3f ms)\n\n", v2_best_block, v2_best_avg);

    // ═════════════════════════════════════════════════════════════════════
    // SECTION 3: v3 — warp-shuffle + warp-leaders → shared → thread-0 global
    // ═════════════════════════════════════════════════════════════════════
    phdr("SECTION 3 — v3: warp-shuffle reduction  (BLOCK=256, 8 warps)");
    {
        int  grid    = (N_VECS + V3_BLOCK - 1) / V3_BLOCK;
        long g_atoms = (long)grid * DIM;
        long s_atoms = (long)(V3_BLOCK / 32) * DIM;  // warp leaders only

        printf("  BLOCK=%d  warps/block=%d\n", V3_BLOCK, V3_BLOCK/32);
        printf("  global_atomics=%ld  (same as v2@256)\n", g_atoms);
        printf("  shared_atomics/block=%ld  (32× fewer than v2@256 shared atomics)\n", s_atoms);
        printf("  shfl ops/thread: %d × 10 = %d  (register-to-register, ~1-2 cycles each)\n\n",
               DIM, DIM * 10);
        printf("  %-8s %-6s %-12s\n", "trial", "ms", "eff_BW_GBs");
        print_sep('-', 40);

        for (int w = 0; w < N_WARMUP; ++w) {
            reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
            accumulate_v3<<<grid,V3_BLOCK,smem>>>(d_vecs,N_VECS,DIM,d_labels,
                                                   d_sumA,d_sumB,d_cntA,d_cntB);
        }
        GpuTimer t3;
        float v3_avg = 0.f;
        for (int r = 0; r < N_TRIALS; ++r) {
            reset_accumulators(d_sumA, d_sumB, d_cntA, d_cntB, cen_bytes);
            t3.start();
            accumulate_v3<<<grid,V3_BLOCK,smem>>>(d_vecs,N_VECS,DIM,d_labels,
                                                   d_sumA,d_sumB,d_cntA,d_cntB);
            float ms = t3.stop();
            v3_avg += ms;
            double bw = accum_read_bytes / (ms * 1e-3) / 1e9;
            printf("  %-8d %-6.3f %-12.1f\n", r, ms, bw);
            csv << "v3," << V3_BLOCK << "," << r << "," << ms << "," << assign_avg << ","
                << bw << "," << g_atoms << "," << s_atoms << ",warp_shuffle\n";
        }
        v3_avg /= N_TRIALS;
        printf("  ─── avg: %.3f ms\n\n", v3_avg);
    }

    // ═════════════════════════════════════════════════════════════════════
    // SECTION 4: v4 — scatter + cuBLAS sgemv  (zero atomics)
    // ═════════════════════════════════════════════════════════════════════
    phdr("SECTION 4 — v4: scatter + cuBLAS sgemv  (zero accumulate atomics)");
    {
        // Total memory moved by v4 accumulate:
        //   scatter_pos:  n labels (read) + n pos (write) = 2×n×4
        //   scatter_data: n×dim vecs (read) + n×dim writes to partA/partB
        //   gemv A:       countA×dim (read) + dim (write)
        //   gemv B:       countB×dim (read) + dim (write)
        double scatter_bytes =
              2.0 * N_VECS * sizeof(int)              // labels + positions
            + 2.0 * (double)N_VECS * DIM * sizeof(float);  // read + write vecs
        double gemv_bytes =
              ((double)h_cntA_v4 + h_cntB_v4) * DIM * sizeof(float)  // reads partA/B
            + (double)(h_cntA_v4 + h_cntB_v4) * sizeof(float)         // ones
            + 2.0 * DIM * sizeof(float);                               // sumA/B writes
        double total_v4_bytes = scatter_bytes + gemv_bytes;

        printf("  countA=%d  countB=%d\n", h_cntA_v4, h_cntB_v4);
        printf("  scatter mem: %.1f MB  gemv mem: %.1f MB  total: %.1f MB\n",
               scatter_bytes/1e6, gemv_bytes/1e6, total_v4_bytes/1e6);
        printf("  vs v1/v2/v3 accum read: %.1f MB  (v4 moves ~%.1f× more data)\n",
               accum_read_bytes/1e6, total_v4_bytes / accum_read_bytes);
        printf("  Global atomics (accumulate): 0  (scatter_positions: n=%d, negligible)\n\n",
               N_VECS);
        printf("  %-8s %-9s %-9s %-9s %-12s\n",
               "trial", "scatter_ms", "gemv_ms", "total_ms", "eff_BW_GBs");
        print_sep('-', 60);

        const float alpha = 1.f, beta = 0.f;
        GpuTimer ts_pos, ts_dat, tg;
        float v4_avg = 0.f;

        for (int w = 0; w < N_WARMUP; ++w) {
            CUDA_CHECK(cudaMemset(d_cntA, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_cntB, 0, sizeof(int)));
            scatter_positions<<<ga256,V4_BLOCK>>>(d_labels,N_VECS,d_pos_a,d_pos_b,d_cntA,d_cntB);
            scatter_data<<<ga256,V4_BLOCK>>>(d_vecs,N_VECS,DIM,d_labels,d_pos_a,d_pos_b,d_partA,d_partB);
            cublasSgemv(cublas,CUBLAS_OP_N,DIM,h_cntA_v4,&alpha,d_partA,DIM,d_ones,1,&beta,d_sumA,1);
            cublasSgemv(cublas,CUBLAS_OP_N,DIM,h_cntB_v4,&alpha,d_partB,DIM,d_ones,1,&beta,d_sumB,1);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        for (int r = 0; r < N_TRIALS; ++r) {
            CUDA_CHECK(cudaMemset(d_cntA, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_cntB, 0, sizeof(int)));

            ts_pos.start();
            scatter_positions<<<ga256,V4_BLOCK>>>(d_labels,N_VECS,d_pos_a,d_pos_b,d_cntA,d_cntB);
            float ms_spos = ts_pos.stop();

            ts_dat.start();
            scatter_data<<<ga256,V4_BLOCK>>>(d_vecs,N_VECS,DIM,d_labels,d_pos_a,d_pos_b,d_partA,d_partB);
            float ms_sdat = ts_dat.stop();

            tg.start();
            CUBLAS_CHECK(cublasSgemv(cublas,CUBLAS_OP_N,DIM,h_cntA_v4,&alpha,
                                     d_partA,DIM,d_ones,1,&beta,d_sumA,1));
            CUBLAS_CHECK(cublasSgemv(cublas,CUBLAS_OP_N,DIM,h_cntB_v4,&alpha,
                                     d_partB,DIM,d_ones,1,&beta,d_sumB,1));
            float ms_gemv = tg.stop();

            float total = ms_spos + ms_sdat + ms_gemv;
            v4_avg += total;
            double bw = total_v4_bytes / (total * 1e-3) / 1e9;

            printf("  %-8d %-9.3f %-9.3f %-9.3f %-12.1f\n",
                   r, ms_spos + ms_sdat, ms_gemv, total, bw);
            csv << "v4," << V4_BLOCK << "," << r << ","
                << total << "," << assign_avg << ","
                << bw << ",0,0,scatter_gemv\n";
        }
        v4_avg /= N_TRIALS;
        printf("  ─── avg total: %.3f ms\n\n", v4_avg);
    }

    // ═════════════════════════════════════════════════════════════════════
    // FINAL SUMMARY TABLE  (per-accumulate-call averages)
    // ═════════════════════════════════════════════════════════════════════
    csv.close();
    printf("\n");
    phdr("FINAL SUMMARY (accumulate phase only, avg over trials)");

    // Gather averages from CSV (re-read)
    struct Row { std::string ver; int blk; float ms; long gatom; };
    std::vector<Row> rows;
    {
        std::ifstream f("bench_results.csv");
        std::string line;
        std::getline(f, line); // header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            // version,block_size,trial,accum_ms,...
            char ver[32]; int blk, trial; float ams, asms, bw; long ga, sa;
            char notes[64];
            if (sscanf(line.c_str(), "%31[^,],%d,%d,%f,%f,%f,%ld,%ld,%63s",
                       ver, &blk, &trial, &ams, &asms, &bw, &ga, &sa, notes) >= 4) {
                if (strcmp(ver,"assign") != 0)
                    rows.push_back({ver, blk, ams, ga});
            }
        }
    }
    // Compute per-(version,block) averages
    struct Summary { std::string key; float sum; int cnt; long gatom; };
    std::vector<Summary> sums;
    auto get = [&](const std::string& k, int blk, long ga) -> Summary& {
        char buf[64]; snprintf(buf, sizeof(buf), "%s@%d", k.c_str(), blk);
        for (auto& s : sums) if (s.key == buf) return s;
        sums.push_back({buf, 0.f, 0, ga});
        return sums.back();
    };
    for (auto& r : rows) { auto& s = get(r.ver, r.blk, r.gatom); s.sum += r.ms; s.cnt++; }

    printf("  %-20s %10s %14s %16s\n",
           "version@block", "avg_ms", "global_atomics", "speedup_vs_v1");
    print_sep();
    float v1_ref = 0.f;
    for (auto& s : sums)
        if (s.key == "v1@256") { v1_ref = s.sum / s.cnt; break; }
    for (auto& s : sums) {
        float avg = s.sum / s.cnt;
        printf("  %-20s %10.3f %14ld %16.1fx\n",
               s.key.c_str(), avg, s.gatom, v1_ref / avg);
    }
    print_sep();
    printf("\n  Global memory read bytes per accumulate (v1/v2/v3): %.1f MB\n",
           accum_read_bytes / 1e6);
    printf("  Low effective BW in v1 = global L2 serialising %ld atomics.\n",
           (long)N_VECS * DIM);
    printf("  v2/v3 recover BW by reducing global atomic pressure.\n");
    printf("  v4 bypasses atomics entirely at cost of ~3× more memory traffic.\n\n");
    printf("  Results saved to bench_results.csv\n");
    printf("  Run: python3 test/plot_bench_results.py\n\n");

    // ── Cleanup ──────────────────────────────────────────────────────────
    cublasDestroy(cublas);
    cudaFree(d_vecs); cudaFree(d_cA);   cudaFree(d_cB);
    cudaFree(d_sumA); cudaFree(d_sumB); cudaFree(d_labels);
    cudaFree(d_cntA); cudaFree(d_cntB);
    cudaFree(d_partA); cudaFree(d_partB);
    cudaFree(d_pos_a); cudaFree(d_pos_b); cudaFree(d_ones);
    return 0;
}

#else
#error "Compile with nvcc -DHAVE_CUDA"
#endif
