// bench_kernel_comparison.cu
//
// Standalone benchmark: gpu_split_kernel (v1) vs split_kernel_v2 (v2).
//
// Designed to show exactly where v2 wins by isolating the accumulate
// phase and logging global-memory read traffic in both cases.
//
// ═══════════════════════════════════════════════════════════════════════
// Compile (from repo root):
//   nvcc -O3 -DHAVE_CUDA --gpu-architecture=sm_86 \
//        -I AgentMemory/M3/include               \
//        test/bench_kernel_comparison.cu          \
//        -o bench_kernel_comparison
//   ./bench_kernel_comparison
// ═══════════════════════════════════════════════════════════════════════
//
// Methodology
// ───────────
// Both kernels read the same data from global memory:
//   n × dim × 4 bytes (vector data) + n × 4 bytes (label array)
//
// They differ only in the WRITE path (accumulation):
//   v1: n × dim  global atomicAdds  (all threads, high contention)
//   v2: ceil(n/BLOCK) × dim  global atomicAdds  (thread-0 per block)
//
// Contention in v1 stalls warps waiting for the global L2 bus,
// which is visible as:
//   (a) much longer accumulate_kernel wall time
//   (b) low effective read bandwidth (pipeline stalled by write serialisation)
//
// We measure per-kernel-phase GPU time with cudaEvent pairs and compute:
//   effective_read_GB/s = global_read_bytes / kernel_time_s
//
// A low bandwidth on v1 despite fixed read data → pipeline stalled by atomics.

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────
// Benchmark configuration
// ─────────────────────────────────────────────────────────────────────
static constexpr int   N_VECS   = 65536;  // large enough to saturate global atomics
static constexpr int   DIM      = 1024;   // same as IVF.index (d=1024)
static constexpr int   BLOCK    = 256;
static constexpr int   N_WARMUP = 3;      // kernel warm-up runs (not timed)
static constexpr int   N_TRIALS = 10;     // timed runs per phase

// ─────────────────────────────────────────────────────────────────────
// CUDA error check
// ─────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────────────────────────────
// Helper: create a cudaEvent pair and time a lambda
// ─────────────────────────────────────────────────────────────────────
struct Timer {
    cudaEvent_t start, stop;
    Timer()  { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void begin() { CUDA_CHECK(cudaEventRecord(start)); }
    float end()  {                                  // returns ms
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ─────────────────────────────────────────────────────────────────────
// Kernel definitions (self-contained, mirroring both source files)
// ─────────────────────────────────────────────────────────────────────

// ── assign kernel (identical in both versions) ────────────────────────
__global__ static void assign_kernel(
        const float* __restrict__ vecs, int n, int dim,
        const float* __restrict__ cA,  const float* __restrict__ cB,
        int* __restrict__ labels)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dA = 0.f, dB = 0.f;
    const float* v = vecs + (ptrdiff_t)i * dim;
    for (int d = 0; d < dim; ++d) {
        float da = v[d] - cA[d];
        float db = v[d] - cB[d];
        dA += da * da;
        dB += db * db;
    }
    labels[i] = (dB < dA) ? 1 : 0;
}

// ── v1: global-atomicAdd accumulate (one atomic per (thread, dim)) ────
// BOTTLENECK: n × dim global atomicAdds, all hammering the same 2*dim addresses.
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
        for (int d = 0; d < dim; ++d)
            atomicAdd(&sumA[d], v[d]);   // ← global atomicAdd per (thread, dim)
        atomicAdd(cntA, 1);
    } else {
        for (int d = 0; d < dim; ++d)
            atomicAdd(&sumB[d], v[d]);
        atomicAdd(cntB, 1);
    }
}

// ── v2: shared-memory reduction, then one global flush per block ──────
// Phase 1: accumulate into shared memory  (~5-20 cycle latency)
// Phase 2: thread 0 flushes block totals to global  (ceil(n/BLOCK) × dim atomics)
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

    // Phase 0: zero shared memory (cooperative)
    for (int d = (int)threadIdx.x; d < dim; d += (int)blockDim.x) {
        sh_sumA[d] = 0.f;
        sh_sumB[d] = 0.f;
    }
    if (threadIdx.x == 0) { sh_cnt[0] = 0; sh_cnt[1] = 0; }
    __syncthreads();

    // Phase 1: accumulate into shared memory (fast on-chip atomics)
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* v = vecs + (ptrdiff_t)i * dim;
        if (labels[i] == 0) {
            for (int d = 0; d < dim; ++d)
                atomicAdd(&sh_sumA[d], v[d]);  // shared-memory atomic ~5-20 cycles
            atomicAdd(&sh_cnt[0], 1);
        } else {
            for (int d = 0; d < dim; ++d)
                atomicAdd(&sh_sumB[d], v[d]);
            atomicAdd(&sh_cnt[1], 1);
        }
    }
    __syncthreads();

    // Phase 2: thread 0 per block flushes to global  (1 writer per block)
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
// Printing helpers
// ─────────────────────────────────────────────────────────────────────
static void print_sep(char c = '─', int w = 72) {
    for (int i = 0; i < w; ++i) putchar(c);
    putchar('\n');
}

static void print_header(const char* title) {
    print_sep('═');
    printf("  %s\n", title);
    print_sep('═');
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────
int main()
{
    // ── Device info ─────────────────────────────────────────────────
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    print_header("Kernel Comparison: v1 (global atomics) vs v2 (shared-mem reduction)");
    printf("  Device : %s\n", prop.name);
    printf("  SM count     : %d\n", prop.multiProcessorCount);
    printf("  Peak BW      : %.1f GB/s\n",
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);
    printf("  Shared mem/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");
    printf("  N (vectors)  : %d\n", N_VECS);
    printf("  DIM          : %d\n", DIM);
    printf("  BLOCK size   : %d\n", BLOCK);
    printf("  Grid size    : %d\n", (N_VECS + BLOCK - 1) / BLOCK);
    printf("  Timed trials : %d (after %d warm-up)\n\n", N_TRIALS, N_WARMUP);

    // ── Theoretical operation counts ────────────────────────────────
    const long grid = (N_VECS + BLOCK - 1) / BLOCK;

    const long v1_global_atomic_accumulate = (long)N_VECS * DIM;      // per call
    const long v2_global_atomic_accumulate = grid * DIM;               // per call
    const long shared_atomic_accumulate    = (long)N_VECS * DIM;      // v2 phase-1
    const long global_read_bytes_accum     =
            (long)N_VECS * DIM * sizeof(float)                         // vectors
          + (long)N_VECS       * sizeof(int);                          // labels
    const long global_read_bytes_assign    =
            (long)N_VECS * DIM * sizeof(float)                         // vectors
          + (long)2 * DIM      * sizeof(float);                        // cA + cB

    print_header("THEORETICAL OPERATION COUNTS — accumulate kernel");
    printf("  Global reads (both):  %ld floats  (%ld MB)\n",
           global_read_bytes_accum / 4, global_read_bytes_accum / (1<<20));
    printf("\n");
    printf("  v1 global atomicAdds: %ld   (n × dim = %d × %d)\n",
           v1_global_atomic_accumulate, N_VECS, DIM);
    printf("  v2 global atomicAdds: %ld   (ceil(n/BLOCK) × dim = %ld × %d)\n",
           v2_global_atomic_accumulate, grid, DIM);
    printf("  v2 shared atomicAdds: %ld   (phase-1, ~5-20 cycle latency)\n",
           shared_atomic_accumulate);
    printf("\n  v2 global atomic REDUCTION FACTOR: %.0f×\n\n",
           (double)v1_global_atomic_accumulate / v2_global_atomic_accumulate);

    // ── Allocate and fill host data ──────────────────────────────────
    size_t vec_bytes = (size_t)N_VECS * DIM * sizeof(float);
    size_t cen_bytes = (size_t)DIM        * sizeof(float);

    std::vector<float> h_vecs(N_VECS * DIM);
    std::vector<float> h_cA(DIM), h_cB(DIM);

    // Random fill — seed fixed for reproducibility
    srand(0xdeadbeef);
    for (auto& x : h_vecs) x = (float)rand() / RAND_MAX;
    for (auto& x : h_cA)   x = (float)rand() / RAND_MAX;
    for (auto& x : h_cB)   x = (float)rand() / RAND_MAX;

    // ── Allocate device memory ───────────────────────────────────────
    float* d_vecs; CUDA_CHECK(cudaMalloc(&d_vecs, vec_bytes));
    float* d_cA;   CUDA_CHECK(cudaMalloc(&d_cA,   cen_bytes));
    float* d_cB;   CUDA_CHECK(cudaMalloc(&d_cB,   cen_bytes));
    int*   d_labels; CUDA_CHECK(cudaMalloc(&d_labels, (size_t)N_VECS * sizeof(int)));
    float* d_sumA; CUDA_CHECK(cudaMalloc(&d_sumA, cen_bytes));
    float* d_sumB; CUDA_CHECK(cudaMalloc(&d_sumB, cen_bytes));
    int*   d_cntA; CUDA_CHECK(cudaMalloc(&d_cntA, sizeof(int)));
    int*   d_cntB; CUDA_CHECK(cudaMalloc(&d_cntB, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_vecs, h_vecs.data(), vec_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cA,   h_cA.data(),   cen_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cB,   h_cB.data(),   cen_bytes, cudaMemcpyHostToDevice));

    const int   g_assign = (N_VECS + BLOCK - 1) / BLOCK;
    const size_t smem_v2 = 2 * (size_t)DIM * sizeof(float) + 2 * sizeof(int);

    // ── Verify shared mem fits ───────────────────────────────────────
    if (smem_v2 > prop.sharedMemPerBlock) {
        fprintf(stderr,
            "ERROR: v2 requires %zu B shared mem per block; device has %zu B.\n"
            "       Reduce DIM or BLOCK.\n",
            smem_v2, prop.sharedMemPerBlock);
        return 1;
    }
    printf("  v2 shared mem per block: %zu B / %zu B available  (%.1f%%)\n\n",
           smem_v2, prop.sharedMemPerBlock,
           100.0 * smem_v2 / prop.sharedMemPerBlock);

    // ─────────────────────────────────────────────────────────────────
    // Warm-up both kernels (not timed)
    // ─────────────────────────────────────────────────────────────────
    for (int w = 0; w < N_WARMUP; ++w) {
        assign_kernel<<<g_assign, BLOCK>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
        CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
        CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
        accumulate_v1<<<g_assign, BLOCK>>>(
            d_vecs, N_VECS, DIM, d_labels, d_sumA, d_sumB, d_cntA, d_cntB);
        accumulate_v2<<<g_assign, BLOCK, smem_v2>>>(
            d_vecs, N_VECS, DIM, d_labels, d_sumA, d_sumB, d_cntA, d_cntB);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ─────────────────────────────────────────────────────────────────
    // Phase A: assign_kernel  (identical in both; baseline timing)
    // ─────────────────────────────────────────────────────────────────
    print_header("PHASE A — assign_kernel  (no atomics; same in v1 and v2)");
    printf("  Global reads: %ld MB  (vectors + 2 centroids)\n",
           global_read_bytes_assign / (1<<20));
    printf("  Global writes: %ld KB  (labels array)\n",
           (long)N_VECS * sizeof(int) / 1024);
    printf("  Global atomics: 0\n\n");

    Timer t_assign;
    float assign_total_ms = 0.f;
    for (int r = 0; r < N_TRIALS; ++r) {
        t_assign.begin();
        assign_kernel<<<g_assign, BLOCK>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
        assign_total_ms += t_assign.end();
    }
    float assign_avg_ms = assign_total_ms / N_TRIALS;
    double assign_read_bw = (double)global_read_bytes_assign / (assign_avg_ms * 1e-3) / 1e9;
    printf("  avg wall time : %.3f ms\n", assign_avg_ms);
    printf("  read bandwidth: %.1f GB/s  (out of %.1f GB/s peak)\n\n",
           assign_read_bw,
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);

    // ─────────────────────────────────────────────────────────────────
    // Phase B: accumulate_v1
    // ─────────────────────────────────────────────────────────────────
    print_header("PHASE B — accumulate_v1  (ALL threads → global atomicAdd)");
    printf("  Global reads  : %ld MB   (vectors + labels)\n",
           global_read_bytes_accum / (1<<20));
    printf("  Global atomics: %ld  (n × dim = serialised by L2)\n\n",
           v1_global_atomic_accumulate);

    // Re-run assign to get a stable label array, then isolate accumulate
    assign_kernel<<<g_assign, BLOCK>>>(d_vecs, N_VECS, DIM, d_cA, d_cB, d_labels);
    CUDA_CHECK(cudaDeviceSynchronize());

    Timer t_v1;
    float v1_total_ms = 0.f;
    for (int r = 0; r < N_TRIALS; ++r) {
        CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
        CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
        t_v1.begin();
        accumulate_v1<<<g_assign, BLOCK>>>(
            d_vecs, N_VECS, DIM, d_labels, d_sumA, d_sumB, d_cntA, d_cntB);
        v1_total_ms += t_v1.end();
    }
    float v1_avg_ms = v1_total_ms / N_TRIALS;
    double v1_read_bw = (double)global_read_bytes_accum / (v1_avg_ms * 1e-3) / 1e9;

    // Effective stall estimate: if reads were free, time = read_bytes / peak_bw
    double peak_bw_GBs = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;
    double v1_min_read_ms = (double)global_read_bytes_accum / (peak_bw_GBs * 1e9) * 1e3;
    double v1_atomic_overhead_ms = v1_avg_ms - v1_min_read_ms;

    printf("  avg wall time        : %.3f ms\n", v1_avg_ms);
    printf("  effective read BW    : %.1f GB/s  (should approach peak if no stalls)\n",
           v1_read_bw);
    printf("  min time if BW-bound : %.3f ms  (at %.1f GB/s peak)\n",
           v1_min_read_ms, peak_bw_GBs);
    printf("  >> estimated global atomic stall overhead: %.3f ms  (%.0f%% of total)\n\n",
           v1_atomic_overhead_ms,
           100.0 * v1_atomic_overhead_ms / v1_avg_ms);

    // ─────────────────────────────────────────────────────────────────
    // Phase C: accumulate_v2
    // ─────────────────────────────────────────────────────────────────
    print_header("PHASE C — accumulate_v2  (shared-mem → thread-0 global flush)");
    printf("  Global reads  : %ld MB   (same as v1)\n",
           global_read_bytes_accum / (1<<20));
    printf("  Shared atomics: %ld  (phase-1, on-chip SRAM ~5-20 cycles)\n",
           shared_atomic_accumulate);
    printf("  Global atomics: %ld  (phase-2, %.0f× fewer than v1)\n\n",
           v2_global_atomic_accumulate,
           (double)v1_global_atomic_accumulate / v2_global_atomic_accumulate);

    Timer t_v2;
    float v2_total_ms = 0.f;
    for (int r = 0; r < N_TRIALS; ++r) {
        CUDA_CHECK(cudaMemset(d_sumA, 0, cen_bytes));
        CUDA_CHECK(cudaMemset(d_sumB, 0, cen_bytes));
        t_v2.begin();
        accumulate_v2<<<g_assign, BLOCK, smem_v2>>>(
            d_vecs, N_VECS, DIM, d_labels, d_sumA, d_sumB, d_cntA, d_cntB);
        v2_total_ms += t_v2.end();
    }
    float v2_avg_ms = v2_total_ms / N_TRIALS;
    double v2_read_bw = (double)global_read_bytes_accum / (v2_avg_ms * 1e-3) / 1e9;
    double v2_atomic_overhead_ms = v2_avg_ms - v1_min_read_ms;

    printf("  avg wall time        : %.3f ms\n", v2_avg_ms);
    printf("  effective read BW    : %.1f GB/s\n", v2_read_bw);
    printf("  min time if BW-bound : %.3f ms  (at %.1f GB/s peak)\n",
           v1_min_read_ms, peak_bw_GBs);
    printf("  >> estimated remaining overhead: %.3f ms  (%.0f%% of total)\n\n",
           (v2_atomic_overhead_ms > 0 ? v2_atomic_overhead_ms : 0.0),
           100.0 * (v2_atomic_overhead_ms > 0 ? v2_atomic_overhead_ms : 0.0) / v2_avg_ms);

    // ─────────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────────
    print_header("SUMMARY");
    printf("  %-32s %10s %10s %10s\n", "Metric", "v1", "v2", "ratio");
    print_sep();
    printf("  %-32s %10.3f %10.3f %9.1fx\n",
           "accumulate kernel (ms)", v1_avg_ms, v2_avg_ms,
           v1_avg_ms / v2_avg_ms);
    printf("  %-32s %10ld %10ld %9.0fx\n",
           "global atomicAdds",
           v1_global_atomic_accumulate,
           v2_global_atomic_accumulate,
           (double)v1_global_atomic_accumulate / v2_global_atomic_accumulate);
    printf("  %-32s %10.1f %10.1f\n",
           "effective read BW (GB/s)", v1_read_bw, v2_read_bw);
    printf("  %-32s %10.3f %10.3f\n",
           "assign kernel (ms)", assign_avg_ms, assign_avg_ms);
    printf("  %-32s %10.3f %10.3f\n",
           "total per iteration (ms)",
           assign_avg_ms + v1_avg_ms, assign_avg_ms + v2_avg_ms);
    print_sep();
    printf("\n  Global memory read bytes per accumulate call (both): %ld MB\n",
           global_read_bytes_accum / (1<<20));
    printf("\n  NOTE: both kernels read the SAME bytes from global memory.\n");
    printf("  v1 low BW = pipeline stalled by %ld serialised L2 atomics.\n",
           v1_global_atomic_accumulate);
    printf("  v2 achieves higher BW because only %ld global atomics fire\n",
           v2_global_atomic_accumulate);
    printf("  (%.0fx fewer), eliminating the L2 serialisation bottleneck.\n\n",
           (double)v1_global_atomic_accumulate / v2_global_atomic_accumulate);

    // ─────────────────────────────────────────────────────────────────
    // Global memory read time breakdown
    // ─────────────────────────────────────────────────────────────────
    print_header("GLOBAL MEMORY READ TIME BREAKDOWN (estimated)");
    printf("  Both kernels must read %ld MB to do the accumulation.\n",
           global_read_bytes_accum / (1<<20));
    printf("  At peak bandwidth (%.1f GB/s) that takes %.3f ms.\n\n",
           peak_bw_GBs, v1_min_read_ms);
    printf("  v1: wall=%.3f ms | est. read=%.3f ms | est. atomic stall=%.3f ms\n",
           v1_avg_ms, v1_min_read_ms,
           v1_atomic_overhead_ms > 0 ? v1_atomic_overhead_ms : 0.0);
    printf("  v2: wall=%.3f ms | est. read=%.3f ms | est. smem+flush overhead=%.3f ms\n\n",
           v2_avg_ms, v1_min_read_ms,
           v2_atomic_overhead_ms > 0 ? v2_atomic_overhead_ms : 0.0);

    // ─────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────
    cudaFree(d_vecs); cudaFree(d_cA); cudaFree(d_cB);
    cudaFree(d_labels); cudaFree(d_sumA); cudaFree(d_sumB);
    cudaFree(d_cntA); cudaFree(d_cntB);

    return 0;
}

#else // !HAVE_CUDA
#error "This benchmark requires CUDA. Compile with: nvcc -DHAVE_CUDA ..."
#endif
