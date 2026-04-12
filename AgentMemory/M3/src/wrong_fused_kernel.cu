// gpu_cluster_index.cu — real CUDA implementation of GpuClusterIndex.
//
// 1. Snapshot clusters (CPU)
//         ↓
// 2. Upload all queries to GPU (H2D)
//         ↓
// 3. Pack all cluster vectors into d_vecs_packed_ (D2D)
//         ↓
// 4. k_query_norms — compute ||Q[qi]||² for every query on GPU
//    (overlaps D2D packing on same stream; no host loop, no extra H2D)
//         ↓
// 5. cublasSgemm — dot products for ALL queries × ALL vectors (GPU)
//         ↓
// 6. k_l2_fused — fused vnorm + L2 correction in one pass (GPU)
//    (reads vecs once inline; eliminates d_norms_scratch_ + k_squared_norms)
//         ↓
// 7. k_topk_per_query — per-query top-k on GPU → k×q_rows result (GPU)
//         ↓
// 8. Single sync + D2H of k×q_rows scores+indices only (GPU→CPU)
//    (reduction vs full matrix: total_n/k × savings on PCIe)
//         ↓
// 9. Per-query CPU merge: resolve vi→DocId + insert buffer (CPU, OMP parallel)
//         ↓
// 10. sort_heap → write output

#include "gpu_cluster_index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>   // std::greater
#include <omp.h>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {
inline bool gpu_diag_enabled() {
    static const bool v = (std::getenv("M3_GPU_DIAG") != nullptr &&
                           std::getenv("M3_GPU_DIAG")[0] == '1');
    return v;
}
using clock_t_ = std::chrono::high_resolution_clock;
inline double fms_(clock_t_::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
} // anonymous namespace

namespace m3 {

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error helpers
// ─────────────────────────────────────────────────────────────────────────────

static void cuda_check_(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("[M3 CUDA] ") + cudaGetErrorString(err) +
            " at " + file + ":" + std::to_string(line));
}
#define CUDA_CHECK(expr) cuda_check_((expr), __FILE__, __LINE__)

static void cublas_check_(cublasStatus_t st, const char* file, int line) {
    if (st != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(
            std::string("[M3 cuBLAS] status ") + std::to_string(static_cast<int>(st)) +
            " at " + file + ":" + std::to_string(line));
}
#define CUBLAS_CHECK(expr) cublas_check_((expr), __FILE__, __LINE__)

// ─────────────────────────────────────────────────────────────────────────────
// Distance kernels for search_cluster (single cluster, single query)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_l2_distances(
        const float* __restrict__ query,
        const float* __restrict__ vecs,
        float*       __restrict__ dists,
        int n, int dim)
{
    extern __shared__ float s_query[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        s_query[d] = query[d];
    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = vecs + (size_t)i * dim;
    float acc = 0.f;
    for (int d = 0; d < dim; ++d) {
        const float diff = s_query[d] - v[d];
        acc += diff * diff;
    }
    dists[i] = acc;
}

__global__ void k_ip_distances(
        const float* __restrict__ query,
        const float* __restrict__ vecs,
        float*       __restrict__ dists,
        int n, int dim)
{
    extern __shared__ float s_query[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        s_query[d] = query[d];
    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float* v = vecs + (size_t)i * dim;
    float acc = 0.f;
    for (int d = 0; d < dim; ++d)
        acc += s_query[d] * v[d];
    dists[i] = -acc;
}

static void launch_distance_kernel(Metric metric,
                                   const float* d_query,
                                   const float* d_vecs,
                                   float*       d_dists,
                                   int n, int dim,
                                   cudaStream_t stream = 0)
{
    const int    threads = 256;
    const int    blocks  = (n + threads - 1) / threads;
    const size_t smem    = static_cast<size_t>(dim) * sizeof(float);

    if (metric == Metric::L2)
        k_l2_distances<<<blocks, threads, smem, stream>>>(d_query, d_vecs, d_dists, n, dim);
    else
        k_ip_distances<<<blocks, threads, smem, stream>>>(d_query, d_vecs, d_dists, n, dim);

    CUDA_CHECK(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────────────
// k_query_norms
// Computes ||Q[qi]||² for each query on the GPU using the already-uploaded
// d_queries_batch_.  One block per query; threads stripe across dim and
// reduce via shared memory.  Replaces the host-side loop + extra H2D copy.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_query_norms(const float* __restrict__ queries,
                               float*       __restrict__ qnorms,
                               int q_rows, int dim)
{
    const int qi = blockIdx.x;
    if (qi >= q_rows) return;

    extern __shared__ float s_partial[];
    const float* q = queries + qi * dim;
    float acc = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        acc += q[d] * q[d];

    s_partial[threadIdx.x] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0) qnorms[qi] = s_partial[0];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_l2_fused
// Fuses k_squared_norms + k_l2_from_ip_batch into a single pass.
//
// Input dists (col-major SGEMM output):
//   dists[vi + qi * total_n] = dot(V[vi], Q[qi])
// Output:
//   dists[vi + qi * total_n] = ||V[vi]||² + ||Q[qi]||² - 2·dot  (= L2²)
//
// vnorm is computed inline by re-reading the vector — this costs one extra
// read of vecs per element but removes d_norms_scratch_ entirely (saving an
// allocation, a kernel launch, and k_squared_norms global-write traffic).
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_l2_fused(float*       __restrict__ dists,
                            const float* __restrict__ vecs,
                            const float* __restrict__ qnorms,
                            int total_n, int q_rows, int dim)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_n * q_rows) return;

    const int vi = idx % total_n;
    const int qi = idx / total_n;

    const float* v = vecs + (size_t)vi * dim;
    float vnorm = 0.f;
    for (int d = 0; d < dim; ++d) vnorm += v[d] * v[d];

    dists[idx] = qnorms[qi] + vnorm - 2.f * dists[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// k_topk_per_query
// For each query qi, selects the k smallest distances from the full distance
// matrix and writes (score, vi) pairs to d_topk_scores / d_topk_vis.
//
// Layout of d_dists (input, col-major):
//   d_dists[vi + qi * total_n]
//
// Layout of outputs:
//   d_topk_scores[qi * k + rank]
//   d_topk_vis   [qi * k + rank]  (vi into packed vector array, resolved CPU-side)
//
// One block per query. Phase 1: each thread builds a private top-k over its
// stripe using a simple replace-worst strategy (k ≤ 256 fits in local mem).
// Phase 2: threads merge into a shared heap serially (correct, O(T×k) but
// T=256, k≤256 → 65k ops, negligible vs SGEMM). Phase 3: thread 0 writes.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k_topk_per_query(
        const float* __restrict__ d_dists,
        float*       __restrict__ d_topk_scores,
        int*         __restrict__ d_topk_vis,
        int total_n, int q_rows, int k)
{
    const int qi = blockIdx.x;
    if (qi >= q_rows) return;

    // Shared heap: k slots, max-heap (worst score at root).
    extern __shared__ float s_mem[];
    float* s_scores = s_mem;
    int*   s_vis    = reinterpret_cast<int*>(s_mem + k);

    if (threadIdx.x == 0) {
        for (int i = 0; i < k; ++i) {
            s_scores[i] = __int_as_float(0x7f800000); // +inf
            s_vis[i]    = -1;
        }
    }
    __syncthreads();

    const float* qi_dists = d_dists + (size_t)qi * total_n;

    // Phase 1: private top-k per thread (local memory, k ≤ 256)
    float l_scores[256];
    int   l_vis[256];
    int   l_size = 0;
    float l_worst = -1.f;
    int   l_worst_idx = 0;

    for (int vi = threadIdx.x; vi < total_n; vi += blockDim.x) {
        const float sc = qi_dists[vi];
        if (l_size < k) {
            l_scores[l_size] = sc;
            l_vis[l_size]    = vi;
            if (sc > l_worst) { l_worst = sc; l_worst_idx = l_size; }
            ++l_size;
        } else if (sc < l_worst) {
            l_scores[l_worst_idx] = sc;
            l_vis[l_worst_idx]    = vi;
            l_worst = l_scores[0]; l_worst_idx = 0;
            for (int i = 1; i < k; ++i) {
                if (l_scores[i] > l_worst) { l_worst = l_scores[i]; l_worst_idx = i; }
            }
        }
    }

    // Phase 2: serial merge into shared heap
    for (int t = 0; t < static_cast<int>(blockDim.x); ++t) {
        if (threadIdx.x == t) {
            float sh_worst = s_scores[0]; int sh_worst_idx = 0;
            for (int i = 1; i < k; ++i)
                if (s_scores[i] > sh_worst) { sh_worst = s_scores[i]; sh_worst_idx = i; }
            for (int i = 0; i < l_size; ++i) {
                if (l_scores[i] < sh_worst) {
                    s_scores[sh_worst_idx] = l_scores[i];
                    s_vis[sh_worst_idx]    = l_vis[i];
                    sh_worst = s_scores[0]; sh_worst_idx = 0;
                    for (int j = 1; j < k; ++j)
                        if (s_scores[j] > sh_worst) { sh_worst = s_scores[j]; sh_worst_idx = j; }
                }
            }
        }
        __syncthreads();
    }

    // Phase 3: thread 0 writes output (unsorted; CPU sorts after D2H)
    if (threadIdx.x == 0) {
        float* out_s = d_topk_scores + qi * k;
        int*   out_v = d_topk_vis    + qi * k;
        for (int i = 0; i < k; ++i) { out_s[i] = s_scores[i]; out_v[i] = s_vis[i]; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuClusterIndex implementation
// ─────────────────────────────────────────────────────────────────────────────

GpuClusterIndex::GpuClusterIndex(int dim, Metric metric, bool normalized)
    : dim_(dim), metric_(metric), normalized_(normalized) {}

GpuClusterIndex::~GpuClusterIndex() {
    if (cublas_handle_)    { cublasDestroy(cublas_handle_);        cublas_handle_    = nullptr; }
    if (search_stream_)    { cudaStreamDestroy(search_stream_);    search_stream_    = nullptr; }
    if (d_query_scratch_)  { cudaFree(d_query_scratch_);           d_query_scratch_  = nullptr; }
    if (d_dist_scratch_)   { cudaFree(d_dist_scratch_);            d_dist_scratch_   = nullptr; }
    if (h_dist_scratch_)   { cudaFreeHost(h_dist_scratch_);        h_dist_scratch_   = nullptr; }
    if (d_vecs_packed_)    { cudaFree(d_vecs_packed_);             d_vecs_packed_    = nullptr; }
    if (d_queries_batch_)  { cudaFree(d_queries_batch_);           d_queries_batch_  = nullptr; }
    if (d_qnorms_batch_)   { cudaFree(d_qnorms_batch_);            d_qnorms_batch_   = nullptr; }
    if (d_topk_scores_)    { cudaFree(d_topk_scores_);             d_topk_scores_    = nullptr; }
    if (d_topk_vis_)       { cudaFree(d_topk_vis_);                d_topk_vis_       = nullptr; }
    if (h_topk_scores_)    { cudaFreeHost(h_topk_scores_);         h_topk_scores_    = nullptr; }
    if (h_topk_vis_)       { cudaFreeHost(h_topk_vis_);            h_topk_vis_       = nullptr; }
    scratch_cap_        = 0;
    vecs_packed_cap_    = 0;
    queries_batch_cap_  = 0;
    qnorms_batch_cap_   = 0;
    topk_cap_           = 0;
}

void GpuClusterIndex::ensure_scratch_(size_t need) const {
    if (!search_stream_) {
        CUDA_CHECK(cudaStreamCreate(&search_stream_));
        CUDA_CHECK(cudaMalloc(&d_query_scratch_,
                              static_cast<size_t>(dim_) * sizeof(float)));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, search_stream_));
    }
    if (need <= scratch_cap_) return;

    size_t new_cap = std::max(need, scratch_cap_ * 2);
    new_cap = std::max(new_cap, size_t(4096));

    if (d_dist_scratch_)  { cudaFree(d_dist_scratch_);     d_dist_scratch_  = nullptr; }
    if (h_dist_scratch_)  { cudaFreeHost(h_dist_scratch_); h_dist_scratch_  = nullptr; }

    CUDA_CHECK(cudaMalloc    (&d_dist_scratch_,  new_cap * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_dist_scratch_,  new_cap * sizeof(float)));
    scratch_cap_ = new_cap;
}

void GpuClusterIndex::ensure_vecs_scratch_(size_t need_floats) const {
    if (need_floats <= vecs_packed_cap_) return;
    size_t new_cap = std::max(need_floats, vecs_packed_cap_ * 2);
    new_cap = std::max(new_cap, size_t(65536));

    if (d_vecs_packed_) { cudaFree(d_vecs_packed_); d_vecs_packed_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_vecs_packed_, new_cap * sizeof(float)));
    vecs_packed_cap_ = new_cap;
}

void GpuClusterIndex::ensure_queries_batch_(size_t need_floats) const {
    if (need_floats <= queries_batch_cap_) return;
    size_t new_cap = std::max(need_floats, queries_batch_cap_ * 2);
    new_cap = std::max(new_cap, size_t(4096));
    if (d_queries_batch_) { cudaFree(d_queries_batch_); d_queries_batch_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_queries_batch_, new_cap * sizeof(float)));
    queries_batch_cap_ = new_cap;
}

void GpuClusterIndex::ensure_qnorms_batch_(size_t need_floats) const {
    if (need_floats <= qnorms_batch_cap_) return;
    size_t new_cap = std::max(need_floats, qnorms_batch_cap_ * 2);
    new_cap = std::max(new_cap, size_t(256));
    if (d_qnorms_batch_) { cudaFree(d_qnorms_batch_); d_qnorms_batch_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_qnorms_batch_, new_cap * sizeof(float)));
    qnorms_batch_cap_ = new_cap;
}

void GpuClusterIndex::ensure_topk_scratch_(size_t need_slots) const {
    if (need_slots <= topk_cap_) return;
    size_t new_cap = std::max(need_slots, topk_cap_ * 2);
    new_cap = std::max(new_cap, size_t(1024));
    if (d_topk_scores_) { cudaFree(d_topk_scores_);     d_topk_scores_ = nullptr; }
    if (d_topk_vis_)    { cudaFree(d_topk_vis_);         d_topk_vis_    = nullptr; }
    if (h_topk_scores_) { cudaFreeHost(h_topk_scores_); h_topk_scores_ = nullptr; }
    if (h_topk_vis_)    { cudaFreeHost(h_topk_vis_);    h_topk_vis_    = nullptr; }
    CUDA_CHECK(cudaMalloc    (&d_topk_scores_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc    (&d_topk_vis_,    new_cap * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_topk_scores_, new_cap * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_topk_vis_,    new_cap * sizeof(int)));
    topk_cap_ = new_cap;
}

void* GpuClusterIndex::store_cluster(int cid,
                                      const DocId* ids,
                                      const float* vecs,
                                      size_t n)
{
    const size_t bytes = n * static_cast<size_t>(dim_) * sizeof(float);

    float* d_ptr = nullptr;
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, vecs, bytes, cudaMemcpyHostToDevice));
    }
    auto buf = std::make_shared<DeviceBuffer>(d_ptr, bytes);
    std::vector<DocId> h_ids(ids, ids + n);

    std::lock_guard<std::mutex> lk(mu_);
    ClusterData& cd = clusters_[cid];
    cd.vecs_buf = std::move(buf);
    cd.h_ids    = std::move(h_ids);
    cd.n        = n;
    return static_cast<void*>(cd.d_vecs());
}

bool GpuClusterIndex::remove_cluster(int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.erase(cid) > 0;
}

bool GpuClusterIndex::has_cluster(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.count(cid) > 0;
}

size_t GpuClusterIndex::cluster_size(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clusters_.find(cid);
    return (it != clusters_.end()) ? it->second.n : 0;
}

size_t GpuClusterIndex::num_clusters() const {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.size();
}

bool GpuClusterIndex::export_cluster(int cid,
                                      std::vector<DocId>& out_ids,
                                      std::vector<float>& out_vecs) const
{
    std::shared_ptr<DeviceBuffer> buf;
    std::vector<DocId> h_ids;
    size_t n;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return false;
        buf   = it->second.vecs_buf;
        h_ids = it->second.h_ids;
        n     = it->second.n;
    }
    if (!buf || !buf->ptr || n == 0) return false;

    const size_t bytes = n * static_cast<size_t>(dim_) * sizeof(float);
    out_vecs.resize(n * static_cast<size_t>(dim_));
    CUDA_CHECK(cudaMemcpy(out_vecs.data(), buf->ptr, bytes, cudaMemcpyDeviceToHost));
    out_ids = std::move(h_ids);
    return true;
}

size_t GpuClusterIndex::expand_cluster(int cid,
                                        const DocId* new_ids,
                                        const float* new_vecs,
                                        size_t n_new)
{
    if (n_new == 0 || !new_ids || !new_vecs) return 0;

    std::shared_ptr<DeviceBuffer> old_buf;
    std::vector<DocId>            old_ids;
    size_t                        n_old = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return 0;
        old_buf = it->second.vecs_buf;
        old_ids = it->second.h_ids;
        n_old   = it->second.n;
    }

    const size_t dim_sz    = static_cast<size_t>(dim_);
    const size_t n_total   = n_old + n_new;
    const size_t new_bytes = n_total * dim_sz * sizeof(float);

    float* d_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new, new_bytes));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (n_old > 0 && old_buf && old_buf->ptr)
        CUDA_CHECK(cudaMemcpyAsync(d_new, old_buf->ptr,
                                   n_old * dim_sz * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_new + n_old * dim_sz, new_vecs,
                               n_new * dim_sz * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::vector<DocId> new_id_list = old_ids;
    new_id_list.insert(new_id_list.end(), new_ids, new_ids + n_new);

    auto new_buf = std::make_shared<DeviceBuffer>(d_new, new_bytes);
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return 0;
        it->second.vecs_buf = std::move(new_buf);
        it->second.h_ids    = std::move(new_id_list);
        it->second.n        = n_total;
    }
    return n_new;
}

// ─────────────────────────────────────────────────────────────────────────────
// search_cluster
// ─────────────────────────────────────────────────────────────────────────────
//
// FIX: eliminated per-call cudaStreamCreate / cudaMalloc×2 / cudaFree×2 /
// cudaStreamDestroy.  The stream, d_query_scratch_, and d_dist_scratch_ are
// now persistent, allocated once via ensure_scratch_ and reused across calls.
// scratch_mu_ is held for the full duration of this call (same as
// collaborative_search) so the two paths cannot race on the shared buffers.
//
// topk_smallest (nth_element O(N) + sort O(k log k)) is kept unchanged —
// it is already asymptotically optimal for post-hoc selection on a flat
// distance array that is fully resident in h_dist_scratch_.  Replacing it
// with a heap (O(N log k)) would be strictly slower here.

size_t GpuClusterIndex::search_cluster(int cid,
                                        const float* query,
                                        int k,
                                        std::vector<DocId>&  out_ids,
                                        std::vector<float>&  out_scores,
                                        GpuSearchTiming*     timing) const
{
    // Snapshot (brief lock).
    std::shared_ptr<DeviceBuffer> buf;
    std::vector<DocId>            h_ids;
    size_t                        n;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return 0;
        buf   = it->second.vecs_buf;
        h_ids = it->second.h_ids;
        n     = it->second.n;
    }
    if (!buf || !buf->ptr || n == 0) return 0;

    const size_t dim_sz = static_cast<size_t>(dim_);

    // Acquire persistent scratch — no per-call stream/malloc/free.
    // scratch_mu_ is held for the duration so search_cluster and
    // collaborative_search do not race on the shared buffers.
    std::lock_guard<std::mutex> slk(scratch_mu_);
    { auto _t = clock_t_::now();
      ensure_scratch_(n);   // grows d_dist_scratch_ / h_dist_scratch_ if needed
      if (timing) timing->malloc_ms += fms_(clock_t_::now() - _t); }

    // H2D: copy query into persistent d_query_scratch_.
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaMemcpyAsync(d_query_scratch_, query,
                                 dim_sz * sizeof(float),
                                 cudaMemcpyHostToDevice, search_stream_));
      if (timing) timing->h2d_ms += fms_(clock_t_::now() - _t); }

    // Launch distance kernel into persistent search_stream_.
    { auto _t = clock_t_::now();
      launch_distance_kernel(metric_, d_query_scratch_, buf->ptr, d_dist_scratch_,
                             static_cast<int>(n), dim_, search_stream_);
      if (timing) timing->kernel_ms += fms_(clock_t_::now() - _t); }

    // D2H into persistent pinned h_dist_scratch_.
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaMemcpyAsync(h_dist_scratch_, d_dist_scratch_,
                                 n * sizeof(float),
                                 cudaMemcpyDeviceToHost, search_stream_));
      if (timing) timing->d2h_ms += fms_(clock_t_::now() - _t); }

    // Sync — actual GPU execution time lands here.
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaStreamSynchronize(search_stream_));
      if (timing) timing->sync_ms += fms_(clock_t_::now() - _t); }

    // CPU top-k: nth_element O(N) + sort O(k log k) — optimal for flat array.
    auto _t_topk = clock_t_::now();
    std::vector<Pair> cands;
    cands.reserve(n);
    for (size_t i = 0; i < n; ++i)
        cands.push_back(Pair{h_dist_scratch_[i], h_ids[i]});
    topk_smallest(cands, k);
    if (timing) timing->topk_ms += fms_(clock_t_::now() - _t_topk);

    out_ids.reserve(out_ids.size() + cands.size());
    out_scores.reserve(out_scores.size() + cands.size());
    for (const auto& p : cands) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return cands.size();
}

// ─────────────────────────────────────────────────────────────────────────────
// collaborative_search_batch
// ─────────────────────────────────────────────────────────────────────────────
//
// Batched version of collaborative_search. All q_rows queries are processed in
// one SGEMM (one H2D, one cuBLAS call, one sync, one D2H) instead of q_rows
// serial SGEMV calls. Reduces GPU round-trip overhead from O(q_rows) to O(1).
//
// Distance matrix layout (cuBLAS col-major output):
//   d_dist_scratch_[vi + qi * total_n] = dot(V[vi], Q[qi])
// After L2 correction:
//   d_dist_scratch_[vi + qi * total_n] = ||V[vi]-Q[qi]||²
//
// Per-query top-k: for query qi, its cluster data spans h_dist_scratch_[qi*total_n
// + s.offset .. qi*total_n + s.offset + s.n - 1] — contiguous, cache-friendly.

size_t GpuClusterIndex::collaborative_search_batch(
        const std::vector<std::vector<int>>& per_query_gpu_cids,
        const float* queries,
        size_t q_rows,
        int k,
        const ClusterInsertBuffer& buf,
        std::vector<std::vector<DocId>>&  out_ids,
        std::vector<std::vector<float>>&  out_scores,
        GpuCollabTiming* timing) const
{
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (q_rows == 0) return 0;

    const size_t dim_sz = static_cast<size_t>(dim_);

    // ── 1. Build union of all GPU clusters across all queries ─────────────
    struct Snap {
        std::shared_ptr<DeviceBuffer> vbuf;
        std::vector<DocId>            h_ids;
        size_t                        n;
        size_t                        offset; // offset in d_vecs_packed_ (in vectors)
        int                           cid;
    };
    std::vector<Snap> snaps;
    std::unordered_map<int, size_t> cid_to_snap_idx;
    size_t total_n = 0;

    for (size_t qi = 0; qi < q_rows; ++qi) {
        for (int cid : per_query_gpu_cids[qi]) {
            if (cid_to_snap_idx.count(cid)) continue;
            std::shared_ptr<DeviceBuffer> vbuf;
            std::vector<DocId> h_ids;
            size_t n = 0;
            {
                std::lock_guard<std::mutex> lk(mu_);
                auto it = clusters_.find(cid);
                if (it == clusters_.end()) continue;
                vbuf  = it->second.vecs_buf;
                h_ids = it->second.h_ids;
                n     = it->second.n;
            }
            if (!vbuf || !vbuf->ptr || n == 0) continue;
            cid_to_snap_idx[cid] = snaps.size();
            snaps.push_back({std::move(vbuf), std::move(h_ids), n, total_n, cid});
            total_n += n;
        }
    }

    // ── 2. Acquire scratch ────────────────────────────────────────────────
    std::lock_guard<std::mutex> slk(scratch_mu_);

    auto t_h2d = clock_t_::now();

    if (total_n > 0) {
        // ensure_scratch_ also initialises search_stream_ and cublas_handle_.
        // d_dist_scratch_ holds the full total_n × q_rows dot matrix on GPU
        // but only k × q_rows results cross PCIe (via d_topk_*).
        // h_dist_scratch_ is only used by search_cluster; not needed here.
        ensure_scratch_(total_n * q_rows);
        ensure_vecs_scratch_(total_n * dim_sz);
        ensure_queries_batch_(q_rows * dim_sz);
        ensure_qnorms_batch_(q_rows);
        ensure_topk_scratch_(static_cast<size_t>(k) * q_rows);

        // ── 3. H2D all query vectors ──────────────────────────────────────
        CUDA_CHECK(cudaMemcpyAsync(d_queries_batch_, queries,
                                   q_rows * dim_sz * sizeof(float),
                                   cudaMemcpyHostToDevice, search_stream_));

        // ── 4. Pack all cluster vecs contiguously (D2D) ───────────────────
        for (const auto& s : snaps) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_vecs_packed_ + s.offset * dim_sz,
                s.vbuf->ptr,
                s.n * dim_sz * sizeof(float),
                cudaMemcpyDeviceToDevice, search_stream_));
        }

        // ── 5. Query norms on GPU ─────────────────────────────────────────
        // Replaces host loop + cudaMemcpyAsync(d_qnorms_batch_).
        // Launched after H2D so d_queries_batch_ is ready; overlaps D2D on
        // same stream (both complete before SGEMM below starts).
        {
            const int thr_qn = 256;
            k_query_norms<<<static_cast<int>(q_rows), thr_qn,
                            thr_qn * sizeof(float), search_stream_>>>(
                d_queries_batch_, d_qnorms_batch_,
                static_cast<int>(q_rows), static_cast<int>(dim_sz));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    auto t_kernels = clock_t_::now();

    if (total_n > 0) {
        // ── 6. Batched SGEMM ──────────────────────────────────────────────
        // C[vi + qi*total_n] = dot(V[vi], Q[qi])  (col-major)
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(cublas_handle_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(total_n),
            static_cast<int>(q_rows),
            static_cast<int>(dim_sz),
            &alpha,
            d_vecs_packed_,   static_cast<int>(dim_sz),
            d_queries_batch_, static_cast<int>(dim_sz),
            &beta,
            d_dist_scratch_,  static_cast<int>(total_n)));

        const int thr = 256;

        if (metric_ == Metric::L2) {
            // ── 7a. Fused vnorm + L2 correction ──────────────────────────
            // Single kernel replaces k_squared_norms + k_l2_from_ip_batch.
            // vnorm is computed inline (one re-read of vecs); eliminates
            // d_norms_scratch_ allocation and k_squared_norms launch entirely.
            const int total_elems = static_cast<int>(total_n * q_rows);
            const int blk_e = (total_elems + thr - 1) / thr;
            k_l2_fused<<<blk_e, thr, 0, search_stream_>>>(
                d_dist_scratch_, d_vecs_packed_, d_qnorms_batch_,
                static_cast<int>(total_n), static_cast<int>(q_rows),
                static_cast<int>(dim_sz));
            CUDA_CHECK(cudaGetLastError());
        } else {
            // IP metric: negate all distances
            const float neg_one = -1.f;
            CUBLAS_CHECK(cublasSscal(cublas_handle_,
                static_cast<int>(total_n * q_rows), &neg_one, d_dist_scratch_, 1));
        }

        // ── 7b. GPU top-k ─────────────────────────────────────────────────
        // Reduces total_n × q_rows → k × q_rows on GPU before D2H.
        // PCIe transfer: k×q_rows×8B instead of total_n×q_rows×4B.
        if (k > 256)
            throw std::runtime_error("[M3] collaborative_search_batch: k must be <= 256");
        {
            const int thr_tk = 256;
            const size_t smem_tk = static_cast<size_t>(k) * (sizeof(float) + sizeof(int));
            k_topk_per_query<<<static_cast<int>(q_rows), thr_tk, smem_tk, search_stream_>>>(
                d_dist_scratch_, d_topk_scores_, d_topk_vis_,
                static_cast<int>(total_n), static_cast<int>(q_rows), k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    auto t_sync = clock_t_::now();

    // ── 8. Single sync + compact D2H ──────────────────────────────────────
    // Transfer only k × q_rows floats + k × q_rows ints across PCIe.
    const size_t topk_slots = static_cast<size_t>(k) * q_rows;
    if (total_n > 0) {
        CUDA_CHECK(cudaStreamSynchronize(search_stream_));
        CUDA_CHECK(cudaMemcpy(h_topk_scores_, d_topk_scores_,
                              topk_slots * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_topk_vis_,    d_topk_vis_,
                              topk_slots * sizeof(int),   cudaMemcpyDeviceToHost));
    }

    auto t_topk = clock_t_::now();

    // ── 9. Per-query CPU merge: resolve vi→DocId + insert buffer ──────────
    // h_topk_scores_[qi*k + r] and h_topk_vis_[qi*k + r] hold up to k GPU
    // candidates per query (slots with vi == -1 are unfilled: total_n < k).
    auto heap_cmp = [](const Pair& a, const Pair& b){ return a.score < b.score; };
    size_t total_results = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:total_results) if(q_rows > 1)
    for (int qi_int = 0; qi_int < static_cast<int>(q_rows); ++qi_int) {
        const size_t qi = static_cast<size_t>(qi_int);
        if (per_query_gpu_cids[qi].empty()) continue;

        std::vector<Pair> heap;
        heap.reserve(static_cast<size_t>(k) + 1);
        std::unordered_set<DocId> seen;
        seen.reserve(static_cast<size_t>(k) * 2);

        // Absorb GPU top-k candidates
        if (total_n > 0) {
            const float* gpu_scores = h_topk_scores_ + qi * k;
            const int*   gpu_vis    = h_topk_vis_    + qi * k;
            for (int r = 0; r < k; ++r) {
                const int   vi = gpu_vis[r];
                const float sc = gpu_scores[r];
                if (vi < 0) continue;  // unfilled slot

                // Resolve vi (packed index) → DocId via snap table
                DocId id = 0;
                bool found = false;
                for (const auto& s : snaps) {
                    if (vi >= static_cast<int>(s.offset) &&
                        vi <  static_cast<int>(s.offset + s.n)) {
                        id    = s.h_ids[vi - static_cast<int>(s.offset)];
                        found = true;
                        break;
                    }
                }
                if (!found) continue;

                if (seen.count(id)) continue;
                heap.push_back({sc, id});
                std::push_heap(heap.begin(), heap.end(), heap_cmp);
                seen.insert(id);
            }
        }

        // Insert buffer scan (staged-but-not-yet-flushed vectors)
        for (int cid : per_query_gpu_cids[qi]) {
            if (buf.has_cluster(cid)) {
                std::vector<DocId> b_ids;
                std::vector<float> b_scores;
                buf.scan_insert_buffer(cid, queries + qi * dim_sz, k,
                                       metric_, normalized_, b_ids, b_scores);
                for (size_t i = 0; i < b_ids.size(); ++i) {
                    const float  sc = b_scores[i];
                    const DocId  id = b_ids[i];
                    if (seen.count(id)) continue;
                    if ((int)heap.size() < k) {
                        heap.push_back({sc, id});
                        std::push_heap(heap.begin(), heap.end(), heap_cmp);
                        seen.insert(id);
                    } else if (sc < heap.front().score) {
                        seen.erase(heap.front().id);
                        std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                        heap.back() = {sc, id};
                        std::push_heap(heap.begin(), heap.end(), heap_cmp);
                        seen.insert(id);
                    }
                }
            }
        }

        std::sort_heap(heap.begin(), heap.end(), heap_cmp);
        out_ids[qi].reserve(heap.size());
        out_scores[qi].reserve(heap.size());
        for (const auto& p : heap) {
            out_ids[qi].push_back(p.id);
            out_scores[qi].push_back(p.score);
        }
        total_results += heap.size();
    }

    auto t_end = clock_t_::now();

    if (timing) {
        timing->h2d_ms      += fms_(t_kernels - t_h2d);
        timing->kernel_ms   += fms_(t_sync    - t_kernels);
        timing->sync_d2h_ms += fms_(t_topk    - t_sync);
        timing->topk_ms     += fms_(t_end     - t_topk);
    }

    if (gpu_diag_enabled() && !snaps.empty()) {
        fprintf(stderr,
            "[M3_GPU_DIAG] clusters=%d  total_vecs=%zu  k=%d  q_rows=%zu  "
            "pcie_before=%.1fKB  pcie_after=%.1fKB  "
            "h2d=%.2fms  kernels=%.2fms  sync+d2h=%.2fms  topk_cpu=%.2fms  TOTAL=%.2fms\n",
            (int)snaps.size(), total_n, k, q_rows,
            (total_n * q_rows * sizeof(float)) / 1024.0,
            (topk_slots * (sizeof(float) + sizeof(int))) / 1024.0,
            fms_(t_kernels - t_h2d),
            fms_(t_sync    - t_kernels),
            fms_(t_topk    - t_sync),
            fms_(t_end     - t_topk),
            fms_(t_end     - t_h2d));
        fflush(stderr);
    }

    return total_results;
}

} // namespace m3
