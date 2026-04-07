// gpu_cluster_index.cu — real CUDA implementation of GpuClusterIndex.
//
// Memory model
// ────────────
//   store_cluster()  : cudaMalloc + cudaMemcpy H2D.  Returns d_vecs as opaque
//                      handle for GpuBudgetManager (informational only;
//                      cudaFree is handled by ~DeviceBuffer via shared_ptr).
//   expand_cluster() : zero-downtime swap —
//                        1. read old ptr (brief lock)
//                        2. cudaMalloc new buffer (no lock)
//                        3. cudaMemcpy D2D  old→new  (GPU→GPU, no lock)
//                        4. cudaMemcpyAsync H2D  buffer→new  (no lock)
//                        5. cudaStreamSynchronize  (no lock)
//                        6. swap shared_ptr under lock  (brief)
//                        7. old DeviceBuffer destructs → cudaFree
//   remove_cluster() : erase map entry → shared_ptr ref-count → cudaFree.
//   export_cluster() : cudaMemcpy D2H (snapshot under brief lock).
//   search_cluster() : snapshot shared_ptr (brief lock), launch distance kernel
//                      outside lock, D2H distances, CPU top-k.
//
// Thread safety
// ─────────────
//   mu_ is held only briefly for pointer snapshots and atomic swaps, never
//   across CUDA kernel launches or synchronise calls.  Concurrent searches
//   are safe during expansion because the old DeviceBuffer stays alive via
//   shared_ptr ref-count until all snapshots release it.
//
// Optimisation notes (vs original)
// ─────────────────────────────────
//   search_cluster()      : FIXED — stream, d_query, and d_dists scratch
//                           are now persistent (shared with collaborative_search
//                           via ensure_scratch_). Per-call cudaStreamCreate /
//                           cudaMalloc×2 / cudaFree×2 / cudaStreamDestroy
//                           removed. topk_smallest (nth_element + sort) kept
//                           as-is — O(N) + O(k log k) is optimal for post-hoc
//                           flat-array selection; a heap would be O(N log k).
//   collaborative_search(): FIXED — replaced std::unordered_map<DocId,float>
//                           + final topk_smallest with a single streaming
//                           max-heap of size k maintained across all clusters.
//                           Eliminates: per-query map construction/destruction,
//                           O(k × num_clusters) hash insertions, map→vector
//                           conversion, and the second nth_element + sort pass.
//                           An unordered_set<DocId> handles dedup (insert-buffer
//                           vs flushed cluster overlap) at much lower cost than
//                           the full score-keyed map.

#include "gpu_cluster_index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <functional>   // std::greater
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
// Distance kernels  (unchanged)
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

__global__ void k_squared_norms(const float* __restrict__ vecs,
                                 float*       __restrict__ norms,
                                 int N, int dim)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* v = vecs + (size_t)i * dim;
    float acc = 0.f;
    for (int d = 0; d < dim; ++d) acc += v[d] * v[d];
    norms[i] = acc;
}

__global__ void k_l2_from_ip(float*       __restrict__ dists,
                               const float* __restrict__ norms,
                               float qnorm, int N)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    dists[i] = qnorm + norms[i] - 2.f * dists[i];
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
// GpuClusterIndex implementation
// ─────────────────────────────────────────────────────────────────────────────

GpuClusterIndex::GpuClusterIndex(int dim, Metric metric, bool normalized)
    : dim_(dim), metric_(metric), normalized_(normalized) {}

GpuClusterIndex::~GpuClusterIndex() {
    if (cublas_handle_)   { cublasDestroy(cublas_handle_);        cublas_handle_   = nullptr; }
    if (search_stream_)   { cudaStreamDestroy(search_stream_);    search_stream_   = nullptr; }
    if (d_query_scratch_) { cudaFree(d_query_scratch_);           d_query_scratch_ = nullptr; }
    if (d_dist_scratch_)  { cudaFree(d_dist_scratch_);            d_dist_scratch_  = nullptr; }
    if (d_norms_scratch_) { cudaFree(d_norms_scratch_);           d_norms_scratch_ = nullptr; }
    if (h_dist_scratch_)  { cudaFreeHost(h_dist_scratch_);        h_dist_scratch_  = nullptr; }
    if (d_vecs_packed_)   { cudaFree(d_vecs_packed_);             d_vecs_packed_   = nullptr; }
    scratch_cap_     = 0;
    vecs_packed_cap_ = 0;
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
    if (d_norms_scratch_) { cudaFree(d_norms_scratch_);    d_norms_scratch_ = nullptr; }
    if (h_dist_scratch_)  { cudaFreeHost(h_dist_scratch_); h_dist_scratch_  = nullptr; }

    CUDA_CHECK(cudaMalloc    (&d_dist_scratch_,  new_cap * sizeof(float)));
    CUDA_CHECK(cudaMalloc    (&d_norms_scratch_, new_cap * sizeof(float)));
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
// collaborative_search
// ─────────────────────────────────────────────────────────────────────────────
//
// FIX: replaced std::unordered_map<DocId,float> + final topk_smallest with a
// single streaming max-heap of size k maintained across all clusters.
//
// OLD cost per query:
//   - unordered_map constructed and destroyed every call
//   - k × num_clusters hash insertions (high constant factor, poor cache behaviour)
//   - map → vector conversion before final sort
//   - second nth_element + sort on merged results
//
// NEW cost per query:
//   - std::vector<Pair> pre-reserved to k+1 — one allocation, reused as heap
//   - O(log k) push_heap / pop_heap per candidate that beats current worst
//   - unordered_set<DocId> for dedup (insert-buffer vs flushed cluster overlap)
//     — membership-only, no score storage, much cheaper than full score map
//   - final std::sort_heap O(k log k) to produce sorted output — unavoidable

size_t GpuClusterIndex::collaborative_search(
        const std::vector<int>& probe_cids,
        const float* query,
        int k,
        const ClusterInsertBuffer& buf,
        std::vector<DocId>&  out_ids,
        std::vector<float>&  out_scores,
        GpuCollabTiming*     timing) const
{
    const bool diag = gpu_diag_enabled();

    // ── 1. Snapshot all resident clusters (brief lock per cluster) ────────
    struct Snap {
        std::shared_ptr<DeviceBuffer> vbuf;
        std::vector<DocId>            h_ids;
        size_t                        n;
        size_t                        offset;
        int                           cid;
    };
    std::vector<Snap> snaps;
    snaps.reserve(probe_cids.size());
    size_t total_n = 0;
    for (int cid : probe_cids) {
        std::shared_ptr<DeviceBuffer> vbuf;
        std::vector<DocId>            h_ids;
        size_t                        n = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = clusters_.find(cid);
            if (it == clusters_.end()) continue;
            vbuf  = it->second.vecs_buf;
            h_ids = it->second.h_ids;
            n     = it->second.n;
        }
        if (!vbuf || !vbuf->ptr || n == 0) continue;
        snaps.push_back({std::move(vbuf), std::move(h_ids), n, total_n, cid});
        total_n += n;
    }

    // ── 2. Acquire scratch and ensure capacity ────────────────────────────
    std::lock_guard<std::mutex> slk(scratch_mu_);
    if (total_n > 0) ensure_scratch_(total_n);

    auto t_h2d = clock_t_::now();

    // ── 3. Upload query once ──────────────────────────────────────────────
    const size_t dim_sz = static_cast<size_t>(dim_);
    if (total_n > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_query_scratch_, query,
                                   dim_sz * sizeof(float),
                                   cudaMemcpyHostToDevice, search_stream_));
    }

    auto t_kernels = clock_t_::now();

    // ── 4. cuBLAS SGEMV path ──────────────────────────────────────────────
    if (total_n > 0) {
        ensure_vecs_scratch_(total_n * dim_sz);
        for (const auto& s : snaps) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_vecs_packed_ + s.offset * dim_sz,
                s.vbuf->ptr,
                s.n * dim_sz * sizeof(float),
                cudaMemcpyDeviceToDevice, search_stream_));
        }

        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T,
            static_cast<int>(dim_sz), static_cast<int>(total_n),  // <- total_n = sum of ALL clusters' n
            &alpha, d_vecs_packed_, static_cast<int>(dim_sz),     // <- ALL clusters packed into one matrix
            d_query_scratch_, 1,
            &beta, d_dist_scratch_, 1));                           // <- distances for ALL clusters at once

        const int thr = 256;
        const int blk = (static_cast<int>(total_n) + thr - 1) / thr;
        if (metric_ == Metric::L2) {
            k_squared_norms<<<blk, thr, 0, search_stream_>>>(
                d_vecs_packed_, d_norms_scratch_,
                static_cast<int>(total_n), static_cast<int>(dim_sz));
            CUDA_CHECK(cudaGetLastError());

            float qnorm = 0.f;
            for (int d = 0; d < dim_; ++d) qnorm += query[d] * query[d];

            k_l2_from_ip<<<blk, thr, 0, search_stream_>>>(
                d_dist_scratch_, d_norms_scratch_, qnorm,
                static_cast<int>(total_n));
            CUDA_CHECK(cudaGetLastError());
        } else {
            const float neg_one = -1.f;
            CUBLAS_CHECK(cublasSscal(cublas_handle_,
                static_cast<int>(total_n), &neg_one, d_dist_scratch_, 1));
        }
    }

    auto t_sync = clock_t_::now();

    // ── 5. Single sync + single D2H (pinned) ─────────────────────────────
    if (total_n > 0) {
        CUDA_CHECK(cudaStreamSynchronize(search_stream_));
        CUDA_CHECK(cudaMemcpy(h_dist_scratch_, d_dist_scratch_,
                              total_n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    auto t_topk = clock_t_::now();

    // ── 6. Streaming max-heap top-k across all clusters ───────────────────
    //
    // heap is a max-heap (largest score at top) capped at k elements.
    // For each candidate: if heap not full → push; else if candidate beats
    // current worst (heap.front()) → evict worst, push candidate.
    // Result: heap holds the k smallest scores seen across all clusters.
    //
    // seen tracks DocIds already in the heap so that a DocId appearing in
    // both a flushed cluster and the insert buffer is not double-counted.
    // This is membership-only — no score storage — making it far cheaper
    // than the previous unordered_map<DocId,float>.

    auto cmp = [](const Pair& a, const Pair& b){ return a.score < b.score; };
    // max-heap: cmp reversed so heap.front() is the largest (worst) score.
    auto heap_cmp = [](const Pair& a, const Pair& b){ return a.score < b.score; };

    std::vector<Pair> heap;
    heap.reserve(static_cast<size_t>(k) + 1);
    std::unordered_set<DocId> seen;
    seen.reserve(static_cast<size_t>(k) * 2);

    // Stream linearly through contiguous pinned h_dist_scratch_ — cache friendly.
    for (const auto& s : snaps) {
        const float* dists = h_dist_scratch_ + s.offset;
        for (size_t i = 0; i < s.n; ++i) {
            const float  sc = dists[i];
            const DocId  id = s.h_ids[i];

            if (seen.count(id)) continue;  // already represented by better score

            if ((int)heap.size() < k) {
                heap.push_back({sc, id});
                std::push_heap(heap.begin(), heap.end(), heap_cmp);
                seen.insert(id);
            } else if (sc < heap.front().score) {
                // Evict current worst from heap and seen, insert better candidate.
                seen.erase(heap.front().id);
                std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                heap.back() = {sc, id};
                std::push_heap(heap.begin(), heap.end(), heap_cmp);
                seen.insert(id);
            }
        }
    }

    // ── 7. Insert-buffer scan ─────────────────────────────────────────────
    // Same heap + seen set — insert-buffer candidates participate in the same
    // global top-k, deduped against already-seen flushed-cluster DocIds.
    for (int cid : probe_cids) {
        if (buf.has_cluster(cid)) {
            std::vector<DocId>  b_ids;
            std::vector<float>  b_scores;
            buf.scan_insert_buffer(cid, query, k, metric_, normalized_,
                                   b_ids, b_scores);
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

    auto t_end = clock_t_::now();

    if (timing) {
        timing->h2d_ms      += fms_(t_kernels - t_h2d);
        timing->kernel_ms   += fms_(t_sync    - t_kernels);
        timing->sync_d2h_ms += fms_(t_topk    - t_sync);
        timing->topk_ms     += fms_(t_end     - t_topk);
    }

    if (diag && !snaps.empty()) {
        fprintf(stderr,
            "[M3_GPU_DIAG] clusters=%d  total_vecs=%zu  "
            "h2d=%.2fms  kernels=%.2fms  sync+d2h=%.2fms  topk=%.2fms  TOTAL=%.2fms\n",
            (int)snaps.size(), total_n,
            fms_(t_kernels - t_h2d),
            fms_(t_sync    - t_kernels),
            fms_(t_topk    - t_sync),
            fms_(t_end     - t_topk),
            fms_(t_end     - t_h2d));
        fflush(stderr);
    }

    // ── 8. Heap → sorted output ───────────────────────────────────────────
    // sort_heap produces ascending order (smallest score first).
    std::sort_heap(heap.begin(), heap.end(), heap_cmp);

    out_ids.clear();    out_ids.reserve(heap.size());
    out_scores.clear(); out_scores.reserve(heap.size());
    for (const auto& p : heap) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return heap.size();
}

} // namespace m3
