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

#include "gpu_cluster_index.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace {
// Returns true once per process (checked at first collaborative_search call).
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
// CUDA error helper
// ─────────────────────────────────────────────────────────────────────────────

static void cuda_check_(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[M3 CUDA] ") + cudaGetErrorString(err) +
            " at " + file + ":" + std::to_string(line));
    }
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
// Distance kernels
// ─────────────────────────────────────────────────────────────────────────────

// One thread per candidate vector.
// The query is loaded cooperatively into shared memory so each block reads it
// once from L1 cache rather than having every thread hit global memory.

__global__ void k_l2_distances(
        const float* __restrict__ query,   // [dim]  (device)
        const float* __restrict__ vecs,    // [n × dim] row-major (device)
        float*       __restrict__ dists,   // [n] output (device)
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

// Inner-product: negate so "smallest score = best match" is consistent with L2.
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
    dists[i] = -acc;   // negate: top-k smallest = highest inner product
}

// Compute squared L2 norm for each of N row-vectors of dimension dim.
// Used to convert cuBLAS inner-product output to L2 distances.
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

// Convert inner-product output to squared L2 distance in-place:
//   dists[i] = qnorm + norms[i] - 2 * dists[i]
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
    const int  threads  = 256;
    const int  blocks   = (n + threads - 1) / threads;
    const size_t smem   = static_cast<size_t>(dim) * sizeof(float);

    if (metric == Metric::L2) {
        k_l2_distances<<<blocks, threads, smem, stream>>>(
            d_query, d_vecs, d_dists, n, dim);
    } else {
        k_ip_distances<<<blocks, threads, smem, stream>>>(
            d_query, d_vecs, d_dists, n, dim);
    }
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

// ensure_scratch_ — lazily create persistent stream/cuBLAS handle/query buffer;
// grow distance and norms scratch to hold at least `need` floats.
// Must be called under scratch_mu_.
void GpuClusterIndex::ensure_scratch_(size_t need) const {
    // One-time: create stream, query buffer, and cuBLAS handle.
    if (!search_stream_) {
        CUDA_CHECK(cudaStreamCreate(&search_stream_));
        CUDA_CHECK(cudaMalloc(&d_query_scratch_,
                              static_cast<size_t>(dim_) * sizeof(float)));
        CUBLAS_CHECK(cublasCreate(&cublas_handle_));
        CUBLAS_CHECK(cublasSetStream(cublas_handle_, search_stream_));
    }
    if (need <= scratch_cap_) return;

    // Grow to at least 2× current capacity (minimum 4096 floats = 16 KB).
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

// ensure_vecs_scratch_ — grow the contiguous device packing buffer.
// Must be called under scratch_mu_.
void GpuClusterIndex::ensure_vecs_scratch_(size_t need_floats) const {
    if (need_floats <= vecs_packed_cap_) return;
    size_t new_cap = std::max(need_floats, vecs_packed_cap_ * 2);
    new_cap = std::max(new_cap, size_t(65536)); // minimum 256 KB

    if (d_vecs_packed_) { cudaFree(d_vecs_packed_); d_vecs_packed_ = nullptr; }
    CUDA_CHECK(cudaMalloc(&d_vecs_packed_, new_cap * sizeof(float)));
    vecs_packed_cap_ = new_cap;
}

// store_cluster — H2D upload; replaces any existing data for cid.
// Returns the device pointer as an opaque handle for GpuBudgetManager.
void* GpuClusterIndex::store_cluster(int cid,
                                      const DocId* ids,
                                      const float* vecs,
                                      size_t n)
{
    const size_t bytes = n * static_cast<size_t>(dim_) * sizeof(float);

    // Allocate device buffer and upload (outside lock — H2D can be slow).
    float* d_ptr = nullptr;
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
        CUDA_CHECK(cudaMemcpy(d_ptr, vecs, bytes, cudaMemcpyHostToDevice));
    }
    auto buf = std::make_shared<DeviceBuffer>(d_ptr, bytes);

    // Build host ID list.
    std::vector<DocId> h_ids(ids, ids + n);

    // Install under lock (brief).
    std::lock_guard<std::mutex> lk(mu_);
    ClusterData& cd = clusters_[cid];
    cd.vecs_buf = std::move(buf);
    cd.h_ids    = std::move(h_ids);
    cd.n        = n;
    return static_cast<void*>(cd.d_vecs());
}

// remove_cluster — releases the shared_ptr; cudaFree fires when ref-count hits 0.
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

// export_cluster — D2H copy of the device vectors + host IDs.
bool GpuClusterIndex::export_cluster(int cid,
                                      std::vector<DocId>& out_ids,
                                      std::vector<float>& out_vecs) const
{
    // Snapshot shared_ptr and metadata under brief lock.
    std::shared_ptr<DeviceBuffer> buf;
    std::vector<DocId> h_ids;
    size_t n;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return false;
        buf    = it->second.vecs_buf;
        h_ids  = it->second.h_ids;
        n      = it->second.n;
    }

    if (!buf || !buf->ptr || n == 0) return false;

    const size_t bytes = n * static_cast<size_t>(dim_) * sizeof(float);
    out_vecs.resize(n * static_cast<size_t>(dim_));
    CUDA_CHECK(cudaMemcpy(out_vecs.data(), buf->ptr, bytes, cudaMemcpyDeviceToHost));
    out_ids = std::move(h_ids);
    return true;
}
// expand_cluster — zero-downtime expansion.
//
//   Old cluster stays searchable throughout; new buffer is built in parallel,
//   then atomically swapped in. Old DeviceBuffer destructs (→ cudaFree) only
//   after all shared_ptr holders (including in-flight searches) release it.
//
//   Transfer order (single stream):
//     1. cudaMemcpy D2D  : existing GPU vectors → new buffer head
//     2. cudaMemcpyAsync H2D : CPU insert-buffer vectors → new buffer tail
//   Both complete before the atomic swap.
size_t GpuClusterIndex::expand_cluster(int cid,
                                        const DocId* new_ids,
                                        const float* new_vecs,
                                        size_t n_new)
{
    if (n_new == 0 || !new_ids || !new_vecs) return 0;

    // ── 1. Snapshot old cluster (brief lock) ──────────────────────────────
    std::shared_ptr<DeviceBuffer> old_buf;
    std::vector<DocId>            old_ids;
    size_t                        n_old = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) return 0;
        old_buf  = it->second.vecs_buf;   // hold ref — keeps old data alive
        old_ids  = it->second.h_ids;
        n_old    = it->second.n;
    }  // lock released — old cluster still searchable

    const size_t dim_sz    = static_cast<size_t>(dim_);
    const size_t n_total   = n_old + n_new;
    const size_t new_bytes = n_total * dim_sz * sizeof(float);

    // ── 2. Allocate new device buffer ────────────────────────────────────
    float* d_new = nullptr;
    CUDA_CHECK(cudaMalloc(&d_new, new_bytes));

    // ── 3 & 4. Transfer on a dedicated stream ────────────────────────────
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (n_old > 0 && old_buf && old_buf->ptr) {
        // GPU→GPU: existing vectors into the head of the new buffer.
        CUDA_CHECK(cudaMemcpyAsync(d_new,
                                   old_buf->ptr,
                                   n_old * dim_sz * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // CPU→GPU: insert-buffer vectors into the tail (async on same stream).
    CUDA_CHECK(cudaMemcpyAsync(d_new + n_old * dim_sz,
                               new_vecs,
                               n_new * dim_sz * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // ── 5. Wait for all transfers ─────────────────────────────────────────
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // ── 6. Build new host ID list (CPU side) ──────────────────────────────
    std::vector<DocId> new_id_list = old_ids;
    new_id_list.insert(new_id_list.end(), new_ids, new_ids + n_new);

    // ── 7. Atomic swap under lock ─────────────────────────────────────────
    auto new_buf = std::make_shared<DeviceBuffer>(d_new, new_bytes);
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = clusters_.find(cid);
        if (it == clusters_.end()) {
            // Cluster was removed concurrently — discard the new buffer.
            // DeviceBuffer dtor will cudaFree d_new.
            return 0;
        }
        it->second.vecs_buf = std::move(new_buf);
        it->second.h_ids    = std::move(new_id_list);
        it->second.n        = n_total;
    }

    // ── 8. Release old buffer ─────────────────────────────────────────────
    // old_buf shared_ptr goes out of scope here. If no concurrent search holds
    // a reference, DeviceBuffer dtor fires immediately → cudaFree(old ptr).
    // If a search is still using it, the free is deferred to that search's end.

    return n_new;
}

// search_cluster — GPU distance kernel + CPU top-k.
//   Mutex held only for the brief snapshot; kernel runs outside the lock.
//   If `timing` is non-null, per-phase wall-clock times (ms) are written into it.
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
        buf    = it->second.vecs_buf;
        h_ids  = it->second.h_ids;
        n      = it->second.n;
    }
    if (!buf || !buf->ptr || n == 0) return 0;

    const size_t dim_sz = static_cast<size_t>(dim_);

    // cudaStreamCreate
    cudaStream_t stream;
    { auto _t = clock_t_::now(); CUDA_CHECK(cudaStreamCreate(&stream));
      if (timing) timing->stream_create_ms += fms_(clock_t_::now() - _t); }

    // cudaMalloc x2
    float *d_query = nullptr, *d_dists = nullptr;
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaMalloc(&d_query, dim_sz * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_dists, n * sizeof(float)));
      if (timing) timing->malloc_ms += fms_(clock_t_::now() - _t); }

    // H2D: copy query
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaMemcpyAsync(d_query, query, dim_sz * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      if (timing) timing->h2d_ms += fms_(clock_t_::now() - _t); }

    // Launch distance kernel
    std::vector<float> h_dists(n);
    { auto _t = clock_t_::now();
      launch_distance_kernel(metric_, d_query, buf->ptr, d_dists,
                             static_cast<int>(n), dim_, stream);
      if (timing) timing->kernel_ms += fms_(clock_t_::now() - _t); }

    // D2H distances
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaMemcpyAsync(h_dists.data(), d_dists, n * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
      if (timing) timing->d2h_ms += fms_(clock_t_::now() - _t); }

    // Stream sync
    { auto _t = clock_t_::now();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (timing) timing->sync_ms += fms_(clock_t_::now() - _t); }

    // Free + stream destroy
    { auto _t = clock_t_::now();
      cudaFree(d_query);
      cudaFree(d_dists);
      cudaStreamDestroy(stream);
      if (timing) timing->free_ms += fms_(clock_t_::now() - _t); }

    // Top-k on CPU using the snapshotted host IDs.
    auto _t_topk = clock_t_::now();
    std::vector<Pair> cands;
    cands.reserve(n);
    for (size_t i = 0; i < n; ++i)
        cands.push_back(Pair{h_dists[i], h_ids[i]});
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

// collaborative_search — batched GPU cluster scan + CPU insert-buffer scan.
//
// Optimisation pipeline:
//   1. Snapshot all cluster VRAM pointers under one brief lock pass.
//   2. Upload query once (H2D) to persistent d_query_scratch_.
//   3. D2D-pack all cluster vectors into contiguous d_vecs_packed_ (GPU→GPU,
//      same stream, ~400 GB/s — avoids host round-trip).
//   4. cuBLAS SGEMV: one highly-parallelised matrix-vector multiply computes
//      all inner products in a single kernel launch (replaces N tiny kernels).
//   5. Two small correction kernels convert inner products → L2 distances
//      (or cublasSscal negates for IP/cosine metric).
//   6. One cudaStreamSynchronize + one D2H into pinned h_dist_scratch_.
//   7. CPU top-k per cluster, then merge + insert-buffer scan.
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
        size_t                        offset; // offset into scratch buffer
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
        // 4a. Pack all cluster VRAM buffers into one contiguous slab (D2D, same stream).
        ensure_vecs_scratch_(total_n * dim_sz);
        for (const auto& s : snaps) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_vecs_packed_ + s.offset * dim_sz,
                s.vbuf->ptr,
                s.n * dim_sz * sizeof(float),
                cudaMemcpyDeviceToDevice, search_stream_));
        }

        // 4b. SGEMV: y[i] = dot(vec_i, query) for all i in [0, total_n).
        //     d_vecs_packed_ is row-major [total_n × dim], which cuBLAS sees as
        //     a column-major [dim × total_n] matrix (lda = dim).
        //     OP_T transposes it → y = A^T * x computes one dot-product per row.
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle_, CUBLAS_OP_T,
            static_cast<int>(dim_sz), static_cast<int>(total_n),
            &alpha, d_vecs_packed_, static_cast<int>(dim_sz),
            d_query_scratch_, 1,
            &beta, d_dist_scratch_, 1));

        // 4c. Convert inner products to the requested distance metric.
        const int thr = 256;
        const int blk = (static_cast<int>(total_n) + thr - 1) / thr;
        if (metric_ == Metric::L2) {
            // ||q - v||² = ||q||² + ||v||² - 2·q·v
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
            // IP / cosine: negate so smallest score = best match.
            const float neg_one = -1.f;
            CUBLAS_CHECK(cublasSscal(cublas_handle_,
                static_cast<int>(total_n), &neg_one, d_dist_scratch_, 1));
        }
    }

    auto t_sync = clock_t_::now();

    // ── 5. Single sync + single D2H (pinned — no staging latency) ─────────
    if (total_n > 0) {
        CUDA_CHECK(cudaStreamSynchronize(search_stream_));
        CUDA_CHECK(cudaMemcpy(h_dist_scratch_, d_dist_scratch_,
                              total_n * sizeof(float), cudaMemcpyDeviceToHost));
    }

    auto t_topk = clock_t_::now();

    // ── 6. CPU top-k per cluster, then merge ─────────────────────────────
    std::unordered_map<DocId, float> best;
    for (const auto& s : snaps) {
        const float* dists = h_dist_scratch_ + s.offset;
        std::vector<Pair> cands;
        cands.reserve(s.n);
        for (size_t i = 0; i < s.n; ++i)
            cands.push_back(Pair{dists[i], s.h_ids[i]});
        topk_smallest(cands, k);
        for (const auto& p : cands) {
            auto [it, ins] = best.emplace(p.id, p.score);
            if (!ins && p.score < it->second) it->second = p.score;
        }
    }

    // ── 7. CPU insert-buffer scan (unchanged) ─────────────────────────────
    for (int cid : probe_cids) {
        if (buf.has_cluster(cid)) {
            std::vector<DocId>  b_ids;
            std::vector<float>  b_scores;
            buf.scan_insert_buffer(cid, query, k, metric_, normalized_,
                                   b_ids, b_scores);
            for (size_t i = 0; i < b_ids.size(); ++i) {
                auto [it, ins] = best.emplace(b_ids[i], b_scores[i]);
                if (!ins && b_scores[i] < it->second) it->second = b_scores[i];
            }
        }
    }

    auto t_end = clock_t_::now();

    if (timing) {
        timing->h2d_ms    += fms_(t_kernels - t_h2d);
        timing->kernel_ms += fms_(t_sync    - t_kernels);
        timing->sync_d2h_ms += fms_(t_topk  - t_sync);
        timing->topk_ms   += fms_(t_end     - t_topk);
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

    // ── 8. Convert map -> sorted top-k ───────────────────────────────────
    std::vector<Pair> all;
    all.reserve(best.size());
    for (const auto& [id, sc] : best)
        all.push_back(Pair{sc, id});
    topk_smallest(all, k);

    out_ids.clear();    out_ids.reserve(all.size());
    out_scores.clear(); out_scores.reserve(all.size());
    for (const auto& p : all) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return all.size();
}

} // namespace m3
