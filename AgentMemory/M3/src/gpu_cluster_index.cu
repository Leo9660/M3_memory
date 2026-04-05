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

#include <chrono>
#include <cstdio>
#include <cstdlib>
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

// collaborative_search — GPU cluster scan + CPU insert-buffer scan, merged.
//
// The insert buffer is only active for GPU-resident clusters (activated on
// promotion, deactivated on eviction). For any cid in probe_cids, if there is
// no active buffer slot, buf.scan_insert_buffer() returns 0 immediately, so the
// loop is always correct. The explicit has_cluster() guard makes the intent
// visible and avoids the hash-table lookup for non-resident cids.
size_t GpuClusterIndex::collaborative_search(
        const std::vector<int>& probe_cids,
        const float* query,
        int k,
        const ClusterInsertBuffer& buf,
        std::vector<DocId>&  out_ids,
        std::vector<float>&  out_scores) const
{
    const bool diag = gpu_diag_enabled();
    GpuSearchTiming agg;   // accumulated over all clusters in this call
    int n_clusters_searched = 0;

    std::unordered_map<DocId, float> best;

    for (int cid : probe_cids) {
        // ── GPU path: linear scan of VRAM-resident vectors ──────────────
        std::vector<DocId>  g_ids;
        std::vector<float>  g_scores;
        search_cluster(cid, query, k, g_ids, g_scores, diag ? &agg : nullptr);
        ++n_clusters_searched;
        for (size_t i = 0; i < g_ids.size(); ++i) {
            auto [it, ins] = best.emplace(g_ids[i], g_scores[i]);
            if (!ins && g_scores[i] < it->second)
                it->second = g_scores[i];
        }

        // ── CPU insert-buffer path: scan staged inserts ──────────────────
        // Only GPU-resident clusters have an active buffer slot; check
        // explicitly so non-resident cids skip the lookup.
        if (buf.has_cluster(cid)) {
            std::vector<DocId>  b_ids;
            std::vector<float>  b_scores;
            buf.scan_insert_buffer(cid, query, k, metric_, normalized_,
                                   b_ids, b_scores);
            for (size_t i = 0; i < b_ids.size(); ++i) {
                auto [it, ins] = best.emplace(b_ids[i], b_scores[i]);
                if (!ins && b_scores[i] < it->second)
                    it->second = b_scores[i];
            }
        }
    }

    // Diagnostic: print per-phase breakdown aggregated over all clusters.
    if (diag && n_clusters_searched > 0) {
        double total = agg.stream_create_ms + agg.malloc_ms + agg.h2d_ms
                     + agg.kernel_ms + agg.d2h_ms + agg.sync_ms
                     + agg.free_ms + agg.topk_ms;
        fprintf(stderr,
            "[M3_GPU_DIAG] clusters=%d  "
            "stream_create=%.1fms  malloc=%.1fms  h2d=%.1fms  "
            "kernel=%.1fms  d2h=%.1fms  sync=%.1fms  "
            "free=%.1fms  topk=%.1fms  TOTAL=%.1fms\n",
            n_clusters_searched,
            agg.stream_create_ms, agg.malloc_ms, agg.h2d_ms,
            agg.kernel_ms, agg.d2h_ms, agg.sync_ms,
            agg.free_ms, agg.topk_ms, total);
        fflush(stderr);
    }

    // Convert map -> sorted top-k.
    std::vector<Pair> all;
    all.reserve(best.size());
    for (const auto& [id, sc] : best)
        all.push_back(Pair{sc, id});
    topk_smallest(all, k);

    out_ids.clear();   out_ids.reserve(all.size());
    out_scores.clear(); out_scores.reserve(all.size());
    for (const auto& p : all) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return all.size();
}

} // namespace m3
