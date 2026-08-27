#include "gpu_coordinator.h"
#include "m3_logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <utility>

namespace m3 {

size_t drain_evicted_clusters(const std::vector<EvictedCluster>& evicted,
                               ClusterInsertBuffer& buf,
                               GpuClusterIndex&     gpu_idx,
                               MultiLevelIndex&     idx) {
    size_t total_drained = 0;
    for (const auto& ev : evicted) {
        if (ev.cid < 0) continue;

        M3Logger::instance().log_eviction(ev.cid, ev.freq, ev.bytes);

        // 1. Drain any buffered-but-not-yet-flushed vectors.
        std::vector<DocId>  drained_ids;
        std::vector<float>  drained_vecs;
        if (buf.drain(ev.cid, drained_ids, drained_vecs) && !drained_ids.empty()) {
            // 2. Write drained vectors to L2 so they are not lost.
            idx.load_cluster(ev.cid, drained_ids.data(), drained_vecs.data(),
                             drained_ids.size());
            total_drained += drained_ids.size();
        }

        // 3. Deactivate slot so future inserts fall through to L2.
        buf.deactivate_cluster(ev.cid);

        // 4. Remove cluster data from GpuClusterIndex.
        gpu_idx.remove_cluster(ev.cid);
    }
    return total_drained;
}

bool promote_cluster(int cid,
                     const DocId* ids, const float* vecs, size_t n,
                     GpuClusterIndex&     gpu_idx,
                     GpuBudgetManager&    budget,
                     ClusterInsertBuffer& buf,
                     MultiLevelIndex&     idx,
                     size_t               bytes_override) {
    const size_t bytes = (bytes_override > 0)
                       ? bytes_override
                       : n * static_cast<size_t>(gpu_idx.dim()) * sizeof(float);

    // Upload data to GpuClusterIndex (simulates H2D memcpy).
    void* ptr = gpu_idx.store_cluster(cid, ids, vecs, n);

    // Register in budget; handle any LFU evictions with the drain protocol.
    std::vector<EvictedCluster> evicted;
    bool ok = budget.register_cluster(cid, ptr, bytes, evicted);

    if (!evicted.empty()) {
        drain_evicted_clusters(evicted, buf, gpu_idx, idx);
    }

    if (!ok) {
        // Cluster is too large for the budget even after evictions — undo upload.
        gpu_idx.remove_cluster(cid);
        return false;
    }

    // Activate insert buffer slot so subsequent inserts are buffered.
    buf.activate_cluster(cid);
    return true;
}


GpuCoordinator::GpuCoordinator(MultiLevelIndex& idx,
                                size_t           gpu_budget_bytes,
                                int              dim,
                                Metric           metric,
                                bool             normalized,
                                size_t           insert_buf_cap)
    : idx_(idx)
    , gpu_idx_(dim, metric, normalized)
    , budget_(gpu_budget_bytes, &idx_)   // budget reads access_count directly from idx
    , insert_buf_(dim, insert_buf_cap)
    , flush_coord_(insert_buf_, idx_, 0, &gpu_idx_, &budget_) {}

GpuCoordinator::~GpuCoordinator() {
    stop_background();
}

BufferResult GpuCoordinator::insert(int cid, DocId id, const float* vec) {
    op_count_.fetch_add(1, std::memory_order_relaxed);
    if (!budget_.is_gpu_resident(cid)) {
        // Not GPU-resident: write directly to L2.
        idx_.load_cluster(cid, &id, vec, 1);
        return BufferResult::kBuffered;
    }

    // GPU-resident: try to buffer.
    auto r = insert_buf_.try_buffer(cid, id, vec);
    if (r == BufferResult::kFull) {
        // Buffer is at capacity. Enqueue cid for async GPU expand on the
        // background thread (non-blocking — same pattern as pending_promotes_).
        // Route this vector to L2 so the insert is durable immediately;
        // the background thread will expand the GPU cluster and reopen the
        // buffer slot without stalling the insert path.
        const uint64_t oc = overflow_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        {
            std::lock_guard<std::mutex> lk(pending_mu_);
            pending_flushes_.push_back(cid);
            // Stage the overflow vector so process_pending_() can expand the
            // GPU cluster with it — prevents the vector from being invisible
            // to search while the cluster remains GPU-resident.
            overflow_ids_[cid].push_back(id);
            const size_t d = static_cast<size_t>(gpu_idx_.dim());
            overflow_vecs_[cid].insert(overflow_vecs_[cid].end(), vec, vec + d);
        }
        M3Logger::instance().log_insert_overflow(cid);
        if (M3Profiler::instance().is_enabled()) {
            // buf_n = current buffer size (== capacity, since try_buffer returned kFull).
            // l2_n = 0: not queried here; SEARCH_DIVERGE events will show the gap.
            M3Profiler::instance().log_recall_diag(
                "OVERFLOW", cid,
                gpu_idx_.cluster_size(cid),  // gpu_n
                insert_buf_.size(cid),        // buf_n (== capacity)
                0,                            // l2_n (unknown at insert site)
                oc);
        }
        idx_.load_cluster(cid, &id, vec, 1);
        return BufferResult::kBuffered;
    }
    return r;
}


size_t GpuCoordinator::search_batch(
        const std::vector<std::vector<int>>& per_query_gpu_cids,
        const float* queries,
        size_t q_rows,
        int k,
        std::vector<std::vector<DocId>>&  out_ids,
        std::vector<std::vector<float>>&  out_scores,
        GpuCollabTiming* timing) {
    return gpu_idx_.collaborative_search_batch(per_query_gpu_cids, queries, q_rows, k,
                                               insert_buf_, out_ids, out_scores, timing);
}

bool GpuCoordinator::promote_to_gpu(int cid) {
    // Export current L2 data for the cluster.
    std::vector<DocId>  ids;
    std::vector<float>  vecs;
    if (!idx_.export_l2_cluster(cid, ids, vecs)) return false;

    const size_t n = ids.empty() ? 0 : ids.size();
    const size_t bytes = n * static_cast<size_t>(gpu_idx_.dim()) * sizeof(float);

    // Upload, register (with eviction drain), activate buffer slot.
    void* ptr = gpu_idx_.store_cluster(cid,
                                        ids.empty()  ? nullptr : ids.data(),
                                        vecs.empty() ? nullptr : vecs.data(),
                                        n);

    std::vector<EvictedCluster> evicted;
    bool ok = budget_.register_cluster(cid, ptr, bytes > 0 ? bytes : 1, evicted);

    if (!evicted.empty()) {
        drain_evicted_clusters(evicted, insert_buf_, gpu_idx_, idx_);
    }

    if (!ok) {
        gpu_idx_.remove_cluster(cid);
        return false;
    }

    // No frequency restoration needed: budget reads access_count live from idx_.
    insert_buf_.activate_cluster(cid);
    M3Logger::instance().log_promotion(cid, idx_.get_access_count(cid),
                                       /*auto_promoted=*/false);
    if (M3Profiler::instance().is_enabled()) {
        // gpu_n = vectors actually uploaded to GPU (from export snapshot = n).
        // l2_n  = current L2 vector count — may exceed gpu_n if vectors were
        //         written to L2 between export_l2_cluster() and register_cluster()
        //         (the promotion race). Those (l2_n - gpu_n) vectors are now
        //         invisible while this cluster remains GPU-resident.
        const size_t gpu_n = gpu_idx_.cluster_size(cid);
        const size_t l2_n  = idx_.l2_vector_count(cid);
        M3Profiler::instance().log_recall_diag(
            "PROMOTION", cid,
            gpu_n,
            0,     // buf_n: buffer just activated, starts empty
            l2_n,
            overflow_count_.load(std::memory_order_relaxed));
    }
    return true;
}

size_t GpuCoordinator::demote_from_gpu(int cid) {
    if (!budget_.is_gpu_resident(cid)) return 0;

    // 1. Drain buffer and flush to L2.
    std::vector<DocId>  drained_ids;
    std::vector<float>  drained_vecs;
    size_t n_drained = 0;
    if (insert_buf_.drain(cid, drained_ids, drained_vecs) && !drained_ids.empty()) {
        idx_.load_cluster(cid, drained_ids.data(), drained_vecs.data(),
                         drained_ids.size());
        n_drained = drained_ids.size();
    }

    // 2. Deactivate insert buffer slot.
    insert_buf_.deactivate_cluster(cid);

    // 3. Remove from GpuClusterIndex.
    gpu_idx_.remove_cluster(cid);

    // 4. Unregister from budget.
    budget_.remove_cluster(cid);

    return n_drained;
}

size_t GpuCoordinator::hotspot_rebalance_() {
    // Snapshot all cluster metadata in one atomic read.
    const auto meta = idx_.get_cluster_metadata();
    const int  n_clusters = static_cast<int>(meta.size());
    if (n_clusters == 0) return 0;

    // Collect all non-resident clusters that have been accessed at least once,
    // sorted hottest-first. We consider all of them as candidates regardless of
    // whether they beat the current weakest GPU cluster — the two-pass logic
    // below decides which actually get promoted.
    std::vector<std::pair<uint64_t, int>> candidates;  // (freq, cid)
    for (int cid = 0; cid < n_clusters; ++cid) {
        if (!budget_.is_gpu_resident(cid) && meta[cid].access_count > 0)
            candidates.emplace_back(meta[cid].access_count, cid);
    }
    if (candidates.empty()) return 0;

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    const size_t n_candidates = candidates.size();
    size_t promoted = 0;

    for (auto& [freq, cid] : candidates) {
        const bool budget_full = (budget_.total_bytes_used() >= budget_.budget_bytes());

        if (!budget_full) {
            // Pass 1: free space available — promote regardless of relative rank.
            // Fills the GPU to capacity with the hottest non-resident clusters.
            enqueue_promote(cid);
            ++promoted;
        } else {
            // Pass 2: budget is full — only displace if this candidate is strictly
            // hotter than the weakest currently GPU-resident cluster.
            uint64_t cur_min = std::numeric_limits<uint64_t>::max();
            for (int r : budget_.all_resident_cids())
                cur_min = std::min(cur_min, idx_.get_access_count(r));

            if (freq <= cur_min)
                break;  // remaining candidates are cooler still; no more swaps needed

            // Existing LFU eviction in GpuBudgetManager fires automatically
            // when promote_to_gpu() registers the new cluster.
            enqueue_promote(cid);
            ++promoted;
        }
    }

    M3Logger::instance().log_hotspot_rebalance(promoted, n_candidates);
    return promoted;
}

size_t GpuCoordinator::flush_buffers() {
    process_pending_();
    M3Logger::instance().log_gpu_memory(budget_.total_bytes_used(),
                                        budget_.budget_bytes(),
                                        budget_.resident_count());
    return flush_coord_.force_flush_all(budget_.all_resident_cids());
}

void GpuCoordinator::cpu_maintenance() {
    process_pending_();
    idx_.maintenance_pass();
}

size_t GpuCoordinator::rebalance() {
    process_pending_();
    return hotspot_rebalance_();
}

void GpuCoordinator::drain_pending() {
    process_pending_();
}

void GpuCoordinator::start_background(int flush_ms, int maintenance_ms, int rebalance_ms,
                                       uint64_t split_every_ops, size_t split_threshold) {
    if (bg_running_.load()) return;
    split_every_ops_ = split_every_ops;
    split_threshold_ = split_threshold;
    stop_flag_.store(false);
    bg_running_.store(true);
    bg_thread_ = std::thread(&GpuCoordinator::bg_thread_fn_, this,
                             flush_ms, maintenance_ms, rebalance_ms);
}

void GpuCoordinator::stop_background() {
    if (!bg_running_.load()) return;
    stop_flag_.store(true);
    if (bg_thread_.joinable()) bg_thread_.join();
    bg_running_.store(false);
}

void GpuCoordinator::split_sweep_() {
    // Snapshot cluster metadata so we iterate a stable list.
    const auto meta = idx_.get_cluster_metadata();
    const int n = static_cast<int>(meta.size());
    for (int cid = 0; cid < n; ++cid) {
        if (meta[static_cast<size_t>(cid)].l2_vector_count <= split_threshold_) continue;
        if (budget_.is_gpu_resident(cid)) {
            // GPU-resident: use GPU k-means (avoids H2D re-upload for VRAM data).
            split_gpu_cluster(cid);
        } else {
            // Non-resident: use CPU k-means via IVFIndex::split_cluster(),
            // with full MultiLevelIndex centroid/metadata bookkeeping.
            idx_.l2_split_cluster(cid, split_threshold_);
        }
    }
}

void GpuCoordinator::bg_thread_fn_(int flush_ms, int maintenance_ms, int rebalance_ms) {
    using clock = std::chrono::steady_clock;
    using ms_t  = std::chrono::milliseconds;

    auto     last_maintenance = clock::now();
    auto     last_rebalance   = clock::now();
    uint64_t last_split_op    = op_count_.load(std::memory_order_relaxed);

    while (!stop_flag_.load()) {
        // Always: drain queues and flush buffers (highest frequency).
        process_pending_();
        flush_coord_.flush_clusters(budget_.all_resident_cids());

        auto now = clock::now();

        if (std::chrono::duration_cast<ms_t>(now - last_maintenance).count() >= maintenance_ms) {
            idx_.maintenance_pass();
            last_maintenance = now;
        }

        if (std::chrono::duration_cast<ms_t>(now - last_rebalance).count() >= rebalance_ms) {
            hotspot_rebalance_();   // enqueues promotes; process_pending_ picks them up next iter
            last_rebalance = now;
        }

        if (split_every_ops_ > 0) {
            const uint64_t cur = op_count_.load(std::memory_order_relaxed);
            if (cur - last_split_op >= split_every_ops_) {
                split_sweep_();
                last_split_op = cur;
            }
        }

        std::this_thread::sleep_for(ms_t(flush_ms));
    }

    // Final pass: drain queues and flush before exit.
    process_pending_();
    flush_coord_.flush_clusters(budget_.all_resident_cids());
}

// ======================================================================
// GPU-side cluster split
// ======================================================================

SplitResult GpuCoordinator::split_gpu_cluster(int cid, int max_iters) {
    // 1. Flush pending insert-buffer for this cluster.
    //    If GPU-resident: expand_cluster() migrates buffer vectors to GPU
    //    and load_cluster() writes the delta to L2 for durability.
    flush_coord_.maybe_flush(cid);

    // 2. Get cluster data for k-means.
    //    Prefer GPU-resident copy (data already in VRAM — avoids L2 export
    //    + H2D re-upload; satisfies paper locality optimization).
    //    Fall back to L2 export for non-resident clusters.
    std::vector<DocId> ids;
    std::vector<float> vecs;
    if (budget_.is_gpu_resident(cid)) {
        if (!gpu_idx_.export_cluster(cid, ids, vecs) || ids.size() < 2)
            return {false, -1};
    } else {
        if (!idx_.export_l2_cluster(cid, ids, vecs) || ids.size() < 2)
            return {false, -1};
    }

    const int n   = static_cast<int>(ids.size());
    const int dim = gpu_idx_.dim();
    const bool gpu_path = budget_.is_gpu_resident(cid);

    // 3. Run 2-centroid k-means.
    //    CPU fallback: pass host pointer directly.
    //    CUDA build: gpu_split_kmeans_v4() manages H2D internally.
    const auto split_t0 = std::chrono::steady_clock::now();
    GpuSplitResult split = gpu_split_kmeans_v4(vecs.data(), n, dim, max_iters);
    const double split_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - split_t0).count();
    M3Logger::instance().log_split(cid, static_cast<size_t>(n), split_ms, gpu_path);

    // 4. Partition ids / vecs into A and B groups.
    std::vector<DocId> ids_a, ids_b;
    std::vector<float> vecs_a, vecs_b;
    ids_a.reserve(n);  ids_b.reserve(n);
    vecs_a.reserve(static_cast<size_t>(n) * dim);
    vecs_b.reserve(static_cast<size_t>(n) * dim);

    for (int i = 0; i < n; ++i) {
        const float* v = vecs.data() + i * dim;
        if (split.partition[i] == 0) {
            ids_a.push_back(ids[i]);
            vecs_a.insert(vecs_a.end(), v, v + dim);
        } else {
            ids_b.push_back(ids[i]);
            vecs_b.insert(vecs_b.end(), v, v + dim);
        }
    }

    // Degenerate split — all vectors on one side; nothing useful to do.
    if (ids_a.empty() || ids_b.empty()) return {false, -1};

    // 5. Write partition A back into the original cluster.
    idx_.rebuild_l2_cluster(cid,
                             ids_a.data(), vecs_a.data(), ids_a.size());

    // 6. Create a new L2 cluster for partition B.
    const int new_cid = idx_.add_l2_cluster(split.centroid_b.data(),
                                              ids_b.data(), vecs_b.data(),
                                              ids_b.size());
    if (new_cid < 0) {
        // Rollback: restore original data to cid.
        idx_.rebuild_l2_cluster(cid, ids.data(), vecs.data(), ids.size());
        return {false, -1};
    }

    // 7. Refresh GPU storage for cid if it was resident, and update the budget
    //    to reflect the smaller post-split size (B_a < B_original).
    //    This frees B_b = B_original - B_a bytes in the budget so that step 8
    //    can register new_cid without triggering LFU evictions — total bytes
    //    stays constant (B_a + B_b == B_original).
    if (budget_.is_gpu_resident(cid)) {
        void* new_ptr_a = gpu_idx_.store_cluster(cid,
                                                  ids_a.data(), vecs_a.data(),
                                                  ids_a.size());
        const size_t bytes_a = ids_a.size() * static_cast<size_t>(dim) * sizeof(float);
        budget_.update_cluster(cid, new_ptr_a, bytes_a > 0 ? bytes_a : 1);
    }

    // 8. Promote new cluster (partition B) to GPU.
    //    By design both partitions must remain GPU-resident after a split.
    //    After the budget update above, B_b bytes are available so this should
    //    always succeed for the GPU-resident split path.
    if (!ids_b.empty()) {
        void* ptr = gpu_idx_.store_cluster(new_cid,
                                            ids_b.data(), vecs_b.data(),
                                            ids_b.size());
        const size_t bytes_b = ids_b.size()
                             * static_cast<size_t>(dim) * sizeof(float);
        std::vector<EvictedCluster> evicted;
        bool ok = budget_.register_cluster(new_cid, ptr,
                                            bytes_b > 0 ? bytes_b : 1, evicted);
        if (!evicted.empty())
            drain_evicted_clusters(evicted, insert_buf_, gpu_idx_, idx_);

        if (ok) {
            insert_buf_.activate_cluster(new_cid);
        } else {
            // Should not happen for a GPU-resident split (bytes are conserved).
            // Fallback for non-resident split path if budget is fully exhausted.
            gpu_idx_.remove_cluster(new_cid);
        }
    }

    return {true, new_cid};
}

void GpuCoordinator::enqueue_promote(int cid) {
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_promotes_.push_back(cid);
}

void GpuCoordinator::enqueue_demote(int cid) {
    if (!budget_.is_gpu_resident(cid)) return;

    // Phase 1 — immediate: flip routing to L2 before returning.
    // From this point, is_gpu_resident() returns false so new inserts and
    // searches use L2. In-flight GPU searches hold a shared_ptr<DeviceBuffer>
    // snapshot and will complete safely; cudaFree is deferred to phase 2.
    budget_.remove_cluster(cid);
    insert_buf_.deactivate_cluster(cid);

    // Phase 2 — deferred: drain remaining buffer vectors to L2, then release
    // the GPU memory. Enqueued here; executed by the background thread.
    std::lock_guard<std::mutex> lk(pending_mu_);
    pending_demotes_.push_back(cid);
}

void GpuCoordinator::process_pending_() {
    std::vector<int> promotes, demotes, flushes;
    std::unordered_map<int, std::vector<DocId>> ovf_ids;
    std::unordered_map<int, std::vector<float>> ovf_vecs;
    {
        std::lock_guard<std::mutex> lk(pending_mu_);
        promotes.swap(pending_promotes_);
        demotes.swap(pending_demotes_);
        flushes.swap(pending_flushes_);
        ovf_ids.swap(overflow_ids_);
        ovf_vecs.swap(overflow_vecs_);
    }

    // Demotes first: drain buffered vectors to L2, then release GPU memory.
    // Budget space is freed before promotes run, so promotes are more likely
    // to succeed without triggering additional LFU evictions.
    for (int cid : demotes) {
        std::vector<DocId> ids;
        std::vector<float> vecs;
        if (insert_buf_.drain(cid, ids, vecs) && !ids.empty())
            idx_.load_cluster(cid, ids.data(), vecs.data(), ids.size());
        // shared_ptr ref-count in GpuClusterIndex drops here; cudaFree fires
        // only after any in-flight search snapshots release their references.
        gpu_idx_.remove_cluster(cid);
    }

    // Promotes: full synchronous H2D on the background thread (not the caller).
    for (int cid : promotes)
        promote_to_gpu(cid);

    // Flushes: cids whose insert buffer hit cap during an insert call.
    // force_flush drains the buffer, expands the GPU cluster in-place, and
    // writes the delta to L2 for durability — reopening the buffer slot so
    // subsequent inserts can buffer again without routing to L2.
    if (!flushes.empty())
        flush_coord_.force_flush_all(flushes);

    // Overflow drain: vectors that were written to L2 when the insert buffer
    // was full are now expanded into the GPU cluster so they are visible to
    // collaborative_search. Skip any cluster that was demoted in this same
    // tick (not GPU-resident anymore) — those vectors already live in L2.
    for (auto& [cid, ids] : ovf_ids) {
        if (ids.empty()) continue;
        if (!budget_.is_gpu_resident(cid) || !gpu_idx_.has_cluster(cid)) continue;
        auto& vecs = ovf_vecs[cid];
        gpu_idx_.expand_cluster(cid, ids.data(), vecs.data(), ids.size());
        const size_t new_bytes = gpu_idx_.cluster_size(cid)
                               * static_cast<size_t>(gpu_idx_.dim()) * sizeof(float);
        budget_.update_cluster(cid, nullptr, new_bytes);
    }
}

bool GpuCoordinator::is_gpu_resident(int cid) const {
    return budget_.is_gpu_resident(cid);
}

size_t GpuCoordinator::gpu_bytes_used() const {
    return budget_.total_bytes_used();
}

size_t GpuCoordinator::gpu_budget_bytes() const {
    return budget_.budget_bytes();
}

std::vector<int> GpuCoordinator::gpu_resident_cids() const {
    return budget_.all_resident_cids();
}

uint64_t GpuCoordinator::total_flushed_vectors() const {
    return flush_coord_.total_flushed_vectors();
}

uint64_t GpuCoordinator::total_flush_events() const {
    return flush_coord_.total_flush_events();
}

size_t GpuCoordinator::gpu_cluster_size(int cid) const {
    return gpu_idx_.cluster_size(cid);
}

size_t GpuCoordinator::buffer_size(int cid) const {
    return insert_buf_.size(cid);
}

uint64_t GpuCoordinator::total_overflow_count() const {
    return overflow_count_.load(std::memory_order_relaxed);
}

} // namespace m3
