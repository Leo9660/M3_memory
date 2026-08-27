#include "gpu_flush_coordinator.h"
#include "m3_logger.h"

#include <chrono>
#include <thread>

namespace m3 {

AsyncFlushCoordinator::AsyncFlushCoordinator(ClusterInsertBuffer& buf,
                                              MultiLevelIndex&     idx,
                                              size_t               flush_threshold,
                                              GpuClusterIndex*     gpu_idx,
                                              GpuBudgetManager*    budget)
    : buf_(buf), idx_(idx), flush_threshold_(flush_threshold),
      gpu_idx_(gpu_idx), budget_(budget) {}

AsyncFlushCoordinator::~AsyncFlushCoordinator() {
    // Ensure the background thread is joined before destruction.
    stop_background();
}

size_t AsyncFlushCoordinator::do_flush_(int cid) {
    std::vector<DocId> drained_ids;
    std::vector<float> drained_vecs;
    if (!buf_.drain(cid, drained_ids, drained_vecs)) return 0;
    if (drained_ids.empty()) return 0;

    const size_t n = drained_ids.size();

    // GPU expansion (primary action, per paper spec):
    //   If the cluster is GPU-resident:
    //     1. Allocate a new device buffer (old_size + n_new vectors).
    //     2. GPU→GPU copy: existing cluster vectors → new buffer head.
    //     3. CPU→GPU async copy: insert-buffer vectors → new buffer tail.
    //     4. Atomic swap: cluster pointer updated; old buffer released.
    //   After expansion, update the budget manager's byte count so LFU
    //   eviction decisions stay accurate.
    if (gpu_idx_ && gpu_idx_->has_cluster(cid)) {
        const auto t0 = std::chrono::steady_clock::now();
        gpu_idx_->expand_cluster(cid, drained_ids.data(), drained_vecs.data(), n);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
        M3Logger::instance().log_expansion(cid, n, ms);

        // Sync byte count in budget manager so LFU decisions stay accurate.
        if (budget_) {
            const size_t new_bytes = gpu_idx_->cluster_size(cid)
                                   * static_cast<size_t>(gpu_idx_->dim())
                                   * sizeof(float);
            budget_->update_cluster(cid, /*new_ptr — informational*/nullptr, new_bytes);
        }
    }

    // L2 durability write:
    //   Always append the delta to L2 so that if the cluster is later evicted
    //   from the GPU, drain_evicted_clusters() only needs to flush the live
    //   (not-yet-flushed) insert buffer portion — the rest is already in L2.
    idx_.load_cluster(cid, drained_ids.data(), drained_vecs.data(), n);

    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        total_flushed_ += n;
        ++total_events_;
    }
    return n;
}

size_t AsyncFlushCoordinator::maybe_flush(int cid) {
    const size_t sz = buf_.size(cid);
    if (sz == 0) return 0;

    // Decide whether to flush based on mode.
    bool ready;
    if (flush_threshold_ == 0) {
        ready = buf_.is_full(cid);
    } else {
        ready = (sz >= flush_threshold_);
    }
    if (!ready) return 0;

    return do_flush_(cid);
}

size_t AsyncFlushCoordinator::flush_clusters(const std::vector<int>& cids) {
    size_t total = 0;
    for (int cid : cids)
        total += maybe_flush(cid);
    return total;
}

size_t AsyncFlushCoordinator::force_flush_all(const std::vector<int>& cids) {
    size_t total = 0;
    for (int cid : cids) {
        if (buf_.size(cid) == 0) continue;
        total += do_flush_(cid);
    }
    return total;
}

void AsyncFlushCoordinator::start_background(const std::vector<int>& cids, int interval_ms) {
    if (bg_running_.load()) return;
    stop_flag_.store(false);
    bg_running_.store(true);
    bg_thread_ = std::thread(&AsyncFlushCoordinator::bg_thread_fn_, this, cids, interval_ms);
}

void AsyncFlushCoordinator::stop_background() {
    if (!bg_running_.load()) return;
    stop_flag_.store(true);
    if (bg_thread_.joinable()) bg_thread_.join();
    bg_running_.store(false);
}

void AsyncFlushCoordinator::bg_thread_fn_(std::vector<int> cids, int interval_ms) {
    while (!stop_flag_.load()) {
        flush_clusters(cids);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
    // Final flush pass on exit so no vectors are left stranded.
    flush_clusters(cids);
}

uint64_t AsyncFlushCoordinator::total_flushed_vectors() const {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return total_flushed_;
}

uint64_t AsyncFlushCoordinator::total_flush_events() const {
    std::lock_guard<std::mutex> lk(stats_mu_);
    return total_events_;
}

} // namespace m3
