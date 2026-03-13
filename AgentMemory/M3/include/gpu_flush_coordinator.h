#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu_cluster_index.h"
#include "gpu_insert_buffer.h"
#include "m3_multi_level.h"

namespace m3 {

// ======================================================================
// AsyncFlushCoordinator
//
// Drains ClusterInsertBuffer slots that have reached a threshold and
// expands the GPU cluster in-place (per paper spec) and writes the
// vectors into a MultiLevelIndex via load_cluster() (L2) for durability.
//
// Per the paper: when the insertion buffer becomes full, the GPU-cached
// cluster is resized and the buffered vectors are migrated from the CPU
// to the GPU (CPU simulation: GpuClusterIndex::expand_cluster appends
// in-place). L2 is also updated so that on future eviction, drain only
// needs to flush the current (not yet flushed) buffer portion.
//
// Two usage modes
// ───────────────
// Synchronous (default):
//   maybe_flush(cid)          — check one cluster and flush if ready.
//   flush_clusters(cids)      — iterate a list and flush any ready slot.
//   Both are safe to call from any thread at any time.
//
// Asynchronous (background thread):
//   start_background(cids, interval_ms) — launch a thread that calls
//     flush_clusters(cids) every `interval_ms` milliseconds.
//   stop_background()                   — signal stop, join the thread.
//   The background thread performs one final flush pass on exit so no
//   vectors are stranded when it shuts down.
//
// Flush threshold
// ───────────────
//   flush_threshold == 0  → flush only when slot reports is_full()
//                           (i.e. has reached insert_cap)
//   flush_threshold  > 0  → flush when slot.size() >= flush_threshold
//                           (earlier than cap, for lower-latency drain)
//
// GPU expansion + L2 durability
// ──────────────────────────────
//   If gpu_idx is non-null and the cluster is GPU-resident (has_cluster):
//     expand_cluster() is called first — migrates buffered vectors to GPU.
//   In all cases, load_cluster() writes the delta to L2 for durability,
//   ensuring that eviction drain only needs to flush the live buffer.
//
// Thread-safe: all public methods are safe for concurrent use.
// ======================================================================

class AsyncFlushCoordinator {
public:
    // buf             : ClusterInsertBuffer to drain from.
    // idx             : target MultiLevelIndex; vectors go to L2 via load_cluster().
    // flush_threshold : 0 = flush at cap; N = flush when slot.size() >= N.
    // gpu_idx         : optional GpuClusterIndex; if non-null and the cluster
    //                   is GPU-resident, expand_cluster() is called before
    //                   the L2 write so the GPU data stays current.
    AsyncFlushCoordinator(ClusterInsertBuffer& buf,
                          MultiLevelIndex&     idx,
                          size_t               flush_threshold = 0,
                          GpuClusterIndex*     gpu_idx         = nullptr);

    ~AsyncFlushCoordinator();

    // ---- Synchronous API ----

    // Check cluster `cid`. If its buffer is at/above threshold, drain it
    // and write the vectors to L2 via idx.load_cluster(cid, ...).
    // Returns the number of vectors flushed (0 if not yet at threshold).
    size_t maybe_flush(int cid);

    // Iterate `cids` and call maybe_flush() for each.
    // Returns total vectors flushed across all clusters in this call.
    size_t flush_clusters(const std::vector<int>& cids);

    // ---- Async API ----

    // Start a background thread that calls flush_clusters(cids) every
    // `interval_ms` milliseconds. No-op if already running.
    void start_background(const std::vector<int>& cids, int interval_ms = 50);

    // Signal the background thread to stop and block until it exits.
    // A final flush pass is executed before the thread terminates.
    void stop_background();

    bool background_running() const { return bg_running_.load(); }

    // ---- Stats ----
    uint64_t total_flushed_vectors() const;
    uint64_t total_flush_events()    const;

private:
    void   bg_thread_fn_(std::vector<int> cids, int interval_ms);

    ClusterInsertBuffer& buf_;
    MultiLevelIndex&     idx_;
    size_t               flush_threshold_;
    GpuClusterIndex*     gpu_idx_;   // nullable; owned by GpuCoordinator

    mutable std::mutex   stats_mu_;
    uint64_t             total_flushed_ = 0;
    uint64_t             total_events_  = 0;

    std::atomic<bool>    bg_running_{false};
    std::atomic<bool>    stop_flag_{false};
    std::thread          bg_thread_;
};

} // namespace m3
