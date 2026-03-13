#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "base.h"
#include "gpu_budget.h"
#include "gpu_cluster_index.h"
#include "gpu_flush_coordinator.h"
#include "gpu_insert_buffer.h"
#include "split_kernel_v3.h"
#include "m3_multi_level.h"

namespace m3 {

// ======================================================================
// Block 6 — Eviction Drain Protocol (free functions)
//
// When GpuBudgetManager::register_cluster() evicts clusters to make
// room, the caller MUST handle each evicted entry before freeing the
// (simulated) device pointer. The protocol in order:
//
//   1. drain ClusterInsertBuffer for the evicted cid — prevents data
//      loss for vectors that are buffered but not yet flushed to L2.
//   2. Write drained vectors to L2 via idx.load_cluster().
//   3. Deactivate the ClusterInsertBuffer slot so future inserts route
//      to L2 instead of the now-dead slot.
//   4. Remove the cluster from GpuClusterIndex.
//
// The EvictedCluster.freq value carries the cluster's access frequency
// at eviction time. When the cluster is re-promoted, pass this value to
// GpuBudgetManager::set_frequency() to restore its historical rank and
// prevent it from being immediately re-evicted.
// ======================================================================

// Handle a list of evicted clusters using the drain protocol above.
// Returns the total number of buffered vectors drained and written to L2.
size_t drain_evicted_clusters(const std::vector<EvictedCluster>& evicted,
                               ClusterInsertBuffer& buf,
                               GpuClusterIndex&     gpu_idx,
                               MultiLevelIndex&     idx);

// All-in-one promote helper: upload cluster data to GPU, register in budget
// (handling any auto-evictions via drain_evicted_clusters), and activate the
// insert buffer slot. bytes_override=0 computes size from n * dim * sizeof(float).
// Returns false if the cluster is too large for the budget.
bool promote_cluster(int cid,
                     const DocId* ids, const float* vecs, size_t n,
                     GpuClusterIndex&     gpu_idx,
                     GpuBudgetManager&    budget,
                     ClusterInsertBuffer& buf,
                     MultiLevelIndex&     idx,
                     size_t               bytes_override = 0);

// ======================================================================
// Block 7 — Maintenance tick (free function)
//
// GpuBudgetManager reads ClusterMetadata::access_count directly on every
// eviction decision — there is no separate frequency counter to sync.
// maintenance_tick() therefore reduces to: flush pending buffers +
// idx.maintenance_pass().
// ======================================================================

// Result type for maintenance_tick().
struct TickResult {
    size_t vectors_flushed    = 0;  // total buffer vectors written to L2
    size_t flush_events       = 0;  // number of cluster flushes performed
    size_t clusters_promoted  = 0;  // new hotspot clusters promoted to GPU
};

// Result type for GpuCoordinator::split_gpu_cluster().
struct SplitResult {
    bool success  = false;
    int  new_cid  = -1;  // cluster id for the B partition; -1 if split failed
};

// Unified maintenance tick:
//   1. Flush all pending buffer slots in monitored_cids.
//   2. Run idx.maintenance_pass() (vector eviction, cold-cluster demotion).
TickResult maintenance_tick(const std::vector<int>& monitored_cids,
                             AsyncFlushCoordinator&  flush_coord,
                             MultiLevelIndex&        idx);

// ======================================================================
// Block 8 — GpuCoordinator
//
// Top-level coordinator that integrates all GPU-CPU index coordination
// components (Blocks 1–7) behind a clean high-level API.
//
// Ownership
// ─────────
//   Owns: GpuClusterIndex, GpuBudgetManager, ClusterInsertBuffer,
//         AsyncFlushCoordinator.
//   References (does NOT own): MultiLevelIndex — caller must keep alive.
//
// Insert routing
// ──────────────
//   insert(cid, id, vec):
//     • GPU-resident cluster → ClusterInsertBuffer (buffered insert).
//       If buffer full (expansion pending): route to L2 without blocking;
//       a WARNING is printed. Background thread expands the GPU cluster and
//       re-opens the buffer slot asynchronously (fully non-blocking insert).
//     • Non-resident cluster → idx.load_cluster() (direct L2 write).
//
// Search
// ──────
//   search(probe_cids, query, k):
//     Delegates to GpuClusterIndex::collaborative_search() which searches
//     GPU-stored vectors + ClusterInsertBuffer for each probe cluster.
//     Non-GPU-resident clusters return only their buffer results.
//
// Lifecycle
// ─────────
//   promote_to_gpu(cid) — upload L2 data to GPU (copy semantics: L2
//                         keeps its data), register in budget (auto-
//                         evicting LFU clusters with drain), activate
//                         buffer slot.
//   demote_from_gpu(cid) — drain insert buffer → flush buffer vectors
//                          to L2 (only the new inserts since promotion),
//                          remove from GPU, deactivate buffer slot.
//                          L2 cluster data is intact; only the buffer
//                          delta needs writing.
//
// Maintenance
// ───────────
//   maintenance_tick() — three phases:
//     1. Flush pending buffer slots → GPU expand + L2 durability write.
//     2. idx.maintenance_pass() — CPU-side L0/L1 eviction and writeback.
//     3. hotspot_rebalance_() — promote non-GPU-resident clusters whose
//        access_count exceeds the current minimum GPU-resident cluster;
//        LFU clusters are evicted via the drain protocol automatically.
//        TickResult::clusters_promoted reports how many were promoted.
//   start/stop_background_flush() — background thread that calls
//                        maintenance_tick() at a configurable interval.
//
// Thread-safe: all public methods are safe for concurrent use.
// ======================================================================

class GpuCoordinator {
public:
    // idx              : MultiLevelIndex to read/write (must outlive coordinator).
    // gpu_budget_bytes : hard VRAM cap (simulated on CPU in tests).
    // dim              : vector dimensionality, must match idx.dim().
    // metric / normalized : scoring parameters for collaborative search.
    // insert_buf_cap   : per-cluster insert buffer capacity before auto-flush.
    GpuCoordinator(MultiLevelIndex& idx,
                   size_t           gpu_budget_bytes,
                   int              dim,
                   Metric           metric,
                   bool             normalized,
                   size_t           insert_buf_cap = 128);

    ~GpuCoordinator();

    // ---- Insert ----

    // Route one vector into the index.
    //   GPU-resident cid  → try_buffer(); if kFull (expansion in progress),
    //                        route directly to L2 without blocking — a WARNING
    //                        is printed to stderr. The background flush thread
    //                        drains the buffer and expands the GPU cluster
    //                        asynchronously; subsequent inserts will buffer again.
    //   Non-resident cid  → idx.load_cluster() directly to L2.
    // Never blocks on GPU memory operations. Always returns kBuffered.
    BufferResult insert(int cid, DocId id, const float* vec);

    // ---- Search ----

    // Collaborative search over probe_cids:
    //   GPU-resident clusters → search GpuClusterIndex data.
    //   All clusters → scan ClusterInsertBuffer.
    // Results are merged (de-duplicated by DocId, best score kept), top-k returned.
    size_t search(const std::vector<int>& probe_cids,
                  const float* query, int k,
                  std::vector<DocId>&  out_ids,
                  std::vector<float>&  out_scores);

    // ---- Cluster lifecycle ----

    // Export cluster cid from L2, upload to GpuClusterIndex, register in budget
    // (auto-evicting LFU clusters as needed with full drain protocol), and
    // activate a ClusterInsertBuffer slot.
    // Returns false if cid has no L2 data or is too large for the budget.
    bool promote_to_gpu(int cid);

    // Drain buffer for cid, flush to L2, remove from GpuClusterIndex, deactivate
    // buffer slot. Safe to call even if cid is not GPU-resident (no-op).
    // Returns number of vectors drained from the buffer and written to L2.
    size_t demote_from_gpu(int cid);

    // ---- Block 9 — GPU-side cluster split ----

    // Split cluster `cid` into two using 2-centroid k-means.
    //
    // Steps:
    //   1. Flush pending insert-buffer for cid to L2.
    //   2. Export all L2 vectors for cid.
    //   3. Run gpu_split_kmeans() — GPU kernels if HAVE_CUDA, CPU fallback otherwise.
    //   4. Write partition A back into cid via rebuild_l2_cluster().
    //   5. Create a new L2 cluster for partition B via add_l2_cluster();
    //      new cluster id is returned in SplitResult::new_cid.
    //   6. If cid was GPU-resident, refresh its GPU storage with partition A data.
    //   7. Attempt to promote the new cluster to GPU (budget permitting).
    //
    // Returns SplitResult{false,-1} if cid has fewer than 2 vectors or L2
    // export fails.
    SplitResult split_gpu_cluster(int cid, int max_iters = 20);

    // ---- Maintenance ----

    // Flush pending buffers, run maintenance_pass(), sync access_counts to budget.
    TickResult maintenance_tick();

    // Start a background thread that calls maintenance_tick() every interval_ms ms.
    void start_background_flush(int interval_ms = 50);

    // Stop the background thread (blocks until exit; final tick performed).
    void stop_background_flush();

    bool background_running() const { return bg_running_.load(); }

    // ---- Accessors ----

    bool             is_gpu_resident(int cid)   const;
    size_t           gpu_bytes_used()            const;
    size_t           gpu_budget_bytes()          const;
    std::vector<int> gpu_resident_cids()         const;
    uint64_t         total_flushed_vectors()     const;
    uint64_t         total_flush_events()        const;

private:
    void   bg_thread_fn_(int interval_ms);

    // Scan all known clusters. For each non-GPU-resident cluster whose
    // access_count exceeds the minimum frequency among currently GPU-resident
    // clusters, call promote_to_gpu() (which auto-evicts the LFU cluster via
    // the drain protocol). Candidates are processed hottest-first so the
    // budget always ends up holding the highest-frequency clusters.
    // Returns the number of clusters newly promoted in this call.
    size_t hotspot_rebalance_();

    MultiLevelIndex&      idx_;
    GpuClusterIndex       gpu_idx_;
    GpuBudgetManager      budget_;
    ClusterInsertBuffer   insert_buf_;
    AsyncFlushCoordinator flush_coord_;

    std::atomic<bool>     bg_running_{false};
    std::atomic<bool>     stop_flag_{false};
    std::thread           bg_thread_;
};

} // namespace m3
