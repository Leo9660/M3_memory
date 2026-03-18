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
#include "kmeans_gpu_v4.h"
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

// Result type for GpuCoordinator::split_gpu_cluster().
struct SplitResult {
    bool success  = false;
    int  new_cid  = -1;  // cluster id for the B partition; -1 if split failed
};

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
//   promote_to_gpu(cid)  — synchronous: export L2 → H2D → register → activate buffer.
//                          Blocks caller until H2D transfer completes.
//                          Use enqueue_promote() for non-blocking promotion.
//   demote_from_gpu(cid) — synchronous: drain buffer → L2 → remove GPU → deactivate.
//                          Use enqueue_demote() for non-blocking demotion.
//   enqueue_promote(cid) — async: push to promote queue, return immediately.
//                          Cluster stays CPU-resident (L2 authoritative) until the
//                          background thread processes it and H2D completes.
//   enqueue_demote(cid)  — async: immediately flips routing to L2 (budget removed,
//                          buffer deactivated so new inserts/searches use L2 at once),
//                          then defers drain+cudaFree to the background thread.
//                          In-flight GPU searches complete safely via shared_ptr ref-count.
//
// Maintenance (three independent operations)
// ───────────
//   flush_buffers()   — drain insert buffers → GPU expand + L2 durability write.
//                       High frequency: tied to insert rate.
//   cpu_maintenance() — L0/L1 vector eviction + cold-cluster demotion in MultiLevelIndex.
//                       Low frequency: time/size based.
//   rebalance()       — enqueue promotions for clusters hotter than weakest GPU-resident.
//                       Medium frequency: access-pattern driven.
//   drain_pending()   — process async promote/demote queues immediately.
//                       Called automatically at the start of each maintenance function;
//                       expose publicly so callers can flush queues on demand.
//   start_background(flush_ms, maintenance_ms, rebalance_ms) — background thread runs
//                       each operation on its own independent interval.
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

    // Collaborative search over GPU-resident probe clusters.
    //
    // probe_cids must contain ONLY GPU-resident cluster IDs. The GPU/L2 split
    // happens at the call site:
    //   1. all_probes = idx.get_l2_probe_ids(query, nprobe)
    //   2. gpu_ids = [cid for cid in all_probes if coord.is_gpu_resident(cid)]
    //   3. l2_ids  = [cid for cid in all_probes if not coord.is_gpu_resident(cid)]
    //   4. coord.search(gpu_ids, query, k, ...)      — this method
    //   5. idx.search_l2_clusters(l2_ids, query, k, ...)  — caller merges
    //
    // For each cluster in probe_cids:
    //   • GPU path  : linear scan over GpuClusterIndex device vectors.
    //   • Buffer path: scan ClusterInsertBuffer (buffered-but-not-yet-flushed vectors).
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

    // Async promote: push cid onto the promote queue and return immediately.
    // L2 remains authoritative for this cluster until the background thread
    // completes the H2D transfer and registers the cluster in the budget.
    void enqueue_promote(int cid);

    // Async demote: immediately removes cid from the budget and deactivates its
    // insert buffer slot so all new traffic routes to L2 at once. The buffer
    // drain and cudaFree are deferred to the background thread.
    // No-op if cid is not GPU-resident.
    void enqueue_demote(int cid);

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

    // Flush insert buffers for all GPU-resident clusters (GPU expand + L2 write).
    // Returns total vectors written.
    size_t flush_buffers();

    // Run L0/L1 vector eviction and cold-cluster demotion in MultiLevelIndex.
    void cpu_maintenance();

    // Enqueue promotions for non-resident clusters hotter than the weakest
    // GPU-resident cluster. Returns the number of clusters enqueued.
    // Actual H2D transfers happen when drain_pending() next runs.
    size_t rebalance();

    // Process the async promote/demote queues immediately.
    // Called automatically at the start of flush_buffers(), cpu_maintenance(),
    // and rebalance(); also available for explicit use in tests.
    void drain_pending();

    // Start a background thread with independent intervals for each operation.
    //   flush_ms        — how often to flush insert buffers       (default  50 ms)
    //   maintenance_ms  — how often to run cpu_maintenance()      (default 5000 ms)
    //   rebalance_ms    — how often to rebalance hotspot clusters  (default  500 ms)
    void start_background(int flush_ms       = 50,
                          int maintenance_ms = 5000,
                          int rebalance_ms   = 500);

    // Stop the background thread (blocks until exit; final flush pass performed).
    void stop_background();

    bool background_running() const { return bg_running_.load(); }

    // ---- Accessors ----

    bool             is_gpu_resident(int cid)   const;
    size_t           gpu_bytes_used()            const;
    size_t           gpu_budget_bytes()          const;
    std::vector<int> gpu_resident_cids()         const;
    uint64_t         total_flushed_vectors()     const;
    uint64_t         total_flush_events()        const;

private:
    void   bg_thread_fn_(int flush_ms, int maintenance_ms, int rebalance_ms);
    size_t hotspot_rebalance_();

    // Drain pending_promotes_ and pending_demotes_ queues.
    // Called at the start of each maintenance_tick() so the background thread
    // processes async requests promptly without a separate polling loop.
    // Demotes are processed first (freeing budget space before promotes run).
    void process_pending_();

    MultiLevelIndex&      idx_;
    GpuClusterIndex       gpu_idx_;
    GpuBudgetManager      budget_;
    ClusterInsertBuffer   insert_buf_;
    AsyncFlushCoordinator flush_coord_;

    std::atomic<bool>     bg_running_{false};
    std::atomic<bool>     stop_flag_{false};
    std::thread           bg_thread_;

    std::mutex            pending_mu_;
    std::vector<int>      pending_promotes_;
    std::vector<int>      pending_demotes_;
    std::vector<int>      pending_flushes_;   // cids whose insert buffer hit cap during insert
};

} // namespace m3
