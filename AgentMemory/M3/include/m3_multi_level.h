#pragma once

#include <memory>
#include <shared_mutex>
#include <limits>
#include <deque>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <mutex>
#include "base.h"      // Metric, DocId, topk_smallest
#include "m3_index.h"  // IVFIndex

// FSM-aware search extension — compiled only when -DM3_WITH_FSM is set.
#ifdef M3_WITH_FSM
#include "fsm_table.h"
#endif

namespace m3 {

// Forward declaration to avoid circular include (gpu_coordinator.h includes this file).
class GpuCoordinator;

// ================================================================
// Cache configuration: thresholds for L0/L1 sizes, eviction,
// promotion/demotion, and L1 neighborhood caching.
// ================================================================

// Controls which search results feed dagent_ (the rolling mean of k-th distances
// used to compute the early-exit threshold αet × dagent_).
//
//   cache_level_k — dagent_ is updated from whatever level returned results
//                   (L0, L1, or full L2). Reflects hot-path quality but is
//                   biased low when early exit fires frequently.
//
//   true_k        — dagent_ is updated ONLY from full L0→L1→L2 searches
//                   (either because early exit never fired, or via the
//                   background calibration path). Keeps dagent_ anchored to
//                   ground-truth distances but may go stale if early exit
//                   fires on almost every query.
enum class DagentUpdateMode { cache_level_k, true_k };

struct CacheConfig {
    // L0 limits
    int l0_max_clusters = 16;
    size_t l0_max_vectors_per_cluster = 1000;

    // L1 limits
    int l1_max_clusters = 256; //32;
    size_t l1_max_vectors_per_cluster = 10000;

    // Eviction thresholds (trigger when cluster at X% of max)
    float l0_eviction_ratio = 0.8f;
    float l1_eviction_ratio = 0.9f;

    // Cluster-level demotion: remove whole cluster if not accessed for this long
    uint64_t cold_time_ns = 60'000'000'000ULL;  // 60s in nanoseconds

    // k' for L1 promotion: top-k' results (across clusters) from a query are cached as one
    // new query-centric L1 cluster.
    int l1_neighborhood_k = 20;

    // k'' for L0 promotion: top-k'' subset of the L1 promotion set also written to L0
    // using the same L1 cluster ID. Must be <= l1_neighborhood_k.
    int l0_neighborhood_k = 5;

    // Max result vectors to promote per query (0 = all)
    int max_promote_per_query = 0;

    // nprobe for L0 search over L0's own query-centric clusters.
    // 0 (default) = use the runtime nprobe passed to search() (capped at l0 nlist).
    // >0 = always search only the top-l0_nprobe L0 clusters by centroid distance.
    int l0_nprobe = 0;

    // nprobe for L1 search over L1's own query-centric clusters.
    // 0 (default) = use the runtime nprobe passed to search() (capped at l1 nlist).
    // >0 = always search only the top-l1_nprobe L1 clusters by centroid distance.
    int l1_nprobe = 0;

    // Early-termination: initial αet value. The live value (alpha_et_dynamic_) is
    // initialised from this and then adapted at runtime when calibration is enabled.
    // Set to 0 to disable dynamic threshold and fall back to MultiLevelConfig::search_threshold.
    float alpha_et = 0.7f;
    // Rolling window size (number of recent queries) used to compute dagent.
    int dagent_window = 20;

    // ---- αet calibration (background full-search verification) ----

    // Controls which results feed dagent_. See DagentUpdateMode above.
    DagentUpdateMode dagent_mode = DagentUpdateMode::cache_level_k;

    // How often (in search-batch ops) to dispatch a background full-search to
    // calibrate αet and (when dagent_mode == true_k) dagent_.
    // 0 = disabled. Recommended starting value: 10.
    uint64_t calibration_interval = 10;

    // Learning rate for αet adaptation.
    // alpha_et_dynamic_ is updated as an EMA toward the observed quality ratio r:
    //   alpha_et_dynamic_ += adapt_rate * (r - alpha_et_dynamic_)
    // where r = true_kth / early_kth ∈ (0, 1].
    // Too high → αet oscillates on noisy single-sample calibration events.
    // Too low  → adaptation barely responds to quality drift.
    // Values to test: 0.01 (slow/stable), 0.05 (default), 0.1 (fast/noisy).
    float alpha_et_adapt_rate = 0.05f;
};

// ================================================================
// Per-cluster metadata (one per L2 centroid ID). Used for
// cluster-level promotion/demotion and to know which levels
// contain this cluster.
// ================================================================
// Per-cluster metadata indexed by L2 cluster ID.
// L0/L1 have their own independent query-centric topology and are not tracked here.
struct ClusterMetadata {
    bool in_l2 = true;   // L2 always has the cluster once it exists

    uint64_t last_access_time = 0;
    uint64_t access_count = 0;

    size_t l2_vector_count = 0;
};

// L0 and L1 share a query-centric cluster topology (same cluster IDs, independent of L2).
// L0 holds a smaller subset of each L1 cluster (l0_neighborhood_k <= l1_neighborhood_k).
// remove_cluster(cid) evicts a cluster from the respective layer.

class L1Strategy {
public:
    virtual ~L1Strategy() = default;
    virtual void on_insert(const DocId* /*ids*/, const float* /*vecs*/, size_t /*n_rows*/) {}
    virtual void on_update(const DocId* /*ids*/, const float* /*vecs*/, size_t /*n_rows*/) {}
    virtual void on_erase (const DocId* /*ids*/, size_t /*n_rows*/) {}
};

// ================================================================
// MultiLevelIndex: scaffold for a 3-layer IVF hierarchy (L0/L1/L2)
// - L0: fast-growing frontier that absorbs fresh points
// - L1: lightly aggregated layer that can merge/split L0 clusters
// - L2: long-lived global layer (closest to today's IVFIndex)
//
// This is a coordination shell; the concrete policies for routing,
// promotion, and splitting are left for incremental development.
// ================================================================

struct MultiLevelConfig {
    int l0_nlist = 1;  // number of clusters to pre-create for L0
    int l1_nlist = 1;  // same for L1
    int l2_nlist = 1;  // same for L2
    float l0_new_cluster_threshold = std::numeric_limits<float>::infinity();
    float search_threshold = std::numeric_limits<float>::infinity();
    float l0_merge_threshold = std::numeric_limits<float>::infinity(); // if best <= merge, reuse
    int   l0_max_nlist = 0; // cap; 0 => fallback to l0_nlist
};

class MultiLevelIndex {
public:
    MultiLevelIndex(int dim, Metric metric, bool normalized,
                    MultiLevelConfig cfg = {});

    // ---- topology bootstrap (per-layer centroids) ----
    void set_l0_centroids(const std::vector<float>& centroids);
    void set_l1_centroids(const std::vector<float>& centroids);
    void set_l2_centroids(const std::vector<float>& centroids);
    void set_cache_config(CacheConfig cfg) {
        cache_config_ = std::move(cfg);
        // Re-sync live αet so the new config takes effect immediately.
        std::lock_guard<std::mutex> lk(alpha_et_mu_);
        alpha_et_dynamic_ = cache_config_.alpha_et;
    }

    // ---- cache diagnostics ----
    struct CacheStats {
        int    l0_clusters = 0;
        size_t l0_total_vecs = 0;
        int    l1_clusters = 0;
        size_t l1_total_vecs = 0;
        size_t l1_dedup_set_size = 0;  // doc IDs tracked in l1_cached_ids_
        size_t l0_l1_overlap = 0;      // doc IDs present in both layers (same cluster ID)
        float  dagent = 0.f;
        float  dynamic_threshold = 0.f;
    };
    CacheStats get_cache_stats() const;

    // Wire in a GpuCoordinator so search/insert can route GPU-resident clusters
    // through the GPU tier. Pass nullptr to disable GPU routing.
    void set_gpu_coordinator(GpuCoordinator* gpu) { gpu_coord_ = gpu; }
    void set_l1_strategy(std::shared_ptr<L1Strategy> s) {
        std::unique_lock lk(topo_mu_);
        l1_strategy_ = std::move(s);
    }

    // ---- writes (high-level routing) ----
    void insert(const DocId* ids, const float* vecs, size_t n_rows);
    void update(const DocId* ids, const float* vecs, size_t n_rows,
                bool insert_if_absent = false);
    void erase(const DocId* ids, size_t n_rows);

    // Bulk-load: write directly into L2 cluster cid, bypassing L0/L1.
    // Intended for corpus bootstrap (e.g. rebuild_from_faiss).
    // Requires cache mode (set_l2_centroids called first).
    void load_cluster(int cid, const DocId* ids, const float* vecs, size_t n_rows);

    // Export all live vectors from an L2 cluster into out_ids / out_vecs.
    // Used by GpuCoordinator::promote_to_gpu() to snapshot a cluster before
    // uploading its data to GPU memory.
    // Returns false if cache mode is inactive or cid is out of range.
    bool export_l2_cluster(int cid,
                           std::vector<DocId>& out_ids,
                           std::vector<float>&  out_vecs) const;

    // Replace the content of L2 cluster `cid` with the supplied vectors
    // (discards existing data). Used after a GPU-side split to write
    // partition A back into the original cluster slot.
    // No-op if cache mode is inactive or cid is out of range.
    void rebuild_l2_cluster(int cid,
                             const DocId*  ids,
                             const float*  vecs,
                             size_t        n);

    // Append a brand-new cluster to L2 with the given centroid and
    // initial vector set. Also appends the centroid to l0_ / l1_ tables
    // so future routing includes the new cluster.
    // Returns the new cluster id, or -1 on failure.
    int add_l2_cluster(const float* centroid,
                       const DocId* ids,
                       const float* vecs,
                       size_t       n);

    // ---- search ----
    // Searches all available layers and merges top-k (smaller score is better).
    void search(const float* queries, size_t q_rows, int k, int nprobe,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores) const;

    // Return the IDs of the nprobe L2 clusters nearest to `query` by centroid
    // distance (ascending order). This is the same probe-set selection that
    // search() uses internally via IVFIndex::get_probe_ids(). Expose it here
    // so callers can split the probe set into GPU-resident vs non-resident
    // clusters before routing to GpuCoordinator::search().
    std::vector<int> get_l2_probe_ids(const float* query, int nprobe) const;

    // Search specific L2 clusters by ID (linear scan within each cluster).
    // Results are appended to out_ids / out_scores (not cleared). Used by
    // callers to search the non-GPU portion of the probe set after GPU-resident
    // clusters are handled by GpuCoordinator::search().
    void search_l2_clusters(const std::vector<int>& cids,
                            const float* query, int k,
                            std::vector<DocId>&  out_ids,
                            std::vector<float>&  out_scores) const;

#ifdef M3_WITH_FSM
    // ---- FSM-aware batch search ----
    //
    // For each query in [queries, queries + q_rows*dim]:
    //   1. Calls fsm_table->match_and_predict(*traj) and enqueues GPU promote
    //      for predicted clusters (non-blocking, best-effort prefetch).
    //   2. Runs the standard 3-level search (identical results to search()).
    //   3. Computes the nearest L2 centroid to the query and appends the step
    //      to *traj.
    //
    // fsm_table and traj may be nullptr (search still runs; step not recorded).
    // Thread-safe.  out_ids / out_scores are resized to q_rows on return.
    void search_fsm(const float* queries, size_t q_rows, int k, int nprobe,
                    const fsm::FSMTable*    fsm_table,
                    fsm::RequestTrajectory* traj,
                    std::vector<std::vector<DocId>>& out_ids,
                    std::vector<std::vector<float>>& out_scores) const;

    // Return the index of the nearest L2 centroid to `query` (squared-L2).
    // Thread-safe.
    int nearest_l2_centroid(const float* query) const;
#endif // M3_WITH_FSM

    // ---- maintenance ----
    void maintenance_pass(); // per-layer maintenance hooks

    // Split a single L2 cluster if its live size exceeds `threshold`.
    // Mirrors the new partition-B centroid into L0/L1 routing tables,
    // updates doc_id_to_cid_, and extends metadata — same bookkeeping
    // as add_l2_cluster(). No-op if the cluster is below threshold or
    // the split is degenerate. Called by GpuCoordinator::split_sweep_().
    void l2_split_cluster(int cid, size_t threshold);

    // ---- meta ----
    int    dim()        const noexcept { return dim_; }
    Metric metric()     const noexcept { return metric_; }
    bool   normalized() const noexcept { return normalized_; }

    // Returns a snapshot of all cluster metadata (indexed by L2 cluster id).
    // Thread-safe; safe to call at any time.
    std::vector<ClusterMetadata> get_cluster_metadata() const {
        std::lock_guard<std::mutex> ml(meta_mu_);
        return metadata_;
    }

    // Returns the access_count for a single cluster (0 if cid is out of range).
    // Used by GpuBudgetManager to make LFU eviction decisions without a full snapshot.
    uint64_t get_access_count(int cid) const {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (cid < 0 || static_cast<size_t>(cid) >= metadata_.size()) return 0;
        return metadata_[static_cast<size_t>(cid)].access_count;
    }

    // Returns the L2 vector count for a single cluster (0 if cid is out of range).
    // Used by recall diagnostics to detect GPU/L2 divergence after promotions.
    size_t l2_vector_count(int cid) const {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (cid < 0 || static_cast<size_t>(cid) >= metadata_.size()) return 0;
        return metadata_[static_cast<size_t>(cid)].l2_vector_count;
    }

private:
    struct Layer {
        std::shared_ptr<IVFIndex> index;    // built on demand
        std::vector<float> centroids;       // last configured centroids
        const char* name = "??";            // "L0"/"L1"/"L2" — used in debug logs
    };

    void ensure_layer_initialized_(Layer& layer, int nlist_hint);
    void ensure_layer_centroids_(Layer& layer, const std::vector<float>& centroids);

    // Merge multiple level results for a single query.
    void merge_levels_(const std::vector<std::vector<DocId>>& per_level_ids,
                       const std::vector<std::vector<float>>& per_level_scores,
                       int k,
                       std::vector<DocId>& out_ids,
                       std::vector<float>& out_scores) const;

    // Promotion: per-query, cache top-k' results as a new query-centric L1 cluster,
    // and top-k'' subset into L0 with the same cluster ID.
    // k_caller: the search k used by the caller — L0 is additionally capped at this
    // so that L0 never holds more vectors per cluster than the caller requested.
    // L1 is capped only by l1_neighborhood_k from CacheConfig (may exceed k_caller
    // when L2 was searched with a wider k_promo = l1_neighborhood_k).
    void promote_query_to_l1_(const float* query,
                               const std::vector<DocId>& result_ids,
                               const std::vector<float>& result_scores,
                               int k_caller) const;
    void record_access_(int cid) const;
    void demote_cluster_(int cid) const;
    void run_vector_eviction_per_level_() const;
    void run_cluster_count_demotion_() const;
    bool cache_enabled_() const;
    // Update the dagent rolling average with the k-th distances from a completed search batch.
    void update_dagent_(const std::vector<std::vector<float>>& out_scores, int k) const;

    // Feed a single k-th distance into the dagent_ rolling average.
    // Used by the background calibration path.
    void update_dagent_single_(float kth) const;

    // Run a full L0→L1→L2 search for one query with no early exit and no
    // calibration side-effects. Used exclusively by the background calibration thread.
    // Layer snapshots are passed by value so the caller's shared-lock is not held
    // across the async boundary.
    void search_one_full_(const float* qptr, int k, int nprobe,
                          const Layer& l0, const Layer& l1, const Layer& l2,
                          std::vector<DocId>& out_ids,
                          std::vector<float>& out_scores) const;

    // Adjust alpha_et_dynamic_ given the quality ratio r = true_kth / early_kth.
    // r ∈ (0, 1]: 1.0 = early exit was perfect; <1 = full search found closer results.
    // Update rule: alpha_et_dynamic_ += adapt_rate * (r - alpha_et_dynamic_)
    // Clamped to [0.3, 1.0] (consistent with the paper's αet = 0.6–0.8 range).
    void update_alpha_et_(float r) const;

private:
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;
    MultiLevelConfig cfg_;

    mutable Layer l0_;
    mutable Layer l1_;
    Layer l2_;
    std::shared_ptr<L1Strategy> l1_strategy_;

    CacheConfig cache_config_;
    mutable std::vector<ClusterMetadata> metadata_;   // indexed by L2 cluster id
    mutable std::mutex meta_mu_;                     // protects metadata_
    std::unordered_map<DocId, int> doc_id_to_cid_;   // L2 assignment (canonical)

    mutable std::shared_mutex topo_mu_;              // protects layer pointers/centroids, doc_id_to_cid_

    GpuCoordinator* gpu_coord_ = nullptr;            // optional; not owned

    // dagent: rolling mean of recent per-query k-th distances, used for dynamic αet·dagent threshold.
    mutable std::deque<float> dagent_history_;
    mutable float             dagent_ = 0.0f;
    mutable std::mutex        dagent_mu_;            // protects dagent_history_ and dagent_

    // Live αet value — initialised from cache_config_.alpha_et and adapted at runtime
    // by the background calibration path. Protected by alpha_et_mu_.
    mutable float             alpha_et_dynamic_ = 0.7f;
    mutable std::mutex        alpha_et_mu_;

    // Counts search() calls (cache path only). Used to decide when to dispatch
    // the next background calibration full-search.
    mutable std::atomic<uint64_t> search_op_count_{0};

    // L1/L0 query-centric cluster tracking (independent of L2 cluster IDs).
    // l1_cached_ids_: DocIds currently stored in L1 (for fast dedup on promotion).
    // l1_cluster_access_time_: L1 cluster_id → last access timestamp (for LRU eviction).
    // doc_id_to_l1_cid_: DocId → L1 cluster ID (needed for in-place update of cached vectors).
    // l1_cache_mu_: protects all of the above.
    mutable std::unordered_set<DocId>         l1_cached_ids_;
    mutable std::unordered_map<int, uint64_t> l1_cluster_access_time_;
    mutable std::unordered_map<DocId, int>    doc_id_to_l1_cid_;
    mutable std::mutex                        l1_cache_mu_;
};

} // namespace m3
