#pragma once

#include <memory>
#include <shared_mutex>
#include <limits>
#include <deque>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include "base.h"      // Metric, DocId, topk_smallest
#include "m3_index.h"  // IVFIndex

namespace m3 {

// Forward declaration to avoid circular include (gpu_coordinator.h includes this file).
class GpuCoordinator;

// ================================================================
// Cache configuration: thresholds for L0/L1 sizes, eviction,
// promotion/demotion, and L1 neighborhood caching.
// ================================================================
struct CacheConfig {
    // L0 limits
    int l0_max_clusters = 16;
    size_t l0_max_vectors_per_cluster = 1000;

    // L1 limits
    int l1_max_clusters = 32;
    size_t l1_max_vectors_per_cluster = 10000;

    // Eviction thresholds (trigger when cluster at X% of max)
    float l0_eviction_ratio = 0.8f;
    float l1_eviction_ratio = 0.9f;

    // Cluster-level demotion: remove whole cluster if not accessed for this long
    uint64_t cold_time_ns = 60'000'000'000ULL;  // 60s in nanoseconds

    // k' for L1 promotion: number of nearest neighbours (from L2) cached into L1 per access.
    // L0 stores the directly accessed vector only (no neighbourhood search), per spec.
    int l1_neighborhood_k = 20;

    // Max result vectors to promote per query (0 = all)
    int max_promote_per_query = 0;

    // Early-termination: dynamic threshold = alpha_et * dagent.
    // Set to 0 to disable dynamic threshold and fall back to MultiLevelConfig::search_threshold.
    float alpha_et = 0.7f;
    // Rolling window size (number of recent queries) used to compute dagent.
    int dagent_window = 20;
};

// ================================================================
// Per-cluster metadata (one per L2 centroid ID). Used for
// cluster-level promotion/demotion and to know which levels
// contain this cluster.
// ================================================================
struct ClusterMetadata {
    bool in_l0 = false;
    bool in_l1 = false;
    bool in_l2 = true;   // L2 always has the cluster once it exists

    uint64_t last_access_time = 0;
    uint64_t access_count = 0;

    size_t l0_vector_count = 0;
    size_t l1_vector_count = 0;
    size_t l2_vector_count = 0;

    uint64_t l0_last_eviction_time = 0;
};

// L0 and L1 are IVFIndex with same nlist as L2; valid_[cid]=false and
// remove_cluster(cid) indicate uncached/invalidated slots.

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
    void set_cache_config(CacheConfig cfg) { cache_config_ = std::move(cfg); }

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

    // ---- maintenance ----
    void maintenance_pass(); // per-layer maintenance hooks

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

    // Promotion: on access, promote vector + neighborhood to L0 (narrow) and L1 (wider).
    void promote_vector_neighborhood_(DocId doc_id) const;
    void record_access_(int cid) const;
    void demote_cluster_(int cid) const;
    void run_vector_eviction_per_level_() const;
    void run_cluster_count_demotion_() const;
    bool cache_enabled_() const;
    // Update the dagent rolling average with the k-th distances from a completed search batch.
    void update_dagent_(const std::vector<std::vector<float>>& out_scores, int k) const;

private:
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;
    MultiLevelConfig cfg_;

    Layer l0_;
    Layer l1_;
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
};

} // namespace m3
