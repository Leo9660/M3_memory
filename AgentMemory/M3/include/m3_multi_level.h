#pragma once

#include <memory>
#include <shared_mutex>
#include <limits>
#include <vector>

#include "base.h"      // Metric, DocId, topk_smallest
#include "m3_index.h"  // IVFIndex

namespace m3 {

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
    void set_l1_strategy(std::shared_ptr<L1Strategy> s) {
        std::unique_lock lk(topo_mu_);
        l1_strategy_ = std::move(s);
    }

    // ---- writes (high-level routing) ----
    // Current scaffold: all writes land in L0 cluster 0.
    // Subsequent iterations will fan out and promote.
    void insert(const DocId* ids, const float* vecs, size_t n_rows);
    void update(const DocId* ids, const float* vecs, size_t n_rows,
                bool insert_if_absent = false);
    void erase(const DocId* ids, size_t n_rows);

    // ---- search ----
    // Searches all available layers and merges top-k (smaller score is better).
    void search(const float* queries, size_t q_rows, int k, int nprobe,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores) const;

    // ---- maintenance ----
    void maintenance_pass(); // per-layer maintenance hooks

    // ---- meta ----
    int    dim()        const noexcept { return dim_; }
    Metric metric()     const noexcept { return metric_; }
    bool   normalized() const noexcept { return normalized_; }

private:
    struct Layer {
        std::shared_ptr<IVFIndex> index;    // built on demand
        std::vector<float> centroids;       // last configured centroids
    };

    void ensure_layer_initialized_(Layer& layer, int nlist_hint);
    void ensure_layer_centroids_(Layer& layer, const std::vector<float>& centroids);

    // Merge multiple level results for a single query.
    void merge_levels_(const std::vector<std::vector<DocId>>& per_level_ids,
                       const std::vector<std::vector<float>>& per_level_scores,
                       int k,
                       std::vector<DocId>& out_ids,
                       std::vector<float>& out_scores) const;

private:
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;
    MultiLevelConfig cfg_;

    Layer l0_;
    Layer l1_;
    Layer l2_;
    std::shared_ptr<L1Strategy> l1_strategy_;

    mutable std::shared_mutex topo_mu_; // protects layer pointers/centroids
};

} // namespace m3
