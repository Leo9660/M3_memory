#pragma once

#include <memory>
#include <shared_mutex>
#include <limits>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <deque>
#include <thread>
#include <condition_variable>
#include <atomic>
#include "base.h"      // Metric, DocId, topk_smallest
#include "m3_index.h"  // IVFIndex
#include "m3_fsm.h"    // FSMTable, FSMConfig, RequestTrajectory, FSMLayer

namespace m3 {

// ================================================================
// Cache configuration
// ================================================================
struct CacheConfig {
    int    l0_max_clusters              = 16;
    size_t l0_max_vectors_per_cluster   = 1000;
    int    l1_max_clusters              = 32;
    size_t l1_max_vectors_per_cluster   = 10000;
    float  l0_eviction_ratio            = 0.8f;
    float  l1_eviction_ratio            = 0.9f;
    uint64_t cold_time_ns               = 60'000'000'000ULL;
    int    l0_neighborhood_k            = 10;
    int    l1_neighborhood_k            = 20;
    int    max_promote_per_query        = 0;

    // Prefetch queue capacity. When full, new tasks are dropped silently
    // (prefetch is best-effort — correctness never depends on it).
    size_t prefetch_queue_capacity      = 256;
};

// ================================================================
// Per-cluster metadata
// ================================================================
struct ClusterMetadata {
    bool     in_l0                = false;
    bool     in_l1                = false;
    bool     in_l2                = true;
    uint64_t last_access_time     = 0;
    uint64_t access_count         = 0;
    size_t   l0_vector_count      = 0;
    size_t   l1_vector_count      = 0;
    size_t   l2_vector_count      = 0;
    uint64_t l0_last_eviction_time = 0;
};

class L1Strategy {
public:
    virtual ~L1Strategy() = default;
    virtual void on_insert(const DocId*, const float*, size_t) {}
    virtual void on_update(const DocId*, const float*, size_t) {}
    virtual void on_erase (const DocId*, size_t) {}
};

struct MultiLevelConfig {
    int   l0_nlist                    = 1;
    int   l1_nlist                    = 1;
    int   l2_nlist                    = 1;
    float l0_new_cluster_threshold    = std::numeric_limits<float>::infinity();
    float search_threshold            = std::numeric_limits<float>::infinity();
    float l0_merge_threshold          = std::numeric_limits<float>::infinity();
    int   l0_max_nlist                = 0;
};

// ================================================================
// PrefetchTask
//
// Describes one unit of background promotion work.
//   cluster_id  — which L2 cluster to warm into L0/L1
//   anchor_doc  — vector to use as the neighbourhood search centre;
//                 DocId{-1} means use the cluster centroid (predictive)
//   predictive  — true = FSM-predicted (enqueued before the search);
//                 false = reactive (result returned from a search)
// ================================================================
struct PrefetchTask {
    int   cluster_id = -1;
    DocId anchor_doc = DocId{-1};
    bool  predictive = false;
};

// ================================================================
// MultiLevelIndex
// ================================================================
class MultiLevelIndex {
public:
    MultiLevelIndex(int dim, Metric metric, bool normalized,
                    MultiLevelConfig cfg = {});
    ~MultiLevelIndex();

    // ---- topology bootstrap ----
    void set_l0_centroids(const std::vector<float>& centroids);
    void set_l1_centroids(const std::vector<float>& centroids);
    void set_l2_centroids(const std::vector<float>& centroids);
    void set_cache_config(CacheConfig cfg);
    void set_l1_strategy(std::shared_ptr<L1Strategy> s) {
        std::unique_lock lk(topo_mu_);
        l1_strategy_ = std::move(s);
    }

    // ---- writes ----
    void insert(const DocId* ids, const float* vecs, size_t n_rows);
    void update(const DocId* ids, const float* vecs, size_t n_rows,
                bool insert_if_absent = false);
    void erase(const DocId* ids, size_t n_rows);

    // ---- search ----
    // traj: optional; passing non-null enables FSM trajectory tracking.
    // When traj is non-null, append_step() is called automatically after
    // each layer search, and predictive prefetch is enqueued at the end.
    void search(const float* queries, size_t q_rows, int k, int nprobe,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores,
                RequestTrajectory* traj = nullptr) const;

    // ---- FSM ----
    void            set_fsm_config(FSMConfig cfg);
    FSMTable&       fsm_table();
    const FSMTable& fsm_table() const;

    // ---- maintenance ----
    void maintenance_pass();

    // ---- meta ----
    int    dim()       const noexcept { return dim_; }
    Metric metric()    const noexcept { return metric_; }
    bool   normalized()const noexcept { return normalized_; }

private:
    struct Layer {
        std::shared_ptr<IVFIndex> index;
        std::vector<float> centroids;
    };

    void ensure_layer_initialized_(Layer& layer, int nlist_hint);
    void ensure_layer_centroids_(Layer& layer, const std::vector<float>& centroids);

    void merge_levels_(const std::vector<std::vector<DocId>>& per_level_ids,
                       const std::vector<std::vector<float>>& per_level_scores,
                       int k,
                       std::vector<DocId>& out_ids,
                       std::vector<float>& out_scores) const;

    // ---- promotion helpers ----
    void promote_vector_neighborhood_(DocId doc_id) const;
    void promote_cluster_centroid_(int cluster_id) const; // predictive: use centroid anchor
    void record_access_(int cid) const;
    void demote_cluster_(int cid) const;
    void run_vector_eviction_per_level_() const;
    void run_cluster_count_demotion_() const;
    bool cache_enabled_() const;
    int  find_l2_cluster_for_doc_(DocId id) const;

    // ---- FSM search helpers ----
    std::vector<int> build_probe_list_(
        const float* query,
        const std::vector<int>& fsm_preferred,
        const std::shared_ptr<IVFIndex>& layer_idx,
        const std::vector<float>& layer_centroids,
        int effective_nprobe) const;

    int search_layer_ordered_(
        const float* query, int k,
        const std::vector<int>& probe_list,
        const std::shared_ptr<IVFIndex>& layer_idx,
        std::vector<DocId>& out_ids,
        std::vector<float>& out_scores) const;

    void update_dagent_(float kth_score) const;
    float dagent_() const;

    // ---- prefetch queue ----
    // Enqueue a task; silently drops if queue is full (best-effort).
    void enqueue_prefetch_(PrefetchTask task) const;
    // Worker thread main loop.
    void prefetch_worker_loop_();
    // Start / stop the prefetch thread (called by constructor/destructor).
    void start_prefetch_thread_();
    void stop_prefetch_thread_();

private:
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;
    MultiLevelConfig cfg_;

    Layer l0_, l1_, l2_;
    std::shared_ptr<L1Strategy> l1_strategy_;

    CacheConfig cache_config_;
    mutable std::vector<ClusterMetadata> metadata_;
    mutable std::mutex meta_mu_;
    std::unordered_map<DocId, int> doc_id_to_cid_;

    mutable std::shared_mutex topo_mu_;

    // ---- FSM ----
    FSMTable fsm_table_;

    // ---- dagent rolling average ----
    mutable std::deque<float> dagent_buf_;
    mutable std::mutex        dagent_mu_;

    // ---- prefetch background thread ----
    mutable std::deque<PrefetchTask>  prefetch_queue_;
    mutable std::mutex                prefetch_mu_;
    mutable std::condition_variable   prefetch_cv_;
    std::thread                       prefetch_thread_;
    std::atomic<bool>                 prefetch_running_{false};
};

} // namespace m3
