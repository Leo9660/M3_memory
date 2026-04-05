#pragma once
#include <memory>
#include <shared_mutex>
#include <vector>

#include "base.h"
#include "cluster.h"   // single-cluster storage/search (thread-safe inside)

namespace m3 {

// ======================================================================
// IVFIndex: core multi-cluster index (thread-safe, shared across threads)
// - Maintains a list of clusters and their centroids
// - Read path: search over a selected subset of clusters (nprobe by centroid)
// - Write path: per-cluster batch add/update/erase (single-writer per cluster is assumed by caller)
// - No background threads here (pure core). Asynchrony/scheduling goes to m3_async.*
// ======================================================================
class IVFIndex {
public:
    // ---- Construction ----
    // layer_name: optional short label ("L0","L1","L2") used in debug logs.
    IVFIndex(int dim, Metric metric, bool normalized,
             const char* layer_name = "??");

    const char* layer_name() const noexcept { return layer_name_; }

    // Initialize clusters by centroids (row-major: [nlist, dim]).
    // Existing clusters (if any) will be cleared and rebuilt to match nlist.
    void set_centroids(const std::vector<float>& centroids);

    // Add a new empty cluster at runtime and return its cluster_id (= index in vector).
    int  add_cluster(const std::vector<float>& centroid);

    // Remove a cluster: drop ref (hard erase), mark slot invalid. Cluster_id remains stable.
    void remove_cluster(int cluster_id);

    // Update centroid of an existing cluster.
    void set_centroid(int cluster_id, const std::vector<float>& centroid);

    // ---- Basic info ----
    int    dim()        const noexcept { return dim_; }
    Metric metric()     const noexcept { return metric_; }
    bool   normalized() const noexcept { return normalized_; }
    int    nlist()      const;   // total cluster slots (including ghost slots after eviction)
    int    live_nlist() const;   // live cluster count (valid slots only) — use for display/stats
    const float* centroid_ptr(int cluster_id) const; // nullptr if invalid

    // ---- Per-cluster writes (caller ensures single-writer-per-cluster) ----
    // allow_missing=true: silently no-op if cluster_id is invalid (used for L0 cache writes).
    void add_batch   (int cluster_id, const DocId* ids, const float* vecs, size_t n_rows,
                      bool allow_missing = false);
    void update_batch(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows,
                      bool insert_if_absent=false);
    void erase_batch (int cluster_id, const DocId* ids, size_t n_rows);
    void rebuild_cluster(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows);
    int  nearest_cluster(const float* vec) const;
    void nearest_clusters(const float* vecs, size_t n_rows, std::vector<int>& out) const;

    //Helpers
    // Returns pointer to vector data if found, nullptr otherwise
    const float* cluster_get_vector(int cluster_id, DocId id) const;
    // Returns live size (number of non-deleted vectors) of a cluster
    size_t cluster_live_size(int cluster_id) const;
    // ---- Maintenance ----
    // Compact a single cluster (remove tombstones).
    void compact_cluster(int cluster_id);
    // Best-effort pass: iterate all clusters and compact based on simple heuristics (optional).
    void maintenance_pass();
    // Split one cluster into two (K-means k=2). Returns new cluster_id or -1 if not done.
    int split_cluster(int cluster_id, size_t max_vectors_before_split = 200000);
    // Merge cluster_id_b into cluster_id_a; cluster b is removed. Both must be valid.
    void merge_clusters(int cluster_id_a, int cluster_id_b);

    // Export live vectors from a cluster (for promotion copy). No-op if invalid/empty.
    void export_cluster_live(int cluster_id,
                             std::vector<DocId>& out_ids,
                             std::vector<float>& out_vecs) const;

    // Ensure cluster slot exists: if valid_[cluster_id] is false, create new empty cluster.
    void ensure_cluster(int cluster_id, const std::vector<float>& centroid);

    // ---- Search APIs ----
    // Search explicitly on a subset of clusters.
    void search_on(const std::vector<int>& cluster_ids,
                   const float* queries, size_t q_rows, int k,
                   std::vector<std::vector<DocId>>& out_ids,
                   std::vector<std::vector<float>>& out_scores) const;

    // Select top-nprobe clusters per query by centroid distance, then search on them.
    // Selection strategy: for each query, compute unified_score(query, centroid)
    // and take the nprobe smallest centroids.
    // out_centroid_ms: time for sgemm + top-nprobe selection (centroid scoring phase).
    // out_scan_ms:     time for cluster vector scans (search_into loop).
    void search_nprobe(const float* queries, size_t q_rows, int k, int nprobe,
                       std::vector<std::vector<DocId>>& out_ids,
                       std::vector<std::vector<float>>& out_scores,
                       double* out_centroid_ms = nullptr,
                       double* out_scan_ms     = nullptr) const;

    // For one query, return the top-nprobe cluster ids by centroid distance (for cache probe set).
    void get_probe_ids(const float* query, int nprobe, std::vector<int>& out_ids) const;

    // For a full batch of queries, return top-nprobe cluster ids per query in one sgemm.
    // out_sgemm_ms: time for the BLAS matrix multiply (centroid scoring across all queries).
    // out_topk_ms:  time for per-query top-nprobe selection from the score matrix.
    void batch_get_probe_ids(const float* queries, size_t q_rows, int nprobe,
                             std::vector<std::vector<int>>& out_probe_ids,
                             double* out_sgemm_ms = nullptr,
                             double* out_topk_ms  = nullptr) const;

    // Search within a single cluster (query as vector); returns k nearest doc_ids and scores.
    void search_within_cluster(int cluster_id, const float* query, int k,
                              std::vector<DocId>& out_ids,
                              std::vector<float>& out_scores) const;

    // Get up to n coldest doc_ids in cluster (by last_access_time) for eviction.
    void get_coldest_doc_ids(int cluster_id, size_t n, std::vector<DocId>& out_ids) const;

private:
    // ---- Helpers ----
    // Return a stable snapshot of clusters, centroids, and valid flags under read lock.
    void snapshot(std::vector<std::shared_ptr<Cluster>>& out_clusters,
                  std::vector<float>& out_centroids,
                  std::vector<bool>& out_valid) const;

    // For a single query, select top-nprobe cluster ids by centroid score ("smaller is better").
    // Only considers cids where valid_snapshot[cid] is true.
    void select_nprobe_for_query(const float* q,
                                 const std::vector<float>& centroids_snapshot, // [nlist, dim_]
                                 const std::vector<bool>& valid_snapshot,
                                 int nprobe,
                                 std::vector<int>& out_ids) const;

    // Remove cluster without acquiring topo_mu_ (caller must already hold a unique_lock).
    void remove_cluster_nolock_(int cluster_id);

private:
    // Fixed config
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;
    const char*  layer_name_;   // short debug label, e.g. "L0"/"L1"/"L2"/"??"

    // Topology & data
    mutable std::shared_mutex topo_mu_;       // guards clusters_, centroids_, valid_
    std::vector<std::shared_ptr<Cluster>> clusters_; // cluster_id == index
    std::vector<float> centroids_;            // row-major [nlist, dim_]
    std::vector<bool> valid_;                // valid_[cid] => slot is in use (not removed)
};

} // namespace m3
