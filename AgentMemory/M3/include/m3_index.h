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
    IVFIndex(int dim, Metric metric, bool normalized);

    // Initialize clusters by centroids (row-major: [nlist, dim]).
    // Existing clusters (if any) will be cleared and rebuilt to match nlist.
    void set_centroids(const std::vector<float>& centroids);

    // Add a new empty cluster at runtime and return its cluster_id (= index in vector).
    int  add_cluster(const std::vector<float>& centroid);

    // Update centroid of an existing cluster.
    void set_centroid(int cluster_id, const std::vector<float>& centroid);

    // ---- Basic info ----
    int    dim()        const noexcept { return dim_; }
    Metric metric()     const noexcept { return metric_; }
    bool   normalized() const noexcept { return normalized_; }
    int    nlist()      const;   // number of clusters
    const float* centroid_ptr(int cluster_id) const; // nullptr if invalid

    // ---- Per-cluster writes (caller ensures single-writer-per-cluster) ----
    void add_batch   (int cluster_id, const DocId* ids, const float* vecs, size_t n_rows);
    void update_batch(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows,
                      bool insert_if_absent=false);
    void erase_batch (int cluster_id, const DocId* ids, size_t n_rows);
    void rebuild_cluster(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows);
    int  nearest_cluster(const float* vec) const;
    void nearest_clusters(const float* vecs, size_t n_rows, std::vector<int>& out) const;

    // ---- Maintenance ----
    // Compact a single cluster (remove tombstones).
    void compact_cluster(int cluster_id);
    // Best-effort pass: iterate all clusters and compact based on simple heuristics (optional).
    void maintenance_pass();

    // ---- Search APIs ----
    // Search explicitly on a subset of clusters.
    void search_on(const std::vector<int>& cluster_ids,
                   const float* queries, size_t q_rows, int k,
                   std::vector<std::vector<DocId>>& out_ids,
                   std::vector<std::vector<float>>& out_scores) const;

    // Select top-nprobe clusters per query by centroid distance, then search on them.
    // Selection strategy: for each query, compute unified_score(query, centroid)
    // and take the nprobe smallest centroids.
    void search_nprobe(const float* queries, size_t q_rows, int k, int nprobe,
                       std::vector<std::vector<DocId>>& out_ids,
                       std::vector<std::vector<float>>& out_scores) const;

private:
    // ---- Helpers ----
    // Return a stable snapshot of clusters and centroids under read lock.
    void snapshot(std::vector<std::shared_ptr<Cluster>>& out_clusters,
                  std::vector<float>& out_centroids) const;

    // For a single query, select top-nprobe cluster ids by centroid score ("smaller is better").
    void select_nprobe_for_query(const float* q,
                                 const std::vector<float>& centroids_snapshot, // [nlist, dim_]
                                 int nprobe,
                                 std::vector<int>& out_ids) const;

private:
    // Fixed config
    const int    dim_;
    const Metric metric_;
    const bool   normalized_;

    // Topology & data
    mutable std::shared_mutex topo_mu_;       // guards clusters_ & centroids_
    std::vector<std::shared_ptr<Cluster>> clusters_; // cluster_id == index
    std::vector<float> centroids_;            // row-major [nlist, dim_]
};

} // namespace m3
