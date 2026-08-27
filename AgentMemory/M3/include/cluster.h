#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <shared_mutex>
#include <cmath>
#include <limits>
#include "base.h"

namespace m3 {
// Monotonic time for access times (nanoseconds). Implemented in cluster.cpp.
uint64_t monotonic_time_ns();
// =====================================================================
// Cluster: a single IVF partition, thread model = one writer per cluster
// =====================================================================
class Cluster {
public:
    // Constructor
    // 'centroid' must have length equal to 'dim'.
    Cluster(int dim, Metric metric, bool normalized, int cluster_id,
            const std::vector<float>& centroid);

    // ---- Basic information ----
    int    dim()        const noexcept;
    int    id()         const noexcept;
    Metric metric()     const noexcept;
    bool   normalized() const noexcept;
    const float* centroid_ptr() const noexcept;

    // ---- Capacity and size ----
    void   reserve_rows(size_t rows_hint);
    size_t size()       const noexcept;
    size_t live_size()  const noexcept;

    // ---- Batched write operations (single-writer semantics) ----
    void add_batch(const DocId* ids, const float* vecs, size_t n_rows);
    void update_batch(const DocId* ids, const float* vecs, size_t n_rows,
                      bool insert_if_absent=false);
    void erase_batch(const DocId* ids, size_t n_rows);
    void rebuild_from(const DocId* ids, const float* vecs, size_t n_rows);

    // ---- Read path ----
    void search(const float* queries, size_t q_rows, int k,
                std::vector<std::vector<DocId>>& out_ids,
                std::vector<std::vector<float>>& out_scores) const;

    // Search a single query and update an existing top-k buffer.
    // top_ids/top_scores together represent current candidates for this query.
    // After the call, they will still contain at most k best items (smaller is better).
    //
    // Contract:
    // - top_ids.size() == top_scores.size()
    // - top_ids.size() <= k
    // - We will not clear them; we only insert/bump worse ones out.
    // q_norm_sq: precomputed ||q||² for L2 (pass < 0 to compute internally).
    // When >= 0, uses decomposed L2 = q_norm + norms_[row] - 2·dot(q,v)
    // with cached per-vector norms — avoids recomputing ||q||² across probes
    // and replaces l2_dist (sub+fmadd) with ip_score (fmadd only, ~2x faster).
    //
    // skip_alive_check: when true, skips the alive_[] test per row (all rows
    // are treated as live). Use after compact() or when tombstones are absent
    // (e.g. rebuild_from_faiss path) to eliminate the branch and extra load
    // that prevent outer-loop vectorisation.
    void search_into(const float* query, float q_norm_sq, int k,
                    std::vector<DocId>& top_ids,
                    std::vector<float>& top_scores,
                    bool skip_alive_check = false) const;

    // Profiling variant: same as search_into but accumulates nanoseconds spent
    // waiting on the shared_lock (*lock_ns) and in the inner distance loop
    // (*scan_ns).  Both are added (not set) so the caller can accumulate across
    // multiple clusters.
    void search_into_timed(const float* query, float q_norm_sq, int k,
                           std::vector<DocId>& top_ids,
                           std::vector<float>& top_scores,
                           bool skip_alive_check,
                           int64_t* lock_ns,
                           int64_t* scan_ns) const;

    // Batch scan for L2 metric: compute distances from n_queries queries to all
    // live vectors in this cluster using a single cblas_sgemm call.
    // Efficient when dim or n_queries is large (avoids per-vector ip_score overhead).
    //
    // Output:
    //   dists_out: resized to [n_queries × N_live], row-major
    //   live_ids_out: DocIds of the N_live live vectors (in storage order)
    //
    // Returns N_live (0 if cluster is empty or metric != L2).
    size_t scan_batch_l2(const float* queries,
                         const float* q_norms_sq,
                         size_t n_queries,
                         std::vector<float>& dists_out,
                         std::vector<DocId>& live_ids_out) const;

    // ---- Maintenance ----
    void compact();

    // Export all live (non-deleted) vectors for split/merge. Appends to out_ids and out_vecs.
    void export_live(std::vector<DocId>& out_ids, std::vector<float>& out_vecs) const;

    // Return up to n doc_ids with oldest last_access_time (for LRU eviction). 0 = oldest.
    void get_coldest_doc_ids(size_t n, std::vector<DocId>& out_ids) const;

    // ---- Per-vector access time (for LRU eviction) ----
    // 0 if not found or not set.
    uint64_t get_last_access_time(DocId id) const;
    // Update access time for a vector (e.g. when returned in search or on insert/update).
    void set_last_access_time(DocId id, uint64_t time_ns);

    // Helpers
    // Returns pointer to vector data if found, nullptr otherwise
    const float* get_vector(DocId id) const;

private:
    // Unified scoring function returning “smaller is better”.
    float score_(const float* q, const float* v) const;

    // Raw row accessors (only valid under external lock).
    const float* row_ptr_(size_t row) const;
    float*       row_ptr_(size_t row);

private:
    const int dim_;
    const Metric metric_;
    const bool normalized_;
    const int id_;
    std::vector<float> centroid_;

    mutable std::shared_mutex mu_;

    // Row-major dense matrix; mat_.size() == ids_.size() * dim_.
    std::vector<DocId> ids_;
    std::vector<float> mat_;
    std::vector<float> norms_;              // norms_[row] = ||mat_[row]||²; same size as ids_
    std::vector<uint8_t> alive_;
    std::vector<uint64_t> last_access_time_;  // same size as ids_; 0 = not set
    std::unordered_map<DocId, uint32_t> id2row_;
    size_t live_count_{0};
};

} // namespace m3
