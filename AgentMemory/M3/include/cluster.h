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
    void search_into(const float* query, int k,
                    std::vector<DocId>& top_ids,
                    std::vector<float>& top_scores) const;

    // ---- Maintenance ----
    void compact();

    //Helpers
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
    std::vector<uint8_t> alive_;
    std::unordered_map<DocId, uint32_t> id2row_;
    size_t live_count_{0};
};

} // namespace m3
