#include "cluster.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <queue>
#include <unordered_set>
#include <cstring>   // memcpy, memset
#include <mutex>

namespace m3 {

// ------------------------ Cluster impl ------------------------

Cluster::Cluster(int dim, Metric metric, bool normalized, int cluster_id,
                 const std::vector<float>& centroid)
    : dim_(dim)
    , metric_(metric)
    , normalized_(normalized)
    , id_(cluster_id)
    , centroid_(centroid)
{
    if (dim_ <= 0) throw std::invalid_argument("Cluster: dim must be > 0");
    if ((int)centroid_.size() != dim_) {
        throw std::invalid_argument("Cluster: centroid dimension mismatch");
    }
}

int Cluster::dim() const noexcept { return dim_; }
int Cluster::id()  const noexcept { return id_;  }
Metric Cluster::metric() const noexcept { return metric_; }
bool Cluster::normalized() const noexcept { return normalized_; }
const float* Cluster::centroid_ptr() const noexcept { return centroid_.data(); }

void Cluster::reserve_rows(size_t rows_hint) {
    std::unique_lock lk(mu_);
    const size_t want = rows_hint;
    ids_.reserve(want);
    alive_.reserve(want);
    mat_.reserve(want * (size_t)dim_);
}

size_t Cluster::size() const noexcept {
    std::shared_lock lk(mu_);
    return ids_.size();
}

size_t Cluster::live_size() const noexcept {
    std::shared_lock lk(mu_);
    return live_count_;
}

void Cluster::add_batch(const DocId* ids, const float* vecs, size_t n_rows) {
    if (!ids || !vecs || n_rows == 0) return;

    std::unique_lock lk(mu_);

    // --- Duplicate checks: within-batch and against existing map ---
    {
        std::unordered_set<DocId> seen;
        seen.reserve(n_rows * 2);
        for (size_t r = 0; r < n_rows; ++r) {
            DocId id = ids[r];
            if (!seen.insert(id).second) {
                throw std::runtime_error("Cluster::add_batch: duplicate id in batch");
            }
            if (id2row_.find(id) != id2row_.end()) {
                throw std::runtime_error("Cluster::add_batch: duplicate id exists");
            }
        }
    }

    const size_t old_rows = ids_.size();
    const size_t new_rows = old_rows + n_rows;

    // --- One-shot grow to final sizes ---
    ids_.resize(new_rows);
    alive_.resize(new_rows);
    mat_.resize(new_rows * (size_t)dim_);
    id2row_.reserve(id2row_.size() + n_rows);

    // --- Bulk copy/initialize ---
    std::memcpy(ids_.data() + old_rows, ids, n_rows * sizeof(DocId));
    std::memset(alive_.data() + old_rows, 1, n_rows * sizeof(uint8_t));
    std::memcpy(mat_.data() + old_rows * (size_t)dim_,
                vecs,
                n_rows * (size_t)dim_ * sizeof(float));

    // --- Build id2row_ mapping (single linear pass) ---
    for (size_t r = 0; r < n_rows; ++r) {
        id2row_.emplace(ids[r], static_cast<uint32_t>(old_rows + r));
    }

    live_count_ += n_rows;

    // sanity
    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
}

void Cluster::update_batch(const DocId* ids, const float* vecs, size_t n_rows,
                           bool insert_if_absent) {
    if (!ids || !vecs || n_rows == 0) return;

    std::unique_lock lk(mu_);
    for (size_t r = 0; r < n_rows; ++r) {
        DocId id = ids[r];
        const float* src = vecs + r * (size_t)dim_;
        auto it = id2row_.find(id);
        if (it == id2row_.end()) {
            if (!insert_if_absent) {
                throw std::runtime_error("Cluster::update_batch: id not found");
            }
            // insert as new row (small-batch path keeps per-row insert, ok)
            uint32_t row = static_cast<uint32_t>(ids_.size());
            ids_.push_back(id);
            alive_.push_back(1u);
            mat_.insert(mat_.end(), src, src + dim_);
            id2row_.emplace(id, row);
            ++live_count_;
        } else {
            uint32_t row = it->second;
            float* dst = row_ptr_(row);
            std::copy(src, src + dim_, dst);
        }
    }

    // sanity
    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
}

void Cluster::erase_batch(const DocId* ids, size_t n_rows) {
    if (!ids || n_rows == 0) return;

    std::unique_lock lk(mu_);
    for (size_t r = 0; r < n_rows; ++r) {
        DocId id = ids[r];
        auto it = id2row_.find(id);
        if (it == id2row_.end()) continue;
        uint32_t row = it->second;
        if (row < alive_.size() && alive_[row]) {
            alive_[row] = 0u;
            if (live_count_ > 0) --live_count_;
        }
        id2row_.erase(it);
    }
}

void Cluster::rebuild_from(const DocId* ids, const float* vecs, size_t n_rows) {
    std::unique_lock lk(mu_);

    if (n_rows > 0 && (!ids || !vecs)) {
        throw std::invalid_argument("Cluster::rebuild_from: ids/vecs must be provided");
    }

    ids_.resize(n_rows);
    alive_.assign(n_rows, 1u);
    mat_.resize(n_rows * (size_t)dim_);
    id2row_.clear();
    id2row_.reserve(n_rows);

    if (n_rows > 0) {
        std::memcpy(ids_.data(), ids, n_rows * sizeof(DocId));
        std::memcpy(mat_.data(), vecs, n_rows * (size_t)dim_ * sizeof(float));
    }

    for (size_t row = 0; row < n_rows; ++row) {
        id2row_.emplace(ids_[row], static_cast<uint32_t>(row));
    }

    live_count_ = n_rows;

    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
}

void Cluster::search(const float* queries, size_t q_rows, int k,
                     std::vector<std::vector<DocId>>& out_ids,
                     std::vector<std::vector<float>>& out_scores) const {
    if (!queries || q_rows == 0 || k <= 0) {
        out_ids.assign(q_rows, {});
        out_scores.assign(q_rows, {});
        return;
    }

    std::shared_lock lk(mu_);

    const size_t N = ids_.size();
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (N == 0) return;

    struct Node { float s; DocId id; };
    auto worse_first = [](const Node& a, const Node& b){ return a.s < b.s; }; // max-heap

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* q = queries + qi * (size_t)dim_;
        std::priority_queue<Node, std::vector<Node>, decltype(worse_first)> heap(worse_first);

        for (size_t row = 0; row < N; ++row) {
            if (!alive_[row]) continue;
            const float* v = row_ptr_(row);
            float s = score_(q, v); // smaller is better

            if ((int)heap.size() < k) {
                heap.push({s, ids_[row]});
            } else if (s < heap.top().s) {
                heap.pop();
                heap.push({s, ids_[row]});
            }
        }

        auto& oi = out_ids[qi];
        auto& os = out_scores[qi];
        const int m = (int)heap.size();
        oi.resize(m);
        os.resize(m);
        for (int i = m - 1; i >= 0; --i) {
            Node n = heap.top(); heap.pop();
            oi[i] = n.id;
            os[i] = n.s;
        }
    }
}

void Cluster::search_into(const float* query, int k,
                          std::vector<DocId>& top_ids,
                          std::vector<float>& top_scores) const {
    if (!query || k <= 0) return;

    std::shared_lock lk(mu_);
    const size_t N = ids_.size();
    if (N == 0) return;

    // If caller hasn't pre-initialized, do it here.
    if ((int)top_scores.size() < k) {
        top_scores.assign(k, std::numeric_limits<float>::infinity());
        top_ids.assign(k, -1);  // use -1 as "empty slot"
    }

    // helper lambda: find index of current worst (largest score)
    auto find_worst = [&]() -> int {
        float worst_s = top_scores[0];
        int worst_i = 0;
        for (int i = 1; i < k; ++i) {
            if (top_scores[i] > worst_s) {
                worst_s = top_scores[i];
                worst_i = i;
            }
        }
        return worst_i;
    };

    int worst_idx = find_worst();
    float worst_score = top_scores[worst_idx];

    for (size_t row = 0; row < N; ++row) {
        if (!alive_[row]) continue;
        const float* v = row_ptr_(row);
        float s = score_(query, v); // smaller is better

        // if this score beats current worst, replace
        if (s < worst_score) {
            top_ids[worst_idx] = ids_[row];
            top_scores[worst_idx] = s;
            // recompute current worst slot
            worst_idx = find_worst();
            worst_score = top_scores[worst_idx];
        }
    }
}

const float* Cluster::get_vector(DocId id) const {
    std::shared_lock lk(mu_);
    auto it = id2row_.find(id);
    if (it == id2row_.end()) {
        return nullptr;
    }
    uint32_t row = it->second;
    if (row >= ids_.size() || !alive_[row]) {
        return nullptr;
    }
    return row_ptr_(row);
}

void Cluster::compact() {
    std::unique_lock lk(mu_);

    const size_t N = ids_.size();
    if (N == 0 || live_count_ == N) return;

    std::vector<DocId> new_ids;
    std::vector<float> new_mat;
    std::vector<uint8_t> new_alive;
    new_ids.reserve(live_count_);
    new_mat.reserve(live_count_ * (size_t)dim_);
    new_alive.reserve(live_count_);
    std::unordered_map<DocId, uint32_t> new_map;
    new_map.reserve(live_count_);

    for (size_t row = 0; row < N; ++row) {
        if (!alive_[row]) continue;
        uint32_t new_row = static_cast<uint32_t>(new_ids.size());
        new_ids.push_back(ids_[row]);
        const float* src = &mat_[row * (size_t)dim_];
        new_mat.insert(new_mat.end(), src, src + dim_);
        new_alive.push_back(1u);
        new_map.emplace(new_ids.back(), new_row);
    }

    ids_.swap(new_ids);
    mat_.swap(new_mat);
    alive_.swap(new_alive);
    id2row_.swap(new_map);
    live_count_ = ids_.size();

    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
}

float Cluster::score_(const float* q, const float* v) const {
    // Delegate to base helpers; unified to "smaller is better"
    switch (metric_) {
        case Metric::L2:     return l2_dist(q, v, dim_);
        case Metric::IP:     return -ip_score(q, v, dim_);
        case Metric::COSINE: return normalized_ ? (1.f - ip_score(q, v, dim_))
                                                :  cos_dist(q, v, dim_);
    }
    return std::numeric_limits<float>::infinity();
}

const float* Cluster::row_ptr_(size_t row) const {
    return &mat_[row * (size_t)dim_];
}

float* Cluster::row_ptr_(size_t row) {
    return &mat_[row * (size_t)dim_];
}

} // namespace m3
