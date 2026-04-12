#include "cluster.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <queue>
#include <unordered_set>
#include <cstring>   // memcpy, memset
#include <mutex>
#include <cblas.h>

namespace m3 {

uint64_t monotonic_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

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
    last_access_time_.reserve(want);
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
    const uint64_t now = monotonic_time_ns();
    ids_.resize(new_rows);
    alive_.resize(new_rows);
    mat_.resize(new_rows * (size_t)dim_);
    norms_.resize(new_rows);
    last_access_time_.resize(new_rows);
    id2row_.reserve(id2row_.size() + n_rows);

    // --- Bulk copy/initialize ---
    std::memcpy(ids_.data() + old_rows, ids, n_rows * sizeof(DocId));
    std::memset(alive_.data() + old_rows, 1, n_rows * sizeof(uint8_t));
    std::memcpy(mat_.data() + old_rows * (size_t)dim_,
                vecs,
                n_rows * (size_t)dim_ * sizeof(float));
    for (size_t r = old_rows; r < new_rows; ++r) {
        last_access_time_[r] = now;
        norms_[r] = ip_score(mat_.data() + r * (size_t)dim_,
                             mat_.data() + r * (size_t)dim_, dim_);
    }

    // --- Build id2row_ mapping (single linear pass) ---
    for (size_t r = 0; r < n_rows; ++r) {
        id2row_.emplace(ids[r], static_cast<uint32_t>(old_rows + r));
    }

    live_count_ += n_rows;

    // sanity
    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
    assert(norms_.size() == ids_.size());
    assert(last_access_time_.size() == ids_.size());
}

void Cluster::update_batch(const DocId* ids, const float* vecs, size_t n_rows,
                           bool insert_if_absent) {
    if (!ids || !vecs || n_rows == 0) return;

    const uint64_t now = monotonic_time_ns();
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
            norms_.push_back(ip_score(src, src, dim_));
            last_access_time_.push_back(now);
            id2row_.emplace(id, row);
            ++live_count_;
        } else {
            uint32_t row = it->second;
            float* dst = row_ptr_(row);
            std::copy(src, src + dim_, dst);
            norms_[row] = ip_score(dst, dst, dim_);
            last_access_time_[row] = now;
        }
    }

    // sanity
    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
    assert(norms_.size() == ids_.size());
    assert(last_access_time_.size() == ids_.size());
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

    const uint64_t now = monotonic_time_ns();
    ids_.resize(n_rows);
    alive_.assign(n_rows, 1u);
    mat_.resize(n_rows * (size_t)dim_);
    norms_.resize(n_rows);
    last_access_time_.assign(n_rows, now);
    id2row_.clear();
    id2row_.reserve(n_rows);

    if (n_rows > 0) {
        std::memcpy(ids_.data(), ids, n_rows * sizeof(DocId));
        std::memcpy(mat_.data(), vecs, n_rows * (size_t)dim_ * sizeof(float));
    }

    for (size_t row = 0; row < n_rows; ++row) {
        id2row_.emplace(ids_[row], static_cast<uint32_t>(row));
        norms_[row] = ip_score(mat_.data() + row * (size_t)dim_,
                               mat_.data() + row * (size_t)dim_, dim_);
    }

    live_count_ = n_rows;

    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
    assert(norms_.size() == ids_.size());
    assert(last_access_time_.size() == ids_.size());
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

void Cluster::search_into(const float* query, float q_norm_sq, int k,
                          std::vector<DocId>& top_ids,
                          std::vector<float>& top_scores,
                          bool skip_alive_check) const {
    if (!query || k <= 0) return;

    std::shared_lock lk(mu_);
    const size_t N = ids_.size();
    if (N == 0) return;

    // If caller hasn't pre-initialized, do it here.
    if ((int)top_scores.size() < k) {
        top_scores.assign(k, std::numeric_limits<float>::infinity());
        top_ids.assign(k, -1);  // use -1 as "empty slot"
    }

    // Max-heap over top_scores/top_ids: root (index 0) is always the worst
    // (largest) score. O(1) worst-score lookup, O(log k) update vs O(k) linear.
    auto sift_down = [&](int root) {
        while (true) {
            int largest = root;
            const int l = 2 * root + 1;
            const int r = 2 * root + 2;
            if (l < k && top_scores[l] > top_scores[largest]) largest = l;
            if (r < k && top_scores[r] > top_scores[largest]) largest = r;
            if (largest == root) break;
            std::swap(top_scores[root], top_scores[largest]);
            std::swap(top_ids[root],    top_ids[largest]);
            root = largest;
        }
    };

    // For L2 with precomputed q_norm and cached v_norms:
    //   ||q-v||² = q_norm + norms_[row] - 2·dot(q,v)
    // The inner loop becomes one ip_score (FMA only, no subtract) + 2 scalar ops,
    // vs l2_dist (subtract+FMA per element) — roughly 2x faster for the distance.
    // q_norm is computed once per query across all nprobe clusters (not per-cluster).
    const bool use_decomposed = (metric_ == Metric::L2)
                                && (q_norm_sq >= 0.0f)
                                && (norms_.size() == N);
    const size_t D = static_cast<size_t>(dim_);

    float worst_score = top_scores[0];  // O(1): always at heap root

    for (size_t row = 0; row < N; ++row) {
        if (!skip_alive_check && !alive_[row]) continue;
        const float* v = mat_.data() + row * D;

        float s;
        if (use_decomposed) {
            s = q_norm_sq + norms_[row] - 2.0f * ip_score(query, v, dim_);
        } else {
            s = score_(query, v);
        }

        if (s < worst_score) {
            top_scores[0] = s;
            top_ids[0]    = ids_[row];
            sift_down(0);
            worst_score = top_scores[0];
        }
    }
}

void Cluster::search_into_timed(const float* query, float q_norm_sq, int k,
                                std::vector<DocId>& top_ids,
                                std::vector<float>& top_scores,
                                bool skip_alive_check,
                                int64_t* lock_ns,
                                int64_t* scan_ns) const {
    if (!query || k <= 0) return;

    using clk = std::chrono::steady_clock;

    const auto t0 = clk::now();
    std::shared_lock lk(mu_);
    const auto t1 = clk::now();

    const size_t N = ids_.size();
    if (N == 0) return;

    if ((int)top_scores.size() < k) {
        top_scores.assign(k, std::numeric_limits<float>::infinity());
        top_ids.assign(k, -1);
    }

    auto sift_down = [&](int root) {
        while (true) {
            int largest = root;
            const int l = 2 * root + 1;
            const int r = 2 * root + 2;
            if (l < k && top_scores[l] > top_scores[largest]) largest = l;
            if (r < k && top_scores[r] > top_scores[largest]) largest = r;
            if (largest == root) break;
            std::swap(top_scores[root], top_scores[largest]);
            std::swap(top_ids[root],    top_ids[largest]);
            root = largest;
        }
    };

    const bool use_decomposed = (metric_ == Metric::L2)
                                && (q_norm_sq >= 0.0f)
                                && (norms_.size() == N);
    const size_t D = static_cast<size_t>(dim_);
    float worst_score = top_scores[0];

    for (size_t row = 0; row < N; ++row) {
        if (!skip_alive_check && !alive_[row]) continue;
        const float* v = mat_.data() + row * D;
        float s;
        if (use_decomposed) {
            s = q_norm_sq + norms_[row] - 2.0f * ip_score(query, v, dim_);
        } else {
            s = score_(query, v);
        }
        if (s < worst_score) {
            top_scores[0] = s;
            top_ids[0]    = ids_[row];
            sift_down(0);
            worst_score = top_scores[0];
        }
    }

    const auto t2 = clk::now();

    if (lock_ns) *lock_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    if (scan_ns) *scan_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

size_t Cluster::scan_batch_l2(
        const float* queries,
        const float* q_norms_sq,
        size_t n_queries,
        std::vector<float>& dists_out,
        std::vector<DocId>& live_ids_out) const {
    if (n_queries == 0 || metric_ != Metric::L2) return 0;

    std::shared_lock lk(mu_);
    const size_t N_total = ids_.size();
    if (N_total == 0 || live_count_ == 0) return 0;

    // Fast path: no tombstones — use mat_ directly without copying.
    const bool all_live = (live_count_ == N_total);
    const float* scan_mat   = nullptr;
    const float* scan_norms = nullptr;
    std::vector<float> tmp_mat, tmp_norms;
    size_t N;

    if (all_live) {
        scan_mat   = mat_.data();
        scan_norms = norms_.data();
        live_ids_out.assign(ids_.begin(), ids_.end());
        N = N_total;
    } else {
        live_ids_out.clear();
        live_ids_out.reserve(live_count_);
        tmp_mat.reserve(live_count_ * (size_t)dim_);
        tmp_norms.reserve(live_count_);
        for (size_t row = 0; row < N_total; ++row) {
            if (!alive_[row]) continue;
            live_ids_out.push_back(ids_[row]);
            tmp_mat.insert(tmp_mat.end(),
                           mat_.begin() + (ptrdiff_t)(row * (size_t)dim_),
                           mat_.begin() + (ptrdiff_t)((row + 1) * (size_t)dim_));
            tmp_norms.push_back(norms_[row]);
        }
        scan_mat   = tmp_mat.data();
        scan_norms = tmp_norms.data();
        N = live_ids_out.size();
    }
    if (N == 0) return 0;

    dists_out.resize(n_queries * N);

    // sgemm: dists = -2 * queries × scan_mat^T  →  [n_queries × N]
    // gives -2·dot(q_i, v_j) for each (i,j)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)n_queries, (int)N, dim_,
                -2.0f, queries, dim_,
                scan_mat, dim_,
                0.0f, dists_out.data(), (int)N);

    // L2 fixup: add q_norm²[i] + v_norm²[j]
    for (size_t qi = 0; qi < n_queries; ++qi) {
        float* row = dists_out.data() + qi * N;
        const float qn = q_norms_sq[qi];
        for (size_t vi = 0; vi < N; ++vi)
            row[vi] += qn + scan_norms[vi];
    }
    return N;
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

uint64_t Cluster::get_last_access_time(DocId id) const {
    std::shared_lock lk(mu_);
    auto it = id2row_.find(id);
    if (it == id2row_.end()) return 0;
    size_t row = it->second;
    return row < last_access_time_.size() ? last_access_time_[row] : 0;
}

void Cluster::set_last_access_time(DocId id, uint64_t time_ns) {
    std::unique_lock lk(mu_);
    auto it = id2row_.find(id);
    if (it == id2row_.end()) return;
    size_t row = it->second;
    if (row < last_access_time_.size())
        last_access_time_[row] = time_ns;
}

void Cluster::export_live(std::vector<DocId>& out_ids, std::vector<float>& out_vecs) const {
    std::shared_lock lk(mu_);
    const size_t N = ids_.size();
    out_ids.reserve(out_ids.size() + live_count_);
    out_vecs.reserve(out_vecs.size() + live_count_ * (size_t)dim_);
    for (size_t row = 0; row < N; ++row) {
        if (!alive_[row]) continue;
        out_ids.push_back(ids_[row]);
        const float* v = &mat_[row * (size_t)dim_];
        out_vecs.insert(out_vecs.end(), v, v + dim_);
    }
}

void Cluster::get_coldest_doc_ids(size_t n, std::vector<DocId>& out_ids) const {
    std::shared_lock lk(mu_);
    const size_t N = ids_.size();
    if (n == 0 || live_count_ == 0) return;
    std::vector<std::pair<uint64_t, DocId>> pairs;
    pairs.reserve(live_count_);
    for (size_t row = 0; row < N; ++row) {
        if (!alive_[row]) continue;
        uint64_t t = (row < last_access_time_.size()) ? last_access_time_[row] : 0;
        pairs.emplace_back(t, ids_[row]);
    }
    std::sort(pairs.begin(), pairs.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
    const size_t take = std::min(n, pairs.size());
    for (size_t i = 0; i < take; ++i)
        out_ids.push_back(pairs[i].second);
}

void Cluster::compact() {
    std::unique_lock lk(mu_);

    const size_t N = ids_.size();
    if (N == 0 || live_count_ == N) return;

    std::vector<DocId> new_ids;
    std::vector<float> new_mat;
    std::vector<float> new_norms;
    std::vector<uint8_t> new_alive;
    std::vector<uint64_t> new_access;
    new_ids.reserve(live_count_);
    new_mat.reserve(live_count_ * (size_t)dim_);
    new_norms.reserve(live_count_);
    new_alive.reserve(live_count_);
    new_access.reserve(live_count_);
    std::unordered_map<DocId, uint32_t> new_map;
    new_map.reserve(live_count_);

    for (size_t row = 0; row < N; ++row) {
        if (!alive_[row]) continue;
        uint32_t new_row = static_cast<uint32_t>(new_ids.size());
        new_ids.push_back(ids_[row]);
        const float* src = &mat_[row * (size_t)dim_];
        new_mat.insert(new_mat.end(), src, src + dim_);
        new_norms.push_back(row < norms_.size() ? norms_[row]
                                                 : ip_score(src, src, dim_));
        new_alive.push_back(1u);
        if (row < last_access_time_.size())
            new_access.push_back(last_access_time_[row]);
        else
            new_access.push_back(0);
        new_map.emplace(new_ids.back(), new_row);
    }

    ids_.swap(new_ids);
    mat_.swap(new_mat);
    norms_.swap(new_norms);
    alive_.swap(new_alive);
    last_access_time_.swap(new_access);
    id2row_.swap(new_map);
    live_count_ = ids_.size();

    assert(mat_.size() == ids_.size() * (size_t)dim_);
    assert(alive_.size() == ids_.size());
    assert(norms_.size() == ids_.size());
    assert(last_access_time_.size() == ids_.size());
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
