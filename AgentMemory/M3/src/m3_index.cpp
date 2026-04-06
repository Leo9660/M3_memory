#include "m3_index.h"
#include "m3_logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <mutex>
#include <cblas.h>

namespace {
using idx_clock = std::chrono::steady_clock;
inline double idx_fms(idx_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}
} // anonymous namespace

namespace m3 {

namespace {

// K-means k=2 on row-major vecs [n_rows, dim]. Returns two centroids and assignment (0 or 1) per row.
static void kmeans2(const float* vecs, size_t n_rows, int dim,
                    Metric metric, bool normalized,
                    std::vector<float>& out_c0, std::vector<float>& out_c1,
                    std::vector<int>& assign) {
    assign.resize(n_rows);
    if (n_rows == 0) return;
    out_c0.resize(static_cast<size_t>(dim));
    out_c1.resize(static_cast<size_t>(dim));
    const size_t d = static_cast<size_t>(dim);

    // Initialize: c0 = first vector, c1 = vector at n/2 (or last)
    const float* v0 = vecs;
    const float* v1 = vecs + (n_rows / 2) * d;
    if (n_rows == 1) { v1 = v0; }
    std::copy(v0, v0 + dim, out_c0.begin());
    std::copy(v1, v1 + dim, out_c1.begin());

    const int max_iter = 10;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Assign each vector to nearest centroid
        for (size_t i = 0; i < n_rows; ++i) {
            const float* v = vecs + i * d;
            float s0 = unified_score(v, out_c0.data(), dim, metric, normalized);
            float s1 = unified_score(v, out_c1.data(), dim, metric, normalized);
            assign[i] = (s0 <= s1) ? 0 : 1;
        }
        // Recompute centroids as mean of assigned vectors
        size_t n0 = 0, n1 = 0;
        std::fill(out_c0.begin(), out_c0.end(), 0.f);
        std::fill(out_c1.begin(), out_c1.end(), 0.f);
        for (size_t i = 0; i < n_rows; ++i) {
            const float* v = vecs + i * d;
            if (assign[i] == 0) {
                ++n0;
                for (size_t j = 0; j < d; ++j) out_c0[j] += v[j];
            } else {
                ++n1;
                for (size_t j = 0; j < d; ++j) out_c1[j] += v[j];
            }
        }
        if (n0 > 0)
            for (size_t j = 0; j < d; ++j) out_c0[j] /= static_cast<float>(n0);
        if (n1 > 0)
            for (size_t j = 0; j < d; ++j) out_c1[j] /= static_cast<float>(n1);
    }
}

// merge helper: merge multiple partial top-k lists (already sorted asc by score)
// inputs: per-cluster results for ONE query: vector< vector<Pair> >
// output: final top-k ids/scores
static void merge_cluster_results_for_one_query(
    const std::vector<std::vector<DocId>>& per_ids,
    const std::vector<std::vector<float>>& per_scores,
    int k,
    std::vector<DocId>& out_ids,
    std::vector<float>& out_scores)
{
    // trivial cases
    if (per_ids.empty()) {
        out_ids.clear();
        out_scores.clear();
        return;
    }

    // we'll do a simple k-way merge using a min-heap
    struct Item {
        float score;
        DocId id;
        size_t list_idx;
        size_t elem_idx;
    };
    struct Cmp {
        bool operator()(const Item& a, const Item& b) const {
            return a.score > b.score; // min-heap
        }
    };

    std::priority_queue<Item, std::vector<Item>, Cmp> pq;

    const size_t m = per_ids.size();
    for (size_t li = 0; li < m; ++li) {
        if (!per_ids[li].empty()) {
            pq.push(Item{
                per_scores[li][0],
                per_ids[li][0],
                li,
                0
            });
        }
    }

    out_ids.clear();
    out_scores.clear();
    out_ids.reserve(k);
    out_scores.reserve(k);

    while (!pq.empty() && (int)out_ids.size() < k) {
        Item cur = pq.top(); pq.pop();
        out_ids.push_back(cur.id);
        out_scores.push_back(cur.score);

        // push next element from same list
        const size_t li = cur.list_idx;
        const size_t next_idx = cur.elem_idx + 1;
        if (next_idx < per_ids[li].size()) {
            pq.push(Item{
                per_scores[li][next_idx],
                per_ids[li][next_idx],
                li,
                next_idx
            });
        }
    }
}

} // anonymous namespace

// ======================================================================
// IVFIndex impl
// ======================================================================

IVFIndex::IVFIndex(int dim, Metric metric, bool normalized, const char* layer_name)
    : dim_(dim)
    , metric_(metric)
    , normalized_(normalized)
    , layer_name_(layer_name ? layer_name : "??")
{
    if (dim_ <= 0) {
        throw std::invalid_argument("IVFIndex: dim must be > 0");
    }
}

void IVFIndex::set_centroids(const std::vector<float>& centroids) {
    if (centroids.size() % (size_t)dim_ != 0) {
        throw std::invalid_argument("IVFIndex::set_centroids: size mismatch");
    }
    const int nlist = (int)(centroids.size() / (size_t)dim_);

    std::unique_lock lk(topo_mu_);

    // resize clusters_ to nlist, creating empty clusters if needed
    clusters_.resize(nlist);
    valid_.assign(static_cast<size_t>(nlist), true);
    centroids_ = centroids;

    // Rebuild compact view from scratch — all slots are valid after set_centroids.
    compact_centroids_ = centroids;
    compact_to_orig_.resize(static_cast<size_t>(nlist));
    orig_to_compact_.resize(static_cast<size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        compact_to_orig_[static_cast<size_t>(cid)] = cid;
        orig_to_compact_[static_cast<size_t>(cid)] = cid;
    }

    for (int cid = 0; cid < nlist; ++cid) {
        if (!clusters_[cid]) {
            // create empty cluster with this centroid
            std::vector<float> c(dim_);
            std::copy(centroids_.begin() + cid * dim_,
                      centroids_.begin() + (cid + 1) * dim_,
                      c.begin());
            clusters_[cid] = std::make_shared<Cluster>(dim_, metric_, normalized_, cid, c);
        } else {
            // update centroid in existing cluster
            std::vector<float> c(dim_);
            std::copy(centroids_.begin() + cid * dim_,
                      centroids_.begin() + (cid + 1) * dim_,
                      c.begin());
            // cluster has its own centroid storage; just overwrite it
            // simplest: recreate cluster centroid through a small setter
            // but since current Cluster ctor takes centroid, we'll rely on set_centroid(...)
            // handled below
            clusters_[cid]->compact(); // optional: make sure structure is clean
        }
    }
}

int IVFIndex::add_cluster(const std::vector<float>& centroid) {
    if ((int)centroid.size() != dim_) {
        throw std::invalid_argument("IVFIndex::add_cluster: centroid dim mismatch");
    }

    std::unique_lock lk(topo_mu_);
    const int cid = (int)clusters_.size();
    clusters_.push_back(std::make_shared<Cluster>(dim_, metric_, normalized_, cid, centroid));
    centroids_.insert(centroids_.end(), centroid.begin(), centroid.end());
    valid_.push_back(true);

    // Append to compact view.
    const int compact_idx = (int)compact_to_orig_.size();
    compact_centroids_.insert(compact_centroids_.end(), centroid.begin(), centroid.end());
    compact_to_orig_.push_back(cid);
    if ((int)orig_to_compact_.size() <= cid)
        orig_to_compact_.resize(static_cast<size_t>(cid + 1), -1);
    orig_to_compact_[static_cast<size_t>(cid)] = compact_idx;

    return cid;
}

// Internal helper: remove a cluster without acquiring topo_mu_.
// Caller must already hold a unique_lock on topo_mu_.
void IVFIndex::remove_cluster_nolock_(int cluster_id) {
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size()) return;
    valid_[static_cast<size_t>(cluster_id)] = false;
    clusters_[static_cast<size_t>(cluster_id)].reset();

    // Swap-remove from compact view: move the last compact slot into the gap
    // left by the removed cluster, then pop the (now-duplicate) last slot.
    // Cost: O(dim) centroid copy — independent of total_slots.
    const size_t cid_sz = static_cast<size_t>(cluster_id);
    if (cid_sz >= orig_to_compact_.size() || orig_to_compact_[cid_sz] < 0) return;
    const int ci      = orig_to_compact_[cid_sz];
    const int last_ci = (int)compact_to_orig_.size() - 1;
    if (ci != last_ci) {
        // Overwrite slot ci with last slot's centroid and origin mapping.
        const int last_orig = compact_to_orig_[static_cast<size_t>(last_ci)];
        const size_t D = static_cast<size_t>(dim_);
        std::copy(compact_centroids_.begin() + last_ci * D,
                  compact_centroids_.begin() + (last_ci + 1) * D,
                  compact_centroids_.begin() + ci * D);
        compact_to_orig_[static_cast<size_t>(ci)] = last_orig;
        orig_to_compact_[static_cast<size_t>(last_orig)] = ci;
    }
    compact_centroids_.resize(compact_centroids_.size() - static_cast<size_t>(dim_));
    compact_to_orig_.pop_back();
    orig_to_compact_[cid_sz] = -1;
}

void IVFIndex::remove_cluster(int cluster_id) {
    std::unique_lock lk(topo_mu_);
    remove_cluster_nolock_(cluster_id);
}

void IVFIndex::set_centroid(int cluster_id, const std::vector<float>& centroid) {
    if ((int)centroid.size() != dim_) {
        throw std::invalid_argument("IVFIndex::set_centroid: centroid dim mismatch");
    }

    std::unique_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)]) {
        throw std::out_of_range("IVFIndex::set_centroid: invalid cluster_id");
    }
    // Update raw centroids_ array.
    std::copy(centroid.begin(), centroid.end(),
              centroids_.begin() + cluster_id * (size_t)dim_);

    // Mirror into compact view.
    const int ci = (static_cast<size_t>(cluster_id) < orig_to_compact_.size())
                   ? orig_to_compact_[static_cast<size_t>(cluster_id)] : -1;
    if (ci >= 0) {
        std::copy(centroid.begin(), centroid.end(),
                  compact_centroids_.begin() + ci * static_cast<size_t>(dim_));
    }

    // update underlying cluster's centroid
    // current Cluster does not expose a "set_centroid" method,
    // so for now we rely on the stored centroid only for routing.
    // If you want the cluster to actually change its internal centroid_,
    // add a setter to Cluster.
}

int IVFIndex::nlist() const {
    std::shared_lock lk(topo_mu_);
    return (int)clusters_.size();
}

int IVFIndex::live_nlist() const {
    std::shared_lock lk(topo_mu_);
    return (int)std::count(valid_.begin(), valid_.end(), true);
}

const float* IVFIndex::centroid_ptr(int cluster_id) const {
    std::shared_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size()) return nullptr;
    if (!valid_[static_cast<size_t>(cluster_id)]) return nullptr;
    return &centroids_[cluster_id * (size_t)dim_];
}

void IVFIndex::add_batch(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows,
                         bool allow_missing) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            if (allow_missing) return;
            const bool oob      = (cluster_id < 0 || cluster_id >= (int)clusters_.size());
            const bool not_valid = !oob && !valid_[static_cast<size_t>(cluster_id)];
            const bool null_ptr  = !oob && clusters_[static_cast<size_t>(cluster_id)] == nullptr;
            M3Logger::instance().log_add_batch_invalid(
                layer_name_, cluster_id, clusters_.size(), oob, not_valid, null_ptr);
            throw std::out_of_range("IVFIndex::add_batch: invalid cluster_id");
        }
        c = clusters_[cluster_id];
    }
    c->add_batch(ids, vecs, n_rows);
}

void IVFIndex::update_batch(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows,
                            bool insert_if_absent) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            throw std::out_of_range("IVFIndex::update_batch: invalid cluster_id");
        }
        c = clusters_[cluster_id];
    }
    c->update_batch(ids, vecs, n_rows, insert_if_absent);
}

void IVFIndex::erase_batch(int cluster_id, const DocId* ids, size_t n_rows) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            throw std::out_of_range("IVFIndex::erase_batch: invalid cluster_id");
        }
        c = clusters_[cluster_id];
    }
    c->erase_batch(ids, n_rows);
}

void IVFIndex::rebuild_cluster(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            throw std::out_of_range("IVFIndex::rebuild_cluster: invalid cluster_id");
        }
        c = clusters_[cluster_id];
    }
    c->rebuild_from(ids, vecs, n_rows);
}

int IVFIndex::nearest_cluster(const float* vec) const {
    if (!vec) return -1;
    std::shared_lock lk(topo_mu_);
    const int nlist = (int)clusters_.size();
    if (nlist == 0 || centroids_.empty()) return -1;
    float best = std::numeric_limits<float>::infinity();
    int best_id = -1;
    for (int cid = 0; cid < nlist; ++cid) {
        if (cid >= (int)valid_.size() || !valid_[static_cast<size_t>(cid)]) continue;
        const float* c = &centroids_[cid * (size_t)dim_];
        float s = unified_score(vec, c, dim_, metric_, normalized_);
        if (s < best) {
            best = s;
            best_id = cid;
        }
    }
    return best_id;
}

void IVFIndex::nearest_clusters(const float* vecs, size_t n_rows, std::vector<int>& out) const {
    out.assign(n_rows, -1);
    if (!vecs || n_rows == 0) return;
    std::shared_lock lk(topo_mu_);
    const int nlist = (int)clusters_.size();
    if (nlist == 0 || centroids_.empty()) return;

    for (size_t i = 0; i < n_rows; ++i) {
        const float* v = vecs + i * (size_t)dim_;
        float best = std::numeric_limits<float>::infinity();
        int best_id = -1;
        for (int cid = 0; cid < nlist; ++cid) {
            if (cid >= (int)valid_.size() || !valid_[static_cast<size_t>(cid)]) continue;
            const float* c = &centroids_[cid * (size_t)dim_];
            float s = unified_score(v, c, dim_, metric_, normalized_);
            if (s < best) {
                best = s;
                best_id = cid;
            }
        }
        out[i] = best_id;
    }
}

const float* IVFIndex::cluster_get_vector(int cluster_id, DocId id) const {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            return nullptr;
        }
        c = clusters_[cluster_id];
    }
    return c->get_vector(id);
}

size_t IVFIndex::cluster_live_size(int cluster_id) const {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            return 0;
        }
        c = clusters_[cluster_id];
    }
    return c->live_size();
}

void IVFIndex::compact_cluster(int cluster_id) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id]) {
            return;
        }
        c = clusters_[cluster_id];
    }
    c->compact();
}

void IVFIndex::maintenance_pass() {
    // 1) Compact clusters with high tombstone ratio
    // 2) Split any cluster over threshold (one per pass)
    // 3) Optionally merge two smallest clusters if both below merge threshold
    const size_t MAX_ROWS_BEFORE_SPLIT = 200000;
    const size_t MAX_ROWS_BEFORE_MERGE = 5000;  // merge if both clusters below this
    const double COMPACT_RATIO = 0.7;

    std::vector<std::shared_ptr<Cluster>> snapshot;
    std::vector<bool> valid_snap;
    {
        std::shared_lock lk(topo_mu_);
        snapshot = clusters_;
        valid_snap = valid_;
    }

    for (size_t cid = 0; cid < snapshot.size(); ++cid) {
        if (cid >= valid_snap.size() || !valid_snap[cid]) continue;
        auto& c = snapshot[cid];
        if (!c) continue;

        const size_t sz = c->size();
        const size_t live = c->live_size();

        if (sz == 0) continue;

        if ((double)live / (double)sz < COMPACT_RATIO) {
            c->compact();
        }

        if (c->live_size() > MAX_ROWS_BEFORE_SPLIT) {
            int new_cid = split_cluster(static_cast<int>(cid), MAX_ROWS_BEFORE_SPLIT);
            (void)new_cid;
            return;  // one split per pass
        }
    }

    // Merge pass: find two valid clusters both below threshold, merge smaller into larger
    int smallest_cid = -1, second_cid = -1;
    size_t smallest_size = SIZE_MAX, second_size = SIZE_MAX;
    for (size_t cid = 0; cid < snapshot.size(); ++cid) {
        if (cid >= valid_snap.size() || !valid_snap[cid]) continue;
        auto& c = snapshot[cid];
        if (!c) continue;
        size_t live = c->live_size();
        if (live == 0 || live > MAX_ROWS_BEFORE_MERGE) continue;
        if (live < smallest_size) {
            second_size = smallest_size;
            second_cid = smallest_cid;
            smallest_size = live;
            smallest_cid = static_cast<int>(cid);
        } else if (live < second_size) {
            second_size = live;
            second_cid = static_cast<int>(cid);
        }
    }
    if (smallest_cid >= 0 && second_cid >= 0 && smallest_cid != second_cid) {
        merge_clusters(second_cid, smallest_cid);  // merge smaller into larger
    }
}

int IVFIndex::split_cluster(int cluster_id, size_t max_vectors_before_split) {
    std::unique_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id])
        return -1;

    std::shared_ptr<Cluster> c = clusters_[cluster_id];
    std::vector<DocId> ids;
    std::vector<float> vecs;
    c->export_live(ids, vecs);
    const size_t n = ids.size();
    if (n < 2) return -1;
    if (n <= max_vectors_before_split) return -1;

    const size_t dim_sz = static_cast<size_t>(dim_);
    std::vector<float> c0, c1;
    std::vector<int> assign;
    kmeans2(vecs.data(), n, dim_, metric_, normalized_, c0, c1, assign);

    std::vector<DocId> ids0, ids1;
    std::vector<float> vecs0, vecs1;
    ids0.reserve(n);
    ids1.reserve(n);
    vecs0.reserve(n * dim_sz);
    vecs1.reserve(n * dim_sz);
    for (size_t i = 0; i < n; ++i) {
        if (assign[i] == 0) {
            ids0.push_back(ids[i]);
            vecs0.insert(vecs0.end(), vecs.data() + i * dim_sz, vecs.data() + (i + 1) * dim_sz);
        } else {
            ids1.push_back(ids[i]);
            vecs1.insert(vecs1.end(), vecs.data() + i * dim_sz, vecs.data() + (i + 1) * dim_sz);
        }
    }
    if (ids0.empty() || ids1.empty()) return -1;

    // Rebuild original cluster in-place.
    clusters_[static_cast<size_t>(cluster_id)]->rebuild_from(ids0.data(), vecs0.data(), ids0.size());
    std::copy(c0.begin(), c0.end(), centroids_.begin() + static_cast<size_t>(cluster_id) * dim_sz);

    // Mirror updated centroid into compact view for the original cluster.
    const int ci_orig = (static_cast<size_t>(cluster_id) < orig_to_compact_.size())
                        ? orig_to_compact_[static_cast<size_t>(cluster_id)] : -1;
    if (ci_orig >= 0) {
        std::copy(c0.begin(), c0.end(),
                  compact_centroids_.begin() + ci_orig * dim_sz);
    }

    // Create new cluster without re-locking topo_mu_ (avoid deadlock with add_cluster).
    const int new_cid = static_cast<int>(clusters_.size());
    M3Logger::instance().log_cpu_split(
        layer_name_, cluster_id, new_cid, n, ids0.size(), ids1.size());
    clusters_.push_back(std::make_shared<Cluster>(dim_, metric_, normalized_, new_cid, c1));
    centroids_.insert(centroids_.end(), c1.begin(), c1.end());
    valid_.push_back(true);

    // Append new cluster to compact view.
    const int compact_new = (int)compact_to_orig_.size();
    compact_centroids_.insert(compact_centroids_.end(), c1.begin(), c1.end());
    compact_to_orig_.push_back(new_cid);
    if ((int)orig_to_compact_.size() <= new_cid)
        orig_to_compact_.resize(static_cast<size_t>(new_cid + 1), -1);
    orig_to_compact_[static_cast<size_t>(new_cid)] = compact_new;

    // Fill new cluster with its assigned vectors.
    clusters_[static_cast<size_t>(new_cid)]->add_batch(ids1.data(), vecs1.data(), ids1.size());
    return new_cid;
}

void IVFIndex::merge_clusters(int cluster_id_a, int cluster_id_b) {
    std::unique_lock lk(topo_mu_);
    if (cluster_id_a == cluster_id_b) return;
    if (cluster_id_a < 0 || cluster_id_a >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id_a)] || !clusters_[cluster_id_a])
        return;
    if (cluster_id_b < 0 || cluster_id_b >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id_b)] || !clusters_[cluster_id_b])
        return;

    std::shared_ptr<Cluster> cb = clusters_[cluster_id_b];
    std::vector<DocId> ids_b;
    std::vector<float> vecs_b;
    cb->export_live(ids_b, vecs_b);
    if (ids_b.empty()) {
        M3Logger::instance().log_cpu_merge(
            layer_name_, cluster_id_a, cluster_id_b, 0, 0,
            "src_cluster_fully_tombstoned");
        remove_cluster_nolock_(cluster_id_b);
        return;
    }

    const size_t n_a_old = clusters_[static_cast<size_t>(cluster_id_a)]->live_size();
    const size_t n_b = ids_b.size();
    M3Logger::instance().log_cpu_merge(
        layer_name_, cluster_id_a, cluster_id_b, n_a_old, n_b,
        "both_below_5k_threshold");
    clusters_[static_cast<size_t>(cluster_id_a)]->add_batch(ids_b.data(), vecs_b.data(), ids_b.size());

    const size_t dim_sz = static_cast<size_t>(dim_);
    const float* old_ca = &centroids_[static_cast<size_t>(cluster_id_a) * dim_sz];
    if ((n_a_old + n_b) > 0) {
        float* dst = &centroids_[static_cast<size_t>(cluster_id_a) * dim_sz];
        for (size_t d = 0; d < dim_sz; ++d) {
            float sum = old_ca[d] * static_cast<float>(n_a_old);
            for (size_t i = 0; i < n_b; ++i)
                sum += vecs_b[i * dim_sz + d];
            dst[d] = sum / static_cast<float>(n_a_old + n_b);
        }
    }
    // Remove cluster_b from compact view via swap-remove.
    remove_cluster_nolock_(cluster_id_b);

    // Mirror the updated centroid of cluster_a into compact view.
    const int ci_a = (static_cast<size_t>(cluster_id_a) < orig_to_compact_.size())
                     ? orig_to_compact_[static_cast<size_t>(cluster_id_a)] : -1;
    if (ci_a >= 0) {
        const float* new_cent = &centroids_[static_cast<size_t>(cluster_id_a) * dim_sz];
        std::copy(new_cent, new_cent + dim_sz,
                  compact_centroids_.begin() + ci_a * dim_sz);
    }
}

void IVFIndex::export_cluster_live(int cluster_id,
                                   std::vector<DocId>& out_ids,
                                   std::vector<float>& out_vecs) const {
    std::shared_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id])
        return;
    clusters_[cluster_id]->export_live(out_ids, out_vecs);
}

void IVFIndex::ensure_cluster(int cluster_id, const std::vector<float>& centroid) {
    if ((int)centroid.size() != dim_) {
        throw std::invalid_argument("IVFIndex::ensure_cluster: centroid dim mismatch");
    }
    std::unique_lock lk(topo_mu_);
    if (cluster_id < 0) return;
    const size_t cid = static_cast<size_t>(cluster_id);
    if (cid < clusters_.size() && valid_[cid])
        return;  // already valid
    if (cid >= clusters_.size()) {
        clusters_.resize(cid + 1);
        valid_.resize(cid + 1, false);
        centroids_.resize((cid + 1) * static_cast<size_t>(dim_));
    }
    valid_[cid] = true;
    std::copy(centroid.begin(), centroid.end(), centroids_.begin() + cid * static_cast<size_t>(dim_));
    clusters_[cid] = std::make_shared<Cluster>(dim_, metric_, normalized_, static_cast<int>(cid), centroid);

    // Add to compact view (slot may not have been in compact_ if it was previously invalid).
    if (cid >= orig_to_compact_.size())
        orig_to_compact_.resize(cid + 1, -1);
    if (orig_to_compact_[cid] < 0) {
        const int compact_idx = (int)compact_to_orig_.size();
        compact_centroids_.insert(compact_centroids_.end(), centroid.begin(), centroid.end());
        compact_to_orig_.push_back(static_cast<int>(cid));
        orig_to_compact_[cid] = compact_idx;
    }
}

void IVFIndex::search_on(const std::vector<int>& cluster_ids,
                         const float* queries, size_t q_rows, int k,
                         std::vector<std::vector<DocId>>& out_ids,
                         std::vector<std::vector<float>>& out_scores) const {
    // snapshot clusters so topology stays stable during the search
    std::vector<std::shared_ptr<Cluster>> clusters_snap;
    {
        std::shared_lock lk(topo_mu_);
        clusters_snap = clusters_;
    }

    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;
    if (cluster_ids.empty()) return;

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* qptr = queries + qi * (size_t)dim_;

        // one unified top-k buffer for THIS query
        std::vector<DocId>  top_ids(k, (DocId)-1);
        std::vector<float>  top_scores(k, std::numeric_limits<float>::infinity());

        // let each chosen cluster try to improve this buffer
        for (int cid : cluster_ids) {
            if (cid < 0 || cid >= (int)clusters_snap.size()) continue;
            auto c = clusters_snap[cid];
            if (!c) continue;
            c->search_into(qptr, k, top_ids, top_scores);
        }

        // compact & sort final results (remove empty slots)
        std::vector<int> idx;
        idx.reserve(k);
        for (int i = 0; i < k; ++i) {
            if (top_ids[i] != (DocId)-1) {
                idx.push_back(i);
            }
        }

        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return top_scores[a] < top_scores[b]; });

        auto& oi = out_ids[qi];
        auto& os = out_scores[qi];
        oi.resize(idx.size());
        os.resize(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            oi[i] = top_ids[idx[i]];
            os[i] = top_scores[idx[i]];
        }
    }
}

void IVFIndex::search_nprobe(const float* queries, size_t q_rows, int k, int nprobe,
                             std::vector<std::vector<DocId>>& out_ids,
                             std::vector<std::vector<float>>& out_scores,
                             double* out_centroid_ms, double* out_scan_ms) const {
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;

    const size_t D = static_cast<size_t>(dim_);

    // Snapshot the pre-built compact view — O(live_nlist × D) copy instead of
    // O(total_slots × D).  After N eviction cycles total_slots grows unboundedly
    // while live_nlist stays constant; the old approach was the dominant cost.
    std::vector<float>                   compact_centroids;
    std::vector<int>                     compact_to_orig;
    std::vector<std::shared_ptr<Cluster>> compact_clusters;
    {
        std::shared_lock lk(topo_mu_);
        compact_centroids = compact_centroids_;          // [live_nlist, dim]
        compact_to_orig   = compact_to_orig_;            // [live_nlist]
        const size_t n = compact_to_orig_.size();
        compact_clusters.resize(n);
        for (size_t i = 0; i < n; ++i)
            compact_clusters[i] = clusters_[static_cast<size_t>(compact_to_orig_[i])];
    }
    const int live_nlist = (int)compact_to_orig.size();
    if (live_nlist == 0) return;

    if (nprobe <= 0) nprobe = live_nlist;
    const int real_nprobe = std::min(nprobe, live_nlist);

    // ---------------------------------------------------------------
    // Batched centroid selection via BLAS sgemm over live clusters only.
    //
    // scores = alpha * Q_mat @ C_compact^T  (shape: Q × live_nlist)
    //
    // For IP/COSINE-normalized: score = -dot(q,c)  → alpha = -1
    // For L2:  score = ||q||^2 - 2*dot(q,c) + ||c||^2  → alpha = -2
    // ---------------------------------------------------------------
    const size_t NL = static_cast<size_t>(live_nlist);

    std::vector<float> scores_mat(q_rows * NL);

    double centroid_acc = 0.0;
    double scan_acc     = 0.0;
    const auto t_sgemm0 = (out_centroid_ms || out_scan_ms) ? idx_clock::now() : idx_clock::time_point{};

    if (metric_ == Metric::L2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)q_rows, live_nlist, dim_,
                    -2.0f,
                    queries,                    dim_,
                    compact_centroids.data(),   dim_,
                    0.0f,
                    scores_mat.data(),          live_nlist);

        std::vector<float> c_norms(NL);
        for (size_t ci = 0; ci < NL; ++ci)
            c_norms[ci] = cblas_sdot(dim_, compact_centroids.data() + ci * D, 1,
                                           compact_centroids.data() + ci * D, 1);

        for (size_t qi = 0; qi < q_rows; ++qi) {
            const float* q = queries + qi * D;
            float q_norm = cblas_sdot(dim_, q, 1, q, 1);
            float* row   = scores_mat.data() + qi * NL;
            for (size_t ci = 0; ci < NL; ++ci)
                row[ci] += q_norm + c_norms[ci];
        }
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)q_rows, live_nlist, dim_,
                    -1.0f,
                    queries,                    dim_,
                    compact_centroids.data(),   dim_,
                    0.0f,
                    scores_mat.data(),          live_nlist);
        if (metric_ == Metric::COSINE && normalized_) {
            for (float& s : scores_mat) s += 1.0f;
        }
    }

    if (out_centroid_ms || out_scan_ms) centroid_acc += idx_fms(idx_clock::now() - t_sgemm0);

    // Per-query: pick top-nprobe clusters and search them.
    std::vector<std::pair<float, int>> row_tmp;
    row_tmp.reserve(NL);

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* q         = queries + qi * D;
        const float* score_row = scores_mat.data() + qi * NL;

        const auto t_topk0 = (out_centroid_ms || out_scan_ms) ? idx_clock::now() : idx_clock::time_point{};

        // 1) pick top-nprobe compact indices by centroid distance.
        //    chosen holds compact indices (not original cids) so we can index
        //    compact_clusters[] directly in the scan step without a reverse lookup.
        row_tmp.clear();
        for (size_t ci = 0; ci < NL; ++ci)
            row_tmp.emplace_back(score_row[ci], (int)ci);

        std::vector<int> chosen;   // compact indices, sorted nearest-first
        chosen.reserve(real_nprobe);
        if (real_nprobe >= (int)row_tmp.size()) {
            std::sort(row_tmp.begin(), row_tmp.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (auto& p : row_tmp) chosen.push_back(p.second);
        } else {
            std::nth_element(row_tmp.begin(), row_tmp.begin() + real_nprobe, row_tmp.end(),
                             [](const auto& a, const auto& b){ return a.first < b.first; });
            row_tmp.resize(real_nprobe);
            std::sort(row_tmp.begin(), row_tmp.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (auto& p : row_tmp) chosen.push_back(p.second);
        }

        if (out_centroid_ms || out_scan_ms) centroid_acc += idx_fms(idx_clock::now() - t_topk0);

        // 2) prepare a single top-k buffer for THIS query
        std::vector<DocId> top_ids(k, (DocId)-1);
        std::vector<float> top_scores(k, std::numeric_limits<float>::infinity());

        const auto t_scan0 = (out_centroid_ms || out_scan_ms) ? idx_clock::now() : idx_clock::time_point{};

        // 3) scan each selected cluster (compact index → compact_clusters[ci])
        for (int ci : chosen) {
            if (ci < 0 || ci >= (int)compact_clusters.size()) continue;
            auto& c = compact_clusters[static_cast<size_t>(ci)];
            if (!c) continue;
            c->search_into(q, k, top_ids, top_scores);
        }

        if (out_centroid_ms || out_scan_ms) scan_acc += idx_fms(idx_clock::now() - t_scan0);

        // 4) compact and sort final results (remove empty -1 slots)
        std::vector<int> idx;
        idx.reserve(k);
        for (int i = 0; i < k; ++i) {
            if (top_ids[i] != (DocId)-1) {
                idx.push_back(i);
            }
        }

        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return top_scores[a] < top_scores[b]; });

        // printf(" Final top_scores =\n");
        // for (float s : top_scores) {
        //     printf(", %.4f", s);
        // }
        // for (DocId id : top_ids) {
        //     printf(", %ld", id);
        // }
        // for (size_t i = 0; i < idx.size(); ++i) {
        //     printf(" (%ld, %.4f)", top_ids[idx[i]], top_scores[idx[i]]);
        // }
        // printf("\n");

        auto& oi = out_ids[qi];
        auto& os = out_scores[qi];
        oi.resize(idx.size());
        os.resize(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            oi[i] = top_ids[idx[i]];
            os[i] = top_scores[idx[i]];
        }

    }
    if (out_centroid_ms) *out_centroid_ms = centroid_acc;
    if (out_scan_ms)     *out_scan_ms     = scan_acc;
}

void IVFIndex::get_probe_ids(const float* query, int nprobe, std::vector<int>& out_ids) const {
    std::vector<float> centroids_snap;
    std::vector<bool> valid_snap;
    {
        std::shared_lock lk(topo_mu_);
        centroids_snap = centroids_;
        valid_snap = valid_;
    }
    select_nprobe_for_query(query, centroids_snap, valid_snap, nprobe, out_ids);
}

void IVFIndex::batch_get_probe_ids(const float* queries, size_t q_rows, int nprobe,
                                    std::vector<std::vector<int>>& out_probe_ids,
                                    double* out_sgemm_ms, double* out_topk_ms) const {
    out_probe_ids.assign(q_rows, {});
    if (!queries || q_rows == 0 || nprobe <= 0) return;

    std::vector<float> centroids_snap;
    std::vector<bool>  valid_snap;
    {
        std::shared_lock lk(topo_mu_);
        centroids_snap = centroids_;
        valid_snap     = valid_;
    }

    const size_t D = static_cast<size_t>(dim_);

    // Compact: skip ghost slots so sgemm cost scales with live clusters only.
    std::vector<float> compact_centroids;
    std::vector<int>   compact_to_orig;
    {
        const size_t total_slots = centroids_snap.size() / D;
        compact_centroids.reserve(total_slots * D);
        compact_to_orig.reserve(total_slots);
        for (size_t ci = 0; ci < total_slots; ++ci) {
            if (ci < valid_snap.size() && valid_snap[ci]) {
                compact_centroids.insert(compact_centroids.end(),
                    centroids_snap.data() + ci * D,
                    centroids_snap.data() + ci * D + D);
                compact_to_orig.push_back((int)ci);
            }
        }
    }

    const int live_nlist  = (int)compact_to_orig.size();
    if (live_nlist == 0) return;
    const int real_nprobe = std::min(nprobe, live_nlist);
    const size_t NL       = static_cast<size_t>(live_nlist);

    // One sgemm: scores[q_rows × NL]
    std::vector<float> scores_mat(q_rows * NL);

    const auto t_sgemm_start = (out_sgemm_ms || out_topk_ms) ? idx_clock::now() : idx_clock::time_point{};

    if (metric_ == Metric::L2) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)q_rows, live_nlist, dim_,
                    -2.0f,
                    queries,                  dim_,
                    compact_centroids.data(), dim_,
                    0.0f,
                    scores_mat.data(),        live_nlist);

        std::vector<float> c_norms(NL);
        for (size_t ci = 0; ci < NL; ++ci)
            c_norms[ci] = cblas_sdot(dim_, compact_centroids.data() + ci * D, 1,
                                           compact_centroids.data() + ci * D, 1);
        for (size_t qi = 0; qi < q_rows; ++qi) {
            float q_norm = cblas_sdot(dim_, queries + qi * D, 1, queries + qi * D, 1);
            float* row   = scores_mat.data() + qi * NL;
            for (size_t ci = 0; ci < NL; ++ci)
                row[ci] += q_norm + c_norms[ci];
        }
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)q_rows, live_nlist, dim_,
                    -1.0f,
                    queries,                  dim_,
                    compact_centroids.data(), dim_,
                    0.0f,
                    scores_mat.data(),        live_nlist);
        if (metric_ == Metric::COSINE && normalized_)
            for (float& s : scores_mat) s += 1.0f;
    }

    const auto t_topk_start = (out_sgemm_ms || out_topk_ms) ? idx_clock::now() : idx_clock::time_point{};

    // Per-query top-nprobe selection from the score matrix.
    std::vector<std::pair<float, int>> row_tmp;
    row_tmp.reserve(NL);

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* score_row = scores_mat.data() + qi * NL;
        row_tmp.clear();
        for (size_t ci = 0; ci < NL; ++ci)
            row_tmp.emplace_back(score_row[ci], (int)ci);

        auto& chosen = out_probe_ids[qi];
        chosen.reserve(real_nprobe);
        if (real_nprobe >= (int)row_tmp.size()) {
            std::sort(row_tmp.begin(), row_tmp.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (auto& p : row_tmp) chosen.push_back(compact_to_orig[p.second]);
        } else {
            std::nth_element(row_tmp.begin(), row_tmp.begin() + real_nprobe, row_tmp.end(),
                             [](const auto& a, const auto& b){ return a.first < b.first; });
            row_tmp.resize(real_nprobe);
            std::sort(row_tmp.begin(), row_tmp.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });
            for (auto& p : row_tmp) chosen.push_back(compact_to_orig[p.second]);
        }
    }

    if (out_sgemm_ms) *out_sgemm_ms = idx_fms(t_topk_start  - t_sgemm_start);
    if (out_topk_ms)  *out_topk_ms  = idx_fms(idx_clock::now() - t_topk_start);
}

void IVFIndex::search_within_cluster(int cluster_id, const float* query, int k,
                                    std::vector<DocId>& out_ids,
                                    std::vector<float>& out_scores) const {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id])
            return;
        c = clusters_[cluster_id];
    }
    out_ids.clear();
    out_scores.clear();
    std::vector<std::vector<DocId>> ids(1);
    std::vector<std::vector<float>> scores(1);
    c->search(query, 1, k, ids, scores);
    out_ids = std::move(ids[0]);
    out_scores = std::move(scores[0]);
}

void IVFIndex::get_coldest_doc_ids(int cluster_id, size_t n, std::vector<DocId>& out_ids) const {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !valid_[static_cast<size_t>(cluster_id)] || !clusters_[cluster_id])
            return;
        c = clusters_[cluster_id];
    }
    c->get_coldest_doc_ids(n, out_ids);
}

void IVFIndex::snapshot(std::vector<std::shared_ptr<Cluster>>& out_clusters,
                        std::vector<float>& out_centroids,
                        std::vector<bool>& out_valid) const {
    std::shared_lock lk(topo_mu_);
    out_clusters = clusters_;
    out_centroids = centroids_;
    out_valid = valid_;
}

void IVFIndex::select_nprobe_for_query(const float* q,
                                       const std::vector<float>& centroids_snapshot,
                                       const std::vector<bool>& valid_snapshot,
                                       int nprobe,
                                       std::vector<int>& out_ids) const {
    const int nlist = (int)(centroids_snapshot.size() / (size_t)dim_);
    out_ids.clear();
    if (nlist == 0 || nprobe <= 0) return;

    // compute score to each valid centroid
    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(static_cast<size_t>(nlist));
    for (int cid = 0; cid < nlist; ++cid) {
        if (cid >= (int)valid_snapshot.size() || !valid_snapshot[static_cast<size_t>(cid)]) continue;
        const float* c = &centroids_snapshot[cid * (size_t)dim_];
        float s = unified_score(q, c, dim_, metric_, normalized_);
        tmp.emplace_back(s, cid);
    }

    // select smallest nprobe
    if (nprobe >= nlist) {
        std::sort(tmp.begin(), tmp.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
    } else {
        std::nth_element(tmp.begin(), tmp.begin() + nprobe, tmp.end(),
                         [](auto& a, auto& b){ return a.first < b.first; });
        tmp.resize(nprobe);
        std::sort(tmp.begin(), tmp.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
    }

    out_ids.reserve(tmp.size());
    for (auto& p : tmp) out_ids.push_back(p.second);
}

} // namespace m3
