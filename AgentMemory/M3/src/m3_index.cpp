#include "m3_index.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>

namespace m3 {

namespace {

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

IVFIndex::IVFIndex(int dim, Metric metric, bool normalized)
    : dim_(dim)
    , metric_(metric)
    , normalized_(normalized)
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
    centroids_ = centroids;

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
    return cid;
}

void IVFIndex::set_centroid(int cluster_id, const std::vector<float>& centroid) {
    if ((int)centroid.size() != dim_) {
        throw std::invalid_argument("IVFIndex::set_centroid: centroid dim mismatch");
    }

    std::unique_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
        throw std::out_of_range("IVFIndex::set_centroid: invalid cluster_id");
    }
    // update snapshot centroid
    std::copy(centroid.begin(), centroid.end(),
              centroids_.begin() + cluster_id * (size_t)dim_);

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

const float* IVFIndex::centroid_ptr(int cluster_id) const {
    std::shared_lock lk(topo_mu_);
    if (cluster_id < 0 || cluster_id >= (int)clusters_.size()) return nullptr;
    return &centroids_[cluster_id * (size_t)dim_];
}

void IVFIndex::add_batch(int cluster_id, const DocId* ids, const float* vecs, size_t n_rows) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
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
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
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
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
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
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
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

void IVFIndex::compact_cluster(int cluster_id) {
    std::shared_ptr<Cluster> c;
    {
        std::shared_lock lk(topo_mu_);
        if (cluster_id < 0 || cluster_id >= (int)clusters_.size() || !clusters_[cluster_id]) {
            return;
        }
        c = clusters_[cluster_id];
    }
    c->compact();
}

void IVFIndex::maintenance_pass() {
    // simplest policy:
    // 1) for each cluster, if live_size is far smaller than size → compact
    // 2) if size is too big -> TODO: split
    const size_t MAX_ROWS_BEFORE_SPLIT = 200000; // example number
    const double COMPACT_RATIO = 0.7; // if live/size < 0.7 -> compact

    std::vector<std::shared_ptr<Cluster>> snapshot;
    {
        std::shared_lock lk(topo_mu_);
        snapshot = clusters_; // shallow copy of shared_ptr
    }

    for (size_t cid = 0; cid < snapshot.size(); ++cid) {
        auto& c = snapshot[cid];
        if (!c) continue;

        const size_t sz = c->size();
        const size_t live = c->live_size();

        if (sz == 0) continue;

        if ((double)live / (double)sz < COMPACT_RATIO) {
            c->compact();
        }

        if (c->size() > MAX_ROWS_BEFORE_SPLIT) {
            // TODO: split cluster 'cid' into two clusters:
            //  - need Cluster to expose iteration or export of ids+vectors
            //  - create a new cluster with same dim/metric
            //  - redistribute points
            // For now, just compact (already done above).
        }
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
                             std::vector<std::vector<float>>& out_scores) const {
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;

    // Snapshot once
    std::vector<std::shared_ptr<Cluster>> clusters_snap;
    std::vector<float> centroids_snap;
    {
        std::shared_lock lk(topo_mu_);
        clusters_snap = clusters_;
        centroids_snap = centroids_;
    }

    const int nlist = (int)clusters_snap.size();
    if (nlist == 0) return;

    if (nprobe <= 0) nprobe = nlist;
    const int real_nprobe = std::min(nprobe, nlist);

    // Reusable buffer for selected cluster ids (per query)
    std::vector<int> chosen;
    chosen.reserve(real_nprobe);

    // print cluster_snap for debug
    // for (int ci = 0; ci < nlist; ++ci) {
    //     auto c = clusters_snap[ci];
    //     if (c) {
    //         printf("Cluster %d: size=%zu live=%zu\n", ci, c->size(), c->live_size());
    //     } else {
    //         printf("Cluster %d: <null>\n", ci);
    //     }
    // }

    // return;

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* q = queries + qi * (size_t)dim_;

        // 1) pick top-nprobe clusters by centroid distance
        chosen.clear();
        select_nprobe_for_query(q, centroids_snap, real_nprobe, chosen);
        // chosen.size() <= real_nprobe

        // print chosen for debug
        // printf("Query %zu: chosen clusters:", qi);
        // for (int cid : chosen) {
        //     printf(" %d", cid);
        // }
        // printf("\n");

        // 2) prepare a single top-k buffer for THIS query
        std::vector<DocId> top_ids(k, (DocId)-1);
        std::vector<float> top_scores(k, std::numeric_limits<float>::infinity());

        // 3) let each selected cluster try to improve this buffer
        for (int cid : chosen) {
            // printf(" Searching cluster %d\n", cid);

            if (cid < 0 || cid >= (int)clusters_snap.size()) continue;
            auto c = clusters_snap[cid];
            if (!c) continue;
            c->search_into(q, k, top_ids, top_scores);

            // printf("  After cluster %d: top_scores =", cid);
            // for (float s : top_scores) {
            //     printf(", %.4f", s);
            // }
            // for (DocId id : top_ids) {
            //     printf(", %ld", id);
            // }
            // printf("\n");
        }

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

        // printf(" Query %zu: final results:", qi);
        // for (size_t i = 0; i < oi.size(); ++i) {
        //     printf(" (id=%ld, score=%.4f)", oi[i], os[i]);
        // }
        // printf("\n");
    }
}

void IVFIndex::snapshot(std::vector<std::shared_ptr<Cluster>>& out_clusters,
                        std::vector<float>& out_centroids) const {
    std::shared_lock lk(topo_mu_);
    out_clusters = clusters_;
    out_centroids = centroids_;
}

void IVFIndex::select_nprobe_for_query(const float* q,
                                       const std::vector<float>& centroids_snapshot,
                                       int nprobe,
                                       std::vector<int>& out_ids) const {
    const int nlist = (int)(centroids_snapshot.size() / (size_t)dim_);
    out_ids.clear();
    if (nlist == 0 || nprobe <= 0) return;

    // compute score to each centroid
    std::vector<std::pair<float,int>> tmp;
    tmp.reserve(nlist);
    for (int cid = 0; cid < nlist; ++cid) {
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
