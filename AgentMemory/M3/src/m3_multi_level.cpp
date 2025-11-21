#include "m3_multi_level.h"

#include <algorithm>
#include <limits>

namespace m3 {

namespace {
// Build a zero-centroid grid with nlist rows.
static std::vector<float> make_zero_centroids(int nlist, int dim) {
    return std::vector<float>(static_cast<size_t>(nlist) * static_cast<size_t>(dim), 0.0f);
}
} // namespace

MultiLevelIndex::MultiLevelIndex(int dim, Metric metric, bool normalized,
                                 MultiLevelConfig cfg)
    : dim_(dim)
    , metric_(metric)
    , normalized_(normalized)
    , cfg_(cfg) {
    if (dim_ <= 0) {
        throw std::invalid_argument("MultiLevelIndex: dim must be > 0");
    }
}

void MultiLevelIndex::ensure_layer_initialized_(Layer& layer, int nlist_hint) {
    if (layer.index) return;

    // Build a fresh IVFIndex with placeholder centroids so callers can write immediately.
    layer.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_);
    layer.centroids = make_zero_centroids(nlist_hint, dim_);
    layer.index->set_centroids(layer.centroids);
}

void MultiLevelIndex::ensure_layer_centroids_(Layer& layer, const std::vector<float>& centroids) {
    if (centroids.empty()) return;
    const int nlist = static_cast<int>(centroids.size() / static_cast<size_t>(dim_));
    if (layer.index) {
        layer.index->set_centroids(centroids);
    } else {
        layer.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_);
        layer.index->set_centroids(centroids);
    }
    layer.centroids = centroids;
    // defensive: if caller passed empty nlist, keep at least one cluster to avoid routing errors
    if (nlist == 0 && layer.index) {
        const auto zeros = make_zero_centroids(1, dim_);
        layer.index->set_centroids(zeros);
        layer.centroids = zeros;
    }
}

void MultiLevelIndex::set_l0_centroids(const std::vector<float>& centroids) {
    std::unique_lock lk(topo_mu_);
    ensure_layer_centroids_(l0_, centroids);
}

void MultiLevelIndex::set_l1_centroids(const std::vector<float>& centroids) {
    std::unique_lock lk(topo_mu_);
    ensure_layer_centroids_(l1_, centroids);
}

void MultiLevelIndex::set_l2_centroids(const std::vector<float>& centroids) {
    std::unique_lock lk(topo_mu_);
    ensure_layer_centroids_(l2_, centroids);
}

void MultiLevelIndex::insert(const DocId* ids, const float* vecs, size_t n_rows) {
    if (!ids || !vecs || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);
    ensure_layer_initialized_(l0_, cfg_.l0_nlist > 0 ? cfg_.l0_nlist : 1);

    const size_t dim_sz = static_cast<size_t>(dim_);
    const float threshold = cfg_.l0_new_cluster_threshold;
    const float merge_threshold = cfg_.l0_merge_threshold;
    const int max_nlist = cfg_.l0_max_nlist > 0 ? cfg_.l0_max_nlist
                                                : std::max(cfg_.l0_nlist, 1);

    struct Pending {
        std::vector<DocId> ids;
        std::vector<float> vecs;
        std::vector<float> sum;
        size_t count = 0;
    };

    auto ensure_pending_size = [](std::vector<Pending>& v, int cid) {
        if (cid < 0) return;
        if ((size_t)(cid + 1) > v.size()) v.resize(static_cast<size_t>(cid + 1));
    };

    std::vector<Pending> pending(static_cast<size_t>(l0_.index->nlist()));

    for (size_t i = 0; i < n_rows; ++i) {
        const float* vptr = vecs + i * dim_sz;

        const size_t cur_nlist = l0_.centroids.empty() ? 0 : l0_.centroids.size() / dim_sz;
        float best = std::numeric_limits<float>::infinity();
        int best_cid = -1;
        for (size_t cid = 0; cid < cur_nlist; ++cid) {
            const float* c = l0_.centroids.data() + cid * dim_sz;
            float s = unified_score(vptr, c, dim_, metric_, normalized_);
            if (s < best) {
                best = s;
                best_cid = static_cast<int>(cid);
            }
        }

        bool reuse_due_to_merge = (merge_threshold < std::numeric_limits<float>::infinity()) &&
                                  (best_cid >= 0) && (best <= merge_threshold);

        const bool under_cap = (int)cur_nlist < max_nlist;
        if (!reuse_due_to_merge && (best_cid == -1 || best > threshold) && under_cap) {
            std::vector<float> new_centroid(vptr, vptr + dim_sz);
            int new_cid = l0_.index->add_cluster(new_centroid);
            l0_.centroids.insert(l0_.centroids.end(), new_centroid.begin(), new_centroid.end());
            ensure_pending_size(pending, new_cid);
            best_cid = new_cid;
        } else {
            if (best_cid < 0) best_cid = 0;
            ensure_pending_size(pending, best_cid);
        }

        Pending& p = pending[best_cid];
        if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
        p.ids.push_back(ids[i]);
        p.vecs.insert(p.vecs.end(), vptr, vptr + dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) p.sum[d] += vptr[d];
        ++p.count;
    }

    for (size_t cid = 0; cid < pending.size(); ++cid) {
        Pending& p = pending[cid];
        if (p.count == 0) continue;

        const size_t base = l0_.index->cluster_live_size(static_cast<int>(cid));
        const size_t total = base + p.count;
        if (total == 0) continue;

        const float* old_c = l0_.centroids.data() + cid * dim_sz;
        std::vector<float> updated(dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) {
            float old_sum = static_cast<float>(base) * old_c[d];
            float new_sum = p.sum[d];
            updated[d] = (old_sum + new_sum) / static_cast<float>(total);
        }

        l0_.index->set_centroid(static_cast<int>(cid), updated);
        std::copy(updated.begin(), updated.end(), l0_.centroids.begin() + cid * dim_sz);

        l0_.index->add_batch(static_cast<int>(cid),
                             p.ids.data(),
                             p.vecs.data(),
                             p.count);
    }

    if (l1_strategy_) l1_strategy_->on_insert(ids, vecs, n_rows);
}

void MultiLevelIndex::update(const DocId* ids, const float* vecs, size_t n_rows,
                             bool insert_if_absent) {
    if (!ids || !vecs || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);
    ensure_layer_initialized_(l0_, cfg_.l0_nlist > 0 ? cfg_.l0_nlist : 1);
    l0_.index->update_batch(/*cluster_id=*/0, ids, vecs, n_rows, insert_if_absent);
    if (l1_strategy_) l1_strategy_->on_update(ids, vecs, n_rows);
}

void MultiLevelIndex::erase(const DocId* ids, size_t n_rows) {
    if (!ids || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);
    if (!l0_.index) return;
    l0_.index->erase_batch(/*cluster_id=*/0, ids, n_rows);
    if (l1_strategy_) l1_strategy_->on_erase(ids, n_rows);
}

void MultiLevelIndex::merge_levels_(const std::vector<std::vector<DocId>>& per_level_ids,
                                    const std::vector<std::vector<float>>& per_level_scores,
                                    int k,
                                    std::vector<DocId>& out_ids,
                                    std::vector<float>& out_scores) const {
    std::vector<Pair> buf;
    size_t total = 0;
    for (const auto& ids : per_level_ids) total += ids.size();
    buf.reserve(total);

    for (size_t li = 0; li < per_level_ids.size(); ++li) {
        const auto& ids = per_level_ids[li];
        const auto& scores = per_level_scores[li];
        for (size_t i = 0; i < ids.size() && i < scores.size(); ++i) {
            buf.push_back(Pair{scores[i], ids[i]});
        }
    }

    topk_smallest(buf, k);
    out_ids.clear();
    out_scores.clear();
    out_ids.reserve(buf.size());
    out_scores.reserve(buf.size());
    for (const auto& p : buf) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
}

void MultiLevelIndex::search(const float* queries, size_t q_rows, int k, int nprobe,
                             std::vector<std::vector<DocId>>& out_ids,
                             std::vector<std::vector<float>>& out_scores) const {
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;

    // Snapshot layers under read lock.
    Layer l0, l1, l2;
    {
        std::shared_lock lk(topo_mu_);
        l0 = l0_;
        l1 = l1_;
        l2 = l2_;
    }

    // Prepare buffers per query per level.
    std::vector<std::vector<DocId>> l0_ids(q_rows), l1_ids(q_rows), l2_ids(q_rows);
    std::vector<std::vector<float>> l0_scores(q_rows), l1_scores(q_rows), l2_scores(q_rows);

    auto search_level = [&](const Layer& layer,
                            std::vector<std::vector<DocId>>& ids,
                            std::vector<std::vector<float>>& scores) {
        if (layer.index) {
            layer.index->search_nprobe(queries, q_rows, k, nprobe, ids, scores);
        }
    };

    search_level(l0, l0_ids, l0_scores);

    const float search_threshold = cfg_.search_threshold;
    const bool has_threshold = search_threshold < std::numeric_limits<float>::infinity();

    // L1 strategy is currently empty; skip search to avoid extra work.
    // search_level(l1, l1_ids, l1_scores);

    // Only search L2 if needed by threshold or by default.
    auto kth_score = [&](size_t qi) -> float {
        const auto& s = l0_scores[qi];
        if (s.empty()) return std::numeric_limits<float>::infinity();
        if ((int)s.size() >= k) return s[static_cast<size_t>(k - 1)];
        return s.back();
    };

    bool need_l2 = !has_threshold;
    if (has_threshold) {
        need_l2 = false;
        for (size_t qi = 0; qi < q_rows; ++qi) {
            if (kth_score(qi) > search_threshold) { need_l2 = true; break; }
            else
            {
                printf("Skip L2 search for query %zu due to threshold %.6f\n", qi, search_threshold);
            }
        }
    }
    if (need_l2) {
        search_level(l2, l2_ids, l2_scores);
    }

    for (size_t qi = 0; qi < q_rows; ++qi) {
        if (has_threshold && !l0_scores[qi].empty() && kth_score(qi) <= search_threshold) {
            out_ids[qi] = l0_ids[qi];
            out_scores[qi] = l0_scores[qi];
            continue;
        }

        std::vector<std::vector<DocId>> per_ids = {l0_ids[qi], /*l1 skipped*/ {}, l2_ids[qi]};
        std::vector<std::vector<float>> per_scores = {l0_scores[qi], {}, l2_scores[qi]};
        merge_levels_(per_ids, per_scores, k, out_ids[qi], out_scores[qi]);
    }
}

void MultiLevelIndex::maintenance_pass() {
    std::shared_ptr<IVFIndex> l0_idx, l1_idx, l2_idx;
    {
        std::shared_lock lk(topo_mu_);
        l0_idx = l0_.index;
        l1_idx = l1_.index;
        l2_idx = l2_.index;
    }
    if (l0_idx) l0_idx->maintenance_pass();
    if (l1_idx) l1_idx->maintenance_pass();
    if (l2_idx) l2_idx->maintenance_pass();
}

} // namespace m3
