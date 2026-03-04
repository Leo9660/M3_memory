#include "m3_multi_level.h"
#include <mutex>
#include <algorithm>
#include <limits>
#include <chrono>

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
    if (!centroids.empty()) {
        ensure_layer_centroids_(l0_, centroids);
        ensure_layer_centroids_(l1_, centroids);
        const size_t nlist = centroids.size() / static_cast<size_t>(dim_);
        {
            std::lock_guard<std::mutex> ml(meta_mu_);
            metadata_.resize(nlist);
            for (size_t i = 0; i < nlist; ++i) {
                metadata_[i].in_l2 = true;
            }
        }
    }
}

void MultiLevelIndex::insert(const DocId* ids, const float* vecs, size_t n_rows) {
    if (!ids || !vecs || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);

    bool cache_mode = (l2_.index != nullptr) && !l2_.centroids.empty();
    if (cache_mode) {
        const size_t nlist = l2_.centroids.size() / static_cast<size_t>(dim_);
        std::lock_guard<std::mutex> ml(meta_mu_);
        cache_mode = (metadata_.size() == nlist);
    }
    if (cache_mode && l2_.index && !l2_.centroids.empty()) {
        const size_t dim_sz = static_cast<size_t>(dim_);
        std::vector<int> cids(n_rows);
        l2_.index->nearest_clusters(vecs, n_rows, cids);
        for (size_t i = 0; i < n_rows; ++i) {
            int cid = cids[i];
            if (cid < 0) continue;
            l2_.index->add_batch(cid, &ids[i], vecs + i * dim_sz, 1);
            doc_id_to_cid_[ids[i]] = cid;
        }
        if (l1_strategy_) l1_strategy_->on_insert(ids, vecs, n_rows);
        return;
    }

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

    bool cache_mode = (l2_.index != nullptr) && !l2_.centroids.empty();
    if (cache_mode) {
        const size_t nlist = l2_.centroids.size() / static_cast<size_t>(dim_);
        std::lock_guard<std::mutex> ml(meta_mu_);
        cache_mode = (metadata_.size() == nlist);
    }
    if (cache_mode && l2_.index) {
        const size_t dim_sz = static_cast<size_t>(dim_);
        for (size_t i = 0; i < n_rows; ++i) {
            DocId id = ids[i];
            auto it = doc_id_to_cid_.find(id);
            if (it == doc_id_to_cid_.end()) {
                if (insert_if_absent) {
                    const float* vptr = vecs + i * dim_sz;
                    std::vector<int> cids(1);
                    l2_.index->nearest_clusters(vptr, 1, cids);
                    int cid = cids[0];
                    if (cid >= 0) {
                        l2_.index->add_batch(cid, &id, vptr, 1);
                        doc_id_to_cid_[id] = cid;
                        promote_vector_neighborhood_(id);
                    }
                }
                continue;
            }
            int cid = it->second;
            const float* vptr = vecs + i * dim_sz;
            l2_.index->update_batch(cid, &id, vptr, 1, false);
            if (l0_.index && l0_.index->cluster_get_vector(cid, id))
                l0_.index->update_batch(cid, &id, vptr, 1, false);
            if (l1_.index && l1_.index->cluster_get_vector(cid, id))
                l1_.index->update_batch(cid, &id, vptr, 1, false);
            promote_vector_neighborhood_(id);
        }
        if (l1_strategy_) l1_strategy_->on_update(ids, vecs, n_rows);
        return;
    }

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
    
    std::unordered_map<size_t, Pending> pending;
    for (size_t i = 0; i < n_rows; ++i) {
        const float* vptr = vecs + i * dim_sz;
        size_t id = ids[i];
        bool found = false;
        int old_cid = -1;

        for (int cid = 0; cid < l0_.index->nlist(); ++cid) {
            const float* old_vec = l0_.index->cluster_get_vector(cid, id);
            if (old_vec) {
                old_cid = cid;
                Pending& p = pending[-(cid + 1)];
                if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
                
                p.ids.push_back(ids[i]);
                p.vecs.insert(p.vecs.end(), old_vec, old_vec + dim_sz);
                for (size_t d = 0; d < dim_sz; ++d) p.sum[d] += old_vec[d];
                ++p.count;
                found = true;
                break;
            }
        }
        
        if (!found && !insert_if_absent) {
            continue;
        }

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
            best_cid = new_cid;
        } else {
            if (best_cid < 0) best_cid = 0;
        }

        Pending& p = pending[best_cid];
        if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
        p.ids.push_back(ids[i]);
        p.vecs.insert(p.vecs.end(), vptr, vptr + dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) p.sum[d] += vptr[d];
        ++p.count;
    }

    for (auto& x: pending) {
        int cid = x.first;
        Pending& p = x.second;
        if (p.count == 0) continue;
        
        if (cid < 0){
            int actual_cid = -(cid + 1);
            const size_t base = l0_.index->cluster_live_size(actual_cid);
            const size_t total = base - p.count;
            if (total == 0) continue;
            
            const float* old_c = l0_.centroids.data() + actual_cid * dim_sz;
            std::vector<float> updated(dim_sz);
            for (size_t d = 0; d < dim_sz; ++d) {
                float old_sum = static_cast<float>(base) * old_c[d];
                float removed_sum = p.sum[d]; 
                updated[d] = (old_sum - removed_sum) / static_cast<float>(total);
            }
            
            l0_.index->set_centroid(actual_cid, updated);
            std::copy(updated.begin(), updated.end(), l0_.centroids.begin() + actual_cid * dim_sz);
            
            l0_.index->erase_batch(actual_cid, p.ids.data(), p.count);
        }
        else{
            const size_t base = l0_.index->cluster_live_size(cid);
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
    }
    if (l1_strategy_) l1_strategy_->on_update(ids, vecs, n_rows);
}

void MultiLevelIndex::erase(const DocId* ids, size_t n_rows) {
    if (!ids || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);

    bool cache_mode = (l2_.index != nullptr) && !l2_.centroids.empty();
    if (cache_mode) {
        const size_t nlist = l2_.centroids.size() / static_cast<size_t>(dim_);
        std::lock_guard<std::mutex> ml(meta_mu_);
        cache_mode = (metadata_.size() == nlist);
    }
    if (cache_mode && l2_.index) {
        for (size_t i = 0; i < n_rows; ++i) {
            DocId id = ids[i];
            auto it = doc_id_to_cid_.find(id);
            if (it == doc_id_to_cid_.end()) continue;
            int cid = it->second;
            l2_.index->erase_batch(cid, &id, 1);
            if (l0_.index) l0_.index->erase_batch(cid, &id, 1);
            if (l1_.index) l1_.index->erase_batch(cid, &id, 1);
            doc_id_to_cid_.erase(it);
        }
        if (l1_strategy_) l1_strategy_->on_erase(ids, n_rows);
        return;
    }

    if (!l0_.index) return;
    const size_t dim_sz = static_cast<size_t>(dim_);

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
        DocId id = ids[i];
        bool found = false;
        int cid = -1;
        
        for (int cluster_id = 0; cluster_id < l0_.index->nlist(); ++cluster_id) {
            const float* old_vec = l0_.index->cluster_get_vector(cluster_id, id);
            if (old_vec) {
                cid = cluster_id;
                found = true;
                
                ensure_pending_size(pending, cid);
                Pending& p = pending[cid];
                if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
                
                p.ids.push_back(id);
                p.vecs.insert(p.vecs.end(), old_vec, old_vec + dim_sz);
                for (size_t d = 0; d < dim_sz; ++d) {
                    p.sum[d] += old_vec[d];
                }
                ++p.count;
                break;
            }
        }
        
        if (!found) {
            continue;
        }
    }
    
    for (size_t cid = 0; cid < pending.size(); ++cid) {
        Pending& p = pending[cid];
        if (p.count == 0) continue;
        
        const size_t base = l0_.index->cluster_live_size(static_cast<int>(cid));
        const size_t total = base - p.count;
        
        if (total == 0) {
            l0_.index->erase_batch(static_cast<int>(cid), p.ids.data(), p.count);
            continue;
        }
        
        const float* old_c = l0_.centroids.data() + cid * dim_sz;
        std::vector<float> updated(dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) {
            float old_sum = static_cast<float>(base) * old_c[d];
            float removed_sum = p.sum[d];
            updated[d] = (old_sum - removed_sum) / static_cast<float>(total);
        }
        
        l0_.index->set_centroid(static_cast<int>(cid), updated);
        std::copy(updated.begin(), updated.end(), l0_.centroids.begin() + cid * dim_sz);
        
        l0_.index->erase_batch(static_cast<int>(cid), p.ids.data(), p.count);
    }
    
    if (l1_strategy_) l1_strategy_->on_erase(ids, n_rows);
}

void MultiLevelIndex::merge_levels_(const std::vector<std::vector<DocId>>& per_level_ids,
                                    const std::vector<std::vector<float>>& per_level_scores,
                                    int k,
                                    std::vector<DocId>& out_ids,
                                    std::vector<float>& out_scores) const {
    // De-duplicate doc_ids across levels (L0/L1/L2 can each return same id).
    // Keep the best (smallest) score for each doc_id.
    std::unordered_map<DocId, float> best;
    size_t total = 0;
    for (const auto& ids : per_level_ids) total += ids.size();
    best.reserve(total);

    for (size_t li = 0; li < per_level_ids.size(); ++li) {
        const auto& ids = per_level_ids[li];
        const auto& scores = per_level_scores[li];
        const size_t n = std::min(ids.size(), scores.size());
        for (size_t i = 0; i < n; ++i) {
            DocId id = ids[i];
            float s = scores[i];
            auto it = best.find(id);
            if (it == best.end() || s < it->second) best[id] = s;
        }
    }

    std::vector<Pair> buf;
    buf.reserve(best.size());
    for (const auto& kv : best)
        buf.push_back(Pair{kv.second, kv.first});

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

    const size_t dim_sz = static_cast<size_t>(dim_);

    if (cache_enabled_()) {
        Layer l0, l1, l2;
        {
            std::shared_lock lk(topo_mu_);
            l0 = l0_;
            l1 = l1_;
            l2 = l2_;
        }
        const float search_threshold = cfg_.search_threshold;
        const bool has_threshold = search_threshold < std::numeric_limits<float>::infinity();

        std::vector<int> probe_ids;
        std::vector<std::vector<DocId>> l0_ids(1), l1_ids(1), l2_ids(1);
        std::vector<std::vector<float>> l0_scores(1), l1_scores(1), l2_scores(1);
        // 0 = no results, 1 = satisfied by L0 only, 2 = satisfied after L1, 3 = needed L2
        std::vector<int> stage(q_rows, 0);

        auto kth_score_vec = [&](const std::vector<float>& s) -> float {
            if (s.empty()) return std::numeric_limits<float>::infinity();
            if ((int)s.size() >= k) return s[static_cast<size_t>(k - 1)];
            return s.back();
        };

        for (size_t qi = 0; qi < q_rows; ++qi) {
            const float* qptr = queries + qi * dim_sz;
            probe_ids.clear();
            if (l2.index)
                l2.index->get_probe_ids(qptr, nprobe, probe_ids);
            if (probe_ids.empty()) continue;

            // Stage 1: search L0 only.
            l0_ids[0].clear(); l0_scores[0].clear();
            l1_ids[0].clear(); l1_scores[0].clear();
            l2_ids[0].clear(); l2_scores[0].clear();

            if (l0.index)
                l0.index->search_on(probe_ids, qptr, 1, k, l0_ids, l0_scores);

            bool satisfied = false;
            std::vector<DocId> merged_ids;
            std::vector<float> merged_scores;

            if (has_threshold && !l0_scores[0].empty()) {
                float ks = kth_score_vec(l0_scores[0]);
                if (ks <= search_threshold) {
                    // L0 alone is good enough; no need to touch L1/L2.
                    merged_ids = l0_ids[0];
                    merged_scores = l0_scores[0];
                    satisfied = true;
                        stage[qi] = 1;
                }
            }

            // Stage 2: include L1 if needed.
            if (!satisfied && l1.index) {
                l1.index->search_on(probe_ids, qptr, 1, k, l1_ids, l1_scores);
                std::vector<std::vector<DocId>> per_ids = {l0_ids[0], l1_ids[0]};
                std::vector<std::vector<float>> per_scores = {l0_scores[0], l1_scores[0]};
                merge_levels_(per_ids, per_scores, k, merged_ids, merged_scores);

                if (has_threshold && !merged_scores.empty()) {
                    float ks = kth_score_vec(merged_scores);
                    if (ks <= search_threshold) {
                        satisfied = true;
                        stage[qi] = 2;
                    }
                }
            }

            // Stage 3: include L2 if still not satisfied or if there is no threshold.
            if (!satisfied) {
                if (l2.index) {
                    l2.index->search_on(probe_ids, qptr, 1, k, l2_ids, l2_scores);
                    std::vector<std::vector<DocId>> per_ids = {l0_ids[0], l1_ids[0], l2_ids[0]};
                    std::vector<std::vector<float>> per_scores = {l0_scores[0], l1_scores[0], l2_scores[0]};
                    merge_levels_(per_ids, per_scores, k, merged_ids, merged_scores);
                    stage[qi] = 3;
                } else {
                    // No L2; if we haven't merged yet (e.g. no L1), fallback to L0-only.
                    if (merged_ids.empty() && !l0_ids[0].empty()) {
                        merged_ids = l0_ids[0];
                        merged_scores = l0_scores[0];
                        // Treat this like L0-only satisfaction.
                        if (stage[qi] == 0) stage[qi] = 1;
                    }
                }
            }

            out_ids[qi] = std::move(merged_ids);
            out_scores[qi] = std::move(merged_scores);
        }
        {
            std::unique_lock promo_lk(topo_mu_);
            for (size_t qi = 0; qi < q_rows; ++qi) {
                const int st = (qi < stage.size()) ? stage[qi] : 0;
                int limit = cache_config_.max_promote_per_query;
                if (limit <= 0) limit = static_cast<int>(out_ids[qi].size());
                for (int j = 0; j < limit && j < static_cast<int>(out_ids[qi].size()); ++j) {
                    DocId doc_id = out_ids[qi][j];
                    auto it = doc_id_to_cid_.find(doc_id);
                    if (it == doc_id_to_cid_.end()) continue;
                    // Always update cluster-level access time so demotion logic works.
                    record_access_(it->second);
                    // But if this query was fully satisfied by L0 alone (stage 1),
                    // skip extra promotion work (results already live in the hottest tier).
                    if (st == 1) continue;
                    promote_vector_neighborhood_(doc_id);
                }
            }
        }
        return;
    }

    // Non-cache path: original behavior
    Layer l0, l1, l2;
    {
        std::shared_lock lk(topo_mu_);
        l0 = l0_;
        l1 = l1_;
        l2 = l2_;
    }

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
        std::vector<std::vector<DocId>> per_ids = {l0_ids[qi], {}, l2_ids[qi]};
        std::vector<std::vector<float>> per_scores = {l0_scores[qi], {}, l2_scores[qi]};
        merge_levels_(per_ids, per_scores, k, out_ids[qi], out_scores[qi]);
    }
}

void MultiLevelIndex::maintenance_pass() {
    // ===== 1) Snapshot topology and metadata size under topo_mu_ / meta_mu_ =====
    std::shared_ptr<IVFIndex> l0_idx, l1_idx, l2_idx;
    size_t nlist = 0;
    std::vector<ClusterMetadata> meta_snap;
    bool cache_mode = false;
    {
        std::unique_lock lk(topo_mu_);
        l0_idx = l0_.index;
        l1_idx = l1_.index;
        l2_idx = l2_.index;
        cache_mode = (l2_.index != nullptr) && !l2_.centroids.empty();
        if (cache_mode) {
            nlist = l2_.centroids.size() / static_cast<size_t>(dim_);
            std::lock_guard<std::mutex> ml(meta_mu_);
            if (metadata_.size() == nlist) {
                meta_snap = metadata_;
            } else {
                cache_mode = false;
            }
        }
    }

    // ===== 2) Run per-IVF maintenance (split/merge/compact) =====
    if (l0_idx) l0_idx->maintenance_pass();
    if (l1_idx) l1_idx->maintenance_pass();
    if (l2_idx) l2_idx->maintenance_pass();

    if (!cache_mode || meta_snap.empty() || nlist == 0) {
        return;
    }

    // ===== 3) Vector-level eviction per level (no MultiLevelIndex locks) =====
    const size_t l0_cap = static_cast<size_t>(cache_config_.l0_max_vectors_per_cluster);
    const size_t l1_cap = static_cast<size_t>(cache_config_.l1_max_vectors_per_cluster);

    if (l0_idx && l0_cap > 0) {
        const int nlist0 = l0_idx->nlist();
        for (int cid = 0; cid < nlist0; ++cid) {
            const size_t sz = l0_idx->cluster_live_size(cid);
            if (sz > l0_cap) {
                std::vector<DocId> evict;
                l0_idx->get_coldest_doc_ids(cid, sz - l0_cap, evict);
                if (!evict.empty()) {
                    l0_idx->erase_batch(cid, evict.data(), evict.size());
                }
            }
        }
    }
    if (l1_idx && l1_cap > 0) {
        const int nlist1 = l1_idx->nlist();
        for (int cid = 0; cid < nlist1; ++cid) {
            const size_t sz = l1_idx->cluster_live_size(cid);
            if (sz > l1_cap) {
                std::vector<DocId> evict;
                l1_idx->get_coldest_doc_ids(cid, sz - l1_cap, evict);
                if (!evict.empty()) {
                    l1_idx->erase_batch(cid, evict.data(), evict.size());
                }
            }
        }
    }

    // ===== 4) Recompute live sizes for each level (IVF only) =====
    std::vector<size_t> l0_counts(nlist, 0), l1_counts(nlist, 0);
    if (l0_idx) {
        const int n0 = l0_idx->nlist();
        const int limit = std::min<int>(static_cast<int>(nlist), n0);
        for (int cid = 0; cid < limit; ++cid) {
            l0_counts[static_cast<size_t>(cid)] = l0_idx->cluster_live_size(cid);
        }
    }
    if (l1_idx) {
        const int n1 = l1_idx->nlist();
        const int limit = std::min<int>(static_cast<int>(nlist), n1);
        for (int cid = 0; cid < limit; ++cid) {
            l1_counts[static_cast<size_t>(cid)] = l1_idx->cluster_live_size(cid);
        }
    }

    // ===== 5) Cluster-count demotion based on metadata snapshot =====
    std::vector<uint8_t> removed_l0(nlist, 0), removed_l1(nlist, 0);
    const int l0_max_clusters = cache_config_.l0_max_clusters;
    const int l1_max_clusters = cache_config_.l1_max_clusters;

    if (l0_idx && l0_max_clusters > 0) {
        std::vector<std::pair<uint64_t, int>> by_time;
        by_time.reserve(nlist);
        for (size_t cid = 0; cid < nlist; ++cid) {
            if (l0_counts[cid] > 0) {
                by_time.emplace_back(meta_snap[cid].last_access_time,
                                     static_cast<int>(cid));
            }
        }
        int excess = static_cast<int>(by_time.size()) - l0_max_clusters;
        if (excess > 0) {
            std::sort(by_time.begin(), by_time.end());
            for (int i = 0; i < excess; ++i) {
                int cid = by_time[static_cast<size_t>(i)].second;
                l0_idx->remove_cluster(cid);
                removed_l0[static_cast<size_t>(cid)] = 1;
                l0_counts[static_cast<size_t>(cid)] = 0;
            }
        }
    }
    if (l1_idx && l1_max_clusters > 0) {
        std::vector<std::pair<uint64_t, int>> by_time;
        by_time.reserve(nlist);
        for (size_t cid = 0; cid < nlist; ++cid) {
            if (l1_counts[cid] > 0) {
                by_time.emplace_back(meta_snap[cid].last_access_time,
                                     static_cast<int>(cid));
            }
        }
        int excess = static_cast<int>(by_time.size()) - l1_max_clusters;
        if (excess > 0) {
            std::sort(by_time.begin(), by_time.end());
            for (int i = 0; i < excess; ++i) {
                int cid = by_time[static_cast<size_t>(i)].second;
                l1_idx->remove_cluster(cid);
                removed_l1[static_cast<size_t>(cid)] = 1;
                l1_counts[static_cast<size_t>(cid)] = 0;
            }
        }
    }

    // ===== 6) Cold-cluster demotion based on last_access_time snapshot =====
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const uint64_t cold = cache_config_.cold_time_ns;

    for (size_t cid = 0; cid < nlist; ++cid) {
        const ClusterMetadata& m = meta_snap[cid];
        const uint64_t elapsed = (now_ns >= m.last_access_time)
                                     ? (now_ns - m.last_access_time)
                                     : 0;
        if (elapsed <= cold) continue;

        if (l0_idx && l0_counts[cid] > 0) {
            l0_idx->remove_cluster(static_cast<int>(cid));
            removed_l0[cid] = 1;
            l0_counts[cid] = 0;
        }
        if (l1_idx && l1_counts[cid] > 0) {
            l1_idx->remove_cluster(static_cast<int>(cid));
            removed_l1[cid] = 1;
            l1_counts[cid] = 0;
        }
    }

    // ===== 7) Apply metadata updates under meta_mu_ (no IVF calls) =====
    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (metadata_.size() != nlist) {
            return; // topology changed concurrently; skip applying
        }
        for (size_t cid = 0; cid < nlist; ++cid) {
            ClusterMetadata& m = metadata_[cid];
            if (removed_l0[cid]) {
                m.in_l0 = false;
                m.l0_vector_count = 0;
            } else {
                m.l0_vector_count = l0_counts[cid];
            }
            if (removed_l1[cid]) {
                m.in_l1 = false;
                m.l1_vector_count = 0;
            } else {
                m.l1_vector_count = l1_counts[cid];
            }
        }
    }
}

bool MultiLevelIndex::cache_enabled_() const {
    std::shared_lock lk(topo_mu_);
    if (!l2_.index || l2_.centroids.empty()) return false;
    const size_t nlist = l2_.centroids.size() / static_cast<size_t>(dim_);
    std::lock_guard<std::mutex> ml(meta_mu_);
    return metadata_.size() == nlist;
}

void MultiLevelIndex::record_access_(int cid) const {
    if (cid < 0) return;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) >= metadata_.size()) return;
    ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
    m.last_access_time = now_ns;
}

void MultiLevelIndex::promote_vector_neighborhood_(DocId doc_id) const {
    auto it = doc_id_to_cid_.find(doc_id);
    if (it == doc_id_to_cid_.end()) return;
    const int cid = it->second;
    if (!l2_.index || static_cast<size_t>(cid) >= metadata_.size()) return;

    const float* vec = l2_.index->cluster_get_vector(cid, doc_id);
    if (!vec) return;

    const size_t dim_sz = static_cast<size_t>(dim_);
    const size_t nlist = l2_.centroids.size() / dim_sz;
    if (static_cast<size_t>(cid) >= nlist) return;

    int l0_k = cache_config_.l0_neighborhood_k;
    int l1_k = cache_config_.l1_neighborhood_k;
    if (l0_k <= 0) l0_k = 1;
    if (l1_k <= 0) l1_k = l0_k;

    std::vector<float> cent(l2_.centroids.begin() + cid * dim_sz,
                            l2_.centroids.begin() + (cid + 1) * dim_sz);

    std::vector<DocId> ids0, ids1;
    std::vector<float> scores0, scores1;
    l2_.index->search_within_cluster(cid, vec, l0_k, ids0, scores0);
    if (!ids0.empty()) {
        std::vector<float> vecs0;
        vecs0.reserve(ids0.size() * dim_sz);
        for (DocId id : ids0) {
            const float* v = l2_.index->cluster_get_vector(cid, id);
            if (v) vecs0.insert(vecs0.end(), v, v + dim_sz);
        }
        if (vecs0.size() == ids0.size() * dim_sz && l0_.index) {
            l0_.index->ensure_cluster(cid, cent);
            l0_.index->update_batch(cid, ids0.data(), vecs0.data(), ids0.size(), true);
        }
    }
    l2_.index->search_within_cluster(cid, vec, l1_k, ids1, scores1);
    if (!ids1.empty() && l1_.index) {
        std::vector<float> vecs1;
        vecs1.reserve(ids1.size() * dim_sz);
        for (DocId id : ids1) {
            const float* v = l2_.index->cluster_get_vector(cid, id);
            if (v) vecs1.insert(vecs1.end(), v, v + dim_sz);
        }
        if (vecs1.size() == ids1.size() * dim_sz) {
            l1_.index->ensure_cluster(cid, cent);
            l1_.index->update_batch(cid, ids1.data(), vecs1.data(), ids1.size(), true);
        }
    }

    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) < metadata_.size()) {
        ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
        m.last_access_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (!ids0.empty()) m.in_l0 = true;
        if (!ids1.empty()) m.in_l1 = true;
        if (l0_.index) m.l0_vector_count = l0_.index->cluster_live_size(cid);
        if (l1_.index) m.l1_vector_count = l1_.index->cluster_live_size(cid);
    }
}

void MultiLevelIndex::demote_cluster_(int cid) const {
    if (cid < 0) return;
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) >= metadata_.size()) return;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
    const uint64_t cold = cache_config_.cold_time_ns;
    const uint64_t elapsed = (now_ns >= m.last_access_time) ? (now_ns - m.last_access_time) : 0;

    if (m.in_l0 && elapsed > cold && l0_.index) {
        l0_.index->remove_cluster(cid);
        m.in_l0 = false;
        m.l0_vector_count = 0;
    }
    if (m.in_l1 && elapsed > cold && l1_.index) {
        l1_.index->remove_cluster(cid);
        m.in_l1 = false;
        m.l1_vector_count = 0;
    }
}

void MultiLevelIndex::run_vector_eviction_per_level_() const {
    const size_t l0_cap = static_cast<size_t>(cache_config_.l0_max_vectors_per_cluster);
    const size_t l1_cap = static_cast<size_t>(cache_config_.l1_max_vectors_per_cluster);
    if (l0_.index) {
        const int nlist = l0_.index->nlist();
        for (int cid = 0; cid < nlist; ++cid) {
            const size_t sz = l0_.index->cluster_live_size(cid);
            if (sz > l0_cap) {
                std::vector<DocId> evict;
                l0_.index->get_coldest_doc_ids(cid, sz - l0_cap, evict);
                if (!evict.empty())
                    l0_.index->erase_batch(cid, evict.data(), evict.size());
            }
        }
    }
    if (l1_.index) {
        const int nlist = l1_.index->nlist();
        for (int cid = 0; cid < nlist; ++cid) {
            const size_t sz = l1_.index->cluster_live_size(cid);
            if (sz > l1_cap) {
                std::vector<DocId> evict;
                l1_.index->get_coldest_doc_ids(cid, sz - l1_cap, evict);
                if (!evict.empty())
                    l1_.index->erase_batch(cid, evict.data(), evict.size());
            }
        }
    }
    std::lock_guard<std::mutex> ml(meta_mu_);
    for (size_t cid = 0; cid < metadata_.size(); ++cid) {
        if (l0_.index) metadata_[cid].l0_vector_count = l0_.index->cluster_live_size(static_cast<int>(cid));
        if (l1_.index) metadata_[cid].l1_vector_count = l1_.index->cluster_live_size(static_cast<int>(cid));
    }
}

void MultiLevelIndex::run_cluster_count_demotion_() const {
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (metadata_.empty()) return;
    const int nlist = static_cast<int>(metadata_.size());
    std::vector<std::pair<uint64_t, int>> by_time;
    by_time.reserve(static_cast<size_t>(nlist));
    if (l0_.index) {
        by_time.clear();
        for (int cid = 0; cid < nlist; ++cid) {
            if (l0_.index->cluster_live_size(cid) > 0)
                by_time.emplace_back(metadata_[static_cast<size_t>(cid)].last_access_time, cid);
        }
        int excess = static_cast<int>(by_time.size()) - cache_config_.l0_max_clusters;
        if (excess > 0) {
            std::sort(by_time.begin(), by_time.end());
            for (int i = 0; i < excess; ++i) {
                int cid = by_time[i].second;
                l0_.index->remove_cluster(cid);
                metadata_[static_cast<size_t>(cid)].in_l0 = false;
                metadata_[static_cast<size_t>(cid)].l0_vector_count = 0;
            }
        }
    }
    if (l1_.index) {
        by_time.clear();
        for (int cid = 0; cid < nlist; ++cid) {
            if (l1_.index->cluster_live_size(cid) > 0)
                by_time.emplace_back(metadata_[static_cast<size_t>(cid)].last_access_time, cid);
        }
        int excess = static_cast<int>(by_time.size()) - cache_config_.l1_max_clusters;
        if (excess > 0) {
            std::sort(by_time.begin(), by_time.end());
            for (int i = 0; i < excess; ++i) {
                int cid = by_time[i].second;
                l1_.index->remove_cluster(cid);
                metadata_[static_cast<size_t>(cid)].in_l1 = false;
                metadata_[static_cast<size_t>(cid)].l1_vector_count = 0;
            }
        }
    }
}

} // namespace m3
