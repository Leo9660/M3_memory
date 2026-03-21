#include "m3_multi_level.h"
#include "m3_fsm.h"
#include <mutex>
#include <algorithm>
#include <limits>
#include <chrono>
#include <numeric>

namespace m3 {

namespace {
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
    start_prefetch_thread_();
}

MultiLevelIndex::~MultiLevelIndex() {
    stop_prefetch_thread_();
}

void MultiLevelIndex::set_cache_config(CacheConfig cfg) {
    cache_config_ = std::move(cfg);
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

    // Batch nearest-cluster assignment: one call into IVFIndex instead of
    // a per-vector O(nlist) scalar loop.  Falls back to the scalar path only
    // for new-cluster candidates (which can't be batch-assigned yet).
    const size_t cur_nlist_before = l0_.centroids.empty() ? 0
                                  : l0_.centroids.size() / dim_sz;
    std::vector<int> batch_cids(n_rows, -1);
    std::vector<float> batch_scores(n_rows, std::numeric_limits<float>::infinity());
    if (cur_nlist_before > 0) {
        l0_.index->nearest_clusters_with_scores(vecs, n_rows, batch_cids, batch_scores);
    }

    for (size_t i = 0; i < n_rows; ++i) {
        const float* vptr = vecs + i * dim_sz;
        const float best  = batch_scores[i];
        int best_cid      = batch_cids[i];

        bool reuse_due_to_merge = (merge_threshold < std::numeric_limits<float>::infinity()) &&
                                  (best_cid >= 0) && (best <= merge_threshold);

        const size_t cur_nlist = l0_.centroids.empty() ? 0 : l0_.centroids.size() / dim_sz;
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
            updated[d] = (static_cast<float>(base) * old_c[d] + p.sum[d])
                         / static_cast<float>(total);
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

    // Two separate maps: one for old-position erases, one for new-position inserts.
    // The original code used a single map with negative keys encoded as size_t, which
    // silently broke on the flush loop (size_t key cast to int is UB for large values;
    // the cid < 0 branch was always false, so old vectors were never erased).
    std::unordered_map<int, Pending> erase_pending;  // cid -> vectors to remove
    std::unordered_map<int, Pending> insert_pending; // cid -> vectors to add

    for (size_t i = 0; i < n_rows; ++i) {
        const float* vptr = vecs + i * dim_sz;
        DocId id = ids[i];
        bool found = false;
        int old_cid = -1;

        for (int cid = 0; cid < l0_.index->nlist(); ++cid) {
            const float* old_vec = l0_.index->cluster_get_vector(cid, id);
            if (old_vec) {
                old_cid = cid;
                Pending& p = erase_pending[cid];
                if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
                p.ids.push_back(id);
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
        // Use the IVFIndex batch scorer if there are clusters; scalar fallback otherwise.
        if (cur_nlist > 0) {
            std::vector<int> tmp_cid(1); std::vector<float> tmp_score(1);
            l0_.index->nearest_clusters_with_scores(vptr, 1, tmp_cid, tmp_score);
            best_cid = tmp_cid[0]; best = tmp_score[0];
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

        Pending& p = insert_pending[best_cid];
        if (p.sum.empty()) p.sum.assign(dim_sz, 0.0f);
        p.ids.push_back(id);
        p.vecs.insert(p.vecs.end(), vptr, vptr + dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) p.sum[d] += vptr[d];
        ++p.count;
    }

    // Apply erases first (remove old positions, update centroids).
    for (auto& [cid, p] : erase_pending) {
        if (p.count == 0) continue;
        const size_t base = l0_.index->cluster_live_size(cid);
        if (base < p.count) continue; // guard against underflow
        const size_t total = base - p.count;
        if (total == 0) {
            l0_.index->erase_batch(cid, p.ids.data(), p.count);
            continue;
        }
        const float* old_c = l0_.centroids.data() + static_cast<size_t>(cid) * dim_sz;
        std::vector<float> updated(dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) {
            updated[d] = (static_cast<float>(base) * old_c[d] - p.sum[d])
                         / static_cast<float>(total);
        }
        l0_.index->set_centroid(cid, updated);
        std::copy(updated.begin(), updated.end(),
                  l0_.centroids.begin() + static_cast<size_t>(cid) * dim_sz);
        l0_.index->erase_batch(cid, p.ids.data(), p.count);
    }

    // Apply inserts (add new positions, update centroids).
    for (auto& [cid, p] : insert_pending) {
        if (p.count == 0) continue;
        const size_t base = l0_.index->cluster_live_size(cid);
        const size_t total = base + p.count;
        const float* old_c = l0_.centroids.data() + static_cast<size_t>(cid) * dim_sz;
        std::vector<float> updated(dim_sz);
        for (size_t d = 0; d < dim_sz; ++d) {
            updated[d] = (static_cast<float>(base) * old_c[d] + p.sum[d])
                         / static_cast<float>(total);
        }
        l0_.index->set_centroid(cid, updated);
        std::copy(updated.begin(), updated.end(),
                  l0_.centroids.begin() + static_cast<size_t>(cid) * dim_sz);
        l0_.index->add_batch(cid, p.ids.data(), p.vecs.data(), p.count);
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
        if (base < p.count) {
            // Shouldn't happen, but guard against underflow before subtracting size_t values.
            l0_.index->erase_batch(static_cast<int>(cid), p.ids.data(), p.count);
            continue;
        }
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

// ======================================================================
// FSM accessors
// ======================================================================
void MultiLevelIndex::set_fsm_config(FSMConfig cfg) {
    // FSMTable holds a mutex so it cannot be move-assigned.
    // Update config in-place; existing patterns are preserved.
    fsm_table_.set_config(std::move(cfg));
}
FSMTable& MultiLevelIndex::fsm_table() { return fsm_table_; }
const FSMTable& MultiLevelIndex::fsm_table() const { return fsm_table_; }

// ======================================================================
// dagent rolling average helpers  (TODO-5)
// ======================================================================
void MultiLevelIndex::update_dagent_(float kth_score) const {
    std::lock_guard<std::mutex> lk(dagent_mu_);
    const int window = fsm_table_.config().dagent_window;
    dagent_buf_.push_back(kth_score);
    while (static_cast<int>(dagent_buf_.size()) > window)
        dagent_buf_.pop_front();
}

float MultiLevelIndex::dagent_() const {
    std::lock_guard<std::mutex> lk(dagent_mu_);
    if (dagent_buf_.empty()) return std::numeric_limits<float>::infinity();
    float sum = 0.0f;
    for (float v : dagent_buf_) sum += v;
    return sum / static_cast<float>(dagent_buf_.size());
}

// ======================================================================
// build_probe_list_  (BUG-1 fix + TODO-2 bridge)
//
// Builds the ordered probe list for one layer:
//   Phase 1: FSM-predicted clusters (from fsm_preferred), validated
//            against the layer's own valid cluster set.
//   Phase 2: Fill remaining slots with centroid-distance order.
// ======================================================================
std::vector<int> MultiLevelIndex::build_probe_list_(
        const float* query,
        const std::vector<int>& fsm_preferred,
        const std::shared_ptr<IVFIndex>& layer_idx,
        const std::vector<float>& layer_centroids,
        int effective_nprobe) const {

    if (!layer_idx || effective_nprobe <= 0) return {};

    const int nlist    = layer_idx->nlist();
    const size_t dim_sz = static_cast<size_t>(dim_);

    std::vector<int> probe_list;
    probe_list.reserve(static_cast<size_t>(effective_nprobe));
    std::unordered_set<int> added;
    added.reserve(static_cast<size_t>(effective_nprobe));

    // Phase 1: FSM-predicted clusters first.
    for (int cid : fsm_preferred) {
        if (cid < 0 || cid >= nlist) continue;
        if (!layer_idx->centroid_ptr(cid)) continue;  // cluster slot invalid
        if (!added.insert(cid).second) continue;
        probe_list.push_back(cid);
        if (static_cast<int>(probe_list.size()) >= effective_nprobe) break;
    }

    // Phase 2: centroid-distance fill.
    if (static_cast<int>(probe_list.size()) < effective_nprobe
            && !layer_centroids.empty()) {
        struct CS { int cid; float score; };
        std::vector<CS> remaining;
        remaining.reserve(static_cast<size_t>(nlist));
        for (int c = 0; c < nlist; ++c) {
            if (!layer_idx->centroid_ptr(c)) continue;
            if (added.count(c)) continue;
            const float* cent = layer_centroids.data() + static_cast<size_t>(c) * dim_sz;
            float score = unified_score(query, cent, dim_, metric_, normalized_);
            remaining.push_back({c, score});
        }
        std::sort(remaining.begin(), remaining.end(),
                  [](const CS& a, const CS& b){ return a.score < b.score; });
        for (const auto& cs : remaining) {
            probe_list.push_back(cs.cid);
            if (static_cast<int>(probe_list.size()) >= effective_nprobe) break;
        }
    }
    return probe_list;
}

// ======================================================================
// search_layer_ordered_
//
// Probes the clusters in probe_list order using search_on (which handles
// the within-cluster top-k correctly), collects all results, then
// picks the top-k by merging.  Returns the cluster id that contributed
// the best-scoring result (used for trajectory recording).
// ======================================================================
int MultiLevelIndex::search_layer_ordered_(
        const float* query, int k,
        const std::vector<int>& probe_list,
        const std::shared_ptr<IVFIndex>& layer_idx,
        std::vector<DocId>& out_ids,
        std::vector<float>& out_scores) const {

    if (probe_list.empty() || !layer_idx) return -1;

    // search_on takes a vector of cluster ids and a batch of queries.
    // We run it with all probe_list clusters in one call so the per-cluster
    // search_into shared-buffer pattern is used internally.
    std::vector<std::vector<DocId>>  batch_ids(1);
    std::vector<std::vector<float>>  batch_scores(1);
    layer_idx->search_on(probe_list, query, 1, k, batch_ids, batch_scores);

    out_ids    = std::move(batch_ids[0]);
    out_scores = std::move(batch_scores[0]);

    if (out_ids.empty()) return -1;

    // Determine which cluster in probe_list contributed the top result.
    // We do a single lookup: find which cluster contains the top-1 doc id.
    int winning_cid = -1;
    if (!out_ids.empty()) {
        DocId top_id = out_ids[0];
        for (int cid : probe_list) {
            if (!layer_idx->centroid_ptr(cid)) continue;
            if (layer_idx->cluster_get_vector(cid, top_id)) {
                winning_cid = cid;
                break;
            }
        }
        // Fallback: just use the first valid cluster in the probe list.
        if (winning_cid == -1) winning_cid = probe_list[0];
    }
    return winning_cid;
}

// ======================================================================
// search()  — FSM-integrated cache path  (TODO-7)
//
// Flow per query (cache_enabled_ == true):
//   1. FSM predict → build L0 probe list (FSM-first)
//   2. Search L0 via search_layer_ordered_
//   3. append_step(L0) into traj
//   4. record_access_ for each probed L0 cluster   (TODO-6)
//   5. Early-termination check: kth_score < αet * dagent → skip L1+L2
//   6. FSM predict again (trajectory now has L0 step) → build L1 probe list
//   7. Search L1 via search_layer_ordered_
//   8. append_step(L1) into traj
//   9. record_access_ for L1 clusters
//  10. Early-termination check → skip L2
//  11. Search L2 normally (no FSM reordering)
//  12. Merge all layer results
//  13. Update dagent with k-th result score
//  14. Promote result vectors (same as before)
// ======================================================================
void MultiLevelIndex::search(const float* queries, size_t q_rows, int k, int nprobe,
                             std::vector<std::vector<DocId>>& out_ids,
                             std::vector<std::vector<float>>& out_scores,
                             RequestTrajectory* traj) const {
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;

    const size_t dim_sz = static_cast<size_t>(dim_);
    const float alpha_et = fsm_table_.config().alpha_et;
    const bool  fsm_on   = (traj != nullptr);

    if (cache_enabled_()) {
        std::shared_ptr<IVFIndex> l0_idx, l1_idx, l2_idx;
        std::vector<float> l0_cents, l1_cents, l2_cents;
        {
            std::shared_lock lk(topo_mu_);
            l0_idx   = l0_.index;
            l1_idx   = l1_.index;
            l2_idx   = l2_.index;
            l0_cents = l0_.centroids;
            l1_cents = l1_.centroids;
        }

        const float search_threshold = cfg_.search_threshold;
        const bool  has_threshold    = search_threshold < std::numeric_limits<float>::infinity();
        // 0=none 1=L0 only 2=L0+L1 3=L0+L1+L2
        std::vector<int> stage(q_rows, 0);

        for (size_t qi = 0; qi < q_rows; ++qi) {
            const float* qptr = queries + qi * dim_sz;

            // ---- get L2 probe ids (geometry-based, used for L2 fallback) ----
            std::vector<int> l2_probe_ids;
            if (l2_idx) l2_idx->get_probe_ids(qptr, nprobe, l2_probe_ids);

            // Per-layer result buffers (filled by search_layer_ordered_ / search_on).
            std::vector<DocId>  l0_ids, l1_ids, l2_ids;
            std::vector<float>  l0_scores, l1_scores, l2_scores;

            // ==============================================================
            // Stage 1: L0 with FSM reordering
            // ==============================================================
            int l0_winning_cid = -1;
            if (l0_idx) {
                std::vector<int> fsm_pref;
                if (fsm_on) {
                    auto pr = fsm_table_.match_and_predict(*traj, FSMLayer::L0);
                    fsm_pref = std::move(pr.ranked_clusters);
                }
                std::vector<int> l0_probe = build_probe_list_(
                    qptr, fsm_pref, l0_idx, l0_cents,
                    (nprobe > 0 ? nprobe : l0_idx->nlist()));

                l0_winning_cid = search_layer_ordered_(
                    qptr, k, l0_probe, l0_idx, l0_ids, l0_scores);

                // record_access_ for all probed L0 clusters (TODO-6).
                for (int cid : l0_probe) record_access_(cid);

                // append L0 step to trajectory (TODO-1).
                if (fsm_on && l0_winning_cid >= 0) {
                    std::vector<float> emb(qptr, qptr + dim_sz);
                    traj->append_step(std::move(emb), l0_winning_cid, FSMLayer::L0);
                }
                stage[qi] = 1;
            }

            // ==============================================================
            // Early termination after L0  (TODO-5)
            // ==============================================================
            bool satisfied = false;
            {
                float ks  = l0_scores.empty() ? std::numeric_limits<float>::infinity()
                          : (static_cast<int>(l0_scores.size()) >= k
                             ? l0_scores[static_cast<size_t>(k - 1)] : l0_scores.back());
                float dag = dagent_();
                bool et_fsm = (dag < std::numeric_limits<float>::infinity())
                           && (ks < alpha_et * dag);
                bool et_thr = has_threshold && (ks <= search_threshold);
                if (et_fsm || et_thr) satisfied = true;
            }

            // ==============================================================
            // Stage 2: L1 with FSM reordering
            // ==============================================================
            int l1_winning_cid = -1;
            if (!satisfied && l1_idx) {
                std::vector<int> fsm_pref;
                if (fsm_on) {
                    auto pr = fsm_table_.match_and_predict(*traj, FSMLayer::L1);
                    fsm_pref = std::move(pr.ranked_clusters);
                }
                std::vector<int> l1_probe = build_probe_list_(
                    qptr, fsm_pref, l1_idx, l1_cents,
                    (nprobe > 0 ? nprobe : l1_idx->nlist()));

                l1_winning_cid = search_layer_ordered_(
                    qptr, k, l1_probe, l1_idx, l1_ids, l1_scores);

                for (int cid : l1_probe) record_access_(cid);

                if (fsm_on && l1_winning_cid >= 0) {
                    std::vector<float> emb(qptr, qptr + dim_sz);
                    traj->append_step(std::move(emb), l1_winning_cid, FSMLayer::L1);
                }
                stage[qi] = 2;

                // Early termination after L1.
                std::vector<DocId>  tmp_ids; std::vector<float> tmp_sc;
                merge_levels_({{l0_ids},{l1_ids}},{{l0_scores},{l1_scores}},k,tmp_ids,tmp_sc);
                float ks  = tmp_sc.empty() ? std::numeric_limits<float>::infinity()
                          : (static_cast<int>(tmp_sc.size()) >= k
                             ? tmp_sc[static_cast<size_t>(k - 1)] : tmp_sc.back());
                float dag = dagent_();
                bool et_fsm = (dag < std::numeric_limits<float>::infinity())
                           && (ks < alpha_et * dag);
                bool et_thr = has_threshold && (ks <= search_threshold);
                if (et_fsm || et_thr) satisfied = true;
            }

            // ==============================================================
            // Stage 3: L2 — no FSM reordering, standard geometry order
            // ==============================================================
            if (!satisfied && l2_idx && !l2_probe_ids.empty()) {
                std::vector<std::vector<DocId>>  l2b(1);
                std::vector<std::vector<float>>  l2sb(1);
                l2_idx->search_on(l2_probe_ids, qptr, 1, k, l2b, l2sb);
                l2_ids    = std::move(l2b[0]);
                l2_scores = std::move(l2sb[0]);
                stage[qi] = 3;
            }

            // ==============================================================
            // Merge all layers and emit
            // ==============================================================
            {
                std::vector<std::vector<DocId>>  per_ids    = {l0_ids, l1_ids, l2_ids};
                std::vector<std::vector<float>>  per_scores = {l0_scores, l1_scores, l2_scores};
                merge_levels_(per_ids, per_scores, k, out_ids[qi], out_scores[qi]);
            }

            // Update dagent with k-th score.
            if (!out_scores[qi].empty()) {
                float ks = (static_cast<int>(out_scores[qi].size()) >= k)
                         ? out_scores[qi][static_cast<size_t>(k - 1)]
                         : out_scores[qi].back();
                update_dagent_(ks);
            }
        }  // end per-query loop

        // ================================================================
        // Promotion + predictive prefetch (background, non-blocking)
        //
        // Reactive: enqueue one task per result doc_id. The worker calls
        //   promote_vector_neighborhood_(doc_id), exactly what the old
        //   blocking loop did, but on a background thread.
        //
        // Predictive: after all queries are done, ask the FSM which cluster
        //   comes next and enqueue a centroid-anchored task for it. The
        //   worker calls promote_cluster_centroid_(cid), which searches the
        //   cluster centroid neighbourhood and copies the dense core into
        //   L0/L1 before the next search arrives.
        // ================================================================
        for (size_t qi = 0; qi < q_rows; ++qi) {
            const int st = stage[qi];
            int limit = cache_config_.max_promote_per_query;
            if (limit <= 0) limit = static_cast<int>(out_ids[qi].size());

            for (int j = 0; j < limit && j < static_cast<int>(out_ids[qi].size()); ++j) {
                DocId doc_id = out_ids[qi][j];
                // record_access_ is cheap (just a timestamp); keep it synchronous.
                {
                    std::shared_lock lk(topo_mu_);
                    auto it = doc_id_to_cid_.find(doc_id);
                    if (it != doc_id_to_cid_.end())
                        record_access_(it->second);
                }
                // Reactive promotion: skip if already satisfied by L0 alone
                // (vectors already in the hottest tier — no promotion needed).
                if (st == 1) continue;
                enqueue_prefetch_({/*cluster_id=*/-1, doc_id, /*predictive=*/false});
            }
        }

        // Predictive prefetch: if FSM is active and traj has steps, predict
        // the next cluster and enqueue a centroid-anchored warmup task.
        if (fsm_on && traj && traj->length() > 0) {
            auto pr = fsm_table_.match_and_predict(*traj, FSMLayer::L0);
            for (int pred_cid : pr.ranked_clusters) {
                enqueue_prefetch_({pred_cid, DocId{-1}, /*predictive=*/true});
            }
        }

        return;
    }  // end cache_enabled_ branch

    // ======================================================================
    // Non-cache path: unchanged from original
    // ======================================================================
    Layer l0, l1, l2;
    {
        std::shared_lock lk(topo_mu_);
        l0 = l0_; l1 = l1_; l2 = l2_;
    }

    std::vector<std::vector<DocId>> l0_ids(q_rows), l1_ids(q_rows), l2_ids(q_rows);
    std::vector<std::vector<float>> l0_scores(q_rows), l1_scores(q_rows), l2_scores(q_rows);

    auto search_level = [&](const Layer& layer,
                            std::vector<std::vector<DocId>>& ids,
                            std::vector<std::vector<float>>& scores) {
        if (layer.index) layer.index->search_nprobe(queries, q_rows, k, nprobe, ids, scores);
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

    bool need_deeper = !has_threshold;
    if (has_threshold)
        for (size_t qi = 0; qi < q_rows; ++qi)
            if (kth_score(qi) > search_threshold) { need_deeper = true; break; }
    if (need_deeper) search_level(l1, l1_ids, l1_scores);

    auto kth_score_after_l1 = [&](size_t qi) -> float {
        const auto& s1 = l1_scores[qi];
        if (s1.empty()) return kth_score(qi);
        float s1k = ((int)s1.size() >= k) ? s1[static_cast<size_t>(k - 1)] : s1.back();
        return std::min(kth_score(qi), s1k);
    };

    bool need_l2 = !has_threshold;
    if (has_threshold && need_deeper)
        for (size_t qi = 0; qi < q_rows; ++qi)
            if (kth_score_after_l1(qi) > search_threshold) { need_l2 = true; break; }
    if (need_l2) search_level(l2, l2_ids, l2_scores);

    for (size_t qi = 0; qi < q_rows; ++qi) {
        if (has_threshold && !l0_scores[qi].empty() && kth_score(qi) <= search_threshold) {
            out_ids[qi] = l0_ids[qi]; out_scores[qi] = l0_scores[qi]; continue;
        }
        std::vector<std::vector<DocId>>  per_ids    = {l0_ids[qi], l1_ids[qi], l2_ids[qi]};
        std::vector<std::vector<float>>  per_scores = {l0_scores[qi], l1_scores[qi], l2_scores[qi]};
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

    const int l2_nlist_before = l2_idx ? l2_idx->nlist() : 0;
    if (l2_idx) l2_idx->maintenance_pass();
    const int l2_nlist_after  = l2_idx ? l2_idx->nlist() : 0;

    // ===== 2b) Reconcile L0/L1/metadata_ after L2 topology changes =====
    // Merge: merge_clusters only invalidates a slot (valid_[cid]=false), nlist unchanged.
    // We must always run merge-invalidation so doc_id_to_cid_ is remapped for merged-away slots.
    // Split: adds new slots, so nlist increases; handle new slots and extend metadata_/centroids.
    if (cache_mode && l2_idx) {
        std::unique_lock lk(topo_mu_);

        // --- Merges: any slot that existed before and is now invalid (merged away). ---
        // Use l2_nlist_before so we don't miss slots [l2_nlist_after, l2_nlist_before) if nlist ever shrinks.
        for (int cid = 0; cid < l2_nlist_before; ++cid) {
            if (l2_idx->centroid_ptr(cid)) continue; // still valid

            std::vector<DocId> stale_docs;
            for (auto& kv : doc_id_to_cid_) {
                if (kv.second == cid) stale_docs.push_back(kv.first);
            }
            for (DocId id : stale_docs) {
                int nc = find_l2_cluster_for_doc_(id);
                if (nc >= 0) doc_id_to_cid_[id] = nc;
                else         doc_id_to_cid_.erase(id);
            }
            if (l0_idx) l0_idx->remove_cluster(cid);
            if (l1_idx) l1_idx->remove_cluster(cid);
            // Invalidate FSM states referencing this cluster so the FSM
            // does not predict a cluster id that no longer exists (Q-2).
            fsm_table_.invalidate_cluster(cid);
        }

        // --- Splits: new slots >= l2_nlist_before (only when nlist increased) ---
        if (l2_nlist_after > l2_nlist_before) {
        for (int new_cid = l2_nlist_before; new_cid < l2_nlist_after; ++new_cid) {
            const float* c2 = l2_idx->centroid_ptr(new_cid);
            if (!c2) continue;

            std::vector<DocId> moved_ids;
            std::vector<float> moved_vecs;
            l2_idx->export_cluster_live(new_cid, moved_ids, moved_vecs);

            int origin_cid = -1;
            for (DocId id : moved_ids) {
                auto it = doc_id_to_cid_.find(id);
                if (it != doc_id_to_cid_.end() && it->second != new_cid) {
                    if (origin_cid < 0) origin_cid = it->second;
                    it->second = new_cid;
                }
            }

            // Drop origin from L0/L1: both partitions' neighbourhoods changed.
            if (origin_cid >= 0) {
                if (l0_idx) l0_idx->remove_cluster(origin_cid);
                if (l1_idx) l1_idx->remove_cluster(origin_cid);
                // Invalidate FSM states for the old cluster that was split.
                fsm_table_.invalidate_cluster(origin_cid);
            }

            // Register new slot as empty placeholder in L0/L1.
            std::vector<float> cent(c2, c2 + static_cast<size_t>(dim_));
            if (l0_idx) l0_idx->ensure_cluster(new_cid, cent);
            if (l1_idx) l1_idx->ensure_cluster(new_cid, cent);
        }
        }

        // --- Extend metadata_ and l2_.centroids; refresh nlist/meta_snap for steps 3-7 ---
        {
            std::lock_guard<std::mutex> ml(meta_mu_);
            if (static_cast<int>(metadata_.size()) < l2_nlist_after) {
                metadata_.resize(static_cast<size_t>(l2_nlist_after));
                for (int cid = l2_nlist_before; cid < l2_nlist_after; ++cid)
                    metadata_[static_cast<size_t>(cid)].in_l2 = true;
            }
            l2_.centroids.resize(
                static_cast<size_t>(l2_nlist_after) * static_cast<size_t>(dim_));
            for (int cid = 0; cid < l2_nlist_after; ++cid) {
                const float* c2 = l2_idx->centroid_ptr(cid);
                if (c2)
                    std::copy(c2, c2 + dim_,
                              l2_.centroids.begin() + cid * static_cast<size_t>(dim_));
            }
            nlist     = static_cast<size_t>(l2_nlist_after);
            meta_snap = metadata_;
        }
    }

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

int MultiLevelIndex::find_l2_cluster_for_doc_(DocId id) const {
    if (!l2_.index) return -1;
    const int nlist = l2_.index->nlist();
    for (int c = 0; c < nlist; ++c) {
        if (l2_.index->centroid_ptr(c) && l2_.index->cluster_get_vector(c, id))
            return c;
    }
    return -1;
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
    // Use a relaxed atomic store — we don't need precision here, just recency.
    // This avoids taking meta_mu_ on every cache hit under concurrent query load.
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) >= metadata_.size()) return;
    // Only update if newer — avoids false cache-line dirtying when called repeatedly.
    uint64_t& t = metadata_[static_cast<size_t>(cid)].last_access_time;
    if (now_ns > t) t = now_ns;
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
    // Search once for the wider L1 neighbourhood; L0 candidates are just the first l0_k.
    const int search_k = std::max(l0_k, l1_k);

    std::vector<float> cent(l2_.centroids.begin() + cid * dim_sz,
                            l2_.centroids.begin() + (cid + 1) * dim_sz);

    // Single within-cluster search returns ids sorted by score (nearest first).
    std::vector<DocId> ids_all;
    std::vector<float> scores_all;
    l2_.index->search_within_cluster(cid, vec, search_k, ids_all, scores_all);

    if (ids_all.empty()) return;

    // Fetch all vectors in one pass (avoids per-id hash lookup repeated across L0 and L1).
    std::vector<float> vecs_all;
    vecs_all.reserve(ids_all.size() * dim_sz);
    std::vector<DocId> ids_fetched;
    ids_fetched.reserve(ids_all.size());
    for (DocId id : ids_all) {
        const float* v = l2_.index->cluster_get_vector(cid, id);
        if (v) {
            vecs_all.insert(vecs_all.end(), v, v + dim_sz);
            ids_fetched.push_back(id);
        }
    }
    if (ids_fetched.empty()) return;

    // Promote the narrow L0 neighbourhood (first l0_k results).
    const size_t l0_count = std::min(static_cast<size_t>(l0_k), ids_fetched.size());
    if (l0_count > 0 && l0_.index) {
        l0_.index->ensure_cluster(cid, cent);
        l0_.index->update_batch(cid, ids_fetched.data(), vecs_all.data(), l0_count, true);
    }

    // Promote the wider L1 neighbourhood (all fetched results up to l1_k).
    const size_t l1_count = std::min(static_cast<size_t>(l1_k), ids_fetched.size());
    if (l1_count > 0 && l1_.index) {
        l1_.index->ensure_cluster(cid, cent);
        l1_.index->update_batch(cid, ids_fetched.data(), vecs_all.data(), l1_count, true);
    }

    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) < metadata_.size()) {
        ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
        m.last_access_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (l0_count > 0) m.in_l0 = true;
        if (l1_count > 0) m.in_l1 = true;
        if (l0_.index) m.l0_vector_count = l0_.index->cluster_live_size(cid);
        if (l1_.index) m.l1_vector_count = l1_.index->cluster_live_size(cid);
    }
}

void MultiLevelIndex::demote_cluster_(int cid) const {
    if (cid < 0) return;
    // Snapshot index pointers under topo_mu_ first; never read l0_.index / l1_.index
    // while only holding meta_mu_ (topo_mu_ protects the shared_ptrs themselves).
    std::shared_ptr<IVFIndex> l0_idx, l1_idx;
    {
        std::shared_lock topo_lk(topo_mu_);
        l0_idx = l0_.index;
        l1_idx = l1_.index;
    }
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) >= metadata_.size()) return;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
    const uint64_t cold = cache_config_.cold_time_ns;
    const uint64_t elapsed = (now_ns >= m.last_access_time) ? (now_ns - m.last_access_time) : 0;

    if (m.in_l0 && elapsed > cold && l0_idx) {
        l0_idx->remove_cluster(cid);
        m.in_l0 = false;
        m.l0_vector_count = 0;
    }
    if (m.in_l1 && elapsed > cold && l1_idx) {
        l1_idx->remove_cluster(cid);
        m.in_l1 = false;
        m.l1_vector_count = 0;
    }
}

// ------------------------------------------------------------
// (search_with_ordering removed — logic absorbed into search())

// ======================================================================
// enqueue_prefetch_
//
// Push a PrefetchTask onto the bounded prefetch queue.
// If the queue is full, the task is silently dropped — prefetch is
// best-effort and correctness never depends on it completing.
// ======================================================================
void MultiLevelIndex::enqueue_prefetch_(PrefetchTask task) const {
    {
        std::lock_guard<std::mutex> lk(prefetch_mu_);
        if (prefetch_queue_.size() >= cache_config_.prefetch_queue_capacity) {
            return;  // drop — queue full
        }
        prefetch_queue_.push_back(std::move(task));
    }
    prefetch_cv_.notify_one();
}

// ======================================================================
// promote_cluster_centroid_
//
// Predictive counterpart to promote_vector_neighborhood_.
// Used when no specific anchor doc is known yet (predictive prefetch):
// searches the cluster using its centroid as the query vector and copies
// the densest core of the cluster into L0/L1.
//
// Must be called WITHOUT holding topo_mu_ (it acquires shared lock inside).
// ======================================================================
void MultiLevelIndex::promote_cluster_centroid_(int cluster_id) const {
    if (cluster_id < 0) return;

    std::shared_ptr<IVFIndex> l2_idx, l0_idx, l1_idx;
    std::vector<float> centroid_vec;
    {
        std::shared_lock lk(topo_mu_);
        l2_idx = l2_.index;
        l0_idx = l0_.index;
        l1_idx = l1_.index;
        if (!l2_idx) return;
        const float* c = l2_idx->centroid_ptr(cluster_id);
        if (!c) return;
        centroid_vec.assign(c, c + static_cast<size_t>(dim_));
    }

    // Search the cluster using its centroid — returns the densest core vectors.
    const int search_k = std::max(cache_config_.l0_neighborhood_k,
                                   cache_config_.l1_neighborhood_k);
    std::vector<DocId>  ids_found;
    std::vector<float>  scores_found;
    l2_idx->search_within_cluster(cluster_id, centroid_vec.data(),
                                   search_k, ids_found, scores_found);
    if (ids_found.empty()) return;

    // Fetch vectors in one pass.
    const size_t dim_sz = static_cast<size_t>(dim_);
    std::vector<float> vecs_found;
    vecs_found.reserve(ids_found.size() * dim_sz);
    std::vector<DocId> ids_fetched;
    ids_fetched.reserve(ids_found.size());
    for (DocId id : ids_found) {
        const float* v = l2_idx->cluster_get_vector(cluster_id, id);
        if (v) {
            vecs_found.insert(vecs_found.end(), v, v + dim_sz);
            ids_fetched.push_back(id);
        }
    }
    if (ids_fetched.empty()) return;

    // Promote into L0/L1 using unique_lock for writes.
    std::unique_lock lk(topo_mu_);
    const size_t l0_count = std::min(static_cast<size_t>(cache_config_.l0_neighborhood_k),
                                      ids_fetched.size());
    if (l0_count > 0 && l0_.index) {
        l0_.index->ensure_cluster(cluster_id, centroid_vec);
        l0_.index->update_batch(cluster_id, ids_fetched.data(),
                                 vecs_found.data(), l0_count, true);
    }
    const size_t l1_count = std::min(static_cast<size_t>(cache_config_.l1_neighborhood_k),
                                      ids_fetched.size());
    if (l1_count > 0 && l1_.index) {
        l1_.index->ensure_cluster(cluster_id, centroid_vec);
        l1_.index->update_batch(cluster_id, ids_fetched.data(),
                                 vecs_found.data(), l1_count, true);
    }

    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cluster_id) < metadata_.size()) {
        ClusterMetadata& m = metadata_[static_cast<size_t>(cluster_id)];
        const uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        m.last_access_time = now_ns;
        if (l0_count > 0) { m.in_l0 = true; m.l0_vector_count = l0_count; }
        if (l1_count > 0) { m.in_l1 = true; m.l1_vector_count = l1_count; }
    }
}

// ======================================================================
// prefetch_worker_loop_
//
// Background thread that drains the prefetch queue.
// For reactive tasks (anchor_doc != -1): calls promote_vector_neighborhood_.
// For predictive tasks (anchor_doc == -1): calls promote_cluster_centroid_.
//
// Both paths are identical to what the old blocking promotion loop did,
// just decoupled from the search thread.
// ======================================================================
void MultiLevelIndex::prefetch_worker_loop_() {
    while (true) {
        PrefetchTask task;
        {
            std::unique_lock<std::mutex> lk(prefetch_mu_);
            prefetch_cv_.wait(lk, [this]{
                return !prefetch_queue_.empty() || !prefetch_running_.load();
            });

            if (!prefetch_running_.load() && prefetch_queue_.empty()) break;
            if (prefetch_queue_.empty()) continue;

            task = prefetch_queue_.front();
            prefetch_queue_.pop_front();
        }

        // Execute outside the queue lock so searches can enqueue concurrently.
        if (task.anchor_doc != DocId{-1}) {
            // Reactive: promote the neighbourhood around a specific result vector.
            promote_vector_neighborhood_(task.anchor_doc);
        } else if (task.cluster_id >= 0) {
            // Predictive: warm the densest core of a predicted cluster.
            promote_cluster_centroid_(task.cluster_id);
        }
    }
}

// ======================================================================
// start_prefetch_thread_ / stop_prefetch_thread_
// ======================================================================
void MultiLevelIndex::start_prefetch_thread_() {
    prefetch_running_.store(true);
    prefetch_thread_ = std::thread([this]{ prefetch_worker_loop_(); });
}

void MultiLevelIndex::stop_prefetch_thread_() {
    prefetch_running_.store(false);
    prefetch_cv_.notify_all();
    if (prefetch_thread_.joinable())
        prefetch_thread_.join();
}

} // namespace m3