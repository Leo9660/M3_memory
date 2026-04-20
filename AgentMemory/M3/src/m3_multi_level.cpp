#include "m3_multi_level.h"
#include "gpu_coordinator.h"
#include "m3_logger.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <limits>
#include <mutex>
#include <omp.h>

// Runtime verbose logging: set env var M3_DEBUG=1 before running to see detailed
// per-query, per-promotion, and maintenance events on stderr.
static bool m3_verbose = (std::getenv("M3_DEBUG") != nullptr);

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
    l0_.name = "L0";
    l1_.name = "L1";
    l2_.name = "L2";
    // alpha_et_dynamic_ starts at the configured value; adapted at runtime when
    // calibration_interval > 0 and alpha_et_adapt_rate > 0.
    alpha_et_dynamic_ = cache_config_.alpha_et;
}

void MultiLevelIndex::ensure_layer_initialized_(Layer& layer, int nlist_hint) {
    if (layer.index) return;

    // Build a fresh IVFIndex with placeholder centroids so callers can write immediately.
    layer.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_, layer.name);
    layer.centroids = make_zero_centroids(nlist_hint, dim_);
    layer.index->set_centroids(layer.centroids);
}

void MultiLevelIndex::ensure_layer_centroids_(Layer& layer, const std::vector<float>& centroids) {
    if (centroids.empty()) return;
    const int nlist = static_cast<int>(centroids.size() / static_cast<size_t>(dim_));
    if (layer.index) {
        layer.index->set_centroids(centroids);
    } else {
        layer.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_, layer.name);
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
        // L0 and L1 use their own query-centric topology — do NOT seed from L2 centroids.
        // Just ensure both index objects exist (empty; clusters added dynamically on promotion).
        if (!l0_.index)
            l0_.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_, l0_.name);
        if (!l1_.index)
            l1_.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_, l1_.name);
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
        using clock = std::chrono::steady_clock;
        using fms   = std::chrono::duration<double, std::milli>;
        const bool profiling = M3Profiler::instance().is_enabled();
        const auto t_insert_start = clock::now();

        const size_t dim_sz = static_cast<size_t>(dim_);
        std::vector<int> cids(n_rows);

        // Stage 1: assign each vector to its canonical L2 cluster.
        const auto t_assign0 = profiling ? clock::now() : clock::time_point{};
        l2_.index->nearest_clusters(vecs, n_rows, cids);
        const double p_assign_ms = profiling ? fms(clock::now() - t_assign0).count() : 0.0;

        // Vectors targeting GPU-resident clusters are collected and dispatched
        // after releasing topo_mu_, because the buffer-overflow fallback path
        // inside GpuCoordinator::insert() calls idx_.load_cluster() which also
        // acquires topo_mu_ — holding it here would deadlock.
        struct GpuPending { int cid; DocId id; size_t vec_offset; };
        std::vector<GpuPending> gpu_pending;

        // Per-cid count deltas applied to metadata after the loop (one meta_mu_ lock).
        std::unordered_map<int, size_t> l2_delta;

        // Stage 2: L2 writes only. L0/L1 are populated exclusively via post-search promotion.
        const auto t_l0l2_0 = profiling ? clock::now() : clock::time_point{};
        for (size_t i = 0; i < n_rows; ++i) {
            int cid = cids[i];
            if (cid < 0) continue;
            // Record canonical L2 cluster (used by update/erase/promote).
            doc_id_to_cid_[ids[i]] = cid;
            if (gpu_coord_ && gpu_coord_->is_gpu_resident(cid)) {
                // GPU-resident: defer to after lock release.
                gpu_pending.push_back({cid, ids[i], i * dim_sz});
            } else {
                // Not GPU-resident: write to L2 (ground truth).
                if (l2_.index) {
                    l2_.index->add_batch(cid, &ids[i], vecs + i * dim_sz, 1);
                    ++l2_delta[cid];
                }
            }
        }
        const double p_l0l2_ms = profiling ? fms(clock::now() - t_l0l2_0).count() : 0.0;

        // Apply deltas to metadata under meta_mu_.
        if (!l2_delta.empty()) {
            std::lock_guard<std::mutex> ml(meta_mu_);
            for (auto& [cid, d] : l2_delta)
                if (static_cast<size_t>(cid) < metadata_.size())
                    metadata_[cid].l2_vector_count += d;
        }
        if (m3_verbose) {
            fprintf(stderr, "[M3:insert] cache-mode  n=%zu  gpu_pending=%zu\n",
                    n_rows, gpu_pending.size());
        }
        auto strat = l1_strategy_;
        lk.unlock();

        // Stage 3: GPU dispatch (outside lock — see deadlock note above).
        const auto t_gpu0 = profiling ? clock::now() : clock::time_point{};
        for (const auto& gp : gpu_pending)
            gpu_coord_->insert(gp.cid, gp.id, vecs + gp.vec_offset);
        const double p_gpu_ms = profiling ? fms(clock::now() - t_gpu0).count() : 0.0;

        if (profiling) {
            const double total_ms = fms(clock::now() - t_insert_start).count();
            // nearest_clusters uses scalar unified_score scan (no BLAS); assign_ms IS the
            // centroid scan with no separate topk step (running-best-so-far), so assign_topk_ms=0.
            M3Profiler::instance().log_insert_row(
                n_rows, gpu_pending.size(),
                p_assign_ms, p_assign_ms, 0.0,
                p_l0l2_ms, p_gpu_ms, total_ms);
        }

        if (strat) strat->on_insert(ids, vecs, n_rows);
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
                    }
                }
                continue;
            }
            int cid = it->second;
            const float* vptr = vecs + i * dim_sz;
            l2_.index->update_batch(cid, &id, vptr, 1, false);
            // Update L1/L0 in-place if the vector is already cached there.
            // L0/L1 use their own query-centric cluster IDs so look up via doc_id_to_l1_cid_.
            {
                std::lock_guard<std::mutex> lk(l1_cache_mu_);
                auto l1it = doc_id_to_l1_cid_.find(id);
                if (l1it != doc_id_to_l1_cid_.end()) {
                    const int l1cid = l1it->second;
                    if (l1_.index && l1_.index->cluster_get_vector(l1cid, id))
                        l1_.index->update_batch(l1cid, &id, vptr, 1, false);
                    if (l0_.index && l0_.index->cluster_get_vector(l1cid, id))
                        l0_.index->update_batch(l1cid, &id, vptr, 1, false);
                }
            }
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

// Enable initial corpus insertion directly to L2
void MultiLevelIndex::load_cluster(int cid, const DocId* ids, const float* vecs, size_t n_rows) {
    if (!ids || !vecs || n_rows == 0) return;
    std::unique_lock lk(topo_mu_);
    if (!l2_.index || cid < 0) return;
    const size_t dim_sz = static_cast<size_t>(dim_);
    const size_t nlist = l2_.centroids.size() / dim_sz;
    if (static_cast<size_t>(cid) >= nlist) return;

    l2_.index->add_batch(cid, ids, vecs, n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        doc_id_to_cid_[ids[i]] = cid;
    }
    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (static_cast<size_t>(cid) < metadata_.size()) {
            metadata_[cid].l2_vector_count += n_rows;
        }
    }
}

bool MultiLevelIndex::export_l2_cluster(int cid,
                                         std::vector<DocId>& out_ids,
                                         std::vector<float>&  out_vecs) const {
    std::shared_lock lk(topo_mu_);
    if (!l2_.index || cid < 0) return false;
    const size_t nlist = l2_.centroids.empty() ? 0
                       : l2_.centroids.size() / static_cast<size_t>(dim_);
    if (static_cast<size_t>(cid) >= nlist) return false;
    l2_.index->export_cluster_live(cid, out_ids, out_vecs);
    return true;
}

void MultiLevelIndex::rebuild_l2_cluster(int cid,
                                          const DocId*  ids,
                                          const float*  vecs,
                                          size_t        n) {
    if (cid < 0) return;
    std::unique_lock lk(topo_mu_);
    if (!l2_.index) return;
    const size_t nlist = l2_.centroids.empty() ? 0
                       : l2_.centroids.size() / static_cast<size_t>(dim_);
    if (static_cast<size_t>(cid) >= nlist) return;

    // Replace cluster content; IVFIndex::rebuild_cluster handles tombstone cleanup.
    l2_.index->rebuild_cluster(cid, ids, vecs, n);

    // Rebuild doc_id_to_cid_ for this cluster's new vectors.
    // Remove old entries that point to cid.
    for (auto it = doc_id_to_cid_.begin(); it != doc_id_to_cid_.end(); ) {
        if (it->second == cid) it = doc_id_to_cid_.erase(it);
        else                   ++it;
    }
    for (size_t i = 0; i < n; ++i)
        doc_id_to_cid_[ids[i]] = cid;

    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (static_cast<size_t>(cid) < metadata_.size())
            metadata_[cid].l2_vector_count = n;
    }
}

int MultiLevelIndex::add_l2_cluster(const float* centroid,
                                     const DocId* ids,
                                     const float* vecs,
                                     size_t       n) {
    if (!centroid) return -1;
    std::unique_lock lk(topo_mu_);
    if (!l2_.index) return -1;

    const std::vector<float> c(centroid, centroid + dim_);

    // Add to L2 index; returns the new cluster id.
    const int new_cid = l2_.index->add_cluster(c);
    if (new_cid < 0) return -1;

    // Mirror centroid into l2_.centroids so centroid-based routing sees it.
    l2_.centroids.insert(l2_.centroids.end(), c.begin(), c.end());

    // Mirror into l0_ / l1_ centroid tables for routing consistency.
    if (l0_.index) {
        l0_.index->add_cluster(c);
        l0_.centroids.insert(l0_.centroids.end(), c.begin(), c.end());
    }
    if (l1_.index) {
        l1_.index->add_cluster(c);
        l1_.centroids.insert(l1_.centroids.end(), c.begin(), c.end());
    }

    // Insert initial vectors if provided.
    if (ids && vecs && n > 0) {
        l2_.index->add_batch(new_cid, ids, vecs, n);
        for (size_t i = 0; i < n; ++i)
            doc_id_to_cid_[ids[i]] = new_cid;
    }

    // Extend metadata table.
    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (static_cast<size_t>(new_cid) >= metadata_.size())
            metadata_.resize(static_cast<size_t>(new_cid) + 1);
        metadata_[new_cid].in_l2          = true;
        metadata_[new_cid].l2_vector_count = n;
    }

    return new_cid;
}

void MultiLevelIndex::search(const float* queries, size_t q_rows, int k, int nprobe,
                             std::vector<std::vector<DocId>>& out_ids,
                             std::vector<std::vector<float>>& out_scores) const {
    out_ids.assign(q_rows, {});
    out_scores.assign(q_rows, {});
    if (!queries || q_rows == 0 || k <= 0) return;

    const size_t dim_sz = static_cast<size_t>(dim_);

    if (cache_enabled_()) {
        using clock = std::chrono::steady_clock;
        using fms   = std::chrono::duration<double, std::milli>;
        const bool profiling = M3Profiler::instance().is_enabled();
        const auto t_search_start = clock::now();

        Layer l0, l1, l2;
        {
            std::shared_lock lk(topo_mu_);
            l0 = l0_;
            l1 = l1_;
            l2 = l2_;
        }
        // Dynamic early-termination threshold: alpha_et_dynamic_ · dagent_.
        // alpha_et_dynamic_ is initialised from cache_config_.alpha_et and adapted at
        // runtime by the background calibration path. Falls back to the static
        // cfg_.search_threshold when dagent_ is not yet populated or alpha_et is 0.
        float search_threshold;
        bool has_threshold;
        {
            std::lock_guard<std::mutex> dlk(dagent_mu_);
            std::lock_guard<std::mutex> alk(alpha_et_mu_);
            if (alpha_et_dynamic_ > 0.f && dagent_ > 0.f) {
                search_threshold = alpha_et_dynamic_ * dagent_;
                has_threshold = true;
            } else {
                search_threshold = cfg_.search_threshold;
                has_threshold = search_threshold < std::numeric_limits<float>::infinity();
            }
        }

        // Profile accumulators (only populated when profiling is enabled).
        double p_probe_ms = 0, p_probe_sgemm_ms = 0, p_probe_topk_ms = 0;
        double p_l0_ms = 0, p_l0_centroid_ms = 0, p_l0_scan_ms = 0;
        double p_l1_ms = 0, p_l1_centroid_ms = 0, p_l1_scan_ms = 0;

        // Per-stage query timing buckets.
        // Note: with batched L0/L1, per-query exit wall times are not tracked;
        // these remain 0 — exit counts are still accurate.
        // p_l2_reach_total_ms is the wall time of the entire L2 phase block.
        double p_l0_exit_total_ms = 0, p_l1_exit_total_ms = 0, p_l2_reach_total_ms = 0;
        clock::time_point t_l2_start;

        // GPU sub-phase timing (accumulated across all L2-reaching queries).
        GpuCollabTiming p_gpu_timing;

        // Batch probe selection: one sgemm across all queries instead of q_rows scalar loops.
        std::vector<std::vector<int>> all_probe_ids(q_rows);
        {
            const auto t0 = profiling ? clock::now() : clock::time_point{};
            if (l2.index)
                l2.index->batch_get_probe_ids(queries, q_rows, nprobe, all_probe_ids,
                                              profiling ? &p_probe_sgemm_ms : nullptr,
                                              profiling ? &p_probe_topk_ms  : nullptr);
            if (profiling) p_probe_ms = fms(clock::now() - t0).count();
        }

        // k_promo: how many L2 candidates to retrieve for L1 promotion.
        // L1 uses all k_promo results; L0 gets min(l0_neighborhood_k, k); caller gets top k.
        // When l1_neighborhood_k <= k, k_promo == k and there is no extra L2 work.
        const int k_promo = std::max(k, cache_config_.l1_neighborhood_k);

        // Per-query result arrays for L0 and L1 (populated by batch calls below).
        std::vector<std::vector<DocId>> all_l0_ids(q_rows), all_l1_ids(q_rows);
        std::vector<std::vector<float>> all_l0_scores(q_rows), all_l1_scores(q_rows);

        // L2 scratch (reused per query in the L2 loop below).
        std::vector<std::vector<DocId>> l2_ids(1);
        std::vector<std::vector<float>> l2_scores(1);

        // Raw L2-only results (up to k_promo) saved per query for L1 promotion.
        std::vector<std::vector<DocId>>  promo_ids(q_rows);
        std::vector<std::vector<float>>  promo_scores(q_rows);
        // 0 = not yet processed, 1 = satisfied at L0, 2 = satisfied at L1, 3 = reached L2
        std::vector<int>   stage(q_rows, 0);
        // k-th distance at the point of early exit (inf = no early exit).
        std::vector<float> early_exit_kths(q_rows, std::numeric_limits<float>::infinity());
        double p_l2_gpu_ms = 0, p_l2_cpu_ms = 0, p_merge_ms = 0;
        size_t p_l0_exits = 0, p_l1_exits = 0;
        size_t p_l2_gpu_clusters = 0, p_l2_cpu_clusters = 0;

        auto kth_score_vec = [&](const std::vector<float>& s) -> float {
            if (s.empty()) return std::numeric_limits<float>::infinity();
            if ((int)s.size() >= k) return s[static_cast<size_t>(k - 1)];
            return s.back();
        };

        // ---------------------------------------------------------------
        // Phase 0: Batch L0 search — one search_nprobe call for all queries.
        // ---------------------------------------------------------------
        if (l0.index) {
            const int l0_nprobe_eff = (cache_config_.l0_nprobe > 0)
                                          ? cache_config_.l0_nprobe
                                          : std::min(nprobe, l0.index->nlist());
            if (l0_nprobe_eff > 0) {
                double l0c = 0, l0s = 0;
                const auto t0 = profiling ? clock::now() : clock::time_point{};
                l0.index->search_nprobe(queries, q_rows, k, l0_nprobe_eff,
                                        all_l0_ids, all_l0_scores,
                                        profiling ? &l0c : nullptr,
                                        profiling ? &l0s : nullptr);
                if (profiling) {
                    p_l0_ms         = fms(clock::now() - t0).count();
                    p_l0_centroid_ms = l0c;
                    p_l0_scan_ms     = l0s;
                }
            }
        }

        // Apply L0 early-exit threshold per query.
        if (has_threshold) {
            for (size_t qi = 0; qi < q_rows; ++qi) {
                if ((int)all_l0_scores[qi].size() < k) continue;
                const float ks = kth_score_vec(all_l0_scores[qi]);
                if (ks <= search_threshold) {
                    out_ids[qi]    = all_l0_ids[qi];
                    out_scores[qi] = all_l0_scores[qi];
                    stage[qi]      = 1;
                    early_exit_kths[qi] = ks;
                    if (profiling) ++p_l0_exits;
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 1: Batch L1 search — gather unsatisfied queries, one call.
        // ---------------------------------------------------------------
        if (l1.index) {
            const int l1_nprobe_eff = (cache_config_.l1_nprobe > 0)
                                          ? cache_config_.l1_nprobe
                                          : std::min(nprobe, l1.index->nlist());
            if (l1_nprobe_eff > 0) {
                // Collect indices of queries that still need L1.
                std::vector<size_t> l1_indices;
                l1_indices.reserve(q_rows);
                for (size_t qi = 0; qi < q_rows; ++qi)
                    if (stage[qi] == 0) l1_indices.push_back(qi);

                if (!l1_indices.empty()) {
                    const size_t n1 = l1_indices.size();

                    // Pack query vectors contiguously.
                    std::vector<float> l1_queries(n1 * dim_sz);
                    for (size_t i = 0; i < n1; ++i)
                        std::memcpy(l1_queries.data() + i * dim_sz,
                                    queries + l1_indices[i] * dim_sz,
                                    dim_sz * sizeof(float));

                    std::vector<std::vector<DocId>> tmp_ids(n1);
                    std::vector<std::vector<float>> tmp_scores(n1);
                    double l1c = 0, l1s = 0;
                    const auto t1 = profiling ? clock::now() : clock::time_point{};
                    l1.index->search_nprobe(l1_queries.data(), n1, k, l1_nprobe_eff,
                                            tmp_ids, tmp_scores,
                                            profiling ? &l1c : nullptr,
                                            profiling ? &l1s : nullptr);
                    if (profiling) {
                        p_l1_ms          = fms(clock::now() - t1).count();
                        p_l1_centroid_ms = l1c;
                        p_l1_scan_ms     = l1s;
                    }

                    // Scatter back and apply L1 early-exit threshold.
                    const auto t_merge = profiling ? clock::now() : clock::time_point{};
                    for (size_t i = 0; i < n1; ++i) {
                        const size_t qi = l1_indices[i];
                        all_l1_ids[qi]    = std::move(tmp_ids[i]);
                        all_l1_scores[qi] = std::move(tmp_scores[i]);

                        // Merge L0+L1 for this query to check early-exit.
                        if (has_threshold || !l2.index) {
                            std::vector<DocId> merged_ids;
                            std::vector<float> merged_scores;
                            std::vector<std::vector<DocId>> per_ids   = {all_l0_ids[qi], all_l1_ids[qi]};
                            std::vector<std::vector<float>> per_scores = {all_l0_scores[qi], all_l1_scores[qi]};
                            merge_levels_(per_ids, per_scores, k, merged_ids, merged_scores);

                            if (!l2.index) {
                                // No L2 at all — L1 is the final answer for this query.
                                out_ids[qi]    = std::move(merged_ids);
                                out_scores[qi] = std::move(merged_scores);
                                stage[qi] = 2;
                            } else if (has_threshold && !merged_scores.empty()) {
                                const float ks = kth_score_vec(merged_scores);
                                if (ks <= search_threshold) {
                                    out_ids[qi]         = std::move(merged_ids);
                                    out_scores[qi]      = std::move(merged_scores);
                                    stage[qi]           = 2;
                                    early_exit_kths[qi] = ks;
                                    if (profiling) ++p_l1_exits;
                                }
                            }
                        }
                    }
                    if (profiling) p_merge_ms += fms(clock::now() - t_merge).count();
                }
            }
        }

        // ---------------------------------------------------------------
        // Phase 2: Batched L2 search.
        //
        // Collect all unsatisfied queries → partition per-query probe clusters
        // into GPU-resident / CPU sets → one batched GPU SGEMM (async) +
        // OMP-parallel CPU scan running concurrently → scatter & merge.
        // ---------------------------------------------------------------
        if (l2.index) {
            if (profiling) t_l2_start = clock::now();
            // Collect L2 queries and partition their probe clusters.
            std::vector<size_t>              l2_qis;
            std::vector<std::vector<int>>    per_q_gpu_cids, per_q_cpu_cids;
            l2_qis.reserve(q_rows);

            for (size_t qi = 0; qi < q_rows; ++qi) {
                if (stage[qi] != 0) continue;
                const auto& probe_ids = all_probe_ids[qi];
                if (probe_ids.empty()) continue;

                std::vector<int> gpu_cids, cpu_cids;
                if (gpu_coord_) {
                    for (int cid : probe_ids) {
                        if (gpu_coord_->is_gpu_resident(cid))
                            gpu_cids.push_back(cid);
                        else
                            cpu_cids.push_back(cid);
                    }
                } else {
                    cpu_cids = std::vector<int>(probe_ids.begin(), probe_ids.end());
                }
                if (profiling) {
                    p_l2_gpu_clusters += gpu_cids.size();
                    p_l2_cpu_clusters += cpu_cids.size();
                }
                l2_qis.push_back(qi);
                per_q_gpu_cids.push_back(std::move(gpu_cids));
                per_q_cpu_cids.push_back(std::move(cpu_cids));
            }

            // ── Recall-divergence diagnostic (profiling only) ───────────────
            // For each unique GPU-resident cluster probed in this search batch,
            // check whether l2_vector_count > (gpu_cluster_size + buffer_size).
            // A positive delta means vectors that are in L2 but will NOT be
            // searched (because L2 is skipped for GPU-resident clusters).
            // Root causes: overflow-to-L2 fallbacks and promotion races.
            if (profiling && gpu_coord_) {
                std::unordered_set<int> seen_gpu_cids;
                for (const auto& qcids : per_q_gpu_cids)
                    for (int cid : qcids)
                        seen_gpu_cids.insert(cid);

                for (int cid : seen_gpu_cids) {
                    const size_t gpu_n = gpu_coord_->gpu_cluster_size(cid);
                    const size_t buf_n = gpu_coord_->buffer_size(cid);
                    const size_t l2_n  = (static_cast<size_t>(cid) < metadata_.size())
                                         ? metadata_[cid].l2_vector_count : 0;
                    if (l2_n > gpu_n + buf_n) {
                        M3Profiler::instance().log_recall_diag(
                            "SEARCH_DIVERGE", cid,
                            gpu_n, buf_n, l2_n,
                            gpu_coord_->total_overflow_count());
                    }
                }
            }

            const size_t n_l2 = l2_qis.size();
            if (n_l2 > 0) {
                // Pack L2 query vectors contiguously.
                std::vector<float> l2_qvecs(n_l2 * dim_sz);
                for (size_t i = 0; i < n_l2; ++i)
                    std::memcpy(l2_qvecs.data() + i * dim_sz,
                                queries + l2_qis[i] * dim_sz,
                                dim_sz * sizeof(float));

                // Per-query result arrays (GPU and CPU contributions).
                std::vector<std::vector<DocId>> gpu_out_ids(n_l2), cpu_out_ids(n_l2);
                std::vector<std::vector<float>> gpu_out_sc(n_l2),  cpu_out_sc(n_l2);

                // Check whether any query has GPU-resident clusters to search.
                const bool has_gpu_work = gpu_coord_ &&
                    std::any_of(per_q_gpu_cids.begin(), per_q_gpu_cids.end(),
                                [](const std::vector<int>& v){ return !v.empty(); });

                // Launch batched GPU search asynchronously so it overlaps with CPU.
                const auto t_gpu0 = profiling ? clock::now() : clock::time_point{};
                auto gpu_fut = has_gpu_work
                    ? std::async(std::launch::async, [&]() {
                          gpu_coord_->search_batch(per_q_gpu_cids,
                                                   l2_qvecs.data(), n_l2, k_promo,
                                                   gpu_out_ids, gpu_out_sc,
                                                   profiling ? &p_gpu_timing : nullptr);
                      })
                    : std::future<void>{};

                // OMP-parallel CPU L2 scan across all L2 queries simultaneously.
                const auto t_cpu0 = profiling ? clock::now() : clock::time_point{};
                #pragma omp parallel for schedule(dynamic) if(n_l2 > 1)
                for (int ii = 0; ii < static_cast<int>(n_l2); ++ii) {
                    if (per_q_cpu_cids[static_cast<size_t>(ii)].empty()) continue;
                    std::vector<std::vector<DocId>> tmp_ids(1);
                    std::vector<std::vector<float>> tmp_sc(1);
                    l2.index->search_on(per_q_cpu_cids[static_cast<size_t>(ii)],
                                        l2_qvecs.data() + static_cast<size_t>(ii) * dim_sz,
                                        1, k_promo, tmp_ids, tmp_sc);
                    cpu_out_ids[static_cast<size_t>(ii)] = std::move(tmp_ids[0]);
                    cpu_out_sc[static_cast<size_t>(ii)]  = std::move(tmp_sc[0]);
                }
                if (profiling) p_l2_cpu_ms += fms(clock::now() - t_cpu0).count();

                if (gpu_fut.valid()) gpu_fut.get();
                if (profiling) p_l2_gpu_ms += fms(clock::now() - t_gpu0).count();

                // Scatter results back, merge L0+L1+L2 per query.
                const auto t_merge0 = profiling ? clock::now() : clock::time_point{};
                for (size_t i = 0; i < n_l2; ++i) {
                    const size_t qi = l2_qis[i];

                    // Combine GPU and CPU L2 results into l2_ids[0]/l2_scores[0].
                    l2_ids[0] = std::move(gpu_out_ids[i]);
                    l2_ids[0].insert(l2_ids[0].end(),
                                     cpu_out_ids[i].begin(), cpu_out_ids[i].end());
                    l2_scores[0] = std::move(gpu_out_sc[i]);
                    l2_scores[0].insert(l2_scores[0].end(),
                                        cpu_out_sc[i].begin(), cpu_out_sc[i].end());

                    promo_ids[qi]    = l2_ids[0];
                    promo_scores[qi] = l2_scores[0];

                    std::vector<std::vector<DocId>> per_ids    = {all_l0_ids[qi], all_l1_ids[qi], l2_ids[0]};
                    std::vector<std::vector<float>> per_scores = {all_l0_scores[qi], all_l1_scores[qi], l2_scores[0]};
                    merge_levels_(per_ids, per_scores, k, out_ids[qi], out_scores[qi]);
                    stage[qi] = 3;
                }
                if (profiling) p_merge_ms += fms(clock::now() - t_merge0).count();
            }
            if (profiling) p_l2_reach_total_ms = fms(clock::now() - t_l2_start).count();
        }

        // Verbose per-query logging (covers all stages).
        if (m3_verbose) {
            float cur_dagent, cur_thresh, cur_alpha_et;
            {
                std::lock_guard<std::mutex> dlk(dagent_mu_);
                std::lock_guard<std::mutex> alk(alpha_et_mu_);
                cur_dagent   = dagent_;
                cur_alpha_et = alpha_et_dynamic_;
                cur_thresh   = (cur_alpha_et > 0.f && cur_dagent > 0.f)
                                   ? cur_alpha_et * cur_dagent
                                   : cfg_.search_threshold;
            }
            for (size_t qi = 0; qi < q_rows; ++qi) {
                const char* sname = (stage[qi] == 1) ? "L0-only"
                                  : (stage[qi] == 2) ? "L0+L1 (early-exit)"
                                  :                    "L0+L1+L2 (full)";
                float kth = out_scores[qi].empty() ? -1.f
                          : ((int)out_scores[qi].size() >= k
                                 ? out_scores[qi][static_cast<size_t>(k-1)]
                                 : out_scores[qi].back());
                fprintf(stderr,
                        "[M3:search] qi=%zu  stage=%-22s  kth=%.4f  "
                        "thresh=%.4f (αet=%.2f × dagent=%.4f)  results=%zu\n",
                        qi, sname, kth, cur_thresh,
                        cur_alpha_et, cur_dagent,
                        out_ids[qi].size());
            }
        }

        // Promotion loop: shared lock — we only read the Layer structs (l0_, l1_, l2_)
        // and write into the individual index cluster stores (protected by their own locks).
        // Downgrading from exclusive to shared allows concurrent searches to proceed in
        // parallel while promotion runs, eliminating the 3-7s serialisation gap.
        //
        // L0 promotion: per-result-vector (temporal locality — single accessed vector).
        // L1 promotion: per-query (spatial locality — top-k' results form one new cluster).
        const auto t_promo_start = profiling ? clock::now() : clock::time_point{};
        {
            std::shared_lock promo_lk(topo_mu_);
            for (size_t qi = 0; qi < q_rows; ++qi) {
                const int st = (qi < stage.size()) ? stage[qi] : 0;
                int limit = cache_config_.max_promote_per_query;
                if (limit <= 0) limit = static_cast<int>(out_ids[qi].size());
                // L0: per-result-vector promotion (unchanged).
                for (int j = 0; j < limit && j < static_cast<int>(out_ids[qi].size()); ++j) {
                    DocId doc_id = out_ids[qi][j];
                    auto it = doc_id_to_cid_.find(doc_id);
                    if (it == doc_id_to_cid_.end()) continue;
                    // Always update cluster-level access time so demotion logic works.
                    record_access_(it->second);
                }
                // L1: per-query promotion — top-k' results form one new query-centric cluster.
                // Stage 3 (L2-reaching): promote with the wider k_promo L2 results so
                //   L1 gets up to l1_neighborhood_k vectors and L0 gets min(l0_nbk, k).
                // Stage 2 (L1 early-exit): promote with the existing merged results (top-k)
                //   since no L2 search was performed; behavior unchanged from before.
                // Stage 1 (L0 early-exit): no promotion (already hot in L0).
                if (st == 3 && !promo_ids[qi].empty()) {
                    const float* qptr_qi = queries + qi * dim_sz;
                    promote_query_to_l1_(qptr_qi, promo_ids[qi], promo_scores[qi], k);
                } else if (st == 2 && !out_ids[qi].empty()) {
                    const float* qptr_qi = queries + qi * dim_sz;
                    promote_query_to_l1_(qptr_qi, out_ids[qi], out_scores[qi], k);
                }
            }
        }

        // Compute true_kth_avg: mean k-th distance from queries that reached L2.
        // Used to detect dagent drift when alpha_et is tuned aggressively.
        float p_true_kth_avg = -1.f;
        if (profiling) {
            float kth_sum = 0.f; int kth_n = 0;
            for (size_t qi = 0; qi < q_rows; ++qi) {
                if (stage[qi] == 3 && !out_scores[qi].empty()) {
                    const auto& sc = out_scores[qi];
                    kth_sum += ((int)sc.size() >= k) ? sc[static_cast<size_t>(k-1)] : sc.back();
                    ++kth_n;
                }
            }
            if (kth_n > 0) p_true_kth_avg = kth_sum / static_cast<float>(kth_n);
        }
        // Timing for promotion (runs on this thread, included in total_ms).
        const double p_promo_ms = profiling ? fms(clock::now() - t_promo_start).count() : 0.0;
        const double p_total_ms = profiling ? fms(clock::now() - t_search_start).count() : 0.0;
        // Update dagent_ rolling average.
        // cache_level_k: use all queries regardless of stage.
        // true_k: use only full-search (stage==3) queries; early-exit queries are skipped
        //         here — ground truth is injected by the background calibration path instead.
        if (cache_config_.dagent_mode == DagentUpdateMode::cache_level_k) {
            update_dagent_(out_scores, k);
        } else {
            std::vector<std::vector<float>> full_scores;
            full_scores.reserve(q_rows);
            for (size_t qi = 0; qi < q_rows; ++qi)
                if (stage[qi] == 3) full_scores.push_back(out_scores[qi]);
            if (!full_scores.empty())
                update_dagent_(full_scores, k);
        }

        // Background calibration: every calibration_interval ops, when at least one query
        // exited early, dispatch a full L0→L1→L2 search for one sampled query.
        // The result is used to compute r = true_kth / early_kth and adapt alpha_et_dynamic_.
        // In true_k mode the true k-th distance is also fed into dagent_.
        const uint64_t op = search_op_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (cache_config_.calibration_interval > 0
            && (op % cache_config_.calibration_interval) == 0) {

            // Find the first query that had an early exit this batch.
            int qi_sample = -1;
            for (size_t qi = 0; qi < q_rows; ++qi) {
                if (stage[qi] == 1 || stage[qi] == 2) { qi_sample = (int)qi; break; }
            }

            if (qi_sample >= 0) {
                // Capture everything the background thread needs by value.
                const float early_kth_sample = early_exit_kths[qi_sample];
                std::vector<float> qvec(queries + qi_sample * dim_sz,
                                        queries + qi_sample * dim_sz + dim_sz);
                Layer l0_snap = l0, l1_snap = l1, l2_snap = l2;
                const int nprobe_snap = nprobe;
                const int k_snap     = k;
                const bool true_k_mode =
                    (cache_config_.dagent_mode == DagentUpdateMode::true_k);

                (void)std::async(std::launch::async,
                    [this, qvec = std::move(qvec), early_kth_sample,
                     l0_snap = std::move(l0_snap), l1_snap = std::move(l1_snap),
                     l2_snap = std::move(l2_snap),
                     nprobe_snap, k_snap, true_k_mode]() {

                        std::vector<DocId> full_ids;
                        std::vector<float> full_scores;
                        search_one_full_(qvec.data(), k_snap, nprobe_snap,
                                         l0_snap, l1_snap, l2_snap,
                                         full_ids, full_scores);

                        if (full_scores.empty()) return;

                        const float true_kth = ((int)full_scores.size() >= k_snap)
                            ? full_scores[static_cast<size_t>(k_snap - 1)]
                            : full_scores.back();

                        // r = true_kth / early_kth ∈ (0, 1]:
                        //   r ≈ 1 → early exit was accurate
                        //   r << 1 → full search found closer results; early exit too aggressive
                        if (early_kth_sample > 0.f) {
                            const float r = true_kth / early_kth_sample;
                            update_alpha_et_(r);
                        }

                        if (true_k_mode)
                            update_dagent_single_(true_kth);
                    });
            }
        }

        // Combined search + cache stats CSV row.
        if (profiling) {
            int l0_live = 0, l1_live = 0;
            size_t l0_total = 0, l1_total = 0;
            {
                std::shared_lock lk(topo_mu_);
                l0_live = l0_.index ? l0_.index->live_nlist() : 0;
                l1_live = l1_.index ? l1_.index->live_nlist() : 0;
                const int l0_slots = l0_.index ? l0_.index->nlist() : 0;
                const int l1_slots = l1_.index ? l1_.index->nlist() : 0;
                for (int c = 0; c < l0_slots; ++c) l0_total += l0_.index->cluster_live_size(c);
                for (int c = 0; c < l1_slots; ++c) l1_total += l1_.index->cluster_live_size(c);
            }
            float cur_dagent, cur_alpha_et;
            {
                std::lock_guard<std::mutex> dlk(dagent_mu_);
                std::lock_guard<std::mutex> alk(alpha_et_mu_);
                cur_dagent   = dagent_;
                cur_alpha_et = alpha_et_dynamic_;
            }
            // Compute per-stage averages.
            const size_t p_l2_reach = q_rows - p_l0_exits - p_l1_exits;
            const double p_l0_exit_avg = p_l0_exits > 0
                ? p_l0_exit_total_ms / static_cast<double>(p_l0_exits) : 0.0;
            const double p_l1_exit_avg = p_l1_exits > 0
                ? p_l1_exit_total_ms / static_cast<double>(p_l1_exits) : 0.0;
            const double p_l2_reach_avg = p_l2_reach > 0
                ? p_l2_reach_total_ms / static_cast<double>(p_l2_reach) : 0.0;
            const size_t p_promo_queries = p_l1_exits + p_l2_reach;
            const double p_promo_avg = p_promo_queries > 0
                ? p_promo_ms / static_cast<double>(p_promo_queries) : 0.0;

            M3Profiler::instance().log_search_profile(
                q_rows,
                p_probe_sgemm_ms, p_probe_topk_ms,
                p_l0_ms, p_l0_centroid_ms, p_l0_scan_ms,
                p_l1_ms, p_l1_centroid_ms, p_l1_scan_ms,
                p_l2_gpu_ms, p_l2_cpu_ms,
                p_gpu_timing.h2d_ms, p_gpu_timing.kernel_ms,
                p_gpu_timing.sync_d2h_ms, p_gpu_timing.topk_ms,
                p_merge_ms, p_promo_ms, p_total_ms,
                p_l2_reach, p_l2_gpu_clusters, p_l2_cpu_clusters);

            M3Profiler::instance().log_search_stats(
                q_rows,
                p_l0_exits, p_l0_exit_avg, p_l0_exit_total_ms,
                p_l1_exits, p_l1_exit_avg, p_l1_exit_total_ms,
                p_l2_reach_avg, p_l2_reach_total_ms,
                p_promo_avg,
                l0_live, l0_total,
                l1_live, l1_total,
                cur_dagent, cur_alpha_et, p_true_kth_avg);
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

    float search_threshold;
    bool has_threshold;
    {
        std::lock_guard<std::mutex> dlk(dagent_mu_);
        std::lock_guard<std::mutex> alk(alpha_et_mu_);
        if (alpha_et_dynamic_ > 0.f && dagent_ > 0.f) {
            search_threshold = alpha_et_dynamic_ * dagent_;
            has_threshold = true;
        } else {
            search_threshold = cfg_.search_threshold;
            has_threshold = search_threshold < std::numeric_limits<float>::infinity();
        }
    }

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
    update_dagent_(out_scores, k);
}

std::vector<int> MultiLevelIndex::get_l2_probe_ids(const float* query, int nprobe) const {
    std::shared_ptr<IVFIndex> l2_idx;
    {
        std::shared_lock lk(topo_mu_);
        l2_idx = l2_.index;
    }
    std::vector<int> out;
    if (l2_idx)
        l2_idx->get_probe_ids(query, nprobe, out);
    return out;
}

void MultiLevelIndex::search_l2_clusters(const std::vector<int>& cids,
                                          const float* query, int k,
                                          std::vector<DocId>&  out_ids,
                                          std::vector<float>&  out_scores) const {
    if (cids.empty() || !query || k <= 0) return;
    std::shared_ptr<IVFIndex> l2_idx;
    {
        std::shared_lock lk(topo_mu_);
        l2_idx = l2_.index;
    }
    if (!l2_idx) return;

    // search_on returns results per query; we have a single query here.
    std::vector<std::vector<DocId>>  ids(1);
    std::vector<std::vector<float>>  scores(1);
    l2_idx->search_on(cids, query, /*q_rows=*/1, k, ids, scores);

    out_ids.insert(out_ids.end(), ids[0].begin(), ids[0].end());
    out_scores.insert(out_scores.end(), scores[0].begin(), scores[0].end());
}

void MultiLevelIndex::l2_split_cluster(int cid, size_t threshold) {
    // Phase 1: run the expensive k-means split WITHOUT holding topo_mu_.
    // IVFIndex::split_cluster() takes its own exclusive topo lock internally.
    std::shared_ptr<IVFIndex> l2_idx;
    {
        std::shared_lock lk(topo_mu_);
        l2_idx = l2_.index;
    }
    if (!l2_idx) return;
    if (l2_idx->cluster_live_size(cid) <= threshold) return;

    const int new_cid = l2_idx->split_cluster(cid, threshold);
    if (new_cid < 0) return;  // degenerate split or cluster not found

    // Phase 2: update MultiLevelIndex routing tables and metadata under topo_mu_.
    // Lock ordering: topo_mu_(unique) → IVFIndex::topo_mu_(shared) — consistent
    // with all other MultiLevelIndex methods that hold topo_mu_ and then call
    // into l2_.index / l0_.index / l1_.index.
    std::unique_lock lk(topo_mu_);

    const size_t dim_sz = static_cast<size_t>(dim_);

    // a. Sync l2_.centroids: update partition-A's (possibly shifted) centroid,
    //    then append partition-B's new centroid.
    const float* c0_ptr = l2_idx->centroid_ptr(cid);
    const float* c1_ptr = l2_idx->centroid_ptr(new_cid);

    if (c0_ptr && static_cast<size_t>(cid) * dim_sz + dim_sz <= l2_.centroids.size()) {
        std::copy(c0_ptr, c0_ptr + dim_sz,
                  l2_.centroids.begin() + static_cast<size_t>(cid) * dim_sz);
    }
    if (c1_ptr) {
        l2_.centroids.insert(l2_.centroids.end(), c1_ptr, c1_ptr + dim_sz);
    }

    // b. Mirror partition-B centroid into L0/L1 routing tables so future
    //    inserts/searches see the new cluster.
    if (c1_ptr) {
        const std::vector<float> c1(c1_ptr, c1_ptr + dim_sz);
        if (l0_.index) {
            l0_.index->add_cluster(c1);
            l0_.centroids.insert(l0_.centroids.end(), c1.begin(), c1.end());
        }
        if (l1_.index) {
            l1_.index->add_cluster(c1);
            l1_.centroids.insert(l1_.centroids.end(), c1.begin(), c1.end());
        }
    }

    // c. Update doc_id_to_cid_ for vectors that moved to partition B.
    std::vector<DocId> ids_b;
    std::vector<float> vecs_b;
    l2_idx->export_cluster_live(new_cid, ids_b, vecs_b);
    for (DocId id : ids_b)
        doc_id_to_cid_[id] = new_cid;

    // d. Extend and update metadata (same pattern as add_l2_cluster).
    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (static_cast<size_t>(new_cid) >= metadata_.size())
            metadata_.resize(static_cast<size_t>(new_cid) + 1);
        metadata_[static_cast<size_t>(new_cid)].in_l2 = true;
        metadata_[static_cast<size_t>(new_cid)].l2_vector_count = ids_b.size();
        metadata_[static_cast<size_t>(new_cid)].access_count = 0;
        if (static_cast<size_t>(cid) < metadata_.size())
            metadata_[static_cast<size_t>(cid)].l2_vector_count =
                l2_idx->cluster_live_size(cid);
    }
}

void MultiLevelIndex::maintenance_pass() {
    if (m3_verbose) {
        fprintf(stderr, "[M3:maint] ── maintenance_pass() begin ──\n");
    }
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

    // Note: IVF-level split/merge (l?_idx->maintenance_pass()) intentionally omitted.
    // Maintenance is event-driven: only overflow triggers eviction; no periodic compaction.

    if (m3_verbose || M3Logger::instance().is_enabled()) {
        const int l0n = l0_idx ? l0_idx->nlist() : -1;
        const int l1n = l1_idx ? l1_idx->nlist() : -1;
        const int l2n = l2_idx ? l2_idx->nlist() : -1;
        int cen_n = -1;
        {
            std::shared_lock ck(topo_mu_);
            cen_n = l2_.centroids.empty() ? 0
                  : static_cast<int>(l2_.centroids.size() / static_cast<size_t>(dim_));
        }
        M3Logger::instance().log_maint_topology(l0n, l1n, l2n, cen_n);
    }

    if (!cache_mode || meta_snap.empty() || nlist == 0) {
        return;
    }

    // ===== 2) L0 overflow: erase LRU vectors =====
    // L1 now has its own query-centric cluster topology (independent of L2 cluster IDs),
    // so L0 overflow vectors cannot be written to L1 by cluster ID. Since L2 is always
    // canonical, it is safe to simply drop overflow vectors from L0.
    const size_t l0_cap = static_cast<size_t>(cache_config_.l0_max_vectors_per_cluster);
    const size_t dim_sz = static_cast<size_t>(dim_);

    if (l0_idx && l0_cap > 0) {
        const int nlist0 = l0_idx->nlist();
        for (int cid = 0; cid < nlist0; ++cid) {
            const size_t sz = l0_idx->cluster_live_size(cid);
            if (sz <= l0_cap) continue;

            std::vector<DocId> evict;
            l0_idx->get_coldest_doc_ids(cid, sz - l0_cap, evict);
            if (evict.empty()) continue;

            l0_idx->erase_batch(cid, evict.data(), evict.size());

            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] L0 overflow evict  cid=%d  evicted=%zu  "
                        "(L0 was %zu > cap %zu)\n",
                        cid, evict.size(), sz, l0_cap);
            }
        }
    }

    // ===== 3) L1 per-cluster vector cap: safety erase =====
    // L1 clusters are small (≤ k' vectors each) so this rarely triggers, but kept as a guard.
    // GT is always in L2, so plain erase is safe.
    const size_t l1_cap = static_cast<size_t>(cache_config_.l1_max_vectors_per_cluster);
    if (l1_idx && l1_cap > 0) {
        const int nlist1 = l1_idx->nlist();
        for (int cid = 0; cid < nlist1; ++cid) {
            const size_t sz = l1_idx->cluster_live_size(cid);
            if (sz <= l1_cap) continue;
            std::vector<DocId> evict;
            l1_idx->get_coldest_doc_ids(cid, sz - l1_cap, evict);
            if (!evict.empty()) {
                l1_idx->erase_batch(cid, evict.data(), evict.size());
                // Remove evicted ids from the dedup set.
                std::lock_guard<std::mutex> lk(l1_cache_mu_);
                for (DocId id : evict) l1_cached_ids_.erase(id);
            }
        }
    }

    // ===== 5) L0 + L1 cluster-count demotion (LRU, keyed by shared L1 cluster IDs) =====
    // L0 and L1 share the same query-centric cluster ID space. Eviction is driven by
    // l1_cluster_access_time_ for both layers. L0 is a subset of L1, so when an L1
    // cluster is evicted it is also removed from L0.
    const int l0_max_clusters = cache_config_.l0_max_clusters;
    const int l1_max_clusters = cache_config_.l1_max_clusters;
    {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);

        // L0 cluster-count demotion: evict oldest L0 clusters (not necessarily L1).
        if (l0_idx && l0_max_clusters > 0) {
            std::vector<std::pair<uint64_t, int>> by_time;
            by_time.reserve(l1_cluster_access_time_.size());
            for (const auto& [cid, t] : l1_cluster_access_time_) {
                if (l0_idx->cluster_live_size(cid) > 0)
                    by_time.emplace_back(t, cid);
            }
            int excess = static_cast<int>(by_time.size()) - l0_max_clusters;
            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] cluster-count check  L0: active=%d  max=%d  excess=%d\n",
                        (int)by_time.size(), l0_max_clusters, std::max(0, excess));
            }
            if (excess > 0) {
                std::sort(by_time.begin(), by_time.end());
                for (int i = 0; i < excess; ++i) {
                    int cid = by_time[static_cast<size_t>(i)].second;
                    l0_idx->remove_cluster(cid);
                    if (m3_verbose)
                        fprintf(stderr,
                                "[M3:maint] cluster-count demotion  L0 cid=%d removed (LRU)\n", cid);
                }
            }
        }

        // L1 cluster-count demotion: evict oldest L1 clusters, also remove from L0.
        if (l1_idx && l1_max_clusters > 0) {
            while (static_cast<int>(l1_cluster_access_time_.size()) > l1_max_clusters) {
                auto lru_it = std::min_element(
                    l1_cluster_access_time_.begin(), l1_cluster_access_time_.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                int lru_cid = lru_it->first;
                std::vector<DocId> evicted_ids;
                std::vector<float> evicted_vecs;
                l1_idx->export_cluster_live(lru_cid, evicted_ids, evicted_vecs);
                for (DocId id : evicted_ids) { l1_cached_ids_.erase(id); doc_id_to_l1_cid_.erase(id); }
                l1_idx->remove_cluster(lru_cid);
                if (l0_idx) l0_idx->remove_cluster(lru_cid);
                l1_cluster_access_time_.erase(lru_it);
                if (m3_verbose)
                    fprintf(stderr,
                            "[M3:maint] cluster-count demotion  L1+L0 cid=%d removed (LRU)\n", lru_cid);
            }
        }
    }

    // ===== 6) Cold-cluster demotion =====
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const uint64_t cold = cache_config_.cold_time_ns;

    // L0 + L1: cold demotion by l1_cluster_access_time_ (shared cluster ID space).
    // Removing an L1 cluster also removes it from L0.
    if (l1_idx) {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);
        std::vector<int> cold_cids;
        for (const auto& [cid, last_access] : l1_cluster_access_time_) {
            const uint64_t elapsed = (now_ns >= last_access) ? (now_ns - last_access) : 0;
            if (elapsed > cold) cold_cids.push_back(cid);
        }
        for (int cid : cold_cids) {
            std::vector<DocId> evicted_ids;
            std::vector<float> evicted_vecs;
            l1_idx->export_cluster_live(cid, evicted_ids, evicted_vecs);
            for (DocId id : evicted_ids) { l1_cached_ids_.erase(id); doc_id_to_l1_cid_.erase(id); }
            l1_idx->remove_cluster(cid);
            if (l0_idx) l0_idx->remove_cluster(cid);
            l1_cluster_access_time_.erase(cid);
            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] cold-cluster demotion  L1+L0 cid=%d removed\n", cid);
            }
        }
    }

}

MultiLevelIndex::CacheStats MultiLevelIndex::get_cache_stats() const {
    CacheStats s;
    {
        std::shared_lock lk(topo_mu_);
        s.l0_clusters = l0_.index ? l0_.index->live_nlist() : 0;
        s.l1_clusters = l1_.index ? l1_.index->live_nlist() : 0;
        // Iterate all slots (nlist()) for vec/overlap counting: invalid slots
        // return 0 from cluster_live_size and empty from export_cluster_live.
        const int l0_slots = l0_.index ? l0_.index->nlist() : 0;
        const int l1_slots = l1_.index ? l1_.index->nlist() : 0;
        for (int c = 0; c < l0_slots; ++c)
            s.l0_total_vecs += l0_.index->cluster_live_size(c);
        for (int c = 0; c < l1_slots; ++c)
            s.l1_total_vecs += l1_.index->cluster_live_size(c);
        // Overlap: same cluster ID, doc ID present in both layers.
        for (int c = 0; c < l0_slots; ++c) {
            std::vector<DocId> docs; std::vector<float> vecs;
            l0_.index->export_cluster_live(c, docs, vecs);
            for (DocId id : docs)
                if (l1_.index && l1_.index->cluster_get_vector(c, id)) ++s.l0_l1_overlap;
        }
    }
    {
        std::lock_guard<std::mutex> clk(l1_cache_mu_);
        s.l1_dedup_set_size = l1_cached_ids_.size();
    }
    {
        std::lock_guard<std::mutex> dlk(dagent_mu_);
        std::lock_guard<std::mutex> alk(alpha_et_mu_);
        s.dagent = dagent_;
        s.dynamic_threshold = alpha_et_dynamic_ * dagent_;
    }
    return s;
}

void MultiLevelIndex::update_dagent_single_(float kth) const {
    std::lock_guard<std::mutex> dlk(dagent_mu_);
    // alpha_et_mu_ acquired after dagent_mu_ — maintain consistent lock order everywhere.
    std::lock_guard<std::mutex> alk(alpha_et_mu_);
    const int win = cache_config_.dagent_window > 0 ? cache_config_.dagent_window : 20;
    dagent_history_.push_back(kth);
    while ((int)dagent_history_.size() > win)
        dagent_history_.pop_front();
    float sum = 0.f;
    for (float d : dagent_history_) sum += d;
    dagent_ = sum / static_cast<float>(dagent_history_.size());
}

void MultiLevelIndex::update_alpha_et_(float r) const {
    if (cache_config_.alpha_et_adapt_rate <= 0.f) return;
    std::lock_guard<std::mutex> lk(alpha_et_mu_);
    // EMA toward the observed quality ratio r = true_kth / early_kth ∈ (0, 1].
    // r ≈ 1: early exit was accurate → αet nudges up (more aggressive).
    // r << 1: full search found much closer results → αet nudges down (more conservative).
    alpha_et_dynamic_ += cache_config_.alpha_et_adapt_rate * (r - alpha_et_dynamic_);
    // Clamp to [0.3, 1.0] — consistent with paper's αet = 0.6–0.8 with headroom.
    if (alpha_et_dynamic_ < 0.3f) alpha_et_dynamic_ = 0.3f;
    if (alpha_et_dynamic_ > 1.0f) alpha_et_dynamic_ = 1.0f;
    if (m3_verbose) {
        fprintf(stderr,
                "[M3:alpha_et] r=%.4f  alpha_et_dynamic=%.4f\n",
                r, alpha_et_dynamic_);
    }
}

void MultiLevelIndex::search_one_full_(const float* qptr, int k, int nprobe,
                                       const Layer& l0, const Layer& l1, const Layer& l2,
                                       std::vector<DocId>& out_ids,
                                       std::vector<float>& out_scores) const {
    // Full L0→L1→L2 search with no early exit and no side-effects (no dagent_/alpha_et_
    // updates, no promotion, no calibration dispatch). Used by background calibration only.
    std::vector<std::vector<DocId>> ids0(1), ids1(1), ids2(1);
    std::vector<std::vector<float>> sc0(1), sc1(1), sc2(1);

    if (l0.index) {
        const int l0_nlist = l0.index->nlist();
        const int l0_nprobe_eff = (cache_config_.l0_nprobe > 0)
                                      ? cache_config_.l0_nprobe
                                      : std::min(nprobe, l0_nlist);
        if (l0_nprobe_eff > 0)
            l0.index->search_nprobe(qptr, 1, k, l0_nprobe_eff, ids0, sc0);
    }
    if (l1.index) {
        const int l1_nprobe_eff = (cache_config_.l1_nprobe > 0)
                                      ? cache_config_.l1_nprobe
                                      : std::min(nprobe, l1.index->nlist());
        if (l1_nprobe_eff > 0)
            l1.index->search_nprobe(qptr, 1, k, l1_nprobe_eff, ids1, sc1);
    }
    if (l2.index) {
        std::vector<int> probe_ids;
        l2.index->get_probe_ids(qptr, nprobe, probe_ids);
        if (!probe_ids.empty())
            l2.index->search_on(probe_ids, qptr, 1, k, ids2, sc2);
    }

    std::vector<std::vector<DocId>> per_ids  = {ids0[0], ids1[0], ids2[0]};
    std::vector<std::vector<float>> per_sc   = {sc0[0],  sc1[0],  sc2[0]};
    merge_levels_(per_ids, per_sc, k, out_ids, out_scores);
}

void MultiLevelIndex::update_dagent_(const std::vector<std::vector<float>>& out_scores, int k) const {
    std::lock_guard<std::mutex> dlk(dagent_mu_);
    // alpha_et_mu_ acquired after dagent_mu_ — maintain consistent lock order everywhere.
    std::lock_guard<std::mutex> alk(alpha_et_mu_);
    const int win = cache_config_.dagent_window > 0 ? cache_config_.dagent_window : 20;
    for (const auto& s : out_scores) {
        if (s.empty()) continue;
        float kth = ((int)s.size() >= k) ? s[static_cast<size_t>(k - 1)] : s.back();
        dagent_history_.push_back(kth);
        while ((int)dagent_history_.size() > win)
            dagent_history_.pop_front();
    }
    if (!dagent_history_.empty()) {
        float sum = 0.f;
        for (float d : dagent_history_) sum += d;
        dagent_ = sum / static_cast<float>(dagent_history_.size());
    }
    if (m3_verbose) {
        fprintf(stderr,
                "[M3:dagent] updated dagent=%.4f  window=%d/%d  "
                "-> next dynamic threshold = αet(%.2f) × dagent = %.4f\n",
                dagent_, (int)dagent_history_.size(), cache_config_.dagent_window,
                alpha_et_dynamic_,
                alpha_et_dynamic_ * dagent_);
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
    ++m.access_count;
}


void MultiLevelIndex::promote_query_to_l1_(const float* query,
                                            const std::vector<DocId>& result_ids,
                                            const std::vector<float>& /*result_scores*/,
                                            int k_caller) const {
    if (!l1_.index || result_ids.empty() || !query) return;

    const size_t dim_sz = static_cast<size_t>(dim_);
    int l1_k = cache_config_.l1_neighborhood_k;
    if (l1_k <= 0) l1_k = 1;
    const size_t n_to_consider = std::min(static_cast<size_t>(l1_k), result_ids.size());

    // Step 1: identify which top-k' result DocIds are not yet cached in L1.
    std::vector<DocId> to_fetch;
    to_fetch.reserve(n_to_consider);
    {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);
        for (size_t i = 0; i < n_to_consider; ++i) {
            DocId id = result_ids[i];
            if (!l1_cached_ids_.count(id))
                to_fetch.push_back(id);
        }
    }
    if (to_fetch.empty()) return;

    // Step 2: fetch vectors from L2 (no l1_cache_mu_ held — avoids lock inversion).
    std::vector<DocId>  new_ids;
    std::vector<float>  new_vecs;
    new_ids.reserve(to_fetch.size());
    new_vecs.reserve(to_fetch.size() * dim_sz);
    for (DocId id : to_fetch) {
        auto cit = doc_id_to_cid_.find(id);
        if (cit == doc_id_to_cid_.end()) continue;
        const float* v = l2_.index->cluster_get_vector(cit->second, id);
        if (!v) continue;
        new_ids.push_back(id);
        new_vecs.insert(new_vecs.end(), v, v + dim_sz);
    }
    if (new_ids.empty()) return;

    // Step 3: evict + create + register atomically so concurrent promotions cannot
    // each pass the eviction check before any of them has registered their new cluster.
    // (Race: with two separate lock acquisitions, batch_size threads can all sneak past
    //  the "size >= l1_max" guard before any of them updates l1_cluster_access_time_.)
    const int l1_max = cache_config_.l1_max_clusters;
    std::vector<float> query_cent(query, query + dim_sz);
    int l1_cid = -1;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);

        // LRU eviction: remove oldest L1 cluster(s) until we're under capacity.
        if (l1_max > 0) {
            while (static_cast<int>(l1_cluster_access_time_.size()) >= l1_max
                   && !l1_cluster_access_time_.empty()) {
                auto lru_it = std::min_element(
                    l1_cluster_access_time_.begin(), l1_cluster_access_time_.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; });
                int lru_cid = lru_it->first;
                // Remove evicted DocIds from the dedup set.
                std::vector<DocId> evicted_ids;
                std::vector<float> evicted_vecs;
                l1_.index->export_cluster_live(lru_cid, evicted_ids, evicted_vecs);
                for (DocId eid : evicted_ids) { l1_cached_ids_.erase(eid); doc_id_to_l1_cid_.erase(eid); }
                l1_.index->remove_cluster(lru_cid);
                if (l0_.index) l0_.index->remove_cluster(lru_cid);
                l1_cluster_access_time_.erase(lru_it);
            }
        }

        // Register new DocIds in dedup set.
        for (DocId id : new_ids) l1_cached_ids_.insert(id);

        l1_cid = l1_.index->add_cluster(query_cent);
        if (l1_cid >= 0) {
            l1_cluster_access_time_[l1_cid] = now_ns;
            for (DocId id : new_ids) doc_id_to_l1_cid_[id] = l1_cid;
        }
    }
    if (l1_cid < 0) return;

    // Add vector data outside the lock — add_batch uses IVFIndex's internal mutex.
    l1_.index->add_batch(l1_cid, new_ids.data(), new_vecs.data(), new_ids.size());
    l1_.centroids.resize((static_cast<size_t>(l1_cid) + 1) * dim_sz, 0.0f);
    std::copy(query_cent.begin(), query_cent.end(),
              l1_.centroids.begin() + l1_cid * dim_sz);

    // Populate L0 with the top-k'' subset using the same cluster ID.
    // L0 is capped at min(l0_neighborhood_k, k_caller) so it never holds more
    // vectors per cluster than the caller's search k (keeps L0 lean and fast).
    if (l0_.index) {
        int l0_k = cache_config_.l0_neighborhood_k;
        if (l0_k <= 0) l0_k = 1;
        const int l0_cap = (k_caller > 0) ? std::min(l0_k, k_caller) : l0_k;
        const size_t n_l0 = std::min(static_cast<size_t>(l0_cap), new_ids.size());
        l0_.index->ensure_cluster(l1_cid, query_cent);
        l0_.centroids.resize((static_cast<size_t>(l1_cid) + 1) * dim_sz, 0.0f);
        std::copy(query_cent.begin(), query_cent.end(),
                  l0_.centroids.begin() + l1_cid * dim_sz);
        l0_.index->add_batch(l1_cid, new_ids.data(), new_vecs.data(), n_l0,
                             /*allow_missing=*/true);
    }

    if (m3_verbose) {
        fprintf(stderr,
                "[M3:promote] L1 query-cluster cid=%d  l1_vecs=%zu/%d (k')  "
                "l0_vecs=%zu/%d (k'')  total_l1_clusters=%d\n",
                l1_cid, new_ids.size(), l1_k,
                std::min(static_cast<size_t>(cache_config_.l0_neighborhood_k), new_ids.size()),
                cache_config_.l0_neighborhood_k,
                l1_.index->nlist());
    }
}

void MultiLevelIndex::demote_cluster_(int cid) const {
    // Only demotes L0 (L2-cluster-aligned). L1 has its own query-centric topology
    // and is evicted via l1_cluster_access_time_ in run_vector_eviction_per_level_().
    if (cid < 0) return;
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) >= metadata_.size()) return;
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
    const uint64_t cold = cache_config_.cold_time_ns;
    const uint64_t elapsed = (now_ns >= m.last_access_time) ? (now_ns - m.last_access_time) : 0;

    // L0/L1 cold demotion is handled via l1_cluster_access_time_ in run_maintenance_.
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
    // L0/L1 vector counts are no longer tracked in L2-indexed metadata.
}

void MultiLevelIndex::run_cluster_count_demotion_() const {
    std::lock_guard<std::mutex> ml(meta_mu_);
    if (metadata_.empty()) return;
    const int nlist = static_cast<int>(metadata_.size());
    std::vector<std::pair<uint64_t, int>> by_time;
    by_time.reserve(static_cast<size_t>(nlist));
    // L0/L1 cluster-count demotion is handled via l1_cluster_access_time_ in run_maintenance_.
}

// ============================================================
// FSM-aware search — compiled only when M3_WITH_FSM is defined.
// The regular search() method is completely untouched.
// ============================================================
#ifdef M3_WITH_FSM

int MultiLevelIndex::nearest_l2_centroid(const float* query) const {
    std::shared_lock<std::shared_mutex> lk(topo_mu_);
    const auto& cents = l2_.centroids;
    if (cents.empty()) return 0;
    const int dim     = dim_;
    const int nlist   = static_cast<int>(cents.size()) / dim;
    float best_sq = std::numeric_limits<float>::infinity();
    int   best_i  = 0;
    for (int i = 0; i < nlist; ++i) {
        const float* c = cents.data() + static_cast<size_t>(i) * dim;
        float sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float e = query[d] - c[d];
            sq += e * e;
        }
        if (sq < best_sq) { best_sq = sq; best_i = i; }
    }
    return best_i;
}

void MultiLevelIndex::search_fsm(const float* queries, size_t q_rows,
                                  int k, int nprobe,
                                  const fsm::FSMTable*    fsm_table,
                                  fsm::RequestTrajectory* traj,
                                  std::vector<std::vector<DocId>>& out_ids,
                                  std::vector<std::vector<float>>& out_scores) const {
    out_ids.resize(q_rows);
    out_scores.resize(q_rows);

    const size_t dim_sz = static_cast<size_t>(dim_);

    for (size_t qi = 0; qi < q_rows; ++qi) {
        const float* qptr = queries + qi * dim_sz;

        // 1. Predict next clusters from FSM and enqueue GPU prefetch.
        if (fsm_table && traj && traj->length() > 0) {
            auto predicted = fsm_table->match_and_predict(*traj);
            if (gpu_coord_ && !predicted.empty()) {
                for (int cid : predicted)
                    gpu_coord_->enqueue_promote(cid);
            }
        }

        // 2. Standard L0→L1→L2 search (probe ordering unchanged, results identical).
        std::vector<std::vector<DocId>> ids(1);
        std::vector<std::vector<float>> scores(1);
        search(qptr, 1, k, nprobe, ids, scores);

        out_ids[qi]    = std::move(ids[0]);
        out_scores[qi] = std::move(scores[0]);

        // 3. Record winning cluster in trajectory.
        if (traj) {
            int win_cid = nearest_l2_centroid(qptr);
            traj->append_step(win_cid, qptr, dim_);
        }
    }
}

#endif // M3_WITH_FSM

} // namespace m3
