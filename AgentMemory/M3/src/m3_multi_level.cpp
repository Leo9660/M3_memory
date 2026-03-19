#include "m3_multi_level.h"
#include "gpu_coordinator.h"
#include "m3_logger.h"
#include <chrono>
#include <future>
#include <mutex>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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
        ensure_layer_centroids_(l0_, centroids);
        // L1 uses its own query-centric cluster topology (not inherited from L2).
        // Just ensure the index object exists with 0 pre-existing clusters.
        if (!l1_.index) {
            l1_.index = std::make_shared<IVFIndex>(dim_, metric_, normalized_, l1_.name);
            // l1_.centroids intentionally left empty — clusters added dynamically via add_cluster().
        }
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
        std::unordered_map<int, size_t> l0_delta, l2_delta;

        // Stage 2: L0 + L2 writes for non-GPU-resident vectors.
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
                // Not GPU-resident: write to L0 (recent access) and L2 (ground truth).
                if (l0_.index) {
                    if (m3_verbose || M3Logger::instance().is_enabled()) {
                        const int l0n = l0_.index->nlist();
                        const int l2n = l2_.index ? l2_.index->nlist() : -1;
                        const int cenn = static_cast<int>(
                            l2_.centroids.size() / static_cast<size_t>(dim_sz));
                        const bool l2v = (cid >= 0 && cid < l2n);
                        const bool l0v = (cid >= 0 && cid < l0n);
                        M3Logger::instance().log_insert_routing(
                            cid, l0n, l2n, cenn, l2v, l0v);
                    }
                    // allow_missing=true: if this cluster was cold-demoted from L0 since
                    // routing, silently skip — L2 write below is the ground truth.
                    l0_.index->add_batch(cid, &ids[i], vecs + i * dim_sz, 1, /*allow_missing=*/true);
                    ++l0_delta[cid];
                }
                if (l2_.index) {
                    l2_.index->add_batch(cid, &ids[i], vecs + i * dim_sz, 1);
                    ++l2_delta[cid];
                }
            }
        }
        const double p_l0l2_ms = profiling ? fms(clock::now() - t_l0l2_0).count() : 0.0;

        // Apply deltas to metadata under meta_mu_ (same ordering as record_access_: topo_mu_ → meta_mu_).
        if (!l0_delta.empty() || !l2_delta.empty()) {
            std::lock_guard<std::mutex> ml(meta_mu_);
            for (auto& [cid, d] : l0_delta)
                if (static_cast<size_t>(cid) < metadata_.size())
                    metadata_[cid].l0_vector_count += d;
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
            M3Profiler::instance().log_insert_batch(
                n_rows, gpu_pending.size(),
                p_assign_ms, p_l0l2_ms, p_gpu_ms, total_ms);
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
        // Dynamic early-termination threshold: αet · dagent (rolling mean of recent top-k distances).
        // Falls back to the static cfg_.search_threshold when dagent is not yet populated or
        // alpha_et is disabled (set to 0).
        float search_threshold;
        bool has_threshold;
        {
            std::lock_guard<std::mutex> dlk(dagent_mu_);
            if (cache_config_.alpha_et > 0.f && dagent_ > 0.f) {
                search_threshold = cache_config_.alpha_et * dagent_;
                has_threshold = true;
            } else {
                search_threshold = cfg_.search_threshold;
                has_threshold = search_threshold < std::numeric_limits<float>::infinity();
            }
        }

        std::vector<int> probe_ids;
        std::vector<std::vector<DocId>> l0_ids(1), l1_ids(1), l2_ids(1);
        std::vector<std::vector<float>> l0_scores(1), l1_scores(1), l2_scores(1);
        // 0 = no results, 1 = satisfied by L0 only, 2 = satisfied after L1, 3 = needed L2
        std::vector<int> stage(q_rows, 0);

        // Profile accumulators (only populated when profiling is enabled).
        double p_probe_ms = 0, p_l0_ms = 0, p_l1_ms = 0;
        double p_l2_gpu_ms = 0, p_l2_cpu_ms = 0, p_merge_ms = 0;
        size_t p_l0_exits = 0, p_l1_exits = 0;
        size_t p_l2_gpu_clusters = 0, p_l2_cpu_clusters = 0;

        auto kth_score_vec = [&](const std::vector<float>& s) -> float {
            if (s.empty()) return std::numeric_limits<float>::infinity();
            if ((int)s.size() >= k) return s[static_cast<size_t>(k - 1)];
            return s.back();
        };

        for (size_t qi = 0; qi < q_rows; ++qi) {
            const float* qptr = queries + qi * dim_sz;
            probe_ids.clear();

            {
                const auto t0 = profiling ? clock::now() : clock::time_point{};
                if (l2.index)
                    l2.index->get_probe_ids(qptr, nprobe, probe_ids);
                if (profiling) p_probe_ms += fms(clock::now() - t0).count();
            }
            if (probe_ids.empty()) continue;

            // Stage 1: search L0 only.
            l0_ids[0].clear(); l0_scores[0].clear();
            l1_ids[0].clear(); l1_scores[0].clear();
            l2_ids[0].clear(); l2_scores[0].clear();

            {
                const auto t0 = profiling ? clock::now() : clock::time_point{};
                if (l0.index)
                    l0.index->search_on(probe_ids, qptr, 1, k, l0_ids, l0_scores);
                if (profiling) p_l0_ms += fms(clock::now() - t0).count();
            }

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
                    if (profiling) ++p_l0_exits;
                }
            }

            // Stage 2: include L1 if needed.
            // L1 now has its own query-centric cluster topology, so we use search_nprobe
            // over L1's own cluster centroids rather than the L2 probe_ids.
            // l1_nprobe=0 (default) → linear scan all L1 clusters (safe for small L1).
            if (!satisfied && l1.index) {
                {
                    const auto t0 = profiling ? clock::now() : clock::time_point{};
                    const int l1_nprobe = (cache_config_.l1_nprobe > 0)
                                              ? cache_config_.l1_nprobe
                                              : l1.index->nlist();  // 0 → scan all
                    if (l1_nprobe > 0)
                        l1.index->search_nprobe(qptr, 1, k, l1_nprobe, l1_ids, l1_scores);
                    if (profiling) p_l1_ms += fms(clock::now() - t0).count();
                }
                {
                    const auto t0 = profiling ? clock::now() : clock::time_point{};
                    std::vector<std::vector<DocId>> per_ids = {l0_ids[0], l1_ids[0]};
                    std::vector<std::vector<float>> per_scores = {l0_scores[0], l1_scores[0]};
                    merge_levels_(per_ids, per_scores, k, merged_ids, merged_scores);
                    if (profiling) p_merge_ms += fms(clock::now() - t0).count();
                }

                if (has_threshold && !merged_scores.empty()) {
                    float ks = kth_score_vec(merged_scores);
                    if (ks <= search_threshold) {
                        satisfied = true;
                        stage[qi] = 2;
                        if (profiling) ++p_l1_exits;
                    }
                }
            }

            // Stage 3: include L2 if still not satisfied or if there is no threshold.
            if (!satisfied) {
                if (l2.index) {
                    if (gpu_coord_) {
                        // Partition probe clusters: GPU-resident → GPU kernel + buffer scan,
                        // non-resident → CPU L2 linear scan.
                        std::vector<int> gpu_cids, cpu_cids;
                        for (int cid : probe_ids) {
                            if (gpu_coord_->is_gpu_resident(cid))
                                gpu_cids.push_back(cid);
                            else
                                cpu_cids.push_back(cid);
                        }
                        if (profiling) {
                            p_l2_gpu_clusters += gpu_cids.size();
                            p_l2_cpu_clusters += cpu_cids.size();
                        }
                        // GPU and CPU L2 searches run in parallel:
                        //   - GPU search is dispatched to a background thread so the
                        //     CUDA kernel and D2H transfer overlap with the CPU scan.
                        //   - CPU scan runs on this thread while the GPU thread is active.
                        //   - Results are joined before the merge step below.
                        std::vector<DocId> gpu_l2_ids;
                        std::vector<float> gpu_l2_scores;
                        const auto t_gpu0 = profiling ? clock::now() : clock::time_point{};
                        auto gpu_fut = !gpu_cids.empty()
                            ? std::async(std::launch::async, [&]() {
                                  gpu_coord_->search(gpu_cids, qptr, k,
                                                     gpu_l2_ids, gpu_l2_scores);
                              })
                            : std::future<void>{};

                        // CPU L2 path runs on this thread while GPU thread is active.
                        const auto t_cpu0 = profiling ? clock::now() : clock::time_point{};
                        if (!cpu_cids.empty())
                            l2.index->search_on(cpu_cids, qptr, 1, k,
                                                l2_ids, l2_scores);
                        if (profiling) p_l2_cpu_ms += fms(clock::now() - t_cpu0).count();

                        // Join GPU thread before merge.
                        if (gpu_fut.valid()) gpu_fut.get();
                        if (profiling) p_l2_gpu_ms += fms(clock::now() - t_gpu0).count();
                        // Fold GPU results into l2 result vectors for unified merge.
                        l2_ids[0].insert(l2_ids[0].end(),
                                         gpu_l2_ids.begin(), gpu_l2_ids.end());
                        l2_scores[0].insert(l2_scores[0].end(),
                                            gpu_l2_scores.begin(), gpu_l2_scores.end());
                    } else {
                        if (profiling) p_l2_cpu_clusters += probe_ids.size();
                        const auto t0 = profiling ? clock::now() : clock::time_point{};
                        l2.index->search_on(probe_ids, qptr, 1, k, l2_ids, l2_scores);
                        if (profiling) p_l2_cpu_ms += fms(clock::now() - t0).count();
                    }
                    {
                        const auto t0 = profiling ? clock::now() : clock::time_point{};
                        std::vector<std::vector<DocId>> per_ids = {l0_ids[0], l1_ids[0], l2_ids[0]};
                        std::vector<std::vector<float>> per_scores = {l0_scores[0], l1_scores[0], l2_scores[0]};
                        merge_levels_(per_ids, per_scores, k, merged_ids, merged_scores);
                        if (profiling) p_merge_ms += fms(clock::now() - t0).count();
                    }
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

            if (m3_verbose) {
                const char* sname = (stage[qi] == 1) ? "L0-only"
                                  : (stage[qi] == 2) ? "L0+L1 (early-exit)"
                                  :                    "L0+L1+L2 (full)";
                float kth = out_scores[qi].empty() ? -1.f
                          : ((int)out_scores[qi].size() >= k ? out_scores[qi][static_cast<size_t>(k-1)]
                                                             : out_scores[qi].back());
                float cur_dagent, cur_thresh;
                {
                    std::lock_guard<std::mutex> dlk(dagent_mu_);
                    cur_dagent = dagent_;
                    cur_thresh = (cache_config_.alpha_et > 0.f && dagent_ > 0.f)
                                     ? cache_config_.alpha_et * dagent_
                                     : cfg_.search_threshold;
                }
                fprintf(stderr,
                        "[M3:search] qi=%zu  stage=%-22s  kth=%.4f  "
                        "thresh=%.4f (αet=%.2f × dagent=%.4f)  results=%zu\n",
                        qi, sname, kth, cur_thresh,
                        cache_config_.alpha_et, cur_dagent,
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
                    // If satisfied by L0 alone (stage 1), skip promotion (already in hottest tier).
                    if (st == 1) continue;
                    promote_vector_neighborhood_(doc_id);  // L0 only
                }
                // L1: per-query promotion — top-k' results form one new query-centric cluster.
                if (st != 1 && !out_ids[qi].empty()) {
                    const float* qptr_qi = queries + qi * dim_sz;
                    promote_query_to_l1_(qptr_qi, out_ids[qi], out_scores[qi]);
                }
            }
        }

        // Emit one profile line per batch — includes promotion since it runs on this thread.
        if (profiling) {
            const double p_promo_ms = fms(clock::now() - t_promo_start).count();
            const double total_ms   = fms(clock::now() - t_search_start).count();
            M3Profiler::instance().log_search_batch(
                q_rows,
                p_probe_ms, p_l0_ms, p_l0_exits,
                p_l1_ms,    p_l1_exits,
                p_l2_gpu_clusters, p_l2_cpu_clusters,
                p_l2_gpu_ms, p_l2_cpu_ms,
                p_merge_ms, p_promo_ms, total_ms);
        }
        // Update dagent rolling average with this batch so the next call can use αet·dagent.
        update_dagent_(out_scores, k);
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
        if (cache_config_.alpha_et > 0.f && dagent_ > 0.f) {
            search_threshold = cache_config_.alpha_et * dagent_;
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

    // ===== 4) Recompute live vector counts for L0 (L1 has its own topology) =====
    std::vector<size_t> l0_counts(nlist, 0);
    if (l0_idx) {
        const int n0 = l0_idx->nlist();
        const int limit = std::min<int>(static_cast<int>(nlist), n0);
        for (int cid = 0; cid < limit; ++cid) {
            l0_counts[static_cast<size_t>(cid)] = l0_idx->cluster_live_size(cid);
        }
    }

    // ===== 5) L0 cluster-count demotion (LRU, based on metadata) =====
    std::vector<uint8_t> removed_l0(nlist, 0);
    const int l0_max_clusters = cache_config_.l0_max_clusters;
    const int l1_max_clusters = cache_config_.l1_max_clusters;

    if (l0_idx && l0_max_clusters > 0) {
        std::vector<std::pair<uint64_t, int>> by_time;
        by_time.reserve(nlist);
        for (size_t cid = 0; cid < nlist; ++cid) {
            if (l0_counts[cid] > 0)
                by_time.emplace_back(meta_snap[cid].last_access_time, static_cast<int>(cid));
        }
        int excess = static_cast<int>(by_time.size()) - l0_max_clusters;
        if (m3_verbose) {
            fprintf(stderr,
                    "[M3:maint] cluster-count check  L0: active_clusters=%d  max=%d  "
                    "excess_to_demote=%d\n",
                    (int)by_time.size(), l0_max_clusters, std::max(0, excess));
        }
        if (excess > 0) {
            std::sort(by_time.begin(), by_time.end());
            for (int i = 0; i < excess; ++i) {
                int cid = by_time[static_cast<size_t>(i)].second;
                l0_idx->remove_cluster(cid);
                removed_l0[static_cast<size_t>(cid)] = 1;
                l0_counts[static_cast<size_t>(cid)] = 0;
                if (m3_verbose) {
                    fprintf(stderr,
                            "[M3:maint] cluster-count demotion  L0 cid=%d removed (LRU)\n", cid);
                }
            }
        }
    }

    // ===== 5b) L1 cluster-count demotion (LRU, based on l1_cluster_access_time_) =====
    // L1 now has its own query-centric cluster topology; evict by LRU access time.
    if (l1_idx && l1_max_clusters > 0) {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);
        while (static_cast<int>(l1_cluster_access_time_.size()) > l1_max_clusters) {
            auto lru_it = std::min_element(
                l1_cluster_access_time_.begin(), l1_cluster_access_time_.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            int lru_cid = lru_it->first;
            std::vector<DocId> evicted_ids;
            std::vector<float> evicted_vecs;
            l1_idx->export_cluster_live(lru_cid, evicted_ids, evicted_vecs);
            for (DocId id : evicted_ids) l1_cached_ids_.erase(id);
            l1_idx->remove_cluster(lru_cid);
            l1_cluster_access_time_.erase(lru_it);
            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] cluster-count demotion  L1 cid=%d removed (LRU)\n", lru_cid);
            }
        }
    }

    // ===== 6) Cold-cluster demotion =====
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    const uint64_t cold = cache_config_.cold_time_ns;

    // L0: cold demotion by L2-cluster-aligned metadata.
    for (size_t cid = 0; cid < nlist; ++cid) {
        const ClusterMetadata& m = meta_snap[cid];
        const uint64_t elapsed = (now_ns >= m.last_access_time)
                                     ? (now_ns - m.last_access_time) : 0;
        if (elapsed <= cold) continue;
        if (l0_idx && l0_counts[cid] > 0) {
            l0_idx->remove_cluster(static_cast<int>(cid));
            removed_l0[cid] = 1;
            l0_counts[cid] = 0;
            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] cold-cluster demotion  L0 cid=%zu  "
                        "(elapsed=%.3fs > cold_threshold=%.3fs)\n",
                        cid, static_cast<double>(elapsed) / 1e9,
                        static_cast<double>(cold) / 1e9);
            }
        }
    }

    // L1: cold demotion by l1_cluster_access_time_ (independent of L2 cluster IDs).
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
            for (DocId id : evicted_ids) l1_cached_ids_.erase(id);
            l1_idx->remove_cluster(cid);
            l1_cluster_access_time_.erase(cid);
            if (m3_verbose) {
                fprintf(stderr,
                        "[M3:maint] cold-cluster demotion  L1 cid=%d removed\n", cid);
            }
        }
    }

    // ===== 7) Apply L0 metadata updates under meta_mu_ =====
    {
        std::lock_guard<std::mutex> ml(meta_mu_);
        if (metadata_.size() != nlist) return; // topology changed concurrently
        for (size_t cid = 0; cid < nlist; ++cid) {
            ClusterMetadata& m = metadata_[cid];
            if (removed_l0[cid]) {
                m.in_l0 = false;
                m.l0_vector_count = 0;
            } else {
                m.l0_vector_count = l0_counts[cid];
            }
            // in_l1 / l1_vector_count are no longer tracked per-L2-cluster
            // since L1 has its own query-centric topology.
        }
    }
}

void MultiLevelIndex::update_dagent_(const std::vector<std::vector<float>>& out_scores, int k) const {
    std::lock_guard<std::mutex> dlk(dagent_mu_);
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
                cache_config_.alpha_et,
                cache_config_.alpha_et * dagent_);
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

void MultiLevelIndex::promote_vector_neighborhood_(DocId doc_id) const {
    // L0 only: insert the directly accessed vector (temporal locality).
    // L1 is populated per-query by promote_query_to_l1_(), not per-result-vector.
    auto it = doc_id_to_cid_.find(doc_id);
    if (it == doc_id_to_cid_.end()) return;
    const int cid = it->second;
    if (!l2_.index || static_cast<size_t>(cid) >= metadata_.size()) return;

    const float* vec = l2_.index->cluster_get_vector(cid, doc_id);
    if (!vec) return;

    const size_t dim_sz = static_cast<size_t>(dim_);
    const size_t nlist = l2_.centroids.size() / dim_sz;
    if (static_cast<size_t>(cid) >= nlist) return;

    std::vector<float> cent(l2_.centroids.begin() + cid * dim_sz,
                            l2_.centroids.begin() + (cid + 1) * dim_sz);

    if (l0_.index) {
        l0_.index->ensure_cluster(cid, cent);
        l0_.index->update_batch(cid, &doc_id, vec, 1, /*insert_if_absent=*/true);
    }

    if (m3_verbose) {
        fprintf(stderr,
                "[M3:promote] doc_id=%ld  cid=%d  L0 <- 1 vec\n",
                (long)doc_id, cid);
    }

    std::lock_guard<std::mutex> ml(meta_mu_);
    if (static_cast<size_t>(cid) < metadata_.size()) {
        ClusterMetadata& m = metadata_[static_cast<size_t>(cid)];
        m.last_access_time = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        if (l0_.index) { m.in_l0 = true; m.l0_vector_count = l0_.index->cluster_live_size(cid); }
    }
}

void MultiLevelIndex::promote_query_to_l1_(const float* query,
                                            const std::vector<DocId>& result_ids,
                                            const std::vector<float>& /*result_scores*/) const {
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

    // Step 3: evict LRU L1 cluster(s) if at capacity, then create new query-centric cluster.
    const int l1_max = cache_config_.l1_max_clusters;
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
                for (DocId eid : evicted_ids) l1_cached_ids_.erase(eid);
                l1_.index->remove_cluster(lru_cid);
                l1_cluster_access_time_.erase(lru_it);
                if (m3_verbose) {
                    fprintf(stderr,
                            "[M3:promote] L1 LRU evict cluster cid=%d\n", lru_cid);
                }
            }
        }

        // Register new DocIds in dedup set.
        for (DocId id : new_ids) l1_cached_ids_.insert(id);
    }

    // Create new L1 cluster with centroid = query vector.
    std::vector<float> query_cent(query, query + dim_sz);
    int l1_cid = l1_.index->add_cluster(query_cent);
    if (l1_cid < 0) return;
    l1_.index->add_batch(l1_cid, new_ids.data(), new_vecs.data(), new_ids.size());

    // Record access time for future LRU eviction.
    const uint64_t now_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    {
        std::lock_guard<std::mutex> lk(l1_cache_mu_);
        l1_cluster_access_time_[l1_cid] = now_ns;
    }

    if (m3_verbose) {
        fprintf(stderr,
                "[M3:promote] L1 query-cluster cid=%d  cached=%zu/%d (k')  "
                "total_l1_clusters=%d\n",
                l1_cid, new_ids.size(), l1_k, l1_.index->nlist());
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

    if (m.in_l0 && elapsed > cold && l0_.index) {
        l0_.index->remove_cluster(cid);
        m.in_l0 = false;
        m.l0_vector_count = 0;
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
