#include "gpu_cluster_index.h"

#include <unordered_map>
#include <limits>

namespace m3 {

GpuClusterIndex::GpuClusterIndex(int dim, Metric metric, bool normalized)
    : dim_(dim), metric_(metric), normalized_(normalized) {}

void* GpuClusterIndex::store_cluster(int cid, const DocId* ids, const float* vecs, size_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    ClusterData& d = clusters_[cid];
    d.ids.assign(ids, ids + n);
    const size_t dim_sz = static_cast<size_t>(dim_);
    d.vecs.assign(vecs, vecs + n * dim_sz);
    // Return pointer into internal storage as an opaque handle.
    return static_cast<void*>(d.vecs.data());
}

bool GpuClusterIndex::export_cluster(int cid,
                                      std::vector<DocId>& out_ids,
                                      std::vector<float>& out_vecs) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clusters_.find(cid);
    if (it == clusters_.end()) return false;
    out_ids  = it->second.ids;
    out_vecs = it->second.vecs;
    return true;
}

size_t GpuClusterIndex::expand_cluster(int cid, const DocId* ids, const float* vecs, size_t n) {
    if (n == 0 || !ids || !vecs) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clusters_.find(cid);
    if (it == clusters_.end()) return 0;
    ClusterData& d = it->second;
    d.ids.insert(d.ids.end(), ids, ids + n);
    const size_t dim_sz = static_cast<size_t>(dim_);
    d.vecs.insert(d.vecs.end(), vecs, vecs + n * dim_sz);
    return n;
}

bool GpuClusterIndex::remove_cluster(int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.erase(cid) > 0;
}

bool GpuClusterIndex::has_cluster(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.count(cid) > 0;
}

size_t GpuClusterIndex::cluster_size(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clusters_.find(cid);
    return (it != clusters_.end()) ? it->second.ids.size() : 0;
}

size_t GpuClusterIndex::num_clusters() const {
    std::lock_guard<std::mutex> lk(mu_);
    return clusters_.size();
}

size_t GpuClusterIndex::search_cluster(int cid, const float* query, int k,
                                        std::vector<DocId>&  out_ids,
                                        std::vector<float>&  out_scores) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = clusters_.find(cid);
    if (it == clusters_.end()) return 0;

    const ClusterData& d = it->second;
    const size_t n = d.ids.size();
    if (n == 0) return 0;

    const size_t dim_sz = static_cast<size_t>(dim_);
    std::vector<Pair> buf;
    buf.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        float score = unified_score(query, d.vecs.data() + i * dim_sz,
                                    dim_, metric_, normalized_);
        buf.push_back(Pair{score, d.ids[i]});
    }
    topk_smallest(buf, k);

    out_ids.reserve(out_ids.size() + buf.size());
    out_scores.reserve(out_scores.size() + buf.size());
    for (const auto& p : buf) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return buf.size();
}

size_t GpuClusterIndex::collaborative_search(const std::vector<int>& probe_cids,
                                              const float* query,
                                              int k,
                                              const ClusterInsertBuffer& buf,
                                              std::vector<DocId>&  out_ids,
                                              std::vector<float>&  out_scores) const {
    // Collect candidates with de-duplication: keep best (smallest) score per DocId.
    std::unordered_map<DocId, float> best;

    for (int cid : probe_cids) {
        // GPU path: search stored cluster data.
        std::vector<DocId>  gpu_ids;
        std::vector<float>  gpu_scores;
        if (search_cluster(cid, query, k, gpu_ids, gpu_scores) > 0) {
            for (size_t i = 0; i < gpu_ids.size(); ++i) {
                auto [it, inserted] = best.emplace(gpu_ids[i], gpu_scores[i]);
                if (!inserted && gpu_scores[i] < it->second)
                    it->second = gpu_scores[i];
            }
        }

        // CPU buffer path: include vectors buffered since last flush.
        std::vector<DocId>  buf_ids;
        std::vector<float>  buf_scores;
        buf.search_buffer(cid, query, k, metric_, normalized_, buf_ids, buf_scores);
        for (size_t i = 0; i < buf_ids.size(); ++i) {
            auto [it, inserted] = best.emplace(buf_ids[i], buf_scores[i]);
            if (!inserted && buf_scores[i] < it->second)
                it->second = buf_scores[i];
        }
    }

    // Convert to sorted top-k.
    std::vector<Pair> all;
    all.reserve(best.size());
    for (const auto& kv : best)
        all.push_back(Pair{kv.second, kv.first});
    topk_smallest(all, k);

    out_ids.clear();
    out_scores.clear();
    out_ids.reserve(all.size());
    out_scores.reserve(all.size());
    for (const auto& p : all) {
        out_ids.push_back(p.id);
        out_scores.push_back(p.score);
    }
    return all.size();
}

} // namespace m3
