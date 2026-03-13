#include "gpu_insert_buffer.h"
#include <algorithm>
#include <limits>

namespace m3 {

ClusterInsertBuffer::ClusterInsertBuffer(int dim, size_t binsert_cap)
    : dim_(dim), binsert_cap_(binsert_cap) {}

void ClusterInsertBuffer::activate_cluster(int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    slots_.emplace(cid, Slot{});   // no-op if already present (emplace semantics)
}

void ClusterInsertBuffer::deactivate_cluster(int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    slots_.erase(cid);
}

bool ClusterInsertBuffer::has_cluster(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    return slots_.count(cid) > 0;
}

BufferResult ClusterInsertBuffer::try_buffer(int cid, DocId id, const float* vec) {
    std::lock_guard<std::mutex> lk(mu_);

    auto it = slots_.find(cid);
    if (it == slots_.end()) return BufferResult::kNotResident;

    Slot& slot = it->second;
    if (slot.ids.size() >= binsert_cap_) return BufferResult::kFull;

    slot.ids.push_back(id);
    const float* vend = vec + static_cast<size_t>(dim_);
    slot.vecs.insert(slot.vecs.end(), vec, vend);
    return BufferResult::kBuffered;
}

size_t ClusterInsertBuffer::size(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slots_.find(cid);
    return (it != slots_.end()) ? it->second.ids.size() : 0;
}

bool ClusterInsertBuffer::is_full(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slots_.find(cid);
    if (it == slots_.end()) return false;
    return it->second.ids.size() >= binsert_cap_;
}

bool ClusterInsertBuffer::drain(int cid,
                                 std::vector<DocId>&  out_ids,
                                 std::vector<float>&  out_vecs) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slots_.find(cid);
    if (it == slots_.end()) return false;

    out_ids  = std::move(it->second.ids);
    out_vecs = std::move(it->second.vecs);
    // Reset slot to empty (keep slot alive — cluster is still GPU-resident)
    it->second.ids.clear();
    it->second.vecs.clear();
    return true;
}

size_t ClusterInsertBuffer::search_buffer(int cid,
                                           const float* query,
                                           int k,
                                           Metric metric,
                                           bool normalized,
                                           std::vector<DocId>&  out_ids,
                                           std::vector<float>&  out_scores) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slots_.find(cid);
    if (it == slots_.end()) return 0;

    const Slot& slot = it->second;
    const size_t n = slot.ids.size();
    if (n == 0) return 0;

    // Linear scan — buffer is ≤ binsert_cap vectors, so this is cheap.
    std::vector<Pair> buf;
    buf.reserve(n);
    const size_t dim_sz = static_cast<size_t>(dim_);
    for (size_t i = 0; i < n; ++i) {
        float score = unified_score(query,
                                    slot.vecs.data() + i * dim_sz,
                                    dim_,
                                    metric,
                                    normalized);
        buf.push_back(Pair{score, slot.ids[i]});
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

bool ClusterInsertBuffer::erase_one(int cid, DocId id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slots_.find(cid);
    if (it == slots_.end()) return false;

    Slot& slot = it->second;
    for (size_t i = 0; i < slot.ids.size(); ++i) {
        if (slot.ids[i] == id) {
            // Swap-erase to avoid shifting the whole vector array.
            const size_t dim_sz = static_cast<size_t>(dim_);
            const size_t last   = slot.ids.size() - 1;
            if (i != last) {
                slot.ids[i] = slot.ids[last];
                std::copy(slot.vecs.data() + last * dim_sz,
                          slot.vecs.data() + (last + 1) * dim_sz,
                          slot.vecs.data() + i * dim_sz);
            }
            slot.ids.pop_back();
            slot.vecs.resize(last * dim_sz);
            return true;
        }
    }
    return false;
}

} // namespace m3
