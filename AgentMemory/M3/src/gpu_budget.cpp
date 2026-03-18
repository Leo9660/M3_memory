#include "gpu_budget.h"
#include "m3_multi_level.h"

#include <limits>
#include <stdexcept>

namespace m3 {

GpuBudgetManager::GpuBudgetManager(size_t budget_bytes,
                                    const MultiLevelIndex* idx)
    : budget_bytes_(budget_bytes)
    , idx_(idx) {}

bool GpuBudgetManager::register_cluster(int cid, void* ptr, size_t bytes,
                                         std::vector<EvictedCluster>& evicted_out) {
    if (bytes > budget_bytes_) return false;

    std::lock_guard<std::mutex> lk(mu_);

    // If already registered, replace in place (re-upload / expansion scenario).
    auto it = registry_.find(cid);
    if (it != registry_.end()) {
        evicted_out.push_back({cid, it->second.ptr, it->second.bytes,
                                idx_ ? idx_->get_access_count(cid) : 0});
        used_bytes_ -= it->second.bytes;
        it->second.ptr   = ptr;
        it->second.bytes = bytes;
        used_bytes_ += bytes;
        return true;
    }

    // Evict LFU entries until the budget allows this cluster.
    while (used_bytes_ + bytes > budget_bytes_ && !registry_.empty()) {
        evicted_out.push_back(evict_one_lfu_locked_());
    }

    if (used_bytes_ + bytes > budget_bytes_) return false;

    registry_[cid] = Entry{ptr, bytes};
    used_bytes_ += bytes;
    return true;
}

EvictedCluster GpuBudgetManager::remove_cluster(int cid) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = registry_.find(cid);
    if (it == registry_.end()) return {-1, nullptr, 0, 0};
    EvictedCluster ev{cid, it->second.ptr, it->second.bytes,
                      idx_ ? idx_->get_access_count(cid) : 0};
    used_bytes_ -= it->second.bytes;
    registry_.erase(it);
    return ev;
}

bool GpuBudgetManager::is_gpu_resident(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    return registry_.count(cid) > 0;
}

void* GpuBudgetManager::get_ptr(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = registry_.find(cid);
    return (it != registry_.end()) ? it->second.ptr : nullptr;
}

size_t GpuBudgetManager::get_bytes(int cid) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = registry_.find(cid);
    return (it != registry_.end()) ? it->second.bytes : 0;
}

size_t GpuBudgetManager::total_bytes_used() const {
    std::lock_guard<std::mutex> lk(mu_);
    return used_bytes_;
}

size_t GpuBudgetManager::resident_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return registry_.size();
}

void GpuBudgetManager::update_cluster(int cid, void* new_ptr, size_t new_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = registry_.find(cid);
    if (it == registry_.end()) return;
    used_bytes_ -= it->second.bytes;
    it->second.ptr   = new_ptr;
    it->second.bytes = new_bytes;
    used_bytes_ += new_bytes;
}

std::vector<int> GpuBudgetManager::all_resident_cids() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int> out;
    out.reserve(registry_.size());
    for (const auto& [cid, _] : registry_) out.push_back(cid);
    return out;
}

EvictedCluster GpuBudgetManager::evict_one_lfu_locked_() {
    // Caller holds mu_. Read access_count from MultiLevelIndex (single source
    // of truth for frequency). If idx_ is null (unit-test mode), all counts
    // appear 0 and ties are broken by largest bytes.
    int      best_cid   = -1;
    uint64_t best_freq  = std::numeric_limits<uint64_t>::max();
    size_t   best_bytes = 0;

    for (const auto& [cid, entry] : registry_) {
        const uint64_t freq = idx_ ? idx_->get_access_count(cid) : 0;
        if (freq < best_freq ||
            (freq == best_freq && entry.bytes > best_bytes)) {
            best_cid   = cid;
            best_freq  = freq;
            best_bytes = entry.bytes;
        }
    }

    if (best_cid < 0) return {-1, nullptr, 0, 0};

    auto it = registry_.find(best_cid);
    EvictedCluster ev{best_cid, it->second.ptr, it->second.bytes, best_freq};
    used_bytes_ -= it->second.bytes;
    registry_.erase(it);
    return ev;
}

} // namespace m3
