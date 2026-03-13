#pragma once

#include <cstdint>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace m3 {

// Forward declaration to avoid pulling in the full header.
class MultiLevelIndex;

// ======================================================================
// GpuBudgetManager
//
// CPU-side registry for GPU-resident L2 clusters.
// Tracks which clusters are in VRAM and their sizes. Enforces a hard
// memory budget by evicting the least-frequently-used (LFU) cluster
// when a new registration would exceed the cap.
//
// Frequency source
// ────────────────
// LFU decisions read ClusterMetadata::access_count directly from the
// MultiLevelIndex (passed at construction). This is the single source
// of truth for access frequency — there is no parallel counter here.
// If no MultiLevelIndex is supplied (idx == nullptr, used in unit tests
// that exercise budget mechanics in isolation) the eviction policy
// falls back to largest-bytes-first (since all frequencies appear 0).
//
// GPU pointer lifetime
// ─────────────────────
// This class only records void* handles. The caller is responsible for
// cudaMalloc/cudaFree. When register_cluster() populates evicted_out,
// the caller MUST:
//   1. Drain the ClusterInsertBuffer for each evicted cid (flush pending
//      inserts back to L2 IVFIndex) BEFORE calling cudaFree on the ptr.
//   2. Deactivate the ClusterInsertBuffer slot for each evicted cid so
//      future inserts route to L2 instead of the now-dead slot.
//
// Thread-safe: all public methods are protected by an internal mutex.
// ======================================================================

struct EvictedCluster {
    int      cid   = -1;
    void*    ptr   = nullptr;
    size_t   bytes = 0;
    uint64_t freq  = 0;   // access_count snapshot at eviction time (informational)
};

class GpuBudgetManager {
public:
    // idx: optional source for LFU access_counts. Pass the owning
    //      MultiLevelIndex so eviction decisions reflect actual search
    //      frequency. May be nullptr (unit-test mode: evict by bytes).
    explicit GpuBudgetManager(size_t budget_bytes,
                               const MultiLevelIndex* idx = nullptr);

    // Attempt to register a cluster. If the new registration would push
    // total usage over budget, evicts LFU clusters until there is room,
    // returning each evicted entry in `evicted_out`.
    //
    // CALLER RESPONSIBILITY before calling cudaFree on evicted_out[i].ptr:
    //   drain ClusterInsertBuffer for evicted_out[i].cid → flush to L2
    //   deactivate ClusterInsertBuffer slot for evicted_out[i].cid
    //
    // Returns false if even after all evictions the cluster is too large
    // for the budget (caller should not load it).
    bool register_cluster(int cid, void* ptr, size_t bytes,
                          std::vector<EvictedCluster>& evicted_out);

    // Remove a specific cluster from the registry unconditionally.
    // Returns the evicted entry so caller can drain the insertion buffer
    // and cudaFree the pointer. Returns {-1,nullptr,0,0} if not resident.
    EvictedCluster remove_cluster(int cid);

    bool   is_gpu_resident(int cid) const;
    void*  get_ptr(int cid)         const;
    size_t get_bytes(int cid)       const;

    size_t total_bytes_used() const;
    size_t budget_bytes()     const { return budget_bytes_; }
    size_t resident_count()   const;

    // Returns a snapshot of all resident cluster ids.
    std::vector<int> all_resident_cids() const;

private:
    struct Entry {
        void*  ptr   = nullptr;
        size_t bytes = 0;
    };

    // Find and remove the entry with the lowest access_count (LFU).
    // Ties broken by largest bytes (prefer evicting bigger allocations).
    // Caller must already hold mu_.
    EvictedCluster evict_one_lfu_locked_();

    mutable std::mutex                  mu_;
    std::unordered_map<int, Entry>      registry_;
    size_t                              budget_bytes_;
    size_t                              used_bytes_ = 0;
    const MultiLevelIndex*              idx_;        // nullable, not owned
};

} // namespace m3
