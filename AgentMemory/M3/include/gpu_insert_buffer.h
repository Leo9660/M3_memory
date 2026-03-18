#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "base.h"   // DocId, Metric, unified_score

namespace m3 {

// ======================================================================
// ClusterInsertBuffer
//
// Per-cluster CPU-side insertion buffer for GPU-resident L2 clusters.
//
// When a cluster is GPU-resident, new inserts targeting it are staged
// here instead of going directly into the IVFIndex. This avoids the
// expensive synchronous CPU-to-GPU transfer for every single insert.
//
// When the buffer reaches binsert_cap (default 128, configurable) the
// caller is expected to trigger an async buffer flush.
//
// During search, the buffer is scanned in parallel with the GPU kernel
// and results are merged via merge_levels_().
//
// Non-GPU-resident clusters are not tracked here at all; try_buffer()
// returns kNotResident so the caller falls through to the normal
// IVFIndex::add_batch() path.
//
// Thread-safe: all public methods are protected by an internal mutex.
// ======================================================================

enum class BufferResult {
    kBuffered,      // vector was accepted into the buffer
    kNotResident,   // cluster has no buffer slot (not GPU-resident)
    kFull,          // buffer is at cap — flush must be triggered first
};

class ClusterInsertBuffer {
public:
    // dim        : vector dimensionality (must match IVFIndex::dim())
    // binsert_cap: maximum vectors per cluster slot before kFull is returned
    explicit ClusterInsertBuffer(int dim, size_t binsert_cap = 128);

    // ---- Slot lifecycle ----
    // Create a buffer slot for a cluster that has just become GPU-resident.
    // No-op if the slot already exists.
    void activate_cluster(int cid);

    // Remove a cluster's buffer slot. The caller MUST drain the buffer
    // (via drain()) before calling this, or buffered vectors will be lost.
    void deactivate_cluster(int cid);

    bool has_cluster(int cid) const;

    // ---- Insert path ----
    // Attempt to buffer one vector for cluster `cid`.
    //   kNotResident : no slot for `cid`  → caller uses normal L2 add_batch
    //   kFull        : slot is at cap     → caller triggers async flush first
    //   kBuffered    : accepted           → vec is now staged in CPU RAM
    BufferResult try_buffer(int cid, DocId id, const float* vec);

    // ---- Query helpers ----
    size_t size(int cid)    const;   // current buffered count for cluster
    bool   is_full(int cid) const;   // size(cid) >= binsert_cap
    size_t binsert_cap()    const { return binsert_cap_; }
    int    dim()            const { return dim_; }

    // ---- Drain (called before/during async flush) ----
    // Moves all buffered (id, vec) pairs for `cid` into out_ids / out_vecs
    // and clears the slot. Caller owns the data afterwards.
    // Returns false if the cluster has no slot.
    bool drain(int cid, std::vector<DocId>& out_ids, std::vector<float>& out_vecs);

    // ---- Search path (CPU half of collaborative search) ----
    // Linear scan of the buffer for cluster `cid`, returning top-k candidates.
    // Distances are computed with unified_score() using the supplied metric.
    // Results are appended (not cleared) into out_ids / out_scores.
    // Returns 0 if the cluster has no slot or an empty buffer.
    // Linear scan of the insert buffer for cluster `cid`, returning top-k
    // candidates. Called during search to include staged-but-not-yet-flushed
    // vectors alongside the GPU cluster results. Returns 0 if no active slot.
    size_t scan_insert_buffer(int cid,
                              const float* query,
                              int k,
                              Metric metric,
                              bool normalized,
                              std::vector<DocId>&  out_ids,
                              std::vector<float>&  out_scores) const;

    // Remove a single doc_id from a cluster's buffer (used by erase path).
    // Returns true if the id was found and removed.
    bool erase_one(int cid, DocId id);

private:
    struct Slot {
        std::vector<DocId>  ids;
        std::vector<float>  vecs;   // row-major, stride = dim_
    };

    int    dim_;
    size_t binsert_cap_;

    mutable std::mutex                 mu_;
    std::unordered_map<int, Slot>      slots_;
};

} // namespace m3
