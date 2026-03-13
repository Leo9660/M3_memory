#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "base.h"
#include "gpu_insert_buffer.h"

namespace m3 {

// ======================================================================
// GpuClusterIndex
//
// CPU-side simulation of GPU VRAM cluster storage and linear-scan search.
//
// In a production CUDA build the vector data would live in device memory
// (cudaMalloc'd) and search would dispatch a kernel; here every byte
// stays on the CPU so the coordination logic can be tested and validated
// without a physical GPU.
//
// Coordination protocol
// ─────────────────────
// 1. Call store_cluster() to upload data. The returned void* handle can
//    be passed to GpuBudgetManager::register_cluster() as the `ptr`.
//    GpuBudgetManager will call the caller's eviction handler when it
//    needs to free a cluster; the caller must then:
//      a. Call buf.drain(cid) + buf.deactivate_cluster(cid) to prevent
//         data loss from buffered-but-not-yet-flushed vectors.
//      b. Call remove_cluster(cid) here.
//
// 2. Inserts targeting a GPU-resident cluster go through the
//    ClusterInsertBuffer (try_buffer) instead of touching this index
//    directly. An AsyncFlushCoordinator drains the buffer asynchronously
//    and calls store_cluster() / IVFIndex::add_batch() as appropriate.
//
// 3. During search call collaborative_search() which:
//      a. For each probe cluster that is GPU-resident: runs linear-scan
//         search over stored vectors and calls
//         budget_mgr.increment_frequency(cid).
//      b. For every probe cluster (GPU-resident or not): scans the
//         ClusterInsertBuffer for buffered-but-not-yet-flushed vectors.
//      c. Merges both result sets and returns top-k (smallest score).
//
// Thread-safe: all public methods are protected by an internal mutex.
// ======================================================================

class GpuClusterIndex {
public:
    GpuClusterIndex(int dim, Metric metric, bool normalized);

    // ---- Data management ----

    // Upload cluster data (simulates H2D memcpy). Replaces any existing
    // data for `cid`. Returns an opaque handle (pointer to internal float
    // storage) suitable for GpuBudgetManager::register_cluster() `ptr`.
    void* store_cluster(int cid, const DocId* ids, const float* vecs, size_t n);

    // Remove cluster data (simulates cudaFree).
    // Returns false if the cluster was not present.
    bool remove_cluster(int cid);

    bool   has_cluster(int cid)  const;
    size_t cluster_size(int cid) const;  // number of stored vectors
    size_t num_clusters()        const;

    // Export the stored vector data for a cluster (CPU simulation of D2H memcpy).
    // Returns false if the cluster is not resident.
    // Used by split_gpu_cluster() to read GPU-resident data directly, avoiding
    // the L2 export + H2D re-upload round-trip for clusters already in VRAM.
    bool export_cluster(int cid,
                        std::vector<DocId>& out_ids,
                        std::vector<float>& out_vecs) const;

    // ---- Search ----

    // Append vectors to an existing cluster (CPU simulation of GPU→GPU + H2D
    // copy that occurs when the insertion buffer is drained to the GPU).
    // Per the paper, when the insertion buffer is full the GPU cluster is
    // expanded in-place rather than flushed down to L2.
    // Returns the number of vectors appended; 0 if the cluster is not resident.
    size_t expand_cluster(int cid, const DocId* ids, const float* vecs, size_t n);

    // Search one GPU-resident cluster. Appends results to out_*.
    // Returns number of results added (0 if cluster not present/empty).
    size_t search_cluster(int cid,
                          const float* query,
                          int k,
                          std::vector<DocId>&  out_ids,
                          std::vector<float>&  out_scores) const;

    // Collaborative search over a set of probe clusters.
    //
    // For each cid in probe_cids:
    //   • GPU path   : if cluster is resident, run search_cluster().
    //   • Buffer path: call buf.search_buffer(cid) to include vectors
    //                  that are buffered but not yet flushed to L2.
    //
    // Candidates are de-duplicated by DocId (best score kept) then
    // sorted ascending. Returns top-k in out_ids / out_scores.
    size_t collaborative_search(const std::vector<int>& probe_cids,
                                const float* query,
                                int k,
                                const ClusterInsertBuffer& buf,
                                std::vector<DocId>&  out_ids,
                                std::vector<float>&  out_scores) const;

    int    dim()        const { return dim_; }
    Metric metric()     const { return metric_; }
    bool   normalized() const { return normalized_; }

private:
    struct ClusterData {
        std::vector<DocId>  ids;
        std::vector<float>  vecs;  // row-major, stride = dim_
    };

    int    dim_;
    Metric metric_;
    bool   normalized_;

    mutable std::mutex                    mu_;
    std::unordered_map<int, ClusterData>  clusters_;
};

} // namespace m3
