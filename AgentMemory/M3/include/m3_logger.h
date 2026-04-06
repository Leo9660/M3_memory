#pragma once

// =========================================================================
// M3Logger — structured debug logging for GPU coordination events.
//
// Disabled by default. Enable with:
//   M3Logger::instance().enable("m3_debug.log");
//
// Log file format (tab-separated):
//   [YYYY-MM-DD HH:MM:SS.mmm]  EVENT_TYPE  details...
//
// All methods are no-ops when disabled and thread-safe when enabled.
// =========================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace m3 {

class M3Logger {
public:
    static M3Logger& instance();

    // Open `path` for writing (truncates on open) and start logging.
    void enable(const std::string& path);

    // Flush and close the log file.
    void disable();

    bool is_enabled() const;

    // ---- Structured event helpers ----
    // All are no-ops when the logger is disabled.

    // GPU memory snapshot — call at the start of each maintenance_tick().
    //   used_bytes     : bytes currently occupied in GPU budget
    //   budget_bytes   : total GPU budget cap
    //   resident_count : number of clusters currently GPU-resident
    void log_gpu_memory(size_t used_bytes, size_t budget_bytes,
                        size_t resident_count);

    // A cluster was evicted from the GPU cache (LFU eviction).
    //   access_count : frequency at eviction time
    //   bytes        : bytes freed
    void log_eviction(int cid, uint64_t access_count, size_t bytes);

    // A cluster was promoted to the GPU cache.
    //   auto_promoted : true = triggered by hotspot_rebalance_(); false = manual
    void log_promotion(int cid, uint64_t access_count, bool auto_promoted);

    // Insert routed to L2 because the GPU insert buffer was full
    // (async expansion pending on background thread).
    void log_insert_overflow(int cid);

    // Async cluster expansion completed (buffer drained → GPU expand_cluster).
    //   n_vecs      : number of vectors migrated from CPU buffer to GPU
    //   elapsed_ms  : wall-clock time for expand_cluster() call
    void log_expansion(int cid, size_t n_vecs, double elapsed_ms);

    // Cluster split completed via gpu_split_kmeans().
    //   n_vecs      : number of vectors in the cluster before split
    //   elapsed_ms  : wall-clock time for gpu_split_kmeans() call
    //   gpu_path    : true = data read from GPU-resident copy (Fix 2);
    //                 false = data exported from L2 (legacy path)
    void log_split(int cid, size_t n_vecs, double elapsed_ms, bool gpu_path);

    // Hotspot rebalance completed; reports how many clusters were promoted.
    void log_hotspot_rebalance(size_t clusters_promoted,
                                size_t candidates_considered);

    // General-purpose formatted line (use for one-off events).
    void log(const char* fmt, ...);

    // ---- CPU topology change events (IVFIndex split/merge) ----

    // A CPU IVFIndex cluster was split by maintenance_pass().
    //   layer       : "L0", "L1", or "L2"
    //   cid         : cluster that was split (partition A keeps this id)
    //   new_cid     : newly created cluster id (partition B)
    //   n_total     : total live vectors before split
    //   n_a, n_b    : vectors assigned to each partition
    void log_cpu_split(const char* layer, int cid, int new_cid,
                       size_t n_total, size_t n_a, size_t n_b);

    // Two CPU IVFIndex clusters were merged by maintenance_pass().
    //   layer       : "L0", "L1", or "L2"
    //   cid_dst     : cluster that absorbed the other (survives)
    //   cid_src     : cluster that was merged in and invalidated
    //   n_dst, n_src: live vector counts before merge
    //   reason      : why merge triggered ("both_below_threshold")
    void log_cpu_merge(const char* layer, int cid_dst, int cid_src,
                       size_t n_dst, size_t n_src, const char* reason);

    // add_batch() was called with an invalid cluster_id — root-cause signal.
    //   layer       : "L0", "L1", or "L2" (which IVFIndex was targeted)
    //   cid         : the cluster_id that was rejected
    //   clusters_sz : current clusters_.size() in that IVFIndex
    //   oob         : true if cid >= clusters_sz (out-of-bounds, e.g. after split in sibling)
    //   not_valid   : true if valid_[cid] == false (e.g. merged away)
    //   null_ptr    : true if clusters_[cid] is null
    void log_add_batch_invalid(const char* layer, int cid, size_t clusters_sz,
                               bool oob, bool not_valid, bool null_ptr);

    // insert() routing mismatch snapshot — logged just before calling add_batch.
    //   cid         : target cluster chosen by nearest_clusters on l2_
    //   l0_nlist    : l0_.index->clusters_.size() at time of insert
    //   l2_nlist    : l2_.index->clusters_.size() at time of insert
    //   l2_cen_nlist: l2_.centroids.size()/dim (routing table size)
    //   l2_cid_valid: is cid valid in l2_? l0_cid_valid: is it valid in l0_?
    void log_insert_routing(int cid, int l0_nlist, int l2_nlist,
                            int l2_cen_nlist, bool l2_cid_valid,
                            bool l0_cid_valid);

    // Topology snapshot after each maintenance_pass() layer call.
    //   l0_nlist, l1_nlist, l2_nlist : clusters_.size() for each IVFIndex
    //   l2_cen_nlist : l2_.centroids.size()/dim (routing table, should == l2_nlist)
    void log_maint_topology(int l0_nlist, int l1_nlist,
                            int l2_nlist, int l2_cen_nlist);

private:
    M3Logger()  = default;
    ~M3Logger() { disable(); }

    // Write one complete line (with timestamp prefix) to fp_.
    // Caller must NOT hold mu_.
    void write_(const char* line);

    // Format current wall-clock time as "[YYYY-MM-DD HH:MM:SS.mmm]".
    static void timestamp_(char* buf, size_t buf_sz);

    mutable std::mutex mu_;
    FILE*              fp_      = nullptr;
    bool               enabled_ = false;
};

// =========================================================================
// M3Profiler — per-batch timing logs for search and insert.
//
// Disabled by default. Enable with env var M3_PROFILE=1.
// Log file path: M3_PROFILE_LOG (default: m3_profile.log in CWD).
//
// Each log line is one completed batch. Fields are space-separated key=value
// pairs, easy to parse with awk/pandas. Written per batch (not per query)
// to keep volume manageable at high QPS.
//
// Thread-safe: all public methods are safe for concurrent use.
// =========================================================================

class M3Profiler {
public:
    static M3Profiler& instance();

    bool is_enabled() const;

    // ---- Search profile row (CSV) ----
    // One row per search() batch. Captures wall time for each pipeline stage
    // in execution order — use to see which stage dominates as the cache warms.
    //   probe_sgemm_ms  : BLAS sgemm for L2 centroid scoring across all queries
    //   probe_topk_ms   : per-query top-nprobe selection from score matrix
    //   l0_ms           : cumulative L0 search time (centroid + scan)
    //   l0_centroid_ms  : L0 centroid scoring + nprobe selection (sgemm+topk over L0 clusters)
    //   l0_scan_ms      : L0 cluster vector scan (search_into loop)
    //   l1_ms           : cumulative L1 search time (centroid + scan)
    //   l1_centroid_ms  : L1 centroid scoring + nprobe selection
    //   l1_scan_ms      : L1 cluster vector scan
    //   l2_gpu_ms       : wall time for GPU L2 search (async, overlaps cpu)
    //   l2_cpu_ms       : wall time for CPU L2 search (overlaps gpu)
    //   gpu_h2d_ms      : query H2D upload inside collaborative_search
    //   gpu_kernel_ms   : CUDA kernel execution time
    //   gpu_sync_d2h_ms : cudaStreamSynchronize (kernel exec wait) + D2H download
    //   gpu_topk_ms     : CPU top-k merge after D2H
    //   merge_ms        : merge_levels_ de-duplication
    //   promotion_ms    : post-search L0/L1 promotion
    //   total_ms        : end-to-end batch wall time
    void log_search_profile(size_t q_rows,
                            double probe_sgemm_ms, double probe_topk_ms,
                            double l0_ms, double l0_centroid_ms, double l0_scan_ms,
                            double l1_ms, double l1_centroid_ms, double l1_scan_ms,
                            double l2_gpu_ms, double l2_cpu_ms,
                            double gpu_h2d_ms, double gpu_kernel_ms,
                            double gpu_sync_d2h_ms, double gpu_topk_ms,
                            double merge_ms, double promotion_ms, double total_ms);

    // ---- Search stats row (CSV) ----
    // One row per search() batch. Captures per-stage exit distribution and
    // derived metrics — use to answer "where are throughput gains".
    //   l0_exits           : queries satisfied by L0 alone
    //   l0_exit_avg_ms     : avg total query time for L0-exit queries
    //   l0_exit_total_ms   : total wall time on all L0-exit queries
    //   l1_exits           : queries satisfied at L0+L1
    //   l1_exit_avg_ms     : avg total query time for L1-exit queries
    //   l1_exit_total_ms   : total wall time on all L1-exit queries
    //   l2_reach_avg_ms    : avg total query time for L2-reaching queries
    //   l2_reach_total_ms  : total wall time on all L2-reaching queries
    //   promo_avg_ms       : avg promotion time per non-L0-exit query
    //   true_kth_avg       : mean k-th distance of L2 queries; -1 if none reached L2
    void log_search_stats(size_t q_rows,
                          size_t l0_exits,
                          double l0_exit_avg_ms, double l0_exit_total_ms,
                          size_t l1_exits,
                          double l1_exit_avg_ms, double l1_exit_total_ms,
                          double l2_reach_avg_ms, double l2_reach_total_ms,
                          double promo_avg_ms,
                          int l0_clusters, size_t l0_vecs,
                          int l1_clusters, size_t l1_vecs,
                          float dagent, float alpha_et, float true_kth_avg);

    // ---- Insert row (CSV) ----
    // One row per insert() batch.
    //   assign_sgemm_ms : BLAS matrix multiply for centroid assignment
    //   assign_topk_ms  : per-vector top-1 selection from score matrix
    void log_insert_row(size_t n_rows,
                        size_t gpu_pending,
                        double assign_ms,
                        double assign_sgemm_ms,
                        double assign_topk_ms,
                        double l0l2_write_ms,
                        double gpu_dispatch_ms,
                        double total_ms);

private:
    M3Profiler();
    ~M3Profiler();

    void write_search_profile_(const char* line);
    void write_search_stats_(const char* line);
    void write_insert_(const char* line);
    static void timestamp_(char* buf, size_t buf_sz);

    mutable std::mutex mu_;
    FILE*              fp_search_profile_ = nullptr;
    FILE*              fp_search_stats_   = nullptr;
    FILE*              fp_insert_         = nullptr;
    bool               enabled_           = false;

    // Accumulators for end-of-run summary lines.
    double   search_total_ms_sum_  = 0.0;
    size_t   search_batch_count_   = 0;
    double   insert_total_ms_sum_  = 0.0;
    size_t   insert_batch_count_   = 0;
    double   l0_exit_ms_sum_       = 0.0;
    size_t   l0_exit_count_        = 0;
    double   l1_exit_ms_sum_       = 0.0;
    size_t   l1_exit_count_        = 0;
};

} // namespace m3
