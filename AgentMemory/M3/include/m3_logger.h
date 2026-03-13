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

} // namespace m3
