#include "m3_logger.h"

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>

namespace m3 {

// ---- Singleton ----

M3Logger& M3Logger::instance() {
    static M3Logger inst;
    // Auto-enable: if M3_DEBUG is set, open the file specified by M3_DEBUG_LOG
    // (default: /tmp/m3_debug.log).  Runs once at first use.
    static bool auto_init_done = false;
    if (!auto_init_done) {
        auto_init_done = true;
        const char* debug_env = std::getenv("M3_DEBUG");
        if (debug_env) {
            const char* log_path = std::getenv("M3_DEBUG_LOG");
            inst.enable(log_path ? log_path : "/tmp/m3_debug.log");
        }
    }
    return inst;
}

// ---- Lifecycle ----

void M3Logger::enable(const std::string& path) {
    std::lock_guard<std::mutex> lk(mu_);
    if (fp_) { fclose(fp_); fp_ = nullptr; }
    fp_ = fopen(path.c_str(), "w");
    if (!fp_) {
        fprintf(stderr, "[M3Logger] WARN: could not open log file '%s'\n",
                path.c_str());
        return;
    }
    enabled_ = true;
    // Write header.
    fprintf(fp_, "# M3 GPU Coordination Debug Log\n");
    fprintf(fp_, "# Columns: [timestamp]  EVENT  details\n");
    fprintf(fp_, "#\n");
    fflush(fp_);
}

void M3Logger::disable() {
    std::lock_guard<std::mutex> lk(mu_);
    enabled_ = false;
    if (fp_) { fflush(fp_); fclose(fp_); fp_ = nullptr; }
}

bool M3Logger::is_enabled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return enabled_;
}

// ---- Timestamp helper ----

void M3Logger::timestamp_(char* buf, size_t buf_sz) {
    using namespace std::chrono;
    const auto now   = system_clock::now();
    const auto secs  = time_point_cast<seconds>(now);
    const auto ms    = duration_cast<milliseconds>(now - secs).count();
    const time_t tt  = system_clock::to_time_t(secs);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    char base[32];
    strftime(base, sizeof(base), "%Y-%m-%d %H:%M:%S", &tm_buf);
    snprintf(buf, buf_sz, "[%s.%03lld]", base, static_cast<long long>(ms));
}

// ---- Write helper ----

void M3Logger::write_(const char* line) {
    char ts[40];
    timestamp_(ts, sizeof(ts));
    std::lock_guard<std::mutex> lk(mu_);
    if (!enabled_ || !fp_) return;
    fprintf(fp_, "%s  %s\n", ts, line);
    fflush(fp_);
}

// ---- Structured events ----

void M3Logger::log_gpu_memory(size_t used_bytes, size_t budget_bytes,
                                size_t resident_count) {
    if (!enabled_) return;
    char buf[256];
    const double pct = (budget_bytes > 0)
                     ? 100.0 * static_cast<double>(used_bytes)
                               / static_cast<double>(budget_bytes)
                     : 0.0;
    snprintf(buf, sizeof(buf),
             "GPU_MEMORY  used=%zu  budget=%zu  pct=%.1f%%  resident_clusters=%zu",
             used_bytes, budget_bytes, pct, resident_count);
    write_(buf);
}

void M3Logger::log_eviction(int cid, uint64_t access_count, size_t bytes) {
    if (!enabled_) return;
    char buf[128];
    snprintf(buf, sizeof(buf),
             "EVICTION    cid=%-4d  access_count=%-8llu  bytes_freed=%zu",
             cid, static_cast<unsigned long long>(access_count), bytes);
    write_(buf);
}

void M3Logger::log_promotion(int cid, uint64_t access_count, bool auto_promoted) {
    if (!enabled_) return;
    char buf[128];
    snprintf(buf, sizeof(buf),
             "PROMOTION   cid=%-4d  access_count=%-8llu  trigger=%s",
             cid, static_cast<unsigned long long>(access_count),
             auto_promoted ? "AUTO(hotspot_rebalance)" : "MANUAL(promote_to_gpu)");
    write_(buf);
}

void M3Logger::log_insert_overflow(int cid) {
    if (!enabled_) return;
    char buf[80];
    snprintf(buf, sizeof(buf),
             "INSERT_OVERFLOW  cid=%-4d  routed_to=L2  reason=BUFFER_FULL_ASYNC_PENDING",
             cid);
    write_(buf);
}

void M3Logger::log_expansion(int cid, size_t n_vecs, double elapsed_ms) {
    if (!enabled_) return;
    char buf[128];
    snprintf(buf, sizeof(buf),
             "GPU_EXPAND  cid=%-4d  vectors_migrated=%-6zu  elapsed_ms=%.3f",
             cid, n_vecs, elapsed_ms);
    write_(buf);
}

void M3Logger::log_split(int cid, size_t n_vecs, double elapsed_ms, bool gpu_path) {
    if (!enabled_) return;
    char buf[160];
    snprintf(buf, sizeof(buf),
             "GPU_SPLIT   cid=%-4d  vectors=%-6zu  elapsed_ms=%.3f  data_source=%s",
             cid, n_vecs, elapsed_ms,
             gpu_path ? "GPU_VRAM(no_L2_export)" : "L2_EXPORT(re_upload)");
    write_(buf);
}

void M3Logger::log_hotspot_rebalance(size_t clusters_promoted,
                                      size_t candidates_considered) {
    if (!enabled_) return;
    char buf[128];
    snprintf(buf, sizeof(buf),
             "HOTSPOT     candidates_considered=%-4zu  newly_promoted=%zu",
             candidates_considered, clusters_promoted);
    write_(buf);
}

void M3Logger::log(const char* fmt, ...) {
    if (!enabled_) return;
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    write_(buf);
}

// ---- CPU topology change events ----

void M3Logger::log_cpu_split(const char* layer, int cid, int new_cid,
                              size_t n_total, size_t n_a, size_t n_b) {
    if (!enabled_) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "CPU_SPLIT   layer=%-2s  cid=%-4d  new_cid=%-4d  "
             "vecs_before=%-7zu  partition_A=%-7zu  partition_B=%-7zu  "
             "reason=cluster_exceeds_200k",
             layer, cid, new_cid, n_total, n_a, n_b);
    write_(buf);
}

void M3Logger::log_cpu_merge(const char* layer, int cid_dst, int cid_src,
                              size_t n_dst, size_t n_src, const char* reason) {
    if (!enabled_) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "CPU_MERGE   layer=%-2s  cid_dst=%-4d  cid_src=%-4d(INVALIDATED)  "
             "vecs_dst=%-6zu  vecs_src=%-6zu  vecs_after=%-6zu  reason=%s",
             layer, cid_dst, cid_src, n_dst, n_src, n_dst + n_src, reason);
    write_(buf);
}

void M3Logger::log_add_batch_invalid(const char* layer, int cid, size_t clusters_sz,
                                      bool oob, bool not_valid, bool null_ptr) {
    if (!enabled_) return;
    char buf[256];
    snprintf(buf, sizeof(buf),
             "ADD_BATCH_INVALID  layer=%-2s  cid=%-4d  clusters_size=%-4zu  "
             "out_of_bounds=%-5s  valid_false=%-5s  null_ptr=%-5s  "
             "likely_cause=%s",
             layer, cid, clusters_sz,
             oob       ? "YES" : "NO",
             not_valid ? "YES" : "NO",
             null_ptr  ? "YES" : "NO",
             oob       ? "SPLIT_IN_SIBLING_NOT_MIRRORED"
                       : not_valid ? "MERGED_AWAY_IN_THIS_LAYER"
                                   : "UNKNOWN");
    write_(buf);
}

void M3Logger::log_insert_routing(int cid, int l0_nlist, int l2_nlist,
                                   int l2_cen_nlist, bool l2_cid_valid,
                                   bool l0_cid_valid) {
    if (!enabled_) return;
    // Only log when something looks wrong (avoids flooding on 8M inserts).
    const bool mismatch = (l0_nlist != l2_nlist) || !l0_cid_valid || !l2_cid_valid
                          || (l2_nlist != l2_cen_nlist);
    if (!mismatch) return;
    char buf[320];
    snprintf(buf, sizeof(buf),
             "INSERT_ROUTING_WARN  cid=%-4d  "
             "l0_nlist=%-4d  l2_nlist=%-4d  l2_centroids_nlist=%-4d  "
             "l2_cid_valid=%-5s  l0_cid_valid=%-5s  "
             "diagnosis=%s",
             cid, l0_nlist, l2_nlist, l2_cen_nlist,
             l2_cid_valid ? "YES" : "NO",
             l0_cid_valid ? "YES" : "NO",
             (l2_nlist != l0_nlist) ? "L0_L2_NLIST_DIVERGED"
             : (!l0_cid_valid)      ? "CID_MERGED_IN_L0_NOT_L2"
             : (!l2_cid_valid)      ? "CID_MERGED_IN_L2"
                                    : "CEN_TABLE_STALE");
    write_(buf);
}

void M3Logger::log_maint_topology(int l0_nlist, int l1_nlist,
                                   int l2_nlist, int l2_cen_nlist) {
    if (!enabled_) return;
    const bool diverged = (l0_nlist != l2_nlist) || (l2_nlist != l2_cen_nlist);
    char buf[256];
    snprintf(buf, sizeof(buf),
             "MAINT_TOPOLOGY  l0_nlist=%-4d  l1_nlist=%-4d  l2_nlist=%-4d  "
             "l2_centroids_nlist=%-4d  topology_ok=%-3s%s",
             l0_nlist, l1_nlist, l2_nlist, l2_cen_nlist,
             diverged ? "NO" : "YES",
             diverged ? "  *** DIVERGED ***" : "");
    write_(buf);
}

// =========================================================================
// M3Profiler implementation
// =========================================================================

M3Profiler::M3Profiler() {
    const char* env = std::getenv("M3_PROFILE");
    if (!env) return;

    const char* dir = std::getenv("M3_PROFILE_DIR");
    if (!dir) dir = "profile";
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char base[256];
    snprintf(base, sizeof(base), "%s/m3_%04d%02d%02d_%02d%02d%02d",
             dir,
             tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
             tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec);

    char path[280];
    snprintf(path, sizeof(path), "%s_search.csv", base);
    fp_search_ = fopen(path, "w");
    if (!fp_search_) {
        fprintf(stderr, "[M3Profiler] WARN: could not open '%s'\n", path);
        return;
    }
    fprintf(fp_search_,
        "timestamp,batch,"
        "probe_ms,l0_ms,l0_exits,l1_ms,l1_exits,"
        "l2_gpu_cls,l2_cpu_cls,l2_gpu_ms,l2_cpu_ms,"
        "merge_ms,promotion_ms,total_ms,"
        "l0_clusters,l0_vecs,l1_clusters,l1_vecs,"
        "dagent,alpha_et,true_kth_avg\n");
    fflush(fp_search_);

    snprintf(path, sizeof(path), "%s_insert.csv", base);
    fp_insert_ = fopen(path, "w");
    if (!fp_insert_) {
        fprintf(stderr, "[M3Profiler] WARN: could not open '%s'\n", path);
        fclose(fp_search_); fp_search_ = nullptr;
        return;
    }
    fprintf(fp_insert_,
        "timestamp,batch,total_ms,gpu_pending,assign_ms,l0l2_write_ms,gpu_dispatch_ms\n");
    fflush(fp_insert_);

    enabled_ = true;
}

M3Profiler::~M3Profiler() {
    std::lock_guard<std::mutex> lk(mu_);
    if (fp_search_) { fflush(fp_search_); fclose(fp_search_); fp_search_ = nullptr; }
    if (fp_insert_) { fflush(fp_insert_); fclose(fp_insert_); fp_insert_ = nullptr; }
}

M3Profiler& M3Profiler::instance() {
    static M3Profiler inst;
    return inst;
}

bool M3Profiler::is_enabled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return enabled_;
}

void M3Profiler::timestamp_(char* buf, size_t buf_sz) {
    using namespace std::chrono;
    const auto now  = system_clock::now();
    const auto secs = time_point_cast<seconds>(now);
    const auto ms   = duration_cast<milliseconds>(now - secs).count();
    const time_t tt = system_clock::to_time_t(secs);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    char base[32];
    strftime(base, sizeof(base), "%Y-%m-%d %H:%M:%S", &tm_buf);
    snprintf(buf, buf_sz, "%s.%03lld", base, static_cast<long long>(ms));
}

void M3Profiler::write_(const char* line) {
    // Unused — kept to satisfy any future generic use.
    (void)line;
}

void M3Profiler::write_search_(const char* line) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!enabled_ || !fp_search_) return;
    fprintf(fp_search_, "%s\n", line);
    fflush(fp_search_);
}

void M3Profiler::write_insert_(const char* line) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!enabled_ || !fp_insert_) return;
    fprintf(fp_insert_, "%s\n", line);
    fflush(fp_insert_);
}

void M3Profiler::log_search_row(size_t q_rows,
                                 double probe_ms,
                                 double l0_ms,  size_t l0_exits,
                                 double l1_ms,  size_t l1_exits,
                                 size_t l2_gpu_cls, size_t l2_cpu_cls,
                                 double l2_gpu_ms,  double l2_cpu_ms,
                                 double merge_ms, double promotion_ms, double total_ms,
                                 int l0_clusters, size_t l0_vecs,
                                 int l1_clusters, size_t l1_vecs,
                                 float dagent, float alpha_et, float true_kth_avg) {
    if (!enabled_) return;
    char ts[32]; timestamp_(ts, sizeof(ts));
    char kth[16];
    if (true_kth_avg < 0.f) kth[0] = '\0';
    else snprintf(kth, sizeof(kth), "%.6f", static_cast<double>(true_kth_avg));
    char buf[512];
    snprintf(buf, sizeof(buf),
        "%s,%zu,"
        "%.3f,%.3f,%zu,%.3f,%zu,"
        "%zu,%zu,%.3f,%.3f,"
        "%.3f,%.3f,%.3f,"
        "%d,%zu,%d,%zu,"
        "%.6f,%.6f,%s",
        ts, q_rows,
        probe_ms, l0_ms, l0_exits, l1_ms, l1_exits,
        l2_gpu_cls, l2_cpu_cls, l2_gpu_ms, l2_cpu_ms,
        merge_ms, promotion_ms, total_ms,
        l0_clusters, l0_vecs, l1_clusters, l1_vecs,
        static_cast<double>(dagent), static_cast<double>(alpha_et), kth);
    write_search_(buf);
}

void M3Profiler::log_insert_row(size_t n_rows,
                                 size_t gpu_pending,
                                 double assign_ms,
                                 double l0l2_write_ms,
                                 double gpu_dispatch_ms,
                                 double total_ms) {
    if (!enabled_) return;
    char ts[32]; timestamp_(ts, sizeof(ts));
    char buf[256];
    snprintf(buf, sizeof(buf),
        "%s,%zu,%.3f,%zu,%.3f,%.3f,%.3f",
        ts, n_rows, total_ms,
        gpu_pending, assign_ms, l0l2_write_ms, gpu_dispatch_ms);
    write_insert_(buf);
}

} // namespace m3
