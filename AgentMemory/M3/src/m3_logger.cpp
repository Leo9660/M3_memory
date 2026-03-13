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

} // namespace m3
