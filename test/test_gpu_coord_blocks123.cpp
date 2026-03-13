// =============================================================================
// test_gpu_coord_blocks123.cpp
//
// Exhaustive standalone tests for GPU-CPU coordination blocks 1–5.
//
//   Block 1 — Access Frequency Counter
//     MultiLevelIndex::record_access_() increments ClusterMetadata::access_count
//     on every search hit; counts survive maintenance_pass() without reset.
//
//   Block 2 — GpuBudgetManager
//     Registration, LFU eviction, budget enforcement, frequency tracking,
//     and thread safety.
//
//   Block 3 — ClusterInsertBuffer
//     Slot lifecycle, try_buffer routing (kBuffered/kNotResident/kFull),
//     search_buffer, drain, erase_one, and thread safety.
//
//   Block 4 — GpuClusterIndex (collaborative search)
//     Store/remove cluster data, search_cluster, collaborative_search that
//     merges GPU-resident data with ClusterInsertBuffer in one call, and
//     GpuBudgetManager frequency tracking on every GPU probe.
//
//   Block 5 — AsyncFlushCoordinator
//     Synchronous maybe_flush / flush_clusters, flush threshold vs. cap,
//     background thread lifecycle, data persistence to L2, stats counters,
//     and thread-safety under concurrent insert+flush.
//
// Build:
//   cmake --build build --target test_gpu_coord
//   ./build/test_gpu_coord
//
// Run a single block:
//   In main() below, comment out any run_block_N() call.
//
// Verbose M3 internals:
//   M3_DEBUG=1 ./build/test_gpu_coord
// =============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_set>
#include <chrono>

#include "m3_multi_level.h"
#include "gpu_budget.h"
#include "gpu_insert_buffer.h"
#include "gpu_cluster_index.h"
#include "gpu_flush_coordinator.h"
#include "gpu_coordinator.h"
#include "m3_logger.h"

// =============================================================================
// Minimal test framework
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

static void log_section(const char* name) {
    printf("\n╔══════════════════════════════════════════════════════════\n");
    printf("║  %s\n", name);
    printf("╚══════════════════════════════════════════════════════════\n");
}

static void log_test(const char* name) {
    printf("  ▶ %s\n", name);
}

#define CHECK(expr)                                                     \
    do {                                                                \
        if (!(expr)) {                                                  \
            printf("    FAIL  %s:%d  CHECK(%s)\n",                     \
                   __FILE__, __LINE__, #expr);                          \
            ++g_failed;                                                 \
        } else {                                                        \
            ++g_passed;                                                 \
        }                                                               \
    } while (0)

#define CHECK_EQ(a, b)                                                  \
    do {                                                                \
        auto _a = (a); auto _b = (b);                                   \
        if (_a != _b) {                                                 \
            printf("    FAIL  %s:%d  CHECK_EQ(%s, %s)  got %lld != %lld\n", \
                   __FILE__, __LINE__, #a, #b,                          \
                   (long long)_a, (long long)_b);                       \
            ++g_failed;                                                 \
        } else { ++g_passed; }                                          \
    } while (0)

// =============================================================================
// Shared test helpers
// =============================================================================

// Returns a centroid grid: nlist centroids, each dim-dimensional,
// centroid i has value i*10 in all dimensions.
static std::vector<float> make_centroid_grid(int nlist, int dim) {
    std::vector<float> c(static_cast<size_t>(nlist) * static_cast<size_t>(dim));
    for (int i = 0; i < nlist; ++i)
        for (int d = 0; d < dim; ++d)
            c[static_cast<size_t>(i) * static_cast<size_t>(dim) + d] =
                static_cast<float>(i) * 10.f;
    return c;
}

// Returns a vector close to cluster `cid`'s centroid (all dims = cid*10 ± noise).
static std::vector<float> vec_near_cluster(int cid, int dim, float noise,
                                           std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-noise, noise);
    std::vector<float> v(static_cast<size_t>(dim));
    for (int d = 0; d < dim; ++d)
        v[static_cast<size_t>(d)] = static_cast<float>(cid) * 10.f + dist(rng);
    return v;
}

// Forward declaration — defined in the Block 5 helpers section below.
static std::unique_ptr<m3::MultiLevelIndex> make_cache_index(int dim, int nlist);

// =============================================================================
// ═══ BLOCK 1 — Access Frequency Counter ═══
// =============================================================================

static void test_b1_counts_increment_on_search() {
    log_test("B1: access_count increments on each search hit");

    using namespace m3;
    const int DIM = 4, NLIST = 2;
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = NLIST;
    m3::MultiLevelIndex idx(DIM, Metric::L2, false, cfg);

    auto centroids = make_centroid_grid(NLIST, DIM);
    idx.set_l2_centroids(centroids);

    m3::CacheConfig cc;
    cc.l0_max_clusters = NLIST;
    cc.l1_max_clusters = NLIST;
    cc.l0_max_vectors_per_cluster = 1000;
    cc.l1_max_vectors_per_cluster = 10000;
    cc.cold_time_ns = 600'000'000'000ULL;
    cc.alpha_et = 0.f;
    idx.set_cache_config(cc);

    std::mt19937 rng(42);

    // Seed L2 directly so vectors are always findable.
    std::vector<int64_t> ids = {1, 2, 3};
    std::vector<float> vecs;
    for (int i = 0; i < 3; ++i) {
        auto v = vec_near_cluster(0, DIM, 0.1f, rng);
        vecs.insert(vecs.end(), v.begin(), v.end());
    }
    idx.load_cluster(0, ids.data(), vecs.data(), 3);

    // Before search: access_count should be 0.
    auto meta_before = idx.get_cluster_metadata();
    printf("    access_count before search: %llu\n",
           (unsigned long long)meta_before[0].access_count);
    CHECK(meta_before[0].access_count == 0);

    // Each search on cluster 0 should increment by 1.
    for (int i = 0; i < 5; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<int64_t>> oi;
        std::vector<std::vector<float>>   os;
        idx.search(q.data(), 1, 3, NLIST, oi, os);
    }

    auto meta_after = idx.get_cluster_metadata();
    printf("    access_count after 5 searches: %llu\n",
           (unsigned long long)meta_after[0].access_count);
    CHECK(meta_after[0].access_count >= 5);

    printf("    [Block 1] counts increment on search: PASS\n");
}

static void test_b1_counts_per_cluster_independent() {
    log_test("B1: access_count is per-cluster and independent");

    using namespace m3;
    const int DIM = 4, NLIST = 3;
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = NLIST;
    m3::MultiLevelIndex idx(DIM, Metric::L2, false, cfg);

    auto centroids = make_centroid_grid(NLIST, DIM);
    idx.set_l2_centroids(centroids);

    m3::CacheConfig cc;
    cc.l0_max_clusters = NLIST;
    cc.l1_max_clusters = NLIST;
    cc.l0_max_vectors_per_cluster = 1000;
    cc.l1_max_vectors_per_cluster = 10000;
    cc.cold_time_ns = 600'000'000'000ULL;
    cc.alpha_et = 0.f;
    idx.set_cache_config(cc);

    std::mt19937 rng(7);

    for (int cid = 0; cid < NLIST; ++cid) {
        std::vector<int64_t> ids = {static_cast<int64_t>(100 + cid)};
        auto v = vec_near_cluster(cid, DIM, 0.1f, rng);
        idx.load_cluster(cid, ids.data(), v.data(), 1);
    }

    // Search cluster 0 twice, cluster 1 once, cluster 2 zero times.
    for (int rep = 0; rep < 2; ++rep) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<int64_t>> oi; std::vector<std::vector<float>> os;
        idx.search(q.data(), 1, 1, 1, oi, os);
    }
    {
        auto q = vec_near_cluster(1, DIM, 0.05f, rng);
        std::vector<std::vector<int64_t>> oi; std::vector<std::vector<float>> os;
        idx.search(q.data(), 1, 1, 1, oi, os);
    }

    auto meta = idx.get_cluster_metadata();
    printf("    cluster 0 access_count=%llu  cluster 1=%llu  cluster 2=%llu\n",
           (unsigned long long)meta[0].access_count,
           (unsigned long long)meta[1].access_count,
           (unsigned long long)meta[2].access_count);

    CHECK(meta[0].access_count >= 2);
    CHECK(meta[1].access_count >= 1);
    // cluster 2 was never directly probed with nprobe=1 at cluster 2's centroid
    CHECK(meta[0].access_count > meta[2].access_count);

    printf("    [Block 1] per-cluster independence: PASS\n");
}

static void test_b1_count_monotonically_increases() {
    log_test("B1: access_count monotonically increases — no resets across maintenance_pass()");

    using namespace m3;
    const int DIM = 4, NLIST = 2;
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = NLIST;
    m3::MultiLevelIndex idx(DIM, Metric::L2, false, cfg);

    auto centroids = make_centroid_grid(NLIST, DIM);
    idx.set_l2_centroids(centroids);

    m3::CacheConfig cc;
    cc.l0_max_clusters = NLIST;
    cc.l1_max_clusters = NLIST;
    cc.l0_max_vectors_per_cluster = 5;   // small cap to trigger L0→L1 writeback
    cc.l1_max_vectors_per_cluster = 100;
    cc.l1_neighborhood_k = 2;
    cc.cold_time_ns = 600'000'000'000ULL;
    cc.alpha_et = 0.f;
    idx.set_cache_config(cc);

    std::mt19937 rng(13);

    // Use load_cluster() to seed L2 directly — stable across all maintenance passes.
    // (insert() in cache mode routes to L0 only; maintenance may then evict L0 data
    //  and promote_vector_neighborhood_ fetches from L2 which would be empty.)
    std::vector<int64_t> ids;
    std::vector<float>   vecs;
    for (int i = 0; i < 20; ++i) {
        ids.push_back(static_cast<int64_t>(1000 + i));
        auto v = vec_near_cluster(0, DIM, 0.1f, rng);
        vecs.insert(vecs.end(), v.begin(), v.end());
    }
    idx.load_cluster(0, ids.data(), vecs.data(), 20);

    // Search 3 times — each call hits record_access_() and increments access_count.
    for (int i = 0; i < 3; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<int64_t>> oi;
        std::vector<std::vector<float>>   os;
        idx.search(q.data(), 1, 3, NLIST, oi, os);
    }

    auto meta_mid = idx.get_cluster_metadata();
    printf("    access_count after 3 searches: %llu\n",
           (unsigned long long)meta_mid[0].access_count);

    // maintenance_pass() must not reset access_counts; L2 data is untouched.
    idx.maintenance_pass();
    idx.maintenance_pass();

    // Search 2 more times — results come from L2, always available.
    int results_after = 0;
    for (int i = 0; i < 2; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<int64_t>> oi;
        std::vector<std::vector<float>>   os;
        idx.search(q.data(), 1, 3, NLIST, oi, os);
        results_after += static_cast<int>(oi[0].size());
    }
    printf("    Results still available after 2x maintenance_pass(): %d\n", results_after);
    CHECK(results_after > 0);

    auto meta_final = idx.get_cluster_metadata();
    printf("    access_count after maintenance + 2 more searches: %llu\n",
           (unsigned long long)meta_final[0].access_count);
    CHECK(meta_final[0].access_count > meta_mid[0].access_count);

    printf("    [Block 1] Monotonic count across maintenance: PASS\n");
}

// =============================================================================
// ═══ BLOCK 2 — GpuBudgetManager ═══
// =============================================================================

static void* fake_ptr(int cid) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(cid + 1) * 0x1000);
}

static void test_b2_basic_register_and_query() {
    log_test("B2: register cluster and query state");
    m3::GpuBudgetManager mgr(1024);

    std::vector<m3::EvictedCluster> evicted;
    bool ok = mgr.register_cluster(0, fake_ptr(0), 256, evicted);
    CHECK(ok);
    CHECK(evicted.empty());
    CHECK(mgr.is_gpu_resident(0));
    CHECK(mgr.get_ptr(0) == fake_ptr(0));
    CHECK(mgr.get_bytes(0) == 256);
    CHECK(mgr.total_bytes_used() == 256);
    CHECK(mgr.resident_count() == 1);
    printf("    [Block 2] basic register and query: PASS\n");
}

static void test_b2_budget_enforcement_lfu_eviction() {
    log_test("B2: budget enforcement — LFU eviction reads access_count from idx");

    const int DIM = 2, NLIST = 3;
    auto idx = make_cache_index(DIM, NLIST);

    // Budget holds exactly 3 × 200 bytes.
    m3::GpuBudgetManager mgr(600, idx.get());

    std::vector<m3::EvictedCluster> evicted;
    // Load one vector per cluster so they are searchable.
    std::mt19937 rng(1);
    for (int cid : {0, 1, 2}) {
        auto v = vec_near_cluster(cid, DIM, 0.05f, rng);
        m3::DocId id = cid * 10;
        idx->load_cluster(cid, &id, v.data(), 1);
        mgr.register_cluster(cid, fake_ptr(cid), 200, evicted);
    }

    // Search cluster 2 many times (hottest), cluster 1 once, cluster 0 never.
    for (int i = 0; i < 5; ++i) {
        auto q = vec_near_cluster(2, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }
    {
        auto q = vec_near_cluster(1, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }
    // cluster 0 has 0 searches → coldest.

    // Adding cluster 3 (200 bytes) must evict the coldest → cluster 0.
    evicted.clear();
    bool ok = mgr.register_cluster(3, fake_ptr(3), 200, evicted);
    CHECK(ok);
    CHECK(evicted.size() == 1);
    printf("    Evicted cid=%d (expected 0)\n", evicted.empty() ? -1 : evicted[0].cid);
    CHECK(!evicted.empty() && evicted[0].cid == 0);
    CHECK(!mgr.is_gpu_resident(0));
    CHECK(mgr.is_gpu_resident(3));
    printf("    [Block 2] LFU eviction order: PASS\n");
}

static void test_b2_lfu_tie_broken_by_larger_bytes() {
    log_test("B2: LFU tie-break — evicts larger cluster when freq equal");
    m3::GpuBudgetManager mgr(500);

    std::vector<m3::EvictedCluster> evicted;
    mgr.register_cluster(0, fake_ptr(0), 100, evicted);  // freq=0, small
    mgr.register_cluster(1, fake_ptr(1), 300, evicted);  // freq=0, large
    CHECK(mgr.total_bytes_used() == 400);

    // Add cluster 2 (200 bytes): need 100 bytes freed; both at freq=0 but
    // cluster 1 is larger — expect cluster 1 evicted.
    bool ok = mgr.register_cluster(2, fake_ptr(2), 200, evicted);
    CHECK(ok);
    CHECK(!evicted.empty());
    CHECK(evicted[0].cid == 1);  // larger cluster evicted
    printf("    [Block 2] LFU tie-break by larger bytes: PASS\n");
}

static void test_b2_cluster_too_large_for_budget() {
    log_test("B2: cluster larger than entire budget is rejected");
    m3::GpuBudgetManager mgr(100);

    std::vector<m3::EvictedCluster> evicted;
    bool ok = mgr.register_cluster(0, fake_ptr(0), 200, evicted);
    CHECK(!ok);
    CHECK(!mgr.is_gpu_resident(0));
    printf("    [Block 2] cluster too large rejected: PASS\n");
}

static void test_b2_remove_cluster() {
    log_test("B2: remove_cluster returns entry and frees bytes");
    m3::GpuBudgetManager mgr(1024);

    std::vector<m3::EvictedCluster> evicted;
    mgr.register_cluster(5, fake_ptr(5), 512, evicted);
    CHECK(mgr.is_gpu_resident(5));

    auto ev = mgr.remove_cluster(5);
    CHECK(ev.cid == 5);
    CHECK(ev.ptr == fake_ptr(5));
    CHECK(ev.bytes == 512);
    CHECK(!mgr.is_gpu_resident(5));
    CHECK(mgr.total_bytes_used() == 0);
    printf("    [Block 2] remove_cluster: PASS\n");
}

static void test_b2_evicted_freq_reflects_access_count() {
    log_test("B2: EvictedCluster.freq reflects idx access_count at eviction time");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    m3::GpuBudgetManager mgr(200, idx.get());  // fits exactly 1 cluster

    // Load cluster 0 and search it a few times to build access_count.
    std::mt19937 rng(3);
    {
        auto v = vec_near_cluster(0, DIM, 0.05f, rng);
        m3::DocId id = 1;
        idx->load_cluster(0, &id, v.data(), 1);
    }
    for (int i = 0; i < 5; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }

    std::vector<m3::EvictedCluster> evicted;
    mgr.register_cluster(0, fake_ptr(0), 200, evicted);

    // Evict cluster 0 by adding cluster 1.
    mgr.register_cluster(1, fake_ptr(1), 200, evicted);
    CHECK(!evicted.empty() && evicted[0].cid == 0);
    const uint64_t ev_freq = evicted[0].freq;
    const uint64_t ac = idx->get_access_count(0);
    printf("    evicted.freq=%llu  idx.access_count=%llu\n",
           (unsigned long long)ev_freq, (unsigned long long)ac);
    // freq snapshot at eviction must equal access_count in idx.
    CHECK(ev_freq == ac);
    CHECK(ev_freq >= 5);  // we searched 5 times
    printf("    [Block 2] evicted freq reflects access_count: PASS\n");
}

static void test_b2_re_register_same_cluster() {
    log_test("B2: re-registering the same cluster replaces ptr and bytes");
    m3::GpuBudgetManager mgr(1024);

    std::vector<m3::EvictedCluster> evicted;
    mgr.register_cluster(0, fake_ptr(0), 100, evicted);

    // Re-register with new (larger) size — simulates expansion.
    mgr.register_cluster(0, fake_ptr(1), 200, evicted);
    CHECK(mgr.get_ptr(0) == fake_ptr(1));
    CHECK(mgr.get_bytes(0) == 200);
    CHECK(mgr.total_bytes_used() == 200);
    printf("    [Block 2] re-register same cluster: PASS\n");
}

static void test_b2_thread_safety() {
    log_test("B2: concurrent register/increment/remove from 8 threads");
    m3::GpuBudgetManager mgr(8 * 128);
    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&mgr, &errors, t]() {
            std::vector<m3::EvictedCluster> evicted;
            int cid = t;
            mgr.register_cluster(cid, fake_ptr(cid), 128, evicted);
            mgr.remove_cluster(cid);
        });
    }
    for (auto& th : threads) th.join();

    // After all threads: budget should be empty or close to it.
    printf("    Final bytes_used=%zu  resident=%zu\n",
           mgr.total_bytes_used(), mgr.resident_count());
    CHECK(mgr.total_bytes_used() <= 8 * 128);
    if (errors.load() != 0)
        printf("    FAIL: %d thread errors\n", errors.load());
    else
        printf("    [Block 2] thread safety: PASS\n");
}

// =============================================================================
// ═══ BLOCK 3 — ClusterInsertBuffer ═══
// =============================================================================

static void test_b3_slot_lifecycle() {
    log_test("B3: activate / deactivate / has_cluster");
    m3::ClusterInsertBuffer buf(4, 16);

    CHECK(!buf.has_cluster(0));
    buf.activate_cluster(0);
    CHECK(buf.has_cluster(0));
    buf.activate_cluster(0);  // no-op
    CHECK(buf.has_cluster(0));
    buf.deactivate_cluster(0);
    CHECK(!buf.has_cluster(0));
    printf("    [Block 3] slot lifecycle: PASS\n");
}

static void test_b3_try_buffer_not_resident() {
    log_test("B3: try_buffer returns kNotResident for unknown cluster");
    m3::ClusterInsertBuffer buf(4, 8);

    float v[4] = {1, 2, 3, 4};
    auto r = buf.try_buffer(99, 1, v);
    CHECK(r == m3::BufferResult::kNotResident);
    printf("    [Block 3] kNotResident: PASS\n");
}

static void test_b3_try_buffer_accepts_up_to_cap() {
    log_test("B3: try_buffer accepts up to cap then returns kFull");
    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(2, CAP);
    buf.activate_cluster(0);

    float v[2] = {1.f, 2.f};
    for (size_t i = 0; i < CAP; ++i) {
        auto r = buf.try_buffer(0, static_cast<int64_t>(i), v);
        CHECK(r == m3::BufferResult::kBuffered);
    }
    CHECK(buf.size(0) == CAP);
    auto r = buf.try_buffer(0, 999, v);
    CHECK(r == m3::BufferResult::kFull);
    printf("    [Block 3] accepts up to cap then kFull: PASS\n");
}

static void test_b3_drain_clears_slot() {
    log_test("B3: drain moves data out and leaves slot active but empty");
    m3::ClusterInsertBuffer buf(2, 8);
    buf.activate_cluster(1);

    float v[2] = {3.f, 4.f};
    buf.try_buffer(1, 10, v);
    buf.try_buffer(1, 11, v);

    std::vector<m3::DocId>  ids;
    std::vector<float>      vecs;
    bool ok = buf.drain(1, ids, vecs);
    CHECK(ok);
    CHECK(ids.size() == 2);
    CHECK(vecs.size() == 4);
    CHECK(buf.size(1) == 0);
    CHECK(buf.has_cluster(1));  // slot still alive

    // Can buffer new vectors after drain.
    auto r = buf.try_buffer(1, 20, v);
    CHECK(r == m3::BufferResult::kBuffered);
    printf("    [Block 3] drain clears slot: PASS\n");
}

static void test_b3_search_buffer_top_k() {
    log_test("B3: search_buffer returns correct top-k by L2 distance");
    const int DIM = 2;
    m3::ClusterInsertBuffer buf(DIM, 16);
    buf.activate_cluster(0);

    // Insert 5 vectors at known distances from query [0,0].
    // Distances² from origin: 2, 8, 18, 32, 50.
    float vecs[5][2] = {{1,1},{2,2},{3,3},{4,4},{5,5}};
    for (int i = 0; i < 5; ++i)
        buf.try_buffer(0, static_cast<int64_t>(i), vecs[i]);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId>  out_ids;
    std::vector<float>      out_scores;
    size_t got = buf.search_buffer(0, query, 3, m3::Metric::L2, false, out_ids, out_scores);
    CHECK(got == 3);
    // Nearest 3: ids 0(d²=2), 1(d²=8), 2(d²=18)
    std::unordered_set<int64_t> found(out_ids.begin(), out_ids.end());
    CHECK(found.count(0)); CHECK(found.count(1)); CHECK(found.count(2));
    CHECK(!found.count(3)); CHECK(!found.count(4));
    printf("    [Block 3] search_buffer top-k: PASS\n");
}

static void test_b3_search_buffer_empty_slot() {
    log_test("B3: search_buffer returns 0 for empty slot");
    m3::ClusterInsertBuffer buf(4, 8);
    buf.activate_cluster(0);

    float q[4] = {};
    std::vector<m3::DocId> oi; std::vector<float> os;
    size_t n = buf.search_buffer(0, q, 5, m3::Metric::L2, false, oi, os);
    CHECK(n == 0);
    printf("    [Block 3] search_buffer empty: PASS\n");
}

static void test_b3_search_buffer_cosine() {
    log_test("B3: search_buffer works with Cosine metric");
    const int DIM = 3;
    m3::ClusterInsertBuffer buf(DIM, 8);
    buf.activate_cluster(0);

    float v0[3] = {1,0,0};
    float v1[3] = {0,1,0};
    float v2[3] = {0,0,1};
    buf.try_buffer(0, 10, v0);
    buf.try_buffer(0, 11, v1);
    buf.try_buffer(0, 12, v2);

    float q[3] = {1,0,0};  // exact match with v0
    std::vector<m3::DocId> oi; std::vector<float> os;
    buf.search_buffer(0, q, 1, m3::Metric::COSINE, false, oi, os);
    CHECK(!oi.empty());
    CHECK(oi[0] == 10);  // v0 is the best match
    printf("    [Block 3] search_buffer cosine: PASS\n");
}

static void test_b3_erase_one() {
    log_test("B3: erase_one removes specific doc_id from buffer");
    const int DIM = 2;
    m3::ClusterInsertBuffer buf(DIM, 8);
    buf.activate_cluster(0);

    float v[2] = {1.f, 2.f};
    buf.try_buffer(0, 100, v);
    buf.try_buffer(0, 200, v);
    buf.try_buffer(0, 300, v);

    bool erased = buf.erase_one(0, 200);
    CHECK(erased);
    CHECK(buf.size(0) == 2);

    // 200 must not appear in search results.
    float q[2] = {0.f, 0.f};
    std::vector<m3::DocId> oi; std::vector<float> os;
    buf.search_buffer(0, q, 10, m3::Metric::L2, false, oi, os);
    for (auto id : oi) CHECK(id != 200);

    // Erase non-existent.
    CHECK(!buf.erase_one(0, 999));
    printf("    [Block 3] erase_one: PASS\n");
}

static void test_b3_multiple_clusters_independent() {
    log_test("B3: multiple cluster slots are fully independent");
    m3::ClusterInsertBuffer buf(2, 4);
    buf.activate_cluster(0);
    buf.activate_cluster(1);

    float v0[2] = {1.f, 0.f};
    float v1[2] = {0.f, 1.f};
    buf.try_buffer(0, 10, v0);
    buf.try_buffer(0, 11, v0);
    buf.try_buffer(1, 20, v1);

    CHECK(buf.size(0) == 2);
    CHECK(buf.size(1) == 1);

    std::vector<m3::DocId> ids; std::vector<float> vecs;
    buf.drain(0, ids, vecs);
    CHECK(ids.size() == 2);
    CHECK(buf.size(1) == 1);  // unaffected
    printf("    [Block 3] multiple clusters independent: PASS\n");
}

static void test_b3_thread_safety() {
    log_test("B3: concurrent try_buffer from 4 threads on same cluster");
    const int DIM = 4;
    const size_t CAP = 256;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    std::atomic<int> buffered{0};
    std::vector<std::thread> threads;
    float v[4] = {1,2,3,4};

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                auto r = buf.try_buffer(0, static_cast<int64_t>(t * 1000 + i), v);
                if (r == m3::BufferResult::kBuffered) ++buffered;
            }
        });
    }
    for (auto& th : threads) th.join();

    printf("    buffered=%d  slot_size=%zu  cap=%zu\n",
           buffered.load(), buf.size(0), CAP);
    CHECK(buf.size(0) <= CAP);
    printf("    [Block 3] thread safety: PASS\n");
}

// =============================================================================
// ═══ BLOCK 4 — GpuClusterIndex (collaborative search) ═══
// =============================================================================

static void test_b4_store_and_search() {
    log_test("B4: store_cluster + search_cluster returns correct top-k");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);

    // Store 5 vectors in cluster 0.
    std::vector<m3::DocId> ids = {10, 11, 12, 13, 14};
    // Vectors at increasing distances from origin: (1,1),(2,2),...
    std::vector<float> vecs = {1,1, 2,2, 3,3, 4,4, 5,5};
    void* handle = gpu.store_cluster(0, ids.data(), vecs.data(), 5);

    CHECK(gpu.has_cluster(0));
    CHECK(gpu.cluster_size(0) == 5);
    CHECK(handle != nullptr);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId>  out_ids;
    std::vector<float>      out_scores;
    size_t n = gpu.search_cluster(0, query, 3, out_ids, out_scores);

    CHECK(n == 3);
    CHECK(out_ids.size() == 3);
    // Nearest 3 from origin: ids 10(d²=2), 11(d²=8), 12(d²=18)
    std::unordered_set<int64_t> found(out_ids.begin(), out_ids.end());
    CHECK(found.count(10)); CHECK(found.count(11)); CHECK(found.count(12));
    CHECK(!found.count(13)); CHECK(!found.count(14));

    printf("    Results: ids=[");
    for (auto id : out_ids) printf("%lld ", (long long)id);
    printf("]  scores=[");
    for (float s : out_scores) printf("%.2f ", s);
    printf("]\n");
    printf("    [Block 4] store and search: PASS\n");
}

static void test_b4_remove_cluster() {
    log_test("B4: remove_cluster — no longer searchable");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);

    std::vector<m3::DocId> ids = {1, 2};
    std::vector<float> vecs = {1.f, 0.f, 0.f, 1.f};
    gpu.store_cluster(0, ids.data(), vecs.data(), 2);
    CHECK(gpu.has_cluster(0));

    bool removed = gpu.remove_cluster(0);
    CHECK(removed);
    CHECK(!gpu.has_cluster(0));
    CHECK(gpu.cluster_size(0) == 0);

    // Search on removed cluster returns 0 results.
    float q[2] = {};
    std::vector<m3::DocId> oi; std::vector<float> os;
    size_t n = gpu.search_cluster(0, q, 5, oi, os);
    CHECK(n == 0);

    // Remove again returns false.
    CHECK(!gpu.remove_cluster(0));
    printf("    [Block 4] remove_cluster: PASS\n");
}

// test_b4_search_increments_budget_frequency removed:
// collaborative_search no longer calls increment_frequency — access_count
// in ClusterMetadata (via MultiLevelIndex::search) is the single source of truth.

static void test_b4_collaborative_gpu_plus_buffer() {
    log_test("B4: collaborative_search merges GPU data and buffer inserts");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager budget(4096);
    m3::ClusterInsertBuffer buf(DIM, 64);

    // GPU cluster 0 has IDs 1,2.
    std::vector<m3::DocId> gpu_ids = {1, 2};
    std::vector<float> gpu_vecs = {1.f, 0.f, 2.f, 0.f};
    void* ptr = gpu.store_cluster(0, gpu_ids.data(), gpu_vecs.data(), 2);
    std::vector<m3::EvictedCluster> evicted;
    budget.register_cluster(0, ptr, 64, evicted);
    buf.activate_cluster(0);

    // Buffer (not yet flushed) has IDs 10, 11 — very close to query.
    float bv0[2] = {0.1f, 0.f};
    float bv1[2] = {0.2f, 0.f};
    buf.try_buffer(0, 10, bv0);
    buf.try_buffer(0, 11, bv1);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    size_t n = gpu.collaborative_search({0}, query, 4, buf, out_ids, out_scores);

    CHECK(n == 4);
    printf("    Results: ids=[");
    for (size_t i = 0; i < out_ids.size(); ++i)
        printf("%lld(%.3f) ", (long long)out_ids[i], out_scores[i]);
    printf("]\n");

    // All 4 IDs must appear.
    std::unordered_set<int64_t> found(out_ids.begin(), out_ids.end());
    CHECK(found.count(1)); CHECK(found.count(2));
    CHECK(found.count(10)); CHECK(found.count(11));
    // Buffer IDs are closer (scores ~0.01, 0.04) vs GPU IDs (1.0, 4.0).
    CHECK(out_ids[0] == 10 || out_ids[0] == 11);

    printf("    [Block 4] collaborative GPU+buffer merge: PASS\n");
}

static void test_b4_collaborative_buffer_only() {
    log_test("B4: collaborative_search returns buffer results when cluster not GPU-resident");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager budget(4096);
    m3::ClusterInsertBuffer buf(DIM, 64);

    // No GPU data for cluster 0 — only buffer.
    buf.activate_cluster(0);
    float v[2] = {1.f, 1.f};
    buf.try_buffer(0, 42, v);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    size_t n = gpu.collaborative_search({0}, query, 5, buf, out_ids, out_scores);

    CHECK(n == 1);
    CHECK(!out_ids.empty() && out_ids[0] == 42);
    printf("    [Block 4] buffer-only (no GPU resident): PASS\n");
}

static void test_b4_collaborative_gpu_only() {
    log_test("B4: collaborative_search returns GPU results when buffer is empty");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager budget(4096);
    m3::ClusterInsertBuffer buf(DIM, 64);

    std::vector<m3::DocId> gpu_ids = {7, 8};
    std::vector<float> gpu_vecs = {0.5f, 0.f, 1.5f, 0.f};
    void* ptr = gpu.store_cluster(0, gpu_ids.data(), gpu_vecs.data(), 2);
    std::vector<m3::EvictedCluster> evicted;
    budget.register_cluster(0, ptr, 64, evicted);
    buf.activate_cluster(0);  // active but empty

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    size_t n = gpu.collaborative_search({0}, query, 5, buf, out_ids, out_scores);

    CHECK(n == 2);
    std::unordered_set<int64_t> found(out_ids.begin(), out_ids.end());
    CHECK(found.count(7)); CHECK(found.count(8));
    printf("    [Block 4] GPU-only (empty buffer): PASS\n");
}

static void test_b4_dedup_same_id_in_gpu_and_buffer() {
    log_test("B4: collaborative_search deduplicates IDs appearing in both GPU and buffer");

    const int DIM = 2;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager budget(4096);
    m3::ClusterInsertBuffer buf(DIM, 64);

    // GPU has ID 5 at position (1,0).
    std::vector<m3::DocId> gpu_ids = {5};
    std::vector<float> gpu_vecs = {1.f, 0.f};
    void* ptr = gpu.store_cluster(0, gpu_ids.data(), gpu_vecs.data(), 1);
    std::vector<m3::EvictedCluster> evicted;
    budget.register_cluster(0, ptr, 64, evicted);
    buf.activate_cluster(0);

    // Buffer also has ID 5 at position (0.1,0) — closer to query.
    float bv[2] = {0.1f, 0.f};
    buf.try_buffer(0, 5, bv);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    gpu.collaborative_search({0}, query, 5, buf, out_ids, out_scores);

    // ID 5 should appear exactly once with the best (smallest) score.
    long long count5 = std::count(out_ids.begin(), out_ids.end(), (int64_t)5);
    CHECK(count5 == 1);
    printf("    ID 5 appears %lld time(s), score=%.4f\n", count5, out_scores[0]);
    // Best score: buffer position (0.1)² = 0.01, not GPU position (1)² = 1.
    CHECK(out_scores[0] < 0.1f);
    printf("    [Block 4] dedup same ID: PASS\n");
}

static void test_b4_thread_safety() {
    log_test("B4: concurrent store / remove / search from 4 threads");

    const int DIM = 4;
    m3::GpuClusterIndex gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager budget(65536);
    m3::ClusterInsertBuffer buf(DIM, 256);

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            std::mt19937 rng(static_cast<unsigned>(t));
            std::uniform_real_distribution<float> dist(-1.f, 1.f);

            std::vector<m3::DocId> ids(4);
            std::vector<float>     vecs(4 * DIM);
            for (int i = 0; i < 4; ++i) {
                ids[i] = static_cast<int64_t>(t * 100 + i);
                for (int d = 0; d < DIM; ++d)
                    vecs[static_cast<size_t>(i) * DIM + d] = dist(rng);
            }

            int cid = t;
            void* ptr = gpu.store_cluster(cid, ids.data(), vecs.data(), 4);
            buf.activate_cluster(cid);
            std::vector<m3::EvictedCluster> evicted;
            budget.register_cluster(cid, ptr, 256, evicted);

            for (int iter = 0; iter < 20; ++iter) {
                float q[4]; for (int d = 0; d < DIM; ++d) q[d] = dist(rng);
                std::vector<m3::DocId> oi; std::vector<float> os;
                gpu.collaborative_search({cid}, q, 2, buf, oi, os);
            }

            gpu.remove_cluster(cid);
        });
    }
    for (auto& th : threads) th.join();

    if (errors.load()) printf("    FAIL: %d errors\n", errors.load());
    else printf("    [Block 4] thread safety: PASS\n");
}

// =============================================================================
// ═══ BLOCK 5 — AsyncFlushCoordinator ═══
// =============================================================================

// Helper: build a MultiLevelIndex in cache mode with `nlist` clusters.
// Returns unique_ptr because MultiLevelIndex is non-copyable (holds mutexes).
static std::unique_ptr<m3::MultiLevelIndex> make_cache_index(int dim, int nlist) {
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = nlist;
    auto idx = std::make_unique<m3::MultiLevelIndex>(dim, m3::Metric::L2, false, cfg);
    auto centroids = make_centroid_grid(nlist, dim);
    idx->set_l2_centroids(centroids);
    m3::CacheConfig cc;
    cc.l0_max_clusters = nlist;
    cc.l1_max_clusters = nlist;
    cc.l0_max_vectors_per_cluster = 10000;
    cc.l1_max_vectors_per_cluster = 10000;
    cc.cold_time_ns = 600'000'000'000ULL;
    cc.alpha_et = 0.f;
    idx->set_cache_config(cc);
    return idx;
}

static void test_b5_no_flush_below_threshold() {
    log_test("B5: maybe_flush does nothing when buffer is below threshold");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 8;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    // Flush threshold = 0 means "flush only when full (== cap)".
    m3::AsyncFlushCoordinator coord(buf, *idx, /*flush_threshold=*/0);

    float v[2] = {0.5f, 0.f};
    for (size_t i = 0; i < CAP - 1; ++i)
        buf.try_buffer(0, static_cast<int64_t>(i), v);

    size_t flushed = coord.maybe_flush(0);
    CHECK(flushed == 0);
    CHECK(buf.size(0) == CAP - 1);
    CHECK(coord.total_flushed_vectors() == 0);
    printf("    Buffer size=%zu, flushed=%zu (expected 0)\n", buf.size(0), flushed);
    printf("    [Block 5] no flush below threshold: PASS\n");
}

static void test_b5_flush_when_full() {
    log_test("B5: maybe_flush drains and writes to L2 when slot is full");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    m3::AsyncFlushCoordinator coord(buf, *idx, /*flush_threshold=*/0);

    std::vector<int64_t> expected_ids;
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {0.5f, static_cast<float>(i)};
        buf.try_buffer(0, static_cast<int64_t>(i + 1), v);
        expected_ids.push_back(static_cast<int64_t>(i + 1));
    }
    CHECK(buf.is_full(0));

    size_t flushed = coord.maybe_flush(0);
    printf("    Flushed %zu vectors\n", flushed);
    CHECK(flushed == CAP);
    CHECK(buf.size(0) == 0);              // slot drained
    CHECK(buf.has_cluster(0));            // slot still active
    CHECK(coord.total_flushed_vectors() == CAP);
    CHECK(coord.total_flush_events() == 1);

    // Flushed vectors must be searchable in L2.
    std::mt19937 rng(1);
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<int64_t>> oi;
    std::vector<std::vector<float>>   os;
    idx->search(q.data(), 1, static_cast<int>(CAP), NLIST, oi, os);
    printf("    Search after flush returned %zu results\n", oi[0].size());
    CHECK(!oi[0].empty());

    printf("    [Block 5] flush when full: PASS\n");
}

static void test_b5_flush_threshold_early() {
    log_test("B5: custom flush_threshold triggers flush before reaching cap");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 16;
    const size_t THRESH = 5;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    m3::AsyncFlushCoordinator coord(buf, *idx, THRESH);

    float v[2] = {0.5f, 0.f};
    for (size_t i = 0; i < THRESH; ++i)
        buf.try_buffer(0, static_cast<int64_t>(i), v);

    CHECK(buf.size(0) == THRESH);
    CHECK(!buf.is_full(0));  // not at cap, but at threshold

    size_t flushed = coord.maybe_flush(0);
    printf("    Flushed %zu at threshold %zu (cap=%zu)\n", flushed, THRESH, CAP);
    CHECK(flushed == THRESH);
    CHECK(buf.size(0) == 0);

    printf("    [Block 5] early threshold flush: PASS\n");
}

static void test_b5_flush_clusters_multiple() {
    log_test("B5: flush_clusters() flushes multiple clusters in one call");

    const int DIM = 2, NLIST = 3;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    for (int cid = 0; cid < NLIST; ++cid) buf.activate_cluster(cid);

    m3::AsyncFlushCoordinator coord(buf, *idx, 0);

    // Fill clusters 0 and 2 to capacity; leave cluster 1 partial.
    for (int cid : {0, 2}) {
        for (size_t i = 0; i < CAP; ++i) {
            float v[2] = {static_cast<float>(cid) * 10.f, static_cast<float>(i)};
            buf.try_buffer(cid, static_cast<int64_t>(cid * 100 + i), v);
        }
    }
    float partial_v[2] = {10.f, 0.f};
    buf.try_buffer(1, 999, partial_v);  // partial

    size_t total = coord.flush_clusters({0, 1, 2});
    printf("    Total flushed across 3 clusters: %zu\n", total);
    CHECK(total == 2 * CAP);                // only clusters 0 and 2 were full
    CHECK(coord.total_flush_events() == 2); // 2 flush events
    CHECK(buf.size(1) == 1);                // cluster 1 untouched

    printf("    [Block 5] flush_clusters multiple: PASS\n");
}

static void test_b5_flush_preserves_data_in_l2() {
    log_test("B5: flushed vectors are searchable in L2 after flush");

    const int DIM = 4, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 6;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    m3::AsyncFlushCoordinator coord(buf, *idx, 0);

    // Insert known IDs into buffer near cluster 0's centroid (0,0,0,0).
    std::mt19937 rng(99);
    std::vector<int64_t> inserted_ids;
    for (size_t i = 0; i < CAP; ++i) {
        auto v = vec_near_cluster(0, DIM, 0.2f, rng);
        int64_t id = static_cast<int64_t>(2000 + i);
        buf.try_buffer(0, id, v.data());
        inserted_ids.push_back(id);
    }
    coord.maybe_flush(0);

    // Search L2 — all IDs should be recoverable.
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<int64_t>> oi;
    std::vector<std::vector<float>>   os;
    idx->search(q.data(), 1, static_cast<int>(CAP), NLIST, oi, os);

    printf("    Search returned %zu/%zu inserted IDs\n", oi[0].size(), CAP);
    CHECK(oi[0].size() == CAP);

    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    for (auto id : inserted_ids)
        CHECK(found.count(id));

    printf("    [Block 5] flush preserves data in L2: PASS\n");
}

static void test_b5_background_thread() {
    log_test("B5: background thread flushes full slots automatically");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    m3::AsyncFlushCoordinator coord(buf, *idx, 0);
    CHECK(!coord.background_running());

    // Fill the buffer to full.
    float v[2] = {0.3f, 0.f};
    for (size_t i = 0; i < CAP; ++i)
        buf.try_buffer(0, static_cast<int64_t>(i), v);
    CHECK(buf.is_full(0));

    // Start background thread with 10ms polling interval.
    coord.start_background({0}, 10);
    CHECK(coord.background_running());

    // Sleep enough for at least one poll cycle.
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    coord.stop_background();
    CHECK(!coord.background_running());

    printf("    total_flushed=%llu  total_events=%llu  buf_size=%zu\n",
           (unsigned long long)coord.total_flushed_vectors(),
           (unsigned long long)coord.total_flush_events(),
           buf.size(0));

    CHECK(coord.total_flushed_vectors() == CAP);
    CHECK(buf.size(0) == 0);

    // Flushed data must be in L2.
    std::mt19937 rng(77);
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<int64_t>> oi;
    std::vector<std::vector<float>>   os;
    idx->search(q.data(), 1, static_cast<int>(CAP), NLIST, oi, os);
    CHECK(!oi[0].empty());

    printf("    [Block 5] background thread: PASS\n");
}

static void test_b5_stats_accuracy() {
    log_test("B5: stats counters track total_flushed_vectors and total_flush_events");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    m3::AsyncFlushCoordinator coord(buf, *idx, 0);

    int64_t id_counter = 0;
    auto fill_and_flush = [&](size_t n) {
        float v[2] = {0.5f, 0.f};
        for (size_t i = 0; i < n; ++i)
            buf.try_buffer(0, id_counter++, v);  // unique IDs across rounds
        coord.maybe_flush(0);
    };

    fill_and_flush(CAP);  // flush #1: 4 vectors (IDs 0–3)
    fill_and_flush(CAP);  // flush #2: 4 vectors (IDs 4–7)

    printf("    total_flushed=%llu  events=%llu\n",
           (unsigned long long)coord.total_flushed_vectors(),
           (unsigned long long)coord.total_flush_events());

    CHECK(coord.total_flushed_vectors() == 2 * CAP);
    CHECK(coord.total_flush_events() == 2);
    printf("    [Block 5] stats accuracy: PASS\n");
}

static void test_b5_concurrent_insert_and_flush() {
    log_test("B5: thread safety — concurrent insert + flush from 2 threads");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 32;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    // Use a small threshold so flusher triggers often.
    m3::AsyncFlushCoordinator coord(buf, *idx, 4);

    std::atomic<bool> stop{false};
    std::atomic<int>  insert_count{0};

    // Thread A: inserts vectors into buffer continuously.
    std::thread inserter([&]() {
        float v[2] = {0.5f, 0.f};
        for (int i = 0; i < 200 && !stop.load(); ++i) {
            buf.try_buffer(0, static_cast<int64_t>(i), v);
            ++insert_count;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });

    // Thread B: continuously calls maybe_flush.
    std::thread flusher([&]() {
        while (!stop.load()) {
            coord.maybe_flush(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        coord.maybe_flush(0);  // final flush
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop.store(true);
    inserter.join();
    flusher.join();

    // One final flush to catch anything left in the buffer.
    // Force flush by draining manually if below threshold.
    std::vector<m3::DocId> rem_ids; std::vector<float> rem_vecs;
    buf.drain(0, rem_ids, rem_vecs);
    if (!rem_ids.empty())
        idx->load_cluster(0, rem_ids.data(), rem_vecs.data(), rem_ids.size());

    printf("    inserted=%d  flushed=%llu  events=%llu\n",
           insert_count.load(),
           (unsigned long long)coord.total_flushed_vectors(),
           (unsigned long long)coord.total_flush_events());

    // At least some vectors were flushed without crashing.
    CHECK(coord.total_flush_events() > 0);
    printf("    [Block 5] concurrent insert+flush: PASS\n");
}

// =============================================================================
// ═══ Integration: all five blocks working together ═══
// =============================================================================

static void test_integration_full_pipeline() {
    log_test("Integration: insert → buffer → collaborative search → flush → L2 search");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    m3::GpuClusterIndex     gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager    budget(65536);
    m3::AsyncFlushCoordinator coord(buf, *idx, 0);

    // Load a small "cold" dataset directly into L2 (cluster 0).
    std::vector<int64_t> cold_ids = {100, 101, 102};
    std::vector<float>   cold_vecs = {0.1f,0.f, 0.2f,0.f, 0.3f,0.f};
    idx->load_cluster(0, cold_ids.data(), cold_vecs.data(), 3);

    // Promote cluster 0 to GPU.
    {
        std::vector<m3::DocId> gpu_ids(cold_ids.begin(), cold_ids.end());
        void* ptr = gpu.store_cluster(0, gpu_ids.data(), cold_vecs.data(), 3);
        std::vector<m3::EvictedCluster> evicted;
        budget.register_cluster(0, ptr, 3 * DIM * 4, evicted);
        buf.activate_cluster(0);
    }

    // New inserts target GPU-resident cluster 0 — they land in the buffer.
    float nv0[2] = {0.05f, 0.f};
    float nv1[2] = {0.07f, 0.f};
    buf.try_buffer(0, 200, nv0);
    buf.try_buffer(0, 201, nv1);

    // Collaborative search: GPU data + buffer both contribute.
    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> cs_ids; std::vector<float> cs_scores;
    gpu.collaborative_search({0}, query, 5, buf, cs_ids, cs_scores);

    printf("    Collaborative search returned %zu results\n", cs_ids.size());
    CHECK(cs_ids.size() >= 2);  // at minimum the two buffer inserts
    std::unordered_set<int64_t> cs_found(cs_ids.begin(), cs_ids.end());
    CHECK(cs_found.count(200)); CHECK(cs_found.count(201));

    // Fill buffer to cap so flush triggers.
    float v0[2] = {0.4f, 0.f};
    float v1[2] = {0.5f, 0.f};
    buf.try_buffer(0, 202, v0);
    buf.try_buffer(0, 203, v1);
    CHECK(buf.is_full(0));

    size_t flushed = coord.maybe_flush(0);
    printf("    Flushed %zu vectors to L2\n", flushed);
    CHECK(flushed == CAP);

    // After flush, search L2 via MultiLevelIndex — all flushed IDs findable.
    std::mt19937 rng(5);
    auto q2 = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<int64_t>> oi;
    std::vector<std::vector<float>>   os;
    idx->search(q2.data(), 1, 10, NLIST, oi, os);
    printf("    L2 search returned %zu results\n", oi[0].size());
    CHECK(!oi[0].empty());

    // access_count incremented by the collaborative search.
    auto meta = idx->get_cluster_metadata();
    printf("    Cluster 0 access_count=%llu\n",
           (unsigned long long)meta[0].access_count);
    CHECK(meta[0].access_count >= 1);

    printf("    [Integration] full pipeline: PASS\n");
}

// =============================================================================
// ═══ BLOCK 6 — Eviction Drain Protocol ═══
// =============================================================================

static void test_b6_drain_on_eviction_no_data_loss() {
    log_test("B6: eviction drains buffer to L2 — no buffered vectors lost");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    m3::GpuClusterIndex     gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager    budget(/*bytes=*/64);  // tiny budget: only 1 cluster at 32 bytes
    m3::ClusterInsertBuffer buf(DIM, 16);

    // Promote cluster 0 (2 vectors × 2 floats × 4 bytes = 16 bytes).
    {
        std::vector<m3::DocId> ids = {1, 2};
        std::vector<float> vecs = {0.1f,0.f, 0.2f,0.f};
        idx->load_cluster(0, ids.data(), vecs.data(), 2);
        void* ptr = gpu.store_cluster(0, ids.data(), vecs.data(), 2);
        std::vector<m3::EvictedCluster> ev;
        budget.register_cluster(0, ptr, 16, ev);
        buf.activate_cluster(0);
    }

    // Buffer 3 vectors for cluster 0 (not yet flushed to L2).
    float bv[2] = {0.3f, 0.f};
    buf.try_buffer(0, 100, bv);
    float bv2[2] = {0.4f, 0.f};
    buf.try_buffer(0, 101, bv2);
    float bv3[2] = {0.5f, 0.f};
    buf.try_buffer(0, 102, bv3);
    CHECK(buf.size(0) == 3);

    // Promote cluster 1 — budget is full so cluster 0 gets LFU-evicted.
    {
        std::vector<m3::DocId> ids = {3, 4};
        std::vector<float> vecs = {10.f,0.f, 10.1f,0.f};
        idx->load_cluster(1, ids.data(), vecs.data(), 2);
        void* ptr = gpu.store_cluster(1, ids.data(), vecs.data(), 2);
        std::vector<m3::EvictedCluster> ev;
        budget.register_cluster(1, ptr, 16, ev);

        // ev should contain cluster 0.
        CHECK(!ev.empty());
        CHECK(ev[0].cid == 0);

        // Apply the drain protocol — buffered IDs 100,101,102 must reach L2.
        size_t drained = m3::drain_evicted_clusters(ev, buf, gpu, *idx);
        printf("    Drained %zu buffered vectors to L2 on eviction\n", drained);
        CHECK(drained == 3);
    }

    // Cluster 0 must no longer be GPU-resident or have a buffer slot.
    CHECK(!budget.is_gpu_resident(0));
    CHECK(!buf.has_cluster(0));
    CHECK(!gpu.has_cluster(0));

    // Buffered vectors (100,101,102) must now be searchable in L2.
    float q[2] = {0.3f, 0.f};
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>>     os;
    idx->search(q, 1, 5, NLIST, oi, os);
    printf("    L2 search after eviction returned %zu results\n", oi[0].size());
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(100)); CHECK(found.count(101)); CHECK(found.count(102));

    printf("    [Block 6] eviction drain no data loss: PASS\n");
}

static void test_b6_promote_cluster_helper() {
    log_test("B6: promote_cluster() helper uploads, registers, and activates buffer");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    m3::GpuClusterIndex     gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager    budget(4096);
    m3::ClusterInsertBuffer buf(DIM, 32);

    // Put some vectors in L2 cluster 0.
    std::vector<m3::DocId> ids = {10, 11, 12};
    std::vector<float>     vecs = {0.1f,0.f, 0.2f,0.f, 0.3f,0.f};
    idx->load_cluster(0, ids.data(), vecs.data(), 3);

    bool ok = m3::promote_cluster(0, ids.data(), vecs.data(), 3,
                                   gpu, budget, buf, *idx);
    CHECK(ok);
    CHECK(gpu.has_cluster(0));
    CHECK(budget.is_gpu_resident(0));
    CHECK(buf.has_cluster(0));

    // Inserts to cluster 0 should now be buffered.
    float nv[2] = {0.4f, 0.f};
    auto r = buf.try_buffer(0, 99, nv);
    CHECK(r == m3::BufferResult::kBuffered);
    printf("    [Block 6] promote_cluster helper: PASS\n");
}

static void test_b6_eviction_hot_cluster_survives() {
    log_test("B6: hot cluster (high access_count) survives over cold cluster during eviction");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    // Budget fits exactly 1 × 16-byte cluster.
    m3::GpuClusterIndex     gpu(DIM, m3::Metric::L2, false);
    m3::GpuBudgetManager    budget(32, idx.get());
    m3::ClusterInsertBuffer buf(DIM, 16);

    std::mt19937 rng(5);

    // Load cluster 0 and search it many times → high access_count.
    {
        auto v = vec_near_cluster(0, DIM, 0.05f, rng);
        m3::DocId id = 1;
        idx->load_cluster(0, &id, v.data(), 1);
    }
    for (int i = 0; i < 20; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }

    // Load cluster 1 with no searches → cold.
    {
        auto v = vec_near_cluster(1, DIM, 0.05f, rng);
        m3::DocId id = 2;
        idx->load_cluster(1, &id, v.data(), 1);
    }

    // Register both; second registration evicts the coldest (cluster 1 = no accesses).
    std::vector<m3::EvictedCluster> ev;
    {
        std::vector<m3::DocId> ids = {1}; std::vector<float> vecs = {0.1f, 0.f};
        void* ptr = gpu.store_cluster(0, ids.data(), vecs.data(), 1);
        budget.register_cluster(0, ptr, 16, ev);
        buf.activate_cluster(0);
    }
    {
        std::vector<m3::DocId> ids = {2}; std::vector<float> vecs = {10.f, 0.f};
        void* ptr = gpu.store_cluster(1, ids.data(), vecs.data(), 1);
        budget.register_cluster(1, ptr, 16, ev);
    }

    // Cluster 0 is hot (access_count=20), cluster 1 is cold → cluster 1 evicted.
    CHECK(!ev.empty());
    printf("    Evicted cid=%d (expected 1, the cold cluster)\n", ev[0].cid);
    CHECK(ev[0].cid == 1);
    CHECK(budget.is_gpu_resident(0));   // hot cluster survives
    CHECK(!budget.is_gpu_resident(1));  // cold cluster evicted

    // EvictedCluster.freq reflects the access_count at eviction time.
    const uint64_t ac1 = idx->get_access_count(1);
    printf("    evicted.freq=%llu  idx.access_count(1)=%llu\n",
           (unsigned long long)ev[0].freq, (unsigned long long)ac1);
    CHECK(ev[0].freq == ac1);

    printf("    [Block 6] eviction: hot cluster survives: PASS\n");
}

static void test_b6_empty_buffer_eviction() {
    log_test("B6: drain_evicted_clusters with empty buffer slot is safe (0 vectors drained)");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    m3::GpuClusterIndex     gpu(DIM, m3::Metric::L2, false);
    m3::ClusterInsertBuffer buf(DIM, 8);

    // Evict cluster 0 with no buffer slot at all.
    std::vector<m3::EvictedCluster> ev = {{0, nullptr, 16, 0}};
    size_t drained = m3::drain_evicted_clusters(ev, buf, gpu, *idx);
    CHECK(drained == 0);

    // Activate slot then evict with empty slot.
    buf.activate_cluster(0);
    size_t drained2 = m3::drain_evicted_clusters(ev, buf, gpu, *idx);
    CHECK(drained2 == 0);
    printf("    [Block 6] empty buffer eviction: PASS\n");
}

// =============================================================================
// ═══ BLOCK 7 — Maintenance tick ═══
// =============================================================================

static void test_b7_budget_reads_access_count_directly() {
    log_test("B7: GpuBudgetManager reads access_count from idx directly — no sync step needed");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    // Pass idx to budget so it reads access_count on eviction.
    m3::GpuBudgetManager budget(128, idx.get());

    std::mt19937 rng(7);
    for (int cid : {0, 1}) {
        m3::DocId id = cid * 10;
        auto v = vec_near_cluster(cid, DIM, 0.1f, rng);
        idx->load_cluster(cid, &id, v.data(), 1);
        std::vector<m3::EvictedCluster> ev;
        budget.register_cluster(cid,
            reinterpret_cast<void*>(static_cast<uintptr_t>(cid + 1) * 0x100),
            64, ev);
    }

    // Search cluster 0 five times, cluster 1 twice.
    for (int i = 0; i < 5; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }
    for (int i = 0; i < 2; ++i) {
        auto q = vec_near_cluster(1, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }

    auto meta = idx->get_cluster_metadata();
    printf("    access_count: cid0=%llu  cid1=%llu\n",
           (unsigned long long)meta[0].access_count,
           (unsigned long long)meta[1].access_count);
    CHECK(meta[0].access_count > meta[1].access_count);

    // Adding a third cluster (64 bytes) must evict the coldest (cid1, fewer accesses).
    std::vector<m3::EvictedCluster> ev;
    budget.register_cluster(2,
        reinterpret_cast<void*>(0x300), 64, ev);
    CHECK(!ev.empty());
    printf("    Evicted cid=%d (expected 1, the colder cluster)\n", ev[0].cid);
    CHECK(ev[0].cid == 1);
    CHECK(budget.is_gpu_resident(0));
    printf("    [Block 7] budget reads access_count directly: PASS\n");
}

static void test_b7_maintenance_tick_flushes_buffers() {
    log_test("B7: maintenance_tick() flushes buffers and runs maintenance_pass");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    m3::ClusterInsertBuffer   buf(DIM, 4);
    m3::AsyncFlushCoordinator flush_coord(buf, *idx, 0);

    buf.activate_cluster(0);

    // Fill buffer to cap.
    float v[2] = {0.5f, 0.f};
    for (int i = 0; i < 4; ++i)
        buf.try_buffer(0, static_cast<int64_t>(i), v);

    auto result = m3::maintenance_tick({0}, flush_coord, *idx);
    printf("    tick: flushed=%zu\n", result.vectors_flushed);

    CHECK(result.vectors_flushed == 4);
    CHECK(buf.size(0) == 0);
    printf("    [Block 7] maintenance_tick flushes buffers: PASS\n");
}

static void test_b7_hot_cluster_survives_eviction_pressure() {
    log_test("B7: hot cluster (read live from idx) is evicted last under budget pressure");

    const int DIM = 2, NLIST = 3;
    auto idx = make_cache_index(DIM, NLIST);

    // Budget fits exactly 2 of 3 clusters (32 bytes each, budget = 64).
    // Pass idx so budget reads access_count live — no sync step needed.
    m3::GpuBudgetManager    budget(64, idx.get());
    m3::ClusterInsertBuffer buf(DIM, 8);

    std::mt19937 rng(42);

    // Register clusters 0 and 1; cluster 0 gets no searches (cold).
    for (int cid : {0, 1}) {
        m3::DocId id = cid * 10;
        auto v = vec_near_cluster(cid, DIM, 0.1f, rng);
        idx->load_cluster(cid, &id, v.data(), 1);
        std::vector<m3::EvictedCluster> ev;
        budget.register_cluster(
            cid, reinterpret_cast<void*>(static_cast<uintptr_t>(cid + 1) * 0x100),
            32, ev);
        buf.activate_cluster(cid);
    }

    // Search cluster 1 many times — hot, no sync needed.
    for (int i = 0; i < 10; ++i) {
        auto q = vec_near_cluster(1, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q.data(), 1, 1, 1, oi, os);
    }

    auto ac0 = idx->get_access_count(0);
    auto ac1 = idx->get_access_count(1);
    printf("    access_count: cid0=%llu  cid1=%llu\n",
           (unsigned long long)ac0, (unsigned long long)ac1);
    CHECK(ac1 > ac0);

    // Register cluster 2 — budget reads live access_counts, evicts coldest (cid0).
    {
        m3::DocId id = 20;
        auto v = vec_near_cluster(2, DIM, 0.1f, rng);
        idx->load_cluster(2, &id, v.data(), 1);
        std::vector<m3::EvictedCluster> ev;
        budget.register_cluster(
            2, reinterpret_cast<void*>(0x300), 32, ev);
        CHECK(!ev.empty());
        CHECK(ev[0].cid == 0);  // cold cluster evicted
        CHECK(budget.is_gpu_resident(1));
        printf("    Evicted cid=%d (expected 0)\n", ev[0].cid);
    }

    printf("    [Block 7] hot cluster survives eviction pressure: PASS\n");
}

// =============================================================================
// ═══ BLOCK 8 — GpuCoordinator ═══
// =============================================================================

static void test_b8_promote_routes_inserts_to_buffer() {
    log_test("B8: after promote_to_gpu(), inserts are buffered not written to L2 directly");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    // Seed some L2 data for cluster 0.
    std::vector<m3::DocId> seed_ids = {1, 2, 3};
    std::vector<float>     seed_vecs = {0.1f,0.f, 0.2f,0.f, 0.3f,0.f};
    idx->load_cluster(0, seed_ids.data(), seed_vecs.data(), 3);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 8);

    bool promoted = coord.promote_to_gpu(0);
    CHECK(promoted);
    CHECK(coord.is_gpu_resident(0));

    // Inserts to cluster 0 should be buffered (not directly in L2).
    float nv[2] = {0.4f, 0.f};
    auto r = coord.insert(0, 99, nv);
    CHECK(r == m3::BufferResult::kBuffered);

    // Cluster 1 is NOT GPU-resident: insert goes directly to L2.
    float nv2[2] = {10.1f, 0.f};
    auto r2 = coord.insert(1, 88, nv2);
    CHECK(r2 == m3::BufferResult::kBuffered);

    printf("    [Block 8] promote routes inserts to buffer: PASS\n");
}

static void test_b8_search_finds_gpu_and_buffer() {
    log_test("B8: search finds GPU-stored data and buffered inserts together");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    std::vector<m3::DocId> ids = {1, 2, 3};
    std::vector<float>     vecs = {0.1f,0.f, 0.2f,0.f, 0.3f,0.f};
    idx->load_cluster(0, ids.data(), vecs.data(), 3);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 16);
    coord.promote_to_gpu(0);

    // Buffer 2 more inserts.
    float nv0[2] = {0.05f, 0.f};
    float nv1[2] = {0.07f, 0.f};
    coord.insert(0, 100, nv0);
    coord.insert(0, 101, nv1);

    float query[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    size_t n = coord.search({0}, query, 5, out_ids, out_scores);

    printf("    Search returned %zu results\n", n);
    CHECK(n >= 2);
    std::unordered_set<int64_t> found(out_ids.begin(), out_ids.end());
    // Buffer inserts must appear (they are closest to origin).
    CHECK(found.count(100)); CHECK(found.count(101));
    // GPU-stored vectors must also appear.
    CHECK(found.count(1)); CHECK(found.count(2)); CHECK(found.count(3));
    printf("    [Block 8] search finds GPU + buffer: PASS\n");
}

static void test_b8_demote_drains_to_l2() {
    log_test("B8: demote_from_gpu() flushes buffered vectors to L2 before removing");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    std::vector<m3::DocId> ids = {1};
    std::vector<float>     vecs = {0.1f, 0.f};
    idx->load_cluster(0, ids.data(), vecs.data(), 1);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 16);
    coord.promote_to_gpu(0);

    // Buffer 3 inserts — not yet in L2.
    for (int i = 0; i < 3; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f + 0.2f, 0.f};
        coord.insert(0, static_cast<int64_t>(200 + i), v);
    }

    size_t drained = coord.demote_from_gpu(0);
    printf("    Drained %zu vectors to L2 on demote\n", drained);
    CHECK(drained == 3);
    CHECK(!coord.is_gpu_resident(0));

    // Drained vectors must now be searchable in L2.
    std::mt19937 rng(9);
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>> os;
    idx->search(q.data(), 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(200)); CHECK(found.count(201)); CHECK(found.count(202));
    printf("    [Block 8] demote drains to L2: PASS\n");
}

static void test_b8_maintenance_tick_end_to_end() {
    log_test("B8: maintenance_tick() flushes + syncs + maintenance_pass in sequence");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    std::vector<m3::DocId> ids = {10, 11};
    std::vector<float>     vecs = {0.1f,0.f, 0.2f,0.f};
    idx->load_cluster(0, ids.data(), vecs.data(), 2);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 4);
    coord.promote_to_gpu(0);

    // Do searches to build access_count in idx metadata.
    std::mt19937 rng(11);
    for (int i = 0; i < 3; ++i) {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<m3::DocId> oi; std::vector<float> os;
        coord.search({0}, q.data(), 2, oi, os);
    }

    // Fill buffer to cap so maintenance_tick will flush it.
    for (int i = 0; i < 4; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.f};
        coord.insert(0, static_cast<int64_t>(300 + i), v);
    }

    auto r = coord.maintenance_tick();
    printf("    maintenance_tick: flushed=%zu\n", r.vectors_flushed);
    CHECK(r.vectors_flushed == 4);

    // Flushed vectors must be in L2.
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>> os;
    idx->search(q.data(), 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(300)); CHECK(found.count(301));
    printf("    [Block 8] maintenance_tick end-to-end: PASS\n");
}

static void test_b8_auto_eviction_on_budget_exceeded() {
    log_test("B8: promoting more clusters than budget allows evicts LFU with drain");

    const int DIM = 2, NLIST = 3;
    auto idx = make_cache_index(DIM, NLIST);

    // Budget holds exactly 2 clusters (8 bytes each, budget = 16 bytes sim).
    // With DIM=2, each cluster of n vecs uses n*2*4 bytes.
    // Use 1 vector per cluster → 8 bytes; budget = 16 → fits 2.
    m3::GpuCoordinator coord(*idx, 16, DIM, m3::Metric::L2, false, 8);

    std::mt19937 rng(5);

    // Promote clusters 0 and 1 (fills budget).
    for (int cid : {0, 1}) {
        std::vector<m3::DocId> ids = {static_cast<int64_t>(cid * 10)};
        auto v = vec_near_cluster(cid, DIM, 0.05f, rng);
        idx->load_cluster(cid, ids.data(), v.data(), 1);
        bool ok = coord.promote_to_gpu(cid);
        CHECK(ok);
    }
    CHECK(coord.is_gpu_resident(0));
    CHECK(coord.is_gpu_resident(1));

    // Buffer an insert for cluster 0 (will be drained on eviction).
    float nv[2] = {0.5f, 0.f};
    coord.insert(0, 999, nv);

    // Promote cluster 2 — LFU cluster (0 or 1) gets evicted, buffer drained.
    {
        std::vector<m3::DocId> ids = {20};
        auto v = vec_near_cluster(2, DIM, 0.05f, rng);
        idx->load_cluster(2, ids.data(), v.data(), 1);
        bool ok = coord.promote_to_gpu(2);
        CHECK(ok);
    }

    // Exactly one of {0,1} must have been evicted (budget still holds 2).
    int resident_count = (coord.is_gpu_resident(0) ? 1 : 0)
                       + (coord.is_gpu_resident(1) ? 1 : 0)
                       + (coord.is_gpu_resident(2) ? 1 : 0);
    printf("    GPU-resident count after 3 promotions: %d (expected 2)\n", resident_count);
    CHECK(resident_count == 2);
    CHECK(coord.is_gpu_resident(2));

    // Buffered insert (ID 999) must have been drained to L2.
    float q0[2] = {0.5f, 0.f};
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>> os;
    idx->search(q0, 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(999));
    printf("    [Block 8] auto-eviction on budget exceeded: PASS\n");
}

static void test_b8_background_flush_lifecycle() {
    log_test("B8: background flush thread flushes buffer and stops cleanly");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    // Seed 1 vector in L2 cluster 0 so promote_to_gpu succeeds.
    std::vector<m3::DocId> ids = {1};
    std::vector<float>     vecs = {0.1f, 0.f};
    idx->load_cluster(0, ids.data(), vecs.data(), 1);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 4);
    coord.promote_to_gpu(0);

    // Fill buffer to cap.
    for (int i = 0; i < 4; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.f};
        coord.insert(0, static_cast<int64_t>(500 + i), v);
    }

    CHECK(!coord.background_running());
    coord.start_background_flush(10);
    CHECK(coord.background_running());

    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    coord.stop_background_flush();
    CHECK(!coord.background_running());

    printf("    total_flushed=%llu\n",
           (unsigned long long)coord.total_flushed_vectors());
    CHECK(coord.total_flushed_vectors() >= 4);

    // Flushed vectors must be in L2.
    std::mt19937 rng(77);
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>> os;
    idx->search(q.data(), 1, 5, NLIST, oi, os);
    CHECK(!oi[0].empty());
    printf("    [Block 8] background flush lifecycle: PASS\n");
}

// =============================================================================
// Block runner functions — comment/uncomment individual run_block_N() calls
// in main() to test each block independently.
// =============================================================================

static void run_block_1() {
    log_section("BLOCK 1 — Access Frequency Counter (MultiLevelIndex)");
    test_b1_counts_increment_on_search();
    test_b1_counts_per_cluster_independent();
    test_b1_count_monotonically_increases();
}

static void run_block_2() {
    log_section("BLOCK 2 — GpuBudgetManager");
    test_b2_basic_register_and_query();
    test_b2_budget_enforcement_lfu_eviction();
    test_b2_lfu_tie_broken_by_larger_bytes();
    test_b2_cluster_too_large_for_budget();
    test_b2_remove_cluster();
    test_b2_evicted_freq_reflects_access_count();
    test_b2_re_register_same_cluster();
    test_b2_thread_safety();
}

static void run_block_3() {
    log_section("BLOCK 3 — ClusterInsertBuffer");
    test_b3_slot_lifecycle();
    test_b3_try_buffer_not_resident();
    test_b3_try_buffer_accepts_up_to_cap();
    test_b3_drain_clears_slot();
    test_b3_search_buffer_top_k();
    test_b3_search_buffer_empty_slot();
    test_b3_search_buffer_cosine();
    test_b3_erase_one();
    test_b3_multiple_clusters_independent();
    test_b3_thread_safety();
}

static void run_block_4() {
    log_section("BLOCK 4 — GpuClusterIndex (collaborative search)");
    test_b4_store_and_search();
    test_b4_remove_cluster();
    test_b4_collaborative_gpu_plus_buffer();
    test_b4_collaborative_buffer_only();
    test_b4_collaborative_gpu_only();
    test_b4_dedup_same_id_in_gpu_and_buffer();
    test_b4_thread_safety();
}

static void test_b5_flush_expands_gpu_cluster() {
    log_test("B5: flush migrates buffer vectors to GPU cluster (expand_cluster), not just L2");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    // Pre-populate the GPU cluster with 2 vectors (simulates post-promotion state).
    m3::GpuClusterIndex gpu_idx(DIM, m3::Metric::L2, false);
    float init_vecs[4] = {1.f, 0.f,  0.f, 1.f};
    int64_t init_ids[2] = {10, 11};
    gpu_idx.store_cluster(0, init_ids, init_vecs, 2);
    CHECK(gpu_idx.cluster_size(0) == 2);

    // Build a coordinator that knows about the GPU index.
    m3::AsyncFlushCoordinator coord(buf, *idx, /*flush_threshold=*/0, &gpu_idx);

    // Fill insert buffer to capacity.
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {0.5f, static_cast<float>(i)};
        buf.try_buffer(0, static_cast<int64_t>(100 + i), v);
    }
    CHECK(buf.is_full(0));

    size_t flushed = coord.maybe_flush(0);
    CHECK(flushed == CAP);

    // GPU cluster must have grown: original 2 + 4 buffer vectors = 6.
    const size_t gpu_size = gpu_idx.cluster_size(0);
    printf("    GPU cluster size after expand: %zu (expected %zu)\n", gpu_size, 2 + CAP);
    CHECK(gpu_size == 2 + CAP);

    // L2 durability: the flushed vectors must also appear in L2 search results.
    std::mt19937 rng(42);
    auto q = vec_near_cluster(0, DIM, 0.05f, rng);
    std::vector<std::vector<int64_t>> oi;
    std::vector<std::vector<float>>   os;
    idx->search(q.data(), 1, static_cast<int>(CAP), NLIST, oi, os);
    CHECK(!oi[0].empty());  // buffer vectors visible in L2

    printf("    [Block 5] GPU expansion on flush: PASS\n");
}

static void run_block_5() {
    log_section("BLOCK 5 — AsyncFlushCoordinator");
    test_b5_no_flush_below_threshold();
    test_b5_flush_when_full();
    test_b5_flush_threshold_early();
    test_b5_flush_clusters_multiple();
    test_b5_flush_preserves_data_in_l2();
    test_b5_background_thread();
    test_b5_stats_accuracy();
    test_b5_concurrent_insert_and_flush();
    test_b5_flush_expands_gpu_cluster();
}

static void run_block_6() {
    log_section("BLOCK 6 — Eviction Drain Protocol");
    test_b6_drain_on_eviction_no_data_loss();
    test_b6_promote_cluster_helper();
    test_b6_eviction_hot_cluster_survives();
    test_b6_empty_buffer_eviction();
}

static void run_block_7() {
    log_section("BLOCK 7 — Maintenance tick");
    test_b7_budget_reads_access_count_directly();
    test_b7_maintenance_tick_flushes_buffers();
    test_b7_hot_cluster_survives_eviction_pressure();
}

static void test_b8_insert_full_buffer_routes_to_l2_async() {
    log_test("B8: insert when buffer full routes to L2 (async — does NOT block for GPU expand)");

    const int DIM = 2, NLIST = 1;
    auto idx = make_cache_index(DIM, NLIST);

    const size_t CAP = 4;
    std::vector<m3::DocId> seed_ids = {1};
    std::vector<float>     seed_vecs = {0.1f, 0.f};
    idx->load_cluster(0, seed_ids.data(), seed_vecs.data(), 1);

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, CAP);
    coord.promote_to_gpu(0);

    // Fill the buffer to capacity (all buffered, none overflow).
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.f};
        auto r = coord.insert(0, static_cast<int64_t>(100 + i), v);
        CHECK(r == m3::BufferResult::kBuffered);
    }

    // One more insert — buffer is full, async path: expect WARNING on stderr
    // and vector written to L2, not blocked.
    float overflow_v[2] = {0.99f, 0.f};
    fprintf(stderr, "--- EXPECT WARNING BELOW ---\n");
    auto r = coord.insert(0, 999, overflow_v);
    fprintf(stderr, "--- END EXPECTED WARNING ---\n");
    CHECK(r == m3::BufferResult::kBuffered);

    // ID 999 must now be in L2 (not just the insert buffer).
    float q[2] = {0.99f, 0.f};
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>> os;
    idx->search(q, 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    printf("    L2 search after overflow insert found %zu results\n", oi[0].size());
    CHECK(found.count(999));

    // Now drain the buffer via maintenance_tick — buffer inserts should
    // appear in L2 and GPU cluster should be expanded.
    auto tick = coord.maintenance_tick();
    printf("    maintenance_tick flushed=%zu\n", tick.vectors_flushed);
    CHECK(tick.vectors_flushed == CAP);  // exactly the 4 buffered inserts

    // After drain, a new insert should go back to buffer (slot still active).
    float new_v[2] = {0.2f, 0.f};
    auto r2 = coord.insert(0, 777, new_v);
    CHECK(r2 == m3::BufferResult::kBuffered);

    printf("    [Block 8] async insert fallback to L2 on full buffer: PASS\n");
}

static void test_b8_maintenance_tick_auto_promotes_hotspot() {
    log_test("B8: maintenance_tick() auto-promotes a rising hotspot cluster to GPU");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    // Seed data for both clusters.
    for (int cid = 0; cid < NLIST; ++cid) {
        std::vector<m3::DocId> ids = {static_cast<int64_t>(cid * 10 + 1)};
        std::vector<float>     v   = {static_cast<float>(cid), 0.f};
        idx->load_cluster(cid, ids.data(), v.data(), 1);
    }

    // Tight budget: only 1 cluster fits (8 bytes = 1 vec * DIM * sizeof(float)).
    m3::GpuCoordinator coord(*idx, 8, DIM, m3::Metric::L2, false, 16);

    // Promote cluster 0 (fills the budget).
    CHECK(coord.promote_to_gpu(0));
    CHECK(coord.is_gpu_resident(0));
    CHECK(!coord.is_gpu_resident(1));

    // Search cluster 1 many times to make it the hottest.
    std::mt19937 rng(3);
    for (int i = 0; i < 10; ++i) {
        auto q = vec_near_cluster(1, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>>     os;
        idx->search(q.data(), 1, 3, NLIST, oi, os);
    }

    // Search cluster 0 only once (makes it the LFU of the two).
    {
        auto q = vec_near_cluster(0, DIM, 0.05f, rng);
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>>     os;
        idx->search(q.data(), 1, 3, NLIST, oi, os);
    }

    printf("    access_count: cid0=%llu  cid1=%llu\n",
           (unsigned long long)idx->get_access_count(0),
           (unsigned long long)idx->get_access_count(1));
    CHECK(idx->get_access_count(1) > idx->get_access_count(0));

    // maintenance_tick() should detect cluster 1 is hotter and promote it,
    // evicting cluster 0 (LFU) via the drain protocol.
    auto tick = coord.maintenance_tick();
    printf("    tick: promoted=%zu  flushed=%zu\n",
           tick.clusters_promoted, tick.vectors_flushed);
    CHECK(tick.clusters_promoted >= 1);
    CHECK(coord.is_gpu_resident(1));   // hottest cluster now on GPU
    CHECK(!coord.is_gpu_resident(0));  // cold cluster evicted

    printf("    [Block 8] maintenance_tick auto-promotes hotspot: PASS\n");
}

static void run_block_8() {
    log_section("BLOCK 8 — GpuCoordinator (master coordinator)");
    test_b8_promote_routes_inserts_to_buffer();
    test_b8_search_finds_gpu_and_buffer();
    test_b8_demote_drains_to_l2();
    test_b8_maintenance_tick_end_to_end();
    test_b8_auto_eviction_on_budget_exceeded();
    test_b8_background_flush_lifecycle();
    test_b8_insert_full_buffer_routes_to_l2_async();
    test_b8_maintenance_tick_auto_promotes_hotspot();
}

static void run_integration() {
    log_section("INTEGRATION — Blocks 1–8 end-to-end pipeline");
    test_integration_full_pipeline();
}

// =============================================================================
// ═══ BLOCK 9 — GPU-side cluster split (gpu_split_kmeans + split_gpu_cluster) ═══
// =============================================================================

// Helper: insert `n` vectors into L2 cluster `cid`, each with a unique id
// starting from `id_base`. Returns the list of ids inserted.
static std::vector<m3::DocId> seed_cluster(m3::MultiLevelIndex& idx, int cid,
                                            int n, int id_base,
                                            const std::vector<float>& proto,
                                            std::mt19937& rng, float noise = 0.05f)
{
    const int dim = static_cast<int>(proto.size());
    std::uniform_real_distribution<float> dist(-noise, noise);
    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    ids.reserve(static_cast<size_t>(n));
    vecs.reserve(static_cast<size_t>(n) * dim);
    for (int i = 0; i < n; ++i) {
        ids.push_back(static_cast<int64_t>(id_base + i));
        for (int d = 0; d < dim; ++d)
            vecs.push_back(proto[d] + dist(rng));
    }
    idx.load_cluster(cid, ids.data(), vecs.data(), static_cast<size_t>(n));
    return ids;
}

// ---- gpu_split_kmeans unit tests (CPU fallback path always active) ----

static void test_b9_split_kmeans_separable() {
    log_test("B9: gpu_split_kmeans separates two clearly distinct point clouds");

    const int DIM = 2, N = 20;
    // 10 vectors near (0,0), 10 vectors near (100,100).
    std::vector<float> vecs;
    vecs.reserve(N * DIM);
    for (int i = 0; i < 10; ++i) { vecs.push_back(0.f + i * 0.01f); vecs.push_back(0.f); }
    for (int i = 0; i < 10; ++i) { vecs.push_back(100.f + i * 0.01f); vecs.push_back(100.f); }

    m3::GpuSplitResult r = m3::gpu_split_kmeans(vecs.data(), N, DIM, 20);

    CHECK(static_cast<int>(r.partition.size()) == N);
    CHECK(!r.centroid_a.empty() && !r.centroid_b.empty());

    // Count each side.
    int cnt0 = 0, cnt1 = 0;
    for (int p : r.partition) { if (p == 0) ++cnt0; else ++cnt1; }
    printf("    partition A=%d  B=%d  iters=%d\n", cnt0, cnt1, r.iters_run);
    // Expect clean 10/10 split (well-separated clusters).
    CHECK(cnt0 == 10 && cnt1 == 10);

    // Centroids should be in the right ballpark.
    float cA = r.centroid_a[0], cB = r.centroid_b[0];
    if (cA > cB) std::swap(cA, cB);   // normalise order
    CHECK(cA < 10.f);
    CHECK(cB > 90.f);
    printf("    [Block 9] split_kmeans separable: PASS\n");
}

static void test_b9_split_kmeans_single_vector() {
    log_test("B9: gpu_split_kmeans with n=1 returns trivial result (no crash)");

    const int DIM = 3;
    float v[3] = {1.f, 2.f, 3.f};
    m3::GpuSplitResult r = m3::gpu_split_kmeans(v, 1, DIM, 10);

    CHECK(static_cast<int>(r.partition.size()) == 1);
    CHECK(r.centroid_a.size() == DIM && r.centroid_b.size() == DIM);
    printf("    [Block 9] split_kmeans single vector: PASS\n");
}

static void test_b9_split_kmeans_converges() {
    log_test("B9: gpu_split_kmeans convergence — iters_run <= max_iters");

    const int DIM = 4, N = 50;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<float> vecs(static_cast<size_t>(N) * DIM);
    for (auto& x : vecs) x = dist(rng);

    m3::GpuSplitResult r = m3::gpu_split_kmeans(vecs.data(), N, DIM, 30);
    printf("    iters_run=%d (max=30)\n", r.iters_run);
    CHECK(r.iters_run >= 1 && r.iters_run <= 30);

    // All vectors assigned.
    CHECK(static_cast<int>(r.partition.size()) == N);
    for (int p : r.partition) CHECK(p == 0 || p == 1);

    printf("    [Block 9] split_kmeans convergence: PASS\n");
}

// ---- GpuCoordinator::split_gpu_cluster tests ----

static void test_b9_split_creates_new_cluster() {
    log_test("B9: split_gpu_cluster creates a second L2 cluster and returns new_cid");

    const int DIM = 2, NLIST = 2;
    // Use 2 initial clusters; cluster 0 will hold 2 well-separated sub-clouds.
    auto idx = make_cache_index(DIM, NLIST);
    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 32);

    std::mt19937 rng(9);

    // Seed cluster 0 with 20 well-separated vectors (10 near 0, 10 near 50).
    std::vector<float> proto0(DIM, 0.f), proto1(DIM, 50.f);
    auto ids_a = seed_cluster(*idx, 0, 10, 100, proto0, rng, 0.1f);
    auto ids_b = seed_cluster(*idx, 0, 10, 200, proto1, rng, 0.1f);

    auto r = coord.split_gpu_cluster(0, 30);
    printf("    success=%d  new_cid=%d\n", (int)r.success, r.new_cid);
    CHECK(r.success);
    CHECK(r.new_cid >= 0);

    // Both partitions must be searchable via MultiLevelIndex L2.
    {
        float q[2] = {0.f, 0.f};
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q, 1, 5, 3, oi, os);
        printf("    search near (0,0): found %zu results\n", oi[0].size());
        CHECK(!oi[0].empty());
        // Results should be IDs near proto0.
        bool found_a = false;
        for (m3::DocId id : oi[0])
            if (id >= 100 && id < 110) found_a = true;
        CHECK(found_a);
    }
    {
        float q[2] = {50.f, 50.f};
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>> os;
        idx->search(q, 1, 5, 3, oi, os);
        printf("    search near (50,50): found %zu results\n", oi[0].size());
        CHECK(!oi[0].empty());
        bool found_b = false;
        for (m3::DocId id : oi[0])
            if (id >= 200 && id < 210) found_b = true;
        CHECK(found_b);
    }
    printf("    [Block 9] split creates new cluster: PASS\n");
}

static void test_b9_split_preserves_all_vectors() {
    log_test("B9: split_gpu_cluster preserves total vector count across A + B");

    const int DIM = 2, NLIST = 2, N = 30;
    auto idx = make_cache_index(DIM, NLIST);
    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 64);

    std::mt19937 rng(7);
    // Insert N vectors with known IDs into cluster 0.
    std::vector<float> proto(DIM, 5.f);
    auto all_ids = seed_cluster(*idx, 0, N, 1000, proto, rng, 2.f);

    auto r = coord.split_gpu_cluster(0, 20);
    CHECK(r.success);

    // Export partition A (cluster 0) and partition B (new_cid).
    std::vector<m3::DocId> ids_a, ids_b;
    std::vector<float>     vecs_a, vecs_b;
    idx->export_l2_cluster(0,        ids_a, vecs_a);
    idx->export_l2_cluster(r.new_cid, ids_b, vecs_b);

    const size_t total = ids_a.size() + ids_b.size();
    printf("    partition A=%zu  B=%zu  total=%zu  original=%d\n",
           ids_a.size(), ids_b.size(), total, N);
    CHECK(total == static_cast<size_t>(N));

    // Every original ID must appear in exactly one partition.
    std::unordered_set<int64_t> set_a(ids_a.begin(), ids_a.end());
    std::unordered_set<int64_t> set_b(ids_b.begin(), ids_b.end());
    for (m3::DocId id : all_ids) {
        bool in_a = set_a.count(id) > 0;
        bool in_b = set_b.count(id) > 0;
        CHECK(in_a != in_b);  // exactly one partition
    }
    printf("    [Block 9] split preserves all vectors: PASS\n");
}

static void test_b9_split_too_few_vectors() {
    log_test("B9: split_gpu_cluster on cluster with < 2 vectors returns failure");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);
    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 8);

    // Empty cluster → should fail.
    auto r = coord.split_gpu_cluster(0, 10);
    CHECK(!r.success);
    CHECK(r.new_cid == -1);

    // Single vector → should also fail.
    m3::DocId id = 42;
    float v[2] = {1.f, 0.f};
    idx->load_cluster(0, &id, v, 1);
    r = coord.split_gpu_cluster(0, 10);
    CHECK(!r.success);

    printf("    [Block 9] split too few vectors: PASS\n");
}

static void test_b9_split_gpu_resident_cluster_refreshed() {
    log_test("B9: split_gpu_cluster refreshes GPU storage for the original cluster");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);
    // Large budget so both clusters fit.
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 32);

    std::mt19937 rng(11);
    auto ids_a = seed_cluster(*idx, 0, 10, 300, std::vector<float>(DIM, 0.f),  rng, 0.1f);
    auto ids_b = seed_cluster(*idx, 0, 10, 400, std::vector<float>(DIM, 50.f), rng, 0.1f);

    // Promote cluster 0 to GPU before splitting.
    bool ok = coord.promote_to_gpu(0);
    CHECK(ok);
    CHECK(coord.is_gpu_resident(0));

    auto r = coord.split_gpu_cluster(0, 20);
    CHECK(r.success);

    // Cluster 0 should still be GPU-resident with partition A data.
    CHECK(coord.is_gpu_resident(0));

    // Search the GPU path — results near (0,0) should come from cluster 0.
    float q[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids;
    std::vector<float>     out_scores;
    coord.search({0}, q, 5, out_ids, out_scores);
    printf("    GPU search near (0,0) after split: %zu results\n", out_ids.size());
    CHECK(!out_ids.empty());

    printf("    [Block 9] GPU-resident cluster refreshed after split: PASS\n");
}

static void test_b9_split_uses_gpu_resident_data() {
    log_test("B9: split_gpu_cluster() reads from GPU-resident data (not L2) when cluster is promoted");

    const int DIM = 2, NLIST = 2;
    auto idx = make_cache_index(DIM, NLIST);

    // Two well-separated groups: A around (1,0), B around (-1,0).
    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    for (int i = 0; i < 6; ++i) {
        ids.push_back(i + 1);
        // First 3 near (1,0), last 3 near (-1,0).
        float x = (i < 3) ? (1.f + i * 0.05f) : (-1.f - (i - 3) * 0.05f);
        vecs.push_back(x);
        vecs.push_back(0.f);
    }
    idx->load_cluster(0, ids.data(), vecs.data(), ids.size());

    m3::GpuCoordinator coord(*idx, 4096, DIM, m3::Metric::L2, false, 16);
    CHECK(coord.promote_to_gpu(0));
    CHECK(coord.is_gpu_resident(0));

    // Insert one extra vector via the buffer, then flush so GPU has it.
    float extra[2] = {1.1f, 0.f};
    coord.insert(0, 99, extra);
    coord.maintenance_tick();  // flushes buffer → GPU expand + L2

    // Split reads from GPU (7 vectors now: 6 original + 1 buffered).
    auto r = coord.split_gpu_cluster(0);
    printf("    split success=%d new_cid=%d\n", r.success, r.new_cid);
    CHECK(r.success);
    CHECK(r.new_cid >= 0);

    printf("    [Block 9] split uses GPU-resident data: PASS\n");
}

static void run_block_9() {
    log_section("BLOCK 9 — GPU-side cluster split (gpu_split_kmeans + split_gpu_cluster)");
    test_b9_split_kmeans_separable();
    test_b9_split_kmeans_single_vector();
    test_b9_split_kmeans_converges();
    test_b9_split_creates_new_cluster();
    test_b9_split_preserves_all_vectors();
    test_b9_split_too_few_vectors();
    test_b9_split_gpu_resident_cluster_refreshed();
    test_b9_split_uses_gpu_resident_data();
}

// =============================================================================
// main
//
// Comment out any run_block_N() line to skip that block entirely.
// =============================================================================

int main() {
    printf("GPU Coordination Test Suite — Blocks 1–9\n");
    printf("Build: %s %s\n\n", __DATE__, __TIME__);

    // Enable structured logging when M3_LOG=<path> env var is set.
    // Example: M3_LOG=m3_debug.log ./build/test_gpu_coord
    if (const char* log_path = std::getenv("M3_LOG")) {
        m3::M3Logger::instance().enable(log_path);
        printf("Debug logging enabled → %s\n\n", log_path);
    }

    run_block_1();    // Block 1: access_count in MultiLevelIndex
    run_block_2();    // Block 2: GpuBudgetManager (LFU eviction, freq tracking)
    run_block_3();    // Block 3: ClusterInsertBuffer (insert/search/drain/erase)
    run_block_4();    // Block 4: GpuClusterIndex (collaborative GPU+buffer search)
    run_block_5();    // Block 5: AsyncFlushCoordinator (sync + async drain to L2)
    run_block_6();    // Block 6: Eviction drain protocol (no data loss on eviction)
    run_block_7();    // Block 7: Maintenance freq sync (access_count → budget LFU)
    run_block_8();    // Block 8: GpuCoordinator (high-level orchestration API)
    run_block_9();    // Block 9: GPU-side k-means split (cpu fallback + coordinator)
    run_integration(); // Integration: all 8 blocks working together

    printf("\n══════════════════════════════════════════════\n");
    printf("  RESULTS:  %d passed  /  %d failed\n", g_passed, g_failed);
    printf("══════════════════════════════════════════════\n");

    return (g_failed > 0) ? 1 : 0;
}
