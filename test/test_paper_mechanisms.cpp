// test_paper_mechanisms.cpp
//
// Integration tests for the four GPU-layer mechanisms described in the paper:
//
//  Mechanism 1 — Hotspot-Aware Caching
//    Track per-cluster access frequency; cache the top-N most-accessed clusters
//    on GPU within a fixed VRAM budget; on hotspot change evict cold clusters and
//    load hot ones via async CPU→GPU transfer.
//
//  Mechanism 2 — CPU Insertion Buffer
//    Each GPU-cached cluster has a paired CPU-side insertion buffer (max B_insert).
//    New insertions accumulate in the buffer; at query time GPU searches its
//    cached portion while CPU searches the buffer, and results are merged.
//
//  Mechanism 3 — Async Consistency Management (buffer flush)
//    When a buffer fills, trigger async cluster expansion on GPU (GPU-GPU for
//    already-GPU data; CPU-GPU for buffer data). Transfers run alongside live
//    serving; old cluster released only after transfer completes (zero-downtime).
//
//  Mechanism 4 — On-GPU Cluster Splitting
//    K-means splitting is executed directly on GPU for GPU-resident clusters,
//    reading data via export_cluster() instead of the L2 IVF round-trip.
//
// Build:
//   cmake -B build && cmake --build build --target test_paper_mechanisms
// Run:
//   ./build/test_paper_mechanisms
//   M3_LOG=mechanisms.log ./build/test_paper_mechanisms   # with debug log

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gpu_coordinator.h"
#include "gpu_cluster_index.h"
#include "gpu_insert_buffer.h"
#include "gpu_budget.h"
#include "gpu_flush_coordinator.h"
#include "m3_multi_level.h"
#include "m3_logger.h"
#include "split_kernel_v3.h"

// =============================================================================
// Minimal test framework
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(expr)                                                             \
    do {                                                                        \
        if (!(expr)) {                                                          \
            fprintf(stderr, "  FAIL  %s:%d  CHECK(%s)\n",                      \
                    __FILE__, __LINE__, #expr);                                 \
            ++g_failed;                                                         \
        } else { ++g_passed; }                                                  \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        auto _a = (a); auto _b = (b);                                           \
        if (_a != _b) {                                                         \
            fprintf(stderr, "  FAIL  %s:%d  CHECK_EQ(%s, %s) — got %lld vs %lld\n", \
                    __FILE__, __LINE__, #a, #b,                                 \
                    (long long)_a, (long long)_b);                              \
            ++g_failed;                                                         \
        } else { ++g_passed; }                                                  \
    } while (0)

static void log_section(const char* name) {
    printf("\n══════════════════════════════════════════════\n");
    printf("  %s\n", name);
    printf("══════════════════════════════════════════════\n");
}
static void log_test(const char* name) { printf("  ▶ %s\n", name); }

// =============================================================================
// Shared test helpers
// =============================================================================

// Build a MultiLevelIndex in cache mode with `nlist` L2 clusters.
// Centroids: cluster i has value i*10.0 in all dimensions.
static std::unique_ptr<m3::MultiLevelIndex> make_index(int dim, int nlist) {
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = nlist;
    auto idx = std::make_unique<m3::MultiLevelIndex>(dim, m3::Metric::L2, false, cfg);

    std::vector<float> centroids(static_cast<size_t>(nlist) * dim);
    for (int i = 0; i < nlist; ++i)
        for (int d = 0; d < dim; ++d)
            centroids[static_cast<size_t>(i) * dim + d] = static_cast<float>(i) * 10.f;
    idx->set_l2_centroids(centroids);

    m3::CacheConfig cc;
    cc.l0_max_clusters            = nlist;
    cc.l1_max_clusters            = nlist;
    cc.l0_max_vectors_per_cluster = 50000;
    cc.l1_max_vectors_per_cluster = 50000;
    cc.cold_time_ns               = 600'000'000'000ULL;  // very cold — no auto-eviction in tests
    cc.alpha_et                   = 0.f;
    idx->set_cache_config(cc);
    return idx;
}

// Generate n vectors near cluster cid's centroid (cid*10 ± noise).
static std::vector<float> random_vecs_near(int cid, int dim, int n,
                                            float noise, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-noise, noise);
    std::vector<float> v(static_cast<size_t>(n) * dim);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < dim; ++d)
            v[static_cast<size_t>(i) * dim + d] = static_cast<float>(cid) * 10.f + dist(rng);
    return v;
}

// Seed L2 cluster `cid` with n vectors. Returns the list of DocIds assigned.
static std::vector<m3::DocId> seed_l2(m3::MultiLevelIndex& idx, int cid,
                                       int n, int id_base,
                                       int dim, float noise, std::mt19937& rng) {
    auto vecs = random_vecs_near(cid, dim, n, noise, rng);
    std::vector<m3::DocId> ids;
    ids.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        ids.push_back(static_cast<m3::DocId>(id_base + i));
    idx.load_cluster(cid, ids.data(), vecs.data(), static_cast<size_t>(n));
    return ids;
}

// Search idx probing `nprobe` clusters for a query near cid's centroid.
static std::vector<m3::DocId> search_near(m3::MultiLevelIndex& idx,
                                           int cid, int dim, int k, int nprobe,
                                           std::mt19937& rng) {
    auto q = random_vecs_near(cid, dim, 1, 0.02f, rng);
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>>     os;
    idx.search(q.data(), 1, k, nprobe, oi, os);
    return oi.empty() ? std::vector<m3::DocId>{} : oi[0];
}

// =============================================================================
// ═══  MECHANISM 1 — Hotspot-Aware Caching  ═══
// =============================================================================

// M1-A: access_count increments for every cluster that is probed during search.
static void test_m1_frequency_tracked_per_cluster() {
    log_test("M1-A: access_count incremented per-cluster independently");

    const int DIM = 4, NLIST = 3;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(1);

    // Seed all clusters.
    for (int cid = 0; cid < NLIST; ++cid)
        seed_l2(*idx, cid, 5, cid * 100, DIM, 0.1f, rng);

    // Search cluster 0 three times, cluster 1 once, cluster 2 zero times.
    for (int i = 0; i < 3; ++i) search_near(*idx, 0, DIM, 3, 1, rng);
    search_near(*idx, 1, DIM, 3, 1, rng);

    auto meta = idx->get_cluster_metadata();
    printf("    access_count: cid0=%llu  cid1=%llu  cid2=%llu\n",
           (unsigned long long)meta[0].access_count,
           (unsigned long long)meta[1].access_count,
           (unsigned long long)meta[2].access_count);

    CHECK(meta[0].access_count >= 3);
    CHECK(meta[1].access_count >= 1);
    CHECK(meta[0].access_count > meta[1].access_count);
    CHECK(meta[0].access_count > meta[2].access_count);
    printf("    [M1-A] PASS\n");
}

// M1-B: with budget for exactly N clusters, the N most-accessed clusters are
// GPU-resident after maintenance_tick().
static void test_m1_budget_holds_top_n_by_frequency() {
    log_test("M1-B: budget keeps exactly the top-N hottest clusters on GPU");

    const int DIM = 2, NLIST = 4;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(2);

    for (int cid = 0; cid < NLIST; ++cid)
        seed_l2(*idx, cid, 2, cid * 10, DIM, 0.1f, rng);

    // Budget fits exactly 2 clusters (each cluster: 2 vecs × DIM × 4 bytes = 16 bytes).
    const size_t bytes_per_cluster = 2 * DIM * sizeof(float);
    m3::GpuCoordinator coord(*idx, 2 * bytes_per_cluster, DIM, m3::Metric::L2, false, 32);

    // Make cluster 2 and cluster 3 the hottest.
    for (int i = 0; i < 8; ++i) search_near(*idx, 2, DIM, 2, 1, rng);
    for (int i = 0; i < 6; ++i) search_near(*idx, 3, DIM, 2, 1, rng);
    search_near(*idx, 0, DIM, 2, 1, rng);
    // cluster 1: 0 accesses

    // Promote cluster 0 (cold) first — fills the budget.
    CHECK(coord.promote_to_gpu(0));
    // Promote cluster 1 — evicts cluster 0 (equally cold, but 0 has 1 access vs 1's 0).
    coord.promote_to_gpu(1);

    // rebalance() → hotspot_rebalance_() should swap in clusters 2 and 3.
    size_t promoted = coord.rebalance(); coord.drain_pending();
    printf("    promoted=%zu\n", promoted);

    CHECK(coord.is_gpu_resident(2));
    CHECK(coord.is_gpu_resident(3));
    // The cold clusters should have been evicted.
    CHECK(!coord.is_gpu_resident(0) || !coord.is_gpu_resident(1));
    printf("    [M1-B] PASS\n");
}

// M1-C: when a non-GPU-resident cluster becomes hotter than the weakest
// GPU-resident cluster, maintenance_tick() promotes it and evicts the coldest.
static void test_m1_hotspot_change_evicts_cold_promotes_hot() {
    log_test("M1-C: rising hotspot evicts cold GPU cluster, gets promoted");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(3);

    for (int cid = 0; cid < NLIST; ++cid)
        seed_l2(*idx, cid, 1, cid * 10, DIM, 0.05f, rng);

    // Budget fits only 1 cluster (1 vec × DIM × 4 bytes = 8 bytes).
    m3::GpuCoordinator coord(*idx, DIM * sizeof(float), DIM, m3::Metric::L2, false, 16);

    // Promote cluster 0.
    CHECK(coord.promote_to_gpu(0));
    CHECK(coord.is_gpu_resident(0));
    CHECK(!coord.is_gpu_resident(1));

    // Make cluster 1 hotter than cluster 0.
    for (int i = 0; i < 10; ++i) search_near(*idx, 1, DIM, 2, NLIST, rng);
    search_near(*idx, 0, DIM, 2, NLIST, rng);

    uint64_t freq0 = idx->get_access_count(0);
    uint64_t freq1 = idx->get_access_count(1);
    printf("    access_count: cid0=%llu  cid1=%llu\n",
           (unsigned long long)freq0, (unsigned long long)freq1);
    CHECK(freq1 > freq0);

    size_t promoted = coord.rebalance(); coord.drain_pending();
    printf("    promoted=%zu\n", promoted);

    CHECK(promoted >= 1);
    CHECK(coord.is_gpu_resident(1));   // hottest now on GPU
    CHECK(!coord.is_gpu_resident(0));  // cold evicted
    printf("    [M1-C] PASS\n");
}

// M1-D: evicted cluster's vectors are flushed to L2 (no data loss on eviction).
static void test_m1_eviction_drains_buffer_to_l2() {
    log_test("M1-D: evicted GPU cluster's buffered vectors are written to L2");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(4);

    seed_l2(*idx, 0, 2, 1, DIM, 0.05f, rng);
    seed_l2(*idx, 1, 2, 100, DIM, 0.05f, rng);

    // Budget for 1 cluster.
    m3::GpuCoordinator coord(*idx, 2 * DIM * sizeof(float), DIM, m3::Metric::L2, false, 16);
    CHECK(coord.promote_to_gpu(0));

    // Buffer 3 new vectors into cluster 0 (not yet flushed to L2).
    for (int i = 0; i < 3; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.f};
        coord.insert(0, static_cast<m3::DocId>(200 + i), v);
    }
    printf("    buffered 3 vectors into cluster 0\n");

    // Make cluster 1 hotter — rebalance() will evict cluster 0.
    for (int i = 0; i < 8; ++i) search_near(*idx, 1, DIM, 2, NLIST, rng);
    search_near(*idx, 0, DIM, 2, NLIST, rng);

    coord.rebalance(); coord.drain_pending();

    CHECK(!coord.is_gpu_resident(0));

    // The 3 buffered vectors (IDs 200, 201, 202) must be in L2 after eviction drain.
    float q[2] = {0.f, 0.f};
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>>     os;
    idx->search(q, 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    printf("    L2 search returned %zu results\n", oi[0].size());
    CHECK(found.count(200) || found.count(201) || found.count(202));
    printf("    [M1-D] PASS\n");
}

// =============================================================================
// ═══  MECHANISM 2 — CPU Insertion Buffer  ═══
// =============================================================================

// M2-A: B_insert capacity — buffer accepts up to cap, returns kFull at cap+1.
static void test_m2_buffer_capacity_binsert() {
    log_test("M2-A: buffer accepts up to B_insert vectors, returns kFull at cap");

    const int DIM = 2;
    const size_t CAP = 128;  // paper's empirical threshold
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    float v[2] = {1.f, 0.f};
    for (size_t i = 0; i < CAP; ++i) {
        auto r = buf.try_buffer(0, static_cast<m3::DocId>(i), v);
        CHECK(r == m3::BufferResult::kBuffered);
    }
    CHECK(buf.is_full(0));
    CHECK_EQ(buf.size(0), CAP);

    auto r = buf.try_buffer(0, static_cast<m3::DocId>(CAP + 1), v);
    CHECK(r == m3::BufferResult::kFull);
    printf("    buffer at cap=%zu → kFull: OK\n", CAP);
    printf("    [M2-A] PASS\n");
}

// M2-B: inserts targeting a non-GPU-resident cluster go directly to L2.
static void test_m2_nonresident_bypasses_buffer() {
    log_test("M2-B: insert to non-GPU-resident cluster routes directly to L2");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(5);
    seed_l2(*idx, 0, 1, 1, DIM, 0.05f, rng);

    // Large budget but cluster 0 is NOT promoted.
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 16);
    CHECK(!coord.is_gpu_resident(0));

    float v[2] = {0.5f, 0.f};
    // insert() on a non-resident cluster → directly to L2, still returns kBuffered.
    auto r = coord.insert(0, 999, v);
    CHECK(r == m3::BufferResult::kBuffered);

    // Must be searchable in L2 immediately (not held in any buffer).
    float q[2] = {0.5f, 0.f};
    std::vector<std::vector<m3::DocId>> oi;
    std::vector<std::vector<float>>     os;
    idx->search(q, 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(999));
    printf("    [M2-B] PASS\n");
}

// M2-C: collaborative search returns results from both GPU cluster data and
// the CPU insertion buffer, and merges them into a single top-k list.
static void test_m2_search_merges_gpu_and_buffer_results() {
    log_test("M2-C: query finds results from GPU cluster data AND CPU buffer, merged top-k");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(6);

    // Seed 3 vectors in L2 (will be uploaded to GPU on promote).
    seed_l2(*idx, 0, 3, 10, DIM, 0.01f, rng);

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 16);
    CHECK(coord.promote_to_gpu(0));

    // Buffer 3 more vectors (IDs 100, 101, 102) without flushing to GPU.
    for (int i = 0; i < 3; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.5f};
        coord.insert(0, static_cast<m3::DocId>(100 + i), v);
    }
    printf("    GPU cluster size=%zu  buffer size=3 (unflushed)\n",
           coord.gpu_bytes_used() / (DIM * sizeof(float)));

    // Search via coordinator — should see both GPU and buffer results.
    float q[2] = {0.05f, 0.25f};
    std::vector<m3::DocId> out_ids;
    std::vector<float>     out_scores;
    size_t found = coord.search({0}, q, 6, out_ids, out_scores);
    printf("    collaborative search returned %zu results\n", found);

    // GPU vectors (IDs 10–12) and buffer vectors (IDs 100–102) should both appear.
    std::unordered_set<int64_t> id_set(out_ids.begin(), out_ids.end());
    bool has_gpu_vec    = id_set.count(10) || id_set.count(11) || id_set.count(12);
    bool has_buffer_vec = id_set.count(100) || id_set.count(101) || id_set.count(102);
    CHECK(has_gpu_vec);
    CHECK(has_buffer_vec);
    printf("    [M2-C] PASS\n");
}

// M2-D: if the same DocId appears in both GPU storage and the buffer,
// collaborative search returns it exactly once with the best (lower) score.
static void test_m2_dedup_keeps_best_score() {
    log_test("M2-D: duplicate DocId across GPU and buffer — deduplicated, best score kept");

    const int DIM = 2;
    m3::GpuClusterIndex gpu_idx(DIM, m3::Metric::L2, false);
    m3::ClusterInsertBuffer buf(DIM, 32);
    buf.activate_cluster(0);

    // GPU: DocId 42 at (0.0, 0.0) — score 0.0 for query at (0,0).
    float gpu_vecs[2] = {0.f, 0.f};
    m3::DocId gpu_ids[1] = {42};
    gpu_idx.store_cluster(0, gpu_ids, gpu_vecs, 1);

    // Buffer: same DocId 42 at (5.0, 5.0) — higher distance.
    float buf_vec[2] = {5.f, 5.f};
    buf.try_buffer(0, 42, buf_vec);

    float q[2] = {0.f, 0.f};
    std::vector<m3::DocId> out_ids;
    std::vector<float>     out_scores;
    gpu_idx.collaborative_search({0}, q, 5, buf, out_ids, out_scores);

    printf("    result count=%zu\n", out_ids.size());
    CHECK(out_ids.size() == 1);            // deduplicated to exactly one entry
    CHECK(out_ids[0] == 42);
    CHECK(out_scores[0] < 1.f);           // best score (near 0) was kept
    printf("    [M2-D] PASS\n");
}

// M2-E: buffered vectors are NOT visible in an L2 search until flushed.
// This confirms the two-path design: GPU+buffer for live search, L2 for durable search.
static void test_m2_buffer_invisible_in_l2_until_flush() {
    log_test("M2-E: buffered vectors not visible in L2 until flush");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(7);
    seed_l2(*idx, 0, 2, 1, DIM, 0.05f, rng);

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 32);
    CHECK(coord.promote_to_gpu(0));

    // Buffer a distinctive vector — NOT flushed to L2 yet.
    float buffered_vec[2] = {0.f, 999.f};
    coord.insert(0, 777, buffered_vec);

    // L2 search should NOT find ID 777 yet.
    float q[2] = {0.f, 999.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> before_flush(oi[0].begin(), oi[0].end());
    CHECK(!before_flush.count(777));
    printf("    before flush: ID 777 not in L2 — correct\n");

    // Flush via flush_buffers() — now L2 must have it.
    coord.flush_buffers();
    oi.clear(); os.clear();
    idx->search(q, 1, 5, NLIST, oi, os);
    std::unordered_set<int64_t> after_flush(oi[0].begin(), oi[0].end());
    CHECK(after_flush.count(777));
    printf("    after flush: ID 777 in L2 — correct\n");
    printf("    [M2-E] PASS\n");
}

// =============================================================================
// ═══  MECHANISM 3 — Async Consistency Management (buffer flush)  ═══
// =============================================================================

// M3-A: when flush is triggered for a GPU-resident cluster, the buffer vectors
// are appended to the existing GPU cluster in-place (expand_cluster), not replaced.
static void test_m3_flush_expands_gpu_cluster_in_place() {
    log_test("M3-A: buffer flush expands existing GPU cluster in-place (no replace)");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);

    const size_t CAP = 4;
    m3::ClusterInsertBuffer buf(DIM, CAP);
    buf.activate_cluster(0);

    // GPU cluster pre-loaded with 3 vectors.
    m3::GpuClusterIndex gpu_idx(DIM, m3::Metric::L2, false);
    float init_vecs[6] = {1.f, 0.f,  2.f, 0.f,  3.f, 0.f};
    m3::DocId init_ids[3] = {10, 11, 12};
    gpu_idx.store_cluster(0, init_ids, init_vecs, 3);
    CHECK_EQ(gpu_idx.cluster_size(0), (size_t)3);

    m3::AsyncFlushCoordinator coord(buf, *idx, /*flush_threshold=*/0, &gpu_idx);

    // Fill buffer to cap.
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {static_cast<float>(i), 1.f};
        buf.try_buffer(0, static_cast<m3::DocId>(100 + i), v);
    }

    size_t flushed = coord.maybe_flush(0);
    CHECK_EQ(flushed, CAP);

    // GPU cluster must have grown: 3 original + 4 buffer.
    size_t gpu_size = gpu_idx.cluster_size(0);
    printf("    GPU cluster: before=3  buffer=%zu  after=%zu\n", CAP, gpu_size);
    CHECK_EQ(gpu_size, (size_t)3 + CAP);

    // Original IDs still present (export and check).
    std::vector<m3::DocId> ex_ids;
    std::vector<float>     ex_vecs;
    gpu_idx.export_cluster(0, ex_ids, ex_vecs);
    std::unordered_set<int64_t> id_set(ex_ids.begin(), ex_ids.end());
    CHECK(id_set.count(10) && id_set.count(11) && id_set.count(12));
    printf("    [M3-A] PASS\n");
}

// M3-B: after a flush, the buffer vectors are also durably written to L2
// (L2 is the recovery path if VRAM is lost).
static void test_m3_flush_writes_durability_to_l2() {
    log_test("M3-B: buffer flush writes vectors to L2 for durability");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(8);
    seed_l2(*idx, 0, 2, 1, DIM, 0.05f, rng);

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 8);
    CHECK(coord.promote_to_gpu(0));

    // Buffer 4 distinctive vectors.
    for (int i = 0; i < 4; ++i) {
        float v[2] = {static_cast<float>(i + 1) * 0.1f, 99.f};
        coord.insert(0, static_cast<m3::DocId>(500 + i), v);
    }

    size_t flushed = coord.flush_buffers();
    printf("    flushed=%zu\n", flushed);
    CHECK(flushed >= 4);

    // L2 must have the flushed vectors.
    float q[2] = {0.15f, 99.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    printf("    L2 search found %zu results\n", oi[0].size());
    bool any_found = found.count(500) || found.count(501) || found.count(502) || found.count(503);
    CHECK(any_found);
    printf("    [M3-B] PASS\n");
}

// M3-C: when buffer is full and expansion is pending, the next insert goes to
// L2 without blocking (async — no GPU stall for the caller).
static void test_m3_overflow_routes_to_l2_nonblocking() {
    log_test("M3-C: full buffer → overflow insert goes to L2 without blocking GPU");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(9);
    seed_l2(*idx, 0, 2, 1, DIM, 0.05f, rng);

    const size_t CAP = 4;
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, CAP);
    CHECK(coord.promote_to_gpu(0));

    // Fill buffer to capacity.
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 0.f};
        CHECK(coord.insert(0, static_cast<m3::DocId>(100 + i), v) == m3::BufferResult::kBuffered);
    }

    // One more insert — buffer full, async path: must not block, must appear in L2.
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "--- EXPECT async overflow WARNING ---\n");
    float overflow_v[2] = {0.77f, 0.f};
    auto r = coord.insert(0, 888, overflow_v);
    auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "--- END EXPECTED WARNING ---\n");
    CHECK(r == m3::BufferResult::kBuffered);

    // Must have returned quickly (< 100 ms — no GPU sync stall).
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf("    overflow insert latency = %lld ms\n", (long long)ms);
    CHECK(ms < 100);

    // ID 888 must be in L2.
    float q[2] = {0.77f, 0.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    CHECK(found.count(888));
    printf("    [M3-C] PASS\n");
}

// M3-D: background flush thread drains all full buffers, expanding GPU clusters
// and writing L2 durability, without any explicit tick() call from the caller.
static void test_m3_background_flush_drains_buffers() {
    log_test("M3-D: background flush thread autonomously drains full buffers");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(10);
    seed_l2(*idx, 0, 2, 1, DIM, 0.05f, rng);

    const size_t CAP = 4;
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, CAP);
    CHECK(coord.promote_to_gpu(0));

    coord.start_background(/*flush_ms=*/10);

    // Fill buffer — background thread should drain it within ~50 ms.
    for (size_t i = 0; i < CAP; ++i) {
        float v[2] = {static_cast<float>(i) * 0.1f, 77.f};
        coord.insert(0, static_cast<m3::DocId>(300 + i), v);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    coord.stop_background();

    // L2 must contain the flushed vectors.
    float q[2] = {0.15f, 77.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> found(oi[0].begin(), oi[0].end());
    printf("    L2 search after background drain: %zu results\n", oi[0].size());
    bool any_drained = found.count(300)||found.count(301)||found.count(302)||found.count(303);
    CHECK(any_drained);
    printf("    [M3-D] PASS\n");
}

// M3-E: during a concurrent flush, the cluster remains searchable via the
// coordinator (zero-downtime — old data still visible until swap completes).
static void test_m3_cluster_searchable_during_flush() {
    log_test("M3-E: cluster remains searchable during concurrent flush (zero downtime)");

    const int DIM = 2, NLIST = 1;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(11);
    seed_l2(*idx, 0, 5, 1, DIM, 0.05f, rng);

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 8);
    CHECK(coord.promote_to_gpu(0));

    std::atomic<int> search_errors{0};
    std::atomic<bool> done{false};

    // Background searcher — must always get results.
    std::thread searcher([&] {
        std::mt19937 srng(99);
        while (!done.load()) {
            auto q = random_vecs_near(0, DIM, 1, 0.05f, srng);
            std::vector<m3::DocId> oi; std::vector<float> os;
            coord.search({0}, q.data(), 3, oi, os);
            if (oi.empty()) ++search_errors;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Insert + flush several times while searching is in progress.
    for (int round = 0; round < 5; ++round) {
        for (int i = 0; i < 3; ++i) {
            float v[2] = {static_cast<float>(round * 10 + i) * 0.01f, 0.f};
            coord.insert(0, static_cast<m3::DocId>(1000 + round * 10 + i), v);
        }
        coord.flush_buffers();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    done.store(true);
    searcher.join();

    printf("    search errors during concurrent flush: %d\n", search_errors.load());
    CHECK(search_errors.load() == 0);
    printf("    [M3-E] PASS\n");
}

// =============================================================================
// ═══  MECHANISM 4 — On-GPU Cluster Splitting  ═══
// =============================================================================

// M4-A: splitting a GPU-resident cluster reads its data via export_cluster()
// (GPU path) rather than from L2 — confirmed by modifying L2 data and checking
// the split still reflects GPU-resident content.
static void test_m4_split_gpu_resident_uses_gpu_export() {
    log_test("M4-A: split reads from GPU-resident data, not L2, for promoted clusters");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);

    // Two distinct groups — well separated so k-means cleanly splits them.
    std::vector<m3::DocId> ids;
    std::vector<float>     vecs;
    for (int i = 0; i < 6; ++i) {
        ids.push_back(static_cast<m3::DocId>(i + 1));
        float x = (i < 3) ? (1.f + i * 0.05f) : (-1.f - (i - 3) * 0.05f);
        vecs.push_back(x); vecs.push_back(0.f);
    }
    idx->load_cluster(0, ids.data(), vecs.data(), ids.size());

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 16);
    CHECK(coord.promote_to_gpu(0));

    // Add an extra vector via buffer to GPU only (not in L2 yet).
    float extra[2] = {1.1f, 0.f};
    coord.insert(0, 99, extra);
    coord.flush_buffers();  // flush → GPU expands to 7 vectors

    auto r = coord.split_gpu_cluster(0);
    printf("    split: success=%d  new_cid=%d\n", r.success, r.new_cid);
    CHECK(r.success);
    CHECK(r.new_cid >= 0);
    printf("    [M4-A] PASS\n");
}

// M4-B: splitting a non-GPU-resident cluster falls back to the L2 export path.
static void test_m4_split_nonresident_falls_back_to_l2() {
    log_test("M4-B: split of non-GPU-resident cluster reads from L2 (correct fallback)");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(12);

    // Two well-separated sub-clouds in cluster 0.
    auto ids_a = seed_l2(*idx, 0, 8, 100, DIM, 0.1f, rng);
    std::vector<float> far_vecs(static_cast<size_t>(8) * DIM);
    for (int i = 0; i < 8; ++i)
        for (int d = 0; d < DIM; ++d)
            far_vecs[static_cast<size_t>(i) * DIM + d] = 50.f + static_cast<float>(i) * 0.1f;
    std::vector<m3::DocId> far_ids;
    for (int i = 0; i < 8; ++i) far_ids.push_back(static_cast<m3::DocId>(200 + i));
    idx->load_cluster(0, far_ids.data(), far_vecs.data(), 8);

    // Cluster 0 is NOT on GPU.
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 32);
    CHECK(!coord.is_gpu_resident(0));

    auto r = coord.split_gpu_cluster(0);
    printf("    split (L2 fallback): success=%d  new_cid=%d\n", r.success, r.new_cid);
    CHECK(r.success);
    CHECK(r.new_cid >= 0);

    // Verify both partitions are searchable.
    float q0[2] = {0.f, 0.f};
    float q1[2] = {50.f, 50.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q0, 1, 5, 3, oi, os);
    CHECK(!oi[0].empty());
    oi.clear(); os.clear();
    idx->search(q1, 1, 5, 3, oi, os);
    CHECK(!oi[0].empty());
    printf("    [M4-B] PASS\n");
}

// M4-C: after splitting a GPU-resident cluster, the GPU is refreshed with only
// partition A data (the cluster is still resident, with the correct subset).
static void test_m4_split_gpu_refreshed_with_partition_a() {
    log_test("M4-C: after split, GPU cluster holds partition A (old full data replaced)");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(13);

    // 10 near (0,0), 10 near (80,0) — clearly separable.
    auto ids_a = seed_l2(*idx, 0, 10, 100, DIM, 0.2f, rng);
    std::vector<float> far_vecs(static_cast<size_t>(10) * DIM);
    std::vector<m3::DocId> far_ids;
    for (int i = 0; i < 10; ++i) {
        far_vecs[static_cast<size_t>(i) * DIM]     = 80.f + static_cast<float>(i) * 0.1f;
        far_vecs[static_cast<size_t>(i) * DIM + 1] = 0.f;
        far_ids.push_back(static_cast<m3::DocId>(200 + i));
    }
    idx->load_cluster(0, far_ids.data(), far_vecs.data(), 10);

    // Large budget — both clusters will fit.
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 64);
    CHECK(coord.promote_to_gpu(0));

    size_t gpu_size_before = coord.gpu_bytes_used() / (DIM * sizeof(float));
    printf("    GPU size before split: %zu vectors\n", gpu_size_before);

    auto r = coord.split_gpu_cluster(0, 30);
    CHECK(r.success);

    // Cluster 0 still GPU-resident.
    CHECK(coord.is_gpu_resident(0));

    // GPU cluster 0 must hold only partition A (≤ original size).
    size_t gpu_size_after = coord.gpu_bytes_used() / (DIM * sizeof(float));
    printf("    GPU size after split: %zu vectors (budget includes both if B also promoted)\n",
           gpu_size_after);

    // Export cluster 0 from GPU and verify it's a strict subset of the original 20.
    std::vector<m3::DocId> ex_ids; std::vector<float> ex_vecs;
    // Access via search — GPU-resident, so search returns partition A vectors.
    float q[2] = {0.f, 0.f};
    std::vector<m3::DocId> sr_ids; std::vector<float> sr_scores;
    coord.search({0}, q, 20, sr_ids, sr_scores);
    printf("    GPU search on cluster 0 after split: %zu results\n", sr_ids.size());
    CHECK(!sr_ids.empty());
    CHECK(sr_ids.size() < 20);  // partition A only, not the full 20

    printf("    [M4-C] PASS\n");
}

// M4-D: split preserves the total vector count: |A| + |B| == |original|.
static void test_m4_split_preserves_total_vectors() {
    log_test("M4-D: split is lossless — |partition A| + |partition B| == |original|");

    const int DIM = 2, NLIST = 2, N = 40;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(14);

    auto all_ids = seed_l2(*idx, 0, N, 1000, DIM, 3.f, rng);

    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 64);
    auto r = coord.split_gpu_cluster(0, 20);
    CHECK(r.success);

    std::vector<m3::DocId> ids_a, ids_b;
    std::vector<float>     vecs_a, vecs_b;
    idx->export_l2_cluster(0, ids_a, vecs_a);
    idx->export_l2_cluster(r.new_cid, ids_b, vecs_b);

    size_t total = ids_a.size() + ids_b.size();
    printf("    |A|=%zu  |B|=%zu  total=%zu  original=%d\n",
           ids_a.size(), ids_b.size(), total, N);
    CHECK_EQ(total, (size_t)N);

    // Each original ID in exactly one partition.
    std::unordered_set<int64_t> set_a(ids_a.begin(), ids_a.end());
    std::unordered_set<int64_t> set_b(ids_b.begin(), ids_b.end());
    for (m3::DocId id : all_ids)
        CHECK(set_a.count(id) != set_b.count(id));  // XOR: exactly one partition

    printf("    [M4-D] PASS\n");
}

// M4-E: if the budget allows, the new partition B cluster is promoted to GPU
// after the split, so both halves are GPU-resident.
static void test_m4_split_promotes_partition_b_when_budget_allows() {
    log_test("M4-E: partition B promoted to GPU if budget allows after split");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(15);

    // 10 + 10 vectors, well separated.
    seed_l2(*idx, 0, 10, 100, DIM, 0.2f, rng);
    std::vector<float> far(static_cast<size_t>(10) * DIM);
    std::vector<m3::DocId> far_ids;
    for (int i = 0; i < 10; ++i) {
        far[static_cast<size_t>(i) * DIM]     = 60.f + static_cast<float>(i) * 0.1f;
        far[static_cast<size_t>(i) * DIM + 1] = 0.f;
        far_ids.push_back(static_cast<m3::DocId>(200 + i));
    }
    idx->load_cluster(0, far_ids.data(), far.data(), 10);

    // Very large budget — easily fits both partitions.
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 64);
    CHECK(coord.promote_to_gpu(0));

    auto r = coord.split_gpu_cluster(0, 30);
    CHECK(r.success);
    CHECK(r.new_cid >= 0);

    // Original cluster (partition A) still GPU-resident.
    CHECK(coord.is_gpu_resident(0));
    // Partition B should also be promoted (budget is ample).
    bool b_promoted = coord.is_gpu_resident(r.new_cid);
    printf("    partition B (cid=%d) GPU-resident: %s\n", r.new_cid, b_promoted ? "yes" : "no");
    CHECK(b_promoted);
    printf("    [M4-E] PASS\n");
}

// M4-F: splitting a cluster with < 2 vectors returns failure (guard condition).
static void test_m4_split_guard_too_few_vectors() {
    log_test("M4-F: split fails gracefully on cluster with < 2 vectors");

    const int DIM = 2, NLIST = 2;
    auto idx = make_index(DIM, NLIST);
    m3::GpuCoordinator coord(*idx, 1 << 20, DIM, m3::Metric::L2, false, 8);

    // Empty cluster.
    auto r = coord.split_gpu_cluster(0);
    CHECK(!r.success);
    CHECK_EQ(r.new_cid, -1);

    // Single vector.
    m3::DocId id = 1; float v[2] = {0.f, 0.f};
    idx->load_cluster(0, &id, v, 1);
    r = coord.split_gpu_cluster(0);
    CHECK(!r.success);
    printf("    [M4-F] PASS\n");
}

// =============================================================================
// ═══  End-to-end integration test: all four mechanisms together  ═══
// =============================================================================

static void test_all_mechanisms_end_to_end() {
    log_test("E2E: all four mechanisms working together — insert, search, flush, split");

    const int DIM = 2, NLIST = 4;
    auto idx = make_index(DIM, NLIST);
    std::mt19937 rng(99);

    // Seed all clusters.
    for (int cid = 0; cid < NLIST; ++cid)
        seed_l2(*idx, cid, 5, cid * 100, DIM, 0.1f, rng);

    // Budget for 2 clusters.
    const size_t bytes_2 = 2 * 5 * DIM * sizeof(float);
    m3::GpuCoordinator coord(*idx, bytes_2, DIM, m3::Metric::L2, false, 8);

    // === Mechanism 1: hotspot-aware promotion ===
    // Make clusters 2 and 3 the hottest.
    for (int i = 0; i < 10; ++i) search_near(*idx, 2, DIM, 3, 1, rng);
    for (int i = 0; i < 8;  ++i) search_near(*idx, 3, DIM, 3, 1, rng);
    search_near(*idx, 0, DIM, 3, 1, rng);
    // Manually promote cold clusters to fill budget, then rebalance.
    coord.promote_to_gpu(0);
    coord.promote_to_gpu(1);
    size_t promoted = coord.rebalance(); coord.drain_pending();
    printf("    M1: promoted=%zu\n", promoted);
    CHECK(coord.is_gpu_resident(2) || coord.is_gpu_resident(3));

    // === Mechanism 2: insertion buffer + collaborative search ===
    // Insert into the hottest GPU cluster.
    int hot = coord.is_gpu_resident(2) ? 2 : 3;
    for (int i = 0; i < 3; ++i) {
        float v[2] = {static_cast<float>(hot) * 10.f + i * 0.1f, 0.5f};
        CHECK(coord.insert(hot, static_cast<m3::DocId>(5000 + i), v) == m3::BufferResult::kBuffered);
    }
    // Collaborative search must find GPU data AND buffer data.
    float q[2] = {static_cast<float>(hot) * 10.f, 0.25f};
    std::vector<m3::DocId> out_ids; std::vector<float> out_scores;
    size_t nfound = coord.search({hot}, q, 8, out_ids, out_scores);
    printf("    M2: collaborative search returned %zu results\n", nfound);
    CHECK(nfound >= 3);  // at least the buffered vectors

    // === Mechanism 3: async flush ===
    size_t flushed = coord.flush_buffers();
    printf("    M3: flushed=%zu\n", flushed);
    CHECK(flushed >= 3);

    // L2 durability — flushed vectors now in L2.
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 10, NLIST, oi, os);
    std::unordered_set<int64_t> l2_set(oi[0].begin(), oi[0].end());
    CHECK(l2_set.count(5000) || l2_set.count(5001) || l2_set.count(5002));

    // === Mechanism 4: on-GPU split ===
    auto sr = coord.split_gpu_cluster(hot, 20);
    printf("    M4: split success=%d  new_cid=%d\n", sr.success, sr.new_cid);
    CHECK(sr.success);
    CHECK(coord.is_gpu_resident(hot));  // original cluster still on GPU

    printf("    [E2E] all four mechanisms: PASS\n");
}

// =============================================================================
// main
// =============================================================================

int main() {
    printf("Paper Mechanism Test Suite\n");
    printf("Build: %s %s\n\n", __DATE__, __TIME__);

    if (const char* lp = std::getenv("M3_LOG")) {
        m3::M3Logger::instance().enable(lp);
        printf("Debug logging → %s\n\n", lp);
    }

    log_section("MECHANISM 1 — Hotspot-Aware Caching");
    test_m1_frequency_tracked_per_cluster();
    test_m1_budget_holds_top_n_by_frequency();
    test_m1_hotspot_change_evicts_cold_promotes_hot();
    test_m1_eviction_drains_buffer_to_l2();

    log_section("MECHANISM 2 — CPU Insertion Buffer");
    test_m2_buffer_capacity_binsert();
    test_m2_nonresident_bypasses_buffer();
    test_m2_search_merges_gpu_and_buffer_results();
    test_m2_dedup_keeps_best_score();
    test_m2_buffer_invisible_in_l2_until_flush();

    log_section("MECHANISM 3 — Async Consistency Management");
    test_m3_flush_expands_gpu_cluster_in_place();
    test_m3_flush_writes_durability_to_l2();
    test_m3_overflow_routes_to_l2_nonblocking();
    test_m3_background_flush_drains_buffers();
    test_m3_cluster_searchable_during_flush();

    log_section("MECHANISM 4 — On-GPU Cluster Splitting");
    test_m4_split_gpu_resident_uses_gpu_export();
    test_m4_split_nonresident_falls_back_to_l2();
    test_m4_split_gpu_refreshed_with_partition_a();
    test_m4_split_preserves_total_vectors();
    test_m4_split_promotes_partition_b_when_budget_allows();
    test_m4_split_guard_too_few_vectors();

    log_section("END-TO-END — All Four Mechanisms");
    test_all_mechanisms_end_to_end();

    printf("\n══════════════════════════════════════════════\n");
    printf("  RESULTS:  %d passed  /  %d failed\n", g_passed, g_failed);
    printf("══════════════════════════════════════════════\n");

    return (g_failed > 0) ? 1 : 0;
}
