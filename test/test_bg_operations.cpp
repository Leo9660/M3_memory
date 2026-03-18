// test_bg_operations.cpp
//
// Tests for the three background maintenance operations on GpuCoordinator:
//
//   M-1: flush_buffers()    — drains CPU insert buffer → GPU expand + L2 write
//   M-2: cpu_maintenance()  — L0 vector eviction to L1 when L0 overflows
//   M-3: rebalance()        — promotes hotter non-GPU cluster, evicts coldest GPU cluster
//
// Build:
//   cmake -B build && cmake --build build --target test_bg_operations
// Run:
//   ./build/test_bg_operations

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "gpu_coordinator.h"
#include "m3_multi_level.h"

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
// Shared helpers
// =============================================================================

// Build a MultiLevelIndex with centroid for cluster i = i*10 in all dims.
// l0_vec_cap: override for l0_max_vectors_per_cluster (0 = use default 50000).
static std::unique_ptr<m3::MultiLevelIndex> make_index(int dim, int nlist,
                                                        size_t l0_vec_cap = 0) {
    m3::MultiLevelConfig cfg;
    cfg.l0_nlist = nlist;
    auto idx = std::make_unique<m3::MultiLevelIndex>(dim, m3::Metric::L2, false, cfg);

    std::vector<float> centroids(static_cast<size_t>(nlist) * dim);
    for (int i = 0; i < nlist; ++i)
        for (int d = 0; d < dim; ++d)
            centroids[static_cast<size_t>(i) * dim + d] = static_cast<float>(i) * 10.f;
    // All three levels share the same centroid layout for simplicity.
    idx->set_l0_centroids(centroids);
    idx->set_l1_centroids(centroids);
    idx->set_l2_centroids(centroids);

    m3::CacheConfig cc;
    cc.l0_max_clusters            = nlist;
    cc.l1_max_clusters            = nlist;
    cc.l0_max_vectors_per_cluster = (l0_vec_cap > 0) ? l0_vec_cap : 50000;
    cc.l1_max_vectors_per_cluster = 50000;
    // UINT64_MAX: cold-demotion must never fire spontaneously in tests.
    // last_access_time starts at 0 (never searched); without this guard,
    // (now_ns - 0) >> any finite threshold and vectors are evicted immediately.
    cc.cold_time_ns               = std::numeric_limits<uint64_t>::max();
    cc.alpha_et                   = 0.f;                 // disable early termination
    idx->set_cache_config(cc);
    return idx;
}

// Bulk-seed L2 cluster cid with n vectors starting at id_base.
static void seed_l2(m3::MultiLevelIndex& idx, int cid, int n, int id_base,
                    int dim, float noise, std::mt19937& rng) {
    std::uniform_real_distribution<float> d(-noise, noise);
    std::vector<float>    vecs(static_cast<size_t>(n) * dim);
    std::vector<m3::DocId> ids(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        ids[static_cast<size_t>(i)] = id_base + i;
        for (int dd = 0; dd < dim; ++dd)
            vecs[static_cast<size_t>(i) * dim + dd] =
                static_cast<float>(cid) * 10.f + d(rng);
    }
    idx.load_cluster(cid, ids.data(), vecs.data(), static_cast<size_t>(n));
}

static bool has_id(const std::vector<m3::DocId>& v, m3::DocId id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

// =============================================================================
// M-1: flush_buffers()
//
// Verifies the two effects of flush:
//   a) GPU expand  — vector migrated from CPU buffer into GPU cluster VRAM.
//   b) L2 durability — vector written to L2 so eviction drain is cheap.
//
// Before flush: vector is in the CPU insert buffer only.
//   → searchable only via buffer scan (collaborative_search buffer path).
//   → NOT in L2 (export_l2_cluster doesn't return it).
// After flush:
//   → vector is in the GPU cluster proper (GPU kernel can find it).
//   → vector IS in L2.
//   → flush_buffers() return value ≥ 1.
// =============================================================================
static void test_m1_flush_buffers() {
    log_test("M-1: flush_buffers() — GPU expand + L2 durability write");

    const int DIM = 2, NLIST = 2;
    std::mt19937 rng(42);
    auto idx = make_index(DIM, NLIST);

    // Seed both clusters in L2.
    seed_l2(*idx, 0, 5, 1000, DIM, 0.1f, rng);
    seed_l2(*idx, 1, 5, 2000, DIM, 0.1f, rng);

    // Budget large enough for 1 cluster; insert_buf_cap=32.
    const size_t bytes_per = 5 * DIM * sizeof(float);
    m3::GpuCoordinator coord(*idx, 2 * bytes_per, DIM, m3::Metric::L2, false, 32);
    idx->set_gpu_coordinator(&coord);

    // Promote cluster 0 to GPU.
    CHECK(coord.promote_to_gpu(0));

    // Insert one vector into the GPU-resident cluster via the index.
    // It should land in the CPU insert buffer, not L2.
    const m3::DocId NEW_ID = 9001;
    float new_vec[2] = {0.2f, 0.2f};  // near cluster 0 centroid (0,0)
    m3::DocId arr_id[1] = {NEW_ID};
    idx->insert(arr_id, new_vec, 1);

    // --- Before flush ---
    std::vector<m3::DocId> l2_before; std::vector<float> l2v_before;
    idx->export_l2_cluster(0, l2_before, l2v_before);
    CHECK(!has_id(l2_before, NEW_ID));
    printf("    before flush: %zu vectors in L2 cluster 0 (9001 absent: correct)\n",
           l2_before.size());

    // Buffer scan path: vector should still be searchable via collaborative_search.
    float q[2] = {0.f, 0.f};
    std::vector<std::vector<m3::DocId>> oi; std::vector<std::vector<float>> os;
    idx->search(q, 1, 10, 1, oi, os);
    CHECK(has_id(oi[0], NEW_ID));
    printf("    before flush: 9001 findable via buffer scan: %s\n",
           has_id(oi[0], NEW_ID) ? "yes" : "no");

    // --- Flush ---
    size_t flushed = coord.flush_buffers();
    printf("    flush_buffers() returned: %zu\n", flushed);
    CHECK(flushed >= 1);

    // --- After flush: vector must be in L2 ---
    std::vector<m3::DocId> l2_after; std::vector<float> l2v_after;
    idx->export_l2_cluster(0, l2_after, l2v_after);
    CHECK(has_id(l2_after, NEW_ID));
    printf("    after flush: %zu vectors in L2 cluster 0 (9001 present: %s)\n",
           l2_after.size(), has_id(l2_after, NEW_ID) ? "yes" : "no");

    // --- After flush: vector still searchable (now via GPU cluster proper) ---
    oi.clear(); os.clear();
    idx->search(q, 1, 10, 1, oi, os);
    CHECK(has_id(oi[0], NEW_ID));
    printf("    after flush: 9001 still findable via GPU cluster: %s\n",
           has_id(oi[0], NEW_ID) ? "yes" : "no");

    idx->set_gpu_coordinator(nullptr);
}

// =============================================================================
// M-2: cpu_maintenance()
//
// Verifies L0 vector eviction: when a cluster's L0 vector count exceeds
// l0_max_vectors_per_cluster, maintenance evicts the coldest vectors to L1.
//
// Setup: l0_max_vectors_per_cluster = 3, insert 6 vectors into cluster 0.
//   → L0 has 6 vectors (over cap).
// After cpu_maintenance():
//   → L0 has ≤ 3 vectors (3 coldest evicted).
//   → L1 gains the evicted vectors (written back to L1 per spec).
//   → L2 unchanged — all 6 vectors still present (L2 = ground truth).
// =============================================================================
static void test_m2_cpu_maintenance() {
    log_test("M-2: cpu_maintenance() — L0 overflow eviction into L1");

    const int DIM = 2, NLIST = 2;
    const size_t L0_CAP = 3;
    std::mt19937 rng(7);

    // Build index with tight L0 cap.
    auto idx = make_index(DIM, NLIST, L0_CAP);

    // No GPU coordinator — inserts go to L0 and L2.
    // Budget sized for 1 cluster (unused here, just constructing coordinator
    // to call cpu_maintenance() through it).
    const size_t bytes_per = 6 * DIM * sizeof(float);
    m3::GpuCoordinator coord(*idx, 2 * bytes_per, DIM, m3::Metric::L2, false, 32);

    // Insert 6 vectors into cluster 0 via idx (no GPU coordinator wired,
    // so each insert goes to both L0 and L2 directly).
    const int N = 6;
    std::vector<m3::DocId> ids(N);
    std::vector<float> vecs(static_cast<size_t>(N) * DIM);
    std::uniform_real_distribution<float> noise(-0.1f, 0.1f);
    for (int i = 0; i < N; ++i) {
        ids[static_cast<size_t>(i)] = 5000 + i;
        vecs[static_cast<size_t>(i) * DIM + 0] = noise(rng);  // near cluster 0 (0,0)
        vecs[static_cast<size_t>(i) * DIM + 1] = noise(rng);
    }
    idx->insert(ids.data(), vecs.data(), static_cast<size_t>(N));

    // Check metadata before maintenance.
    auto meta_before = idx->get_cluster_metadata();
    printf("    before maintenance: L0[cid=0]=%zu L1[cid=0]=%zu L2[cid=0]=%zu\n",
           meta_before[0].l0_vector_count,
           meta_before[0].l1_vector_count,
           meta_before[0].l2_vector_count);
    CHECK(meta_before[0].l0_vector_count == static_cast<size_t>(N));

    // Run cpu_maintenance() — triggers L0→L1 eviction for the overflow.
    coord.cpu_maintenance();

    auto meta_after = idx->get_cluster_metadata();
    printf("    after  maintenance: L0[cid=0]=%zu L1[cid=0]=%zu L2[cid=0]=%zu\n",
           meta_after[0].l0_vector_count,
           meta_after[0].l1_vector_count,
           meta_after[0].l2_vector_count);

    // L0 must be at or below cap.
    CHECK(meta_after[0].l0_vector_count <= L0_CAP);

    // Evicted vectors must have been written to L1.
    CHECK(meta_after[0].l1_vector_count > 0);

    // L2 must still hold all 6 vectors (ground truth unaffected by maintenance).
    std::vector<m3::DocId> l2_ids; std::vector<float> l2_vecs;
    idx->export_l2_cluster(0, l2_ids, l2_vecs);
    CHECK(l2_ids.size() == static_cast<size_t>(N));
    printf("    L2 still has all %d vectors: %s\n",
           N, l2_ids.size() == static_cast<size_t>(N) ? "yes" : "no");
}

// =============================================================================
// M-3: rebalance()
//
// Verifies hotspot-driven GPU promotion:
//   - Cluster 0 is GPU-resident with access_count = 0 (never searched).
//   - Cluster 1 is CPU-resident; 10 searches near its centroid drive its
//     access_count above cluster 0's.
//   - rebalance() detects this and enqueues cluster 1 for promotion.
//   - drain_pending() processes the queue synchronously:
//       promote_to_gpu(1) → LFU evicts cluster 0 (weaker count) to make room.
//   - After drain: cluster 1 GPU-resident, cluster 0 not.
// =============================================================================
static void test_m3_rebalance() {
    log_test("M-3: rebalance() — promotes hotter cluster, evicts coldest GPU resident");

    const int DIM = 2, NLIST = 3;
    std::mt19937 rng(13);
    auto idx = make_index(DIM, NLIST);

    // Seed all 3 clusters in L2.
    seed_l2(*idx, 0, 5, 1000, DIM, 0.1f, rng);
    seed_l2(*idx, 1, 5, 2000, DIM, 0.1f, rng);
    seed_l2(*idx, 2, 5, 3000, DIM, 0.1f, rng);

    // Budget fits exactly 1 cluster.
    const size_t bytes_per = 5 * DIM * sizeof(float);
    m3::GpuCoordinator coord(*idx, 1 * bytes_per, DIM, m3::Metric::L2, false, 32);
    idx->set_gpu_coordinator(&coord);

    // Promote cluster 0 to GPU — access_count stays 0 (no searches).
    CHECK(coord.promote_to_gpu(0));
    CHECK(coord.is_gpu_resident(0));
    CHECK(!coord.is_gpu_resident(1));

    printf("    initial: cid0 GPU=%d  cid1 GPU=%d\n",
           coord.is_gpu_resident(0), coord.is_gpu_resident(1));
    printf("    access_count before searches: cid0=%llu  cid1=%llu\n",
           (unsigned long long)idx->get_access_count(0),
           (unsigned long long)idx->get_access_count(1));

    // Drive cluster 1's access_count up via repeated searches near its centroid.
    // record_access_() increments access_count for any cluster whose vectors
    // appear in the top-k result set.
    const int N_SEARCHES = 10;
    float q1[2] = {10.f, 10.f};  // cluster 1 centroid
    for (int i = 0; i < N_SEARCHES; ++i) {
        std::vector<std::vector<m3::DocId>> oi;
        std::vector<std::vector<float>>     os;
        idx->search(q1, 1, 5, /*nprobe=*/2, oi, os);
    }

    printf("    access_count after %d searches near cid1: cid0=%llu  cid1=%llu\n",
           N_SEARCHES,
           (unsigned long long)idx->get_access_count(0),
           (unsigned long long)idx->get_access_count(1));

    CHECK(idx->get_access_count(1) > idx->get_access_count(0));

    // rebalance(): cluster 1 is hotter than the weakest GPU resident (cluster 0).
    // enqueue_promote(1) is added to pending queue — no H2D yet.
    size_t enqueued = coord.rebalance();
    printf("    rebalance() enqueued %zu cluster(s)\n", enqueued);
    CHECK(enqueued >= 1);

    // At this point cluster 1 is NOT yet GPU-resident — async, not processed yet.
    CHECK(!coord.is_gpu_resident(1));

    // drain_pending(): process the queue synchronously.
    // promote_to_gpu(1) runs: budget is full → LFU evicts cluster 0 (access_count=0).
    coord.drain_pending();

    printf("    after drain_pending: cid0 GPU=%d  cid1 GPU=%d\n",
           coord.is_gpu_resident(0), coord.is_gpu_resident(1));

    CHECK(coord.is_gpu_resident(1));
    CHECK(!coord.is_gpu_resident(0));

    idx->set_gpu_coordinator(nullptr);
}

// =============================================================================
// main
// =============================================================================

int main() {
    log_section("M-1: flush_buffers()");
    test_m1_flush_buffers();

    log_section("M-2: cpu_maintenance()");
    test_m2_cpu_maintenance();

    log_section("M-3: rebalance()");
    test_m3_rebalance();

    printf("\n══════════════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", g_passed, g_failed);
    printf("══════════════════════════════════════════════\n");
    return g_failed > 0 ? 1 : 0;
}
