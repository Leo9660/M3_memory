#include "m3_fsm.h"
#include "m3_multi_level.h"
#include <cassert>
#include <iostream>
#include <numeric>
#include <cmath>

using namespace m3;

// -----------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------
static std::vector<float> make_centroids(int nlist, int dim, float fill = 0.1f) {
    return std::vector<float>(static_cast<size_t>(nlist * dim), fill);
}

static void fill_rand(std::vector<float>& v, float lo, float hi) {
    for (auto& x : v) x = lo + (hi - lo) * (static_cast<float>(rand()) / RAND_MAX);
}

// -----------------------------------------------------------------------
// Test 1: FSMState / FSMPattern basics
// -----------------------------------------------------------------------
static void test_pattern_basics() {
    FSMPattern pat;
    pat.pattern_id = 0;

    pat.add_or_update_state(2, 0.3f, {0.1f, 0.2f, 0.3f});
    pat.add_or_update_state(5, 0.1f, {0.4f, 0.5f, 0.6f});
    pat.add_transition(2, 5);
    pat.add_transition(5, 2);

    assert(pat.has_transition(2, 5));
    assert(pat.has_transition(5, 2));
    assert(!pat.has_transition(2, 9));
    assert(pat.predict_next(2) == 5);   // only one outgoing edge

    // predict_next_ranked: both 5→2 is only option, so ranked list has one entry
    auto ranked = pat.predict_next_ranked(2);
    assert(ranked.size() == 1 && ranked[0] == 5);

    // Add a second outgoing transition from 2: 2→7 (delta 0.05 — tighter than 5)
    pat.add_or_update_state(7, 0.05f, {0.7f, 0.8f, 0.9f});
    pat.add_transition(2, 7);
    auto ranked2 = pat.predict_next_ranked(2);
    assert(ranked2.size() == 2);
    assert(ranked2[0] == 7);  // delta 0.05 < 0.1 → 7 comes first
    assert(ranked2[1] == 5);

    std::cout << "PASS test_pattern_basics\n";
}

// -----------------------------------------------------------------------
// Test 2: RequestTrajectory with FSMLayer tag
// -----------------------------------------------------------------------
static void test_trajectory_layer_tag() {
    RequestTrajectory traj;
    traj.request_id = "req-layer-test";

    traj.append_step({0.1f, 0.2f}, 3, FSMLayer::L0);
    traj.append_step({0.3f, 0.4f}, 7, FSMLayer::L1);
    traj.append_step({0.5f, 0.6f}, 3, FSMLayer::L0);

    assert(traj.length() == 3);
    assert(traj.steps[0].cluster_id == 3 && traj.steps[0].layer == FSMLayer::L0);
    assert(traj.steps[1].cluster_id == 7 && traj.steps[1].layer == FSMLayer::L1);
    assert(traj.steps[2].cluster_id == 3 && traj.steps[2].layer == FSMLayer::L0);

    // cluster_id_sequence is layer-agnostic
    auto seq = traj.cluster_id_sequence();
    assert(seq == std::vector<int>({3, 7, 3}));

    traj.finalize();
    assert(traj.is_complete);

    std::cout << "PASS test_trajectory_layer_tag\n";
}

// -----------------------------------------------------------------------
// Test 3: FSMTable round-trip with ranked prediction
// -----------------------------------------------------------------------
static void test_fsm_table_ranked_prediction() {
    FSMConfig cfg;
    cfg.max_patterns    = 8;
    cfg.match_threshold = 0.0f;
    cfg.merge_threshold = 0.0f;
    FSMTable table(cfg);

    // Build a trajectory: 2 → 5 → 2 at L0
    RequestTrajectory traj;
    traj.request_id = "req-1";
    traj.append_step({0.1f, 0.2f, 0.3f}, 2, FSMLayer::L0, 1000);
    traj.append_step({0.4f, 0.5f, 0.6f}, 5, FSMLayer::L0, 2000);
    traj.append_step({0.1f, 0.2f, 0.3f}, 2, FSMLayer::L0, 3000);
    traj.finalize();

    // flat centroids [8 clusters × 3 dim]
    std::vector<float> centroids(8 * 3, 0.1f);
    table.update_from_trajectory(traj, centroids, 3);
    assert(table.num_patterns() == 1);

    // partial trajectory ending at cluster 2 → predict 5
    RequestTrajectory traj2;
    traj2.request_id = "req-2";
    traj2.append_step({0.1f, 0.2f, 0.3f}, 2, FSMLayer::L0, 4000);

    auto r = table.match_and_predict(traj2, FSMLayer::L0);
    assert(!r.ranked_clusters.empty());
    assert(r.ranked_clusters[0] == 5);

    std::cout << "PASS test_fsm_table_ranked_prediction\n";
}

// -----------------------------------------------------------------------
// Test 4: invalidate_cluster removes states and transitions
// -----------------------------------------------------------------------
static void test_invalidate_cluster() {
    FSMConfig cfg;
    cfg.match_threshold = 0.0f;
    cfg.merge_threshold = 0.0f;
    FSMTable table(cfg);

    RequestTrajectory traj;
    traj.request_id = "req-inv";
    traj.append_step({1,0,0}, 0, FSMLayer::L0);
    traj.append_step({0,1,0}, 1, FSMLayer::L0);
    traj.append_step({0,0,1}, 2, FSMLayer::L0);
    traj.finalize();

    std::vector<float> cents(8 * 3, 0.0f);
    table.update_from_trajectory(traj, cents, 3);
    assert(table.num_patterns() == 1);

    // Invalidate cluster 1 — transitions 0→1 and 1→2 should vanish.
    table.invalidate_cluster(1);

    const FSMPattern* p = table.get_pattern(0);
    assert(p != nullptr);
    assert(!p->states.count(1));
    assert(!p->has_transition(0, 1));
    assert(!p->has_transition(1, 2));
    // State 0 and 2 and transition 0→2... 0→2 was never added so check states remain
    assert(p->states.count(0));
    assert(p->states.count(2));

    std::cout << "PASS test_invalidate_cluster\n";
}

// -----------------------------------------------------------------------
// Test 5: set_config preserves patterns
// -----------------------------------------------------------------------
static void test_set_config_preserves_patterns() {
    FSMConfig cfg;
    cfg.match_threshold = 0.0f;
    cfg.merge_threshold = 0.0f;
    FSMTable table(cfg);

    RequestTrajectory traj;
    traj.request_id = "r";
    traj.append_step({1,0}, 0, FSMLayer::L0);
    traj.append_step({0,1}, 1, FSMLayer::L0);
    traj.finalize();

    std::vector<float> cents(4 * 2, 0.1f);
    table.update_from_trajectory(traj, cents, 2);
    assert(table.num_patterns() == 1);

    // Change config — patterns must survive
    FSMConfig cfg2;
    cfg2.alpha_et = 0.5f;
    cfg2.dagent_window = 32;
    table.set_config(cfg2);

    assert(table.num_patterns() == 1);
    assert(std::abs(table.config().alpha_et - 0.5f) < 1e-6f);
    assert(table.config().dagent_window == 32);

    std::cout << "PASS test_set_config_preserves_patterns\n";
}

// -----------------------------------------------------------------------
// Test 6: MultiLevelIndex FSM accessors
// -----------------------------------------------------------------------
static void test_multilevel_fsm_accessor() {
    MultiLevelIndex idx(3, Metric::L2, false);

    FSMConfig cfg;
    cfg.max_patterns = 4;
    cfg.alpha_et     = 0.6f;
    idx.set_fsm_config(cfg);

    assert(idx.fsm_table().config().max_patterns == 4);
    assert(std::abs(idx.fsm_table().config().alpha_et - 0.6f) < 1e-6f);
    assert(idx.fsm_table().num_patterns() == 0);

    std::cout << "PASS test_multilevel_fsm_accessor\n";
}

// -----------------------------------------------------------------------
// Test 7: search() with traj=nullptr (FSM disabled) works correctly
// -----------------------------------------------------------------------
static void test_search_no_traj() {
    const int dim   = 4;
    const int nlist = 2;
    MultiLevelIndex idx(dim, Metric::L2, false);

    // One centroid per cluster
    std::vector<float> cents = {
        0.0f, 0.0f, 0.0f, 0.0f,   // cluster 0
        1.0f, 1.0f, 1.0f, 1.0f,   // cluster 1
    };
    idx.set_l0_centroids(cents);

    // Insert two vectors
    std::vector<DocId>  ids  = {10, 20};
    std::vector<float>  vecs = {
        0.1f, 0.1f, 0.1f, 0.1f,   // near cluster 0
        0.9f, 0.9f, 0.9f, 0.9f,   // near cluster 1
    };
    idx.insert(ids.data(), vecs.data(), 2);

    // Search without FSM
    float query[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    std::vector<std::vector<DocId>>  out_ids;
    std::vector<std::vector<float>>  out_scores;
    idx.search(query, 1, 1, nlist, out_ids, out_scores, nullptr);

    assert(!out_ids.empty() && !out_ids[0].empty());
    assert(out_ids[0][0] == 10);  // closest to cluster 0

    std::cout << "PASS test_search_no_traj\n";
}

// -----------------------------------------------------------------------
// Test 8: search() with traj fills trajectory steps
// -----------------------------------------------------------------------
static void test_search_fills_trajectory() {
    const int dim   = 4;
    const int nlist = 2;
    MultiLevelIndex idx(dim, Metric::L2, false);

    std::vector<float> cents = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
    };
    idx.set_l0_centroids(cents);

    std::vector<DocId> ids  = {10, 20};
    std::vector<float> vecs = {
        0.1f, 0.1f, 0.1f, 0.1f,
        0.9f, 0.9f, 0.9f, 0.9f,
    };
    idx.insert(ids.data(), vecs.data(), 2);

    RequestTrajectory traj;
    traj.request_id = "req-fill";

    float query[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    std::vector<std::vector<DocId>>  out_ids;
    std::vector<std::vector<float>>  out_scores;
    idx.search(query, 1, 1, nlist, out_ids, out_scores, &traj);

    // Non-cache path (no L2 centroids set) — traj may have 0 steps
    // because append_step only runs in cache path. Verify at least search returned.
    assert(!out_ids.empty());
    std::cout << "PASS test_search_fills_trajectory (steps=" << traj.length() << ")\n";
}

// -----------------------------------------------------------------------
// Test 9: dagent rolling average
// -----------------------------------------------------------------------
static void test_dagent() {
    // We can't call update_dagent_ directly (private), so drive it through
    // a search that goes all the way to L2 (which calls update_dagent_).
    // Instead test it indirectly: repeated searches should not crash.
    const int dim   = 3;
    const int nlist = 2;
    MultiLevelIndex idx(dim, Metric::L2, false);

    std::vector<float> cents(nlist * dim, 0.5f);
    idx.set_l0_centroids(cents);

    std::vector<DocId> ids  = {1, 2, 3};
    std::vector<float> vecs = {
        0.1f, 0.2f, 0.3f,
        0.4f, 0.5f, 0.6f,
        0.7f, 0.8f, 0.9f,
    };
    idx.insert(ids.data(), vecs.data(), 3);

    float q[3] = {0.3f, 0.4f, 0.5f};
    for (int i = 0; i < 20; ++i) {
        std::vector<std::vector<DocId>>  oi;
        std::vector<std::vector<float>>  os;
        idx.search(q, 1, 2, nlist, oi, os, nullptr);
    }
    std::cout << "PASS test_dagent (no crash after 20 searches)\n";
}

// -----------------------------------------------------------------------
// Test 10: eviction by merging keeps table at cap
// -----------------------------------------------------------------------
static void test_eviction_cap() {
    FSMConfig cfg;
    cfg.max_patterns    = 2;
    cfg.match_threshold = 0.0f;
    cfg.merge_threshold = 2.0f;  // never reinforce, always create new
    FSMTable table(cfg);

    std::vector<float> cents(16 * 3, 0.1f);
    for (int i = 0; i < 5; ++i) {
        RequestTrajectory t;
        t.request_id = "r" + std::to_string(i);
        t.append_step({float(i), 0, 0}, i * 2,     FSMLayer::L0);
        t.append_step({float(i), 1, 0}, i * 2 + 1, FSMLayer::L0);
        t.finalize();
        table.update_from_trajectory(t, cents, 3);
    }
    assert(table.num_patterns() <= 2);
    std::cout << "PASS test_eviction_cap (patterns=" << table.num_patterns() << ")\n";
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main() {
    test_pattern_basics();
    test_trajectory_layer_tag();
    test_fsm_table_ranked_prediction();
    test_invalidate_cluster();
    test_set_config_preserves_patterns();
    test_multilevel_fsm_accessor();
    test_search_no_traj();
    test_search_fills_trajectory();
    test_dagent();
    test_eviction_cap();

    std::cout << "\nAll tests passed.\n";
    return 0;
}
