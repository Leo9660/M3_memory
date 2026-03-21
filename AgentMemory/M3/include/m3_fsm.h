#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <mutex>
#include <limits>
#include <optional>

#include "base.h"   // Metric, DocId, unified_score

namespace m3 {

// ======================================================================
// FSMState
//
// One state in an FSM pattern corresponds to one cluster (L0 or L1 —
// they share the same cluster-id namespace since both are initialised from
// the same L2 centroids via set_l2_centroids).
//
// delta: average intra-cluster vector deviation from the centroid.
//   Tight clusters (small δ) contribute more weight in the similarity
//   formula: δ_k / (1 + |c_k − v_k|).
// ======================================================================
struct FSMState {
    int      cluster_id = -1;
    float    delta      = 1.0f;  // 1.0 = uninitialised / unknown tightness

    // Representative centroid snapshot at creation / last refresh.
    // Length == index dim.  Empty before first use.
    std::vector<float> centroid;

    FSMState() = default;
    FSMState(int cid, float d, std::vector<float> c)
        : cluster_id(cid), delta(d), centroid(std::move(c)) {}
};

// ======================================================================
// FSMPattern  P = (S, T)
//
// S — set of cluster states, keyed by cluster_id for O(1) lookup.
// T — directed transition table: from_cluster_id → {to_cluster_id, …}
//
// frequency / last_used drive pattern table eviction.
// ======================================================================
struct FSMPattern {
    int      pattern_id  = -1;
    uint64_t frequency   = 0;
    uint64_t last_used   = 0;  // monotonic ns; 0 = never

    std::unordered_map<int, FSMState>                     states;
    std::unordered_map<int, std::unordered_set<int>>      transitions;

    // ---- mutation helpers ----
    void add_or_update_state(int cid, float delta, const std::vector<float>& centroid);
    void add_transition(int from, int to);
    bool has_transition(int from, int to) const;

    // Return the single best next cluster_id from last_cluster
    // (tightest delta wins).  Returns -1 if no outgoing transitions.
    int predict_next(int last_cluster) const;

    // Return ALL outgoing transitions from last_cluster, ranked by
    // delta ascending (tightest / most reliable first).
    // Returns an empty vector if no outgoing transitions exist.
    // This is what gets turned into the preferred_order list passed
    // to search_with_ordering at each layer.
    std::vector<int> predict_next_ranked(int last_cluster) const;

    size_t num_transitions() const;
};

// ======================================================================
// FSMLayer
//
// Tag that indicates which index layer a trajectory step belongs to.
// L0 and L1 share the same cluster-id namespace (both initialised from
// the same L2 centroids), so ids are directly comparable across layers.
// The tag is kept for debugging and future per-layer analysis.
// ======================================================================
enum class FSMLayer : uint8_t { L0 = 0, L1 = 1 };

// ======================================================================
// FSMStep
//
// A single step recorded in a RequestTrajectory.
// ======================================================================
struct FSMStep {
    int      cluster_id = -1;
    FSMLayer layer      = FSMLayer::L0;
};

// ======================================================================
// RequestTrajectory
//
// Tracks the in-flight cluster-access sequence for a single request.
//
// Each search call may contribute up to two steps:
//   1. The winning L0 cluster (when L0 search ran).
//   2. The winning L1 cluster (only when L0 early-termination did NOT fire).
//
// L2 results are NOT recorded — L2 has no FSM-driven probe reordering.
//
// The C++ search path calls append_step() automatically after each layer.
// Python needs only call finalize() and pass the trajectory to
// FSMTable::update_from_trajectory() after the full request is done.
// ======================================================================
struct RequestTrajectory {
    std::string request_id;

    // Parallel arrays — entry k records one layer-search step.
    std::vector<std::vector<float>> embeddings;    // query embedding for this step
    std::vector<FSMStep>            steps;         // cluster_id + layer
    std::vector<uint64_t>           timestamps_ns;

    bool is_complete = false;

    // Record one completed step.
    // embedding: query vector used for this step (length = dim).
    // cluster_id: the cluster that contained the top result.
    // layer: which layer this cluster belongs to.
    // time_ns: monotonic ns (0 = auto-fill with current time).
    void append_step(std::vector<float> embedding,
                     int cluster_id,
                     FSMLayer layer,
                     uint64_t time_ns = 0);

    void finalize();

    // Flat cluster-id sequence (layer-agnostic), used by
    // compute_similarity_transitions which doesn't need layer info.
    std::vector<int> cluster_id_sequence() const;

    size_t length() const { return steps.size(); }
};

// ======================================================================
// FSMConfig
// ======================================================================
struct FSMConfig {
    int    max_patterns       = 16;    // Np in the paper
    int    max_states_per_fsm = 32;    // NS in the paper
    float  merge_threshold    = 0.7f;  // min similarity to reinforce existing pattern
    float  match_threshold    = 0.3f;  // min similarity to accept a prediction
    float  merge_dist         = 0.5f;  // d_merge: max centroid distance for state merging

    // Early-termination factor αet (paper §4.2).
    // After an L0 or L1 search, if max(out_scores[:k]) < alpha_et * dagent,
    // skip the next layer.  Paper recommends 0.6–0.8.
    float  alpha_et            = 0.7f;

    // Rolling window size for dagent (average k-th distance across recent queries).
    int    dagent_window       = 64;
};

// ======================================================================
// FSMTable
//
// The Np-capped pattern table owned by MultiLevelIndex.
// Thread safety: all public methods acquire fsm_mu_ internally.
// ======================================================================
class FSMTable {
public:
    explicit FSMTable(FSMConfig cfg = {});

    // ---- primary interface ----

    struct PredictResult {
        // Ranked cluster ids to probe first at the relevant layer.
        // Empty = no prediction (fall back to centroid-distance ordering).
        std::vector<int> ranked_clusters;

        float best_score      = 0.0f;
        int   best_pattern_id = -1;
    };

    // Given a partial trajectory, find the best matching pattern and return
    // a ranked preferred-cluster list for the next probe at `layer`.
    // Uses the last step in the trajectory to look up outgoing transitions.
    PredictResult match_and_predict(const RequestTrajectory& traj,
                                    FSMLayer layer) const;

    // After a request is complete, update the pattern table.
    // cluster_centroids: flat [nlist * dim] centroid array (same layout as
    //   l0_.centroids / l1_.centroids — they share the same namespace).
    // dim: dimensionality of each centroid vector.
    void update_from_trajectory(const RequestTrajectory& traj,
                                const std::vector<float>& cluster_centroids,
                                int dim,
                                float cluster_delta_default = 1.0f);

    // Invalidate all FSM states and transitions that reference cluster_id.
    // Called whenever L0 topology changes (cluster split / merge / remove)
    // so the FSM does not predict a cluster that no longer exists.
    void invalidate_cluster(int cluster_id);

    // ---- similarity functions (also used internally) ----

    // Transition-only (fast, no embedding distance term).
    float compute_similarity_transitions(const FSMPattern& pat,
                                         const std::vector<int>& cluster_seq) const;

    // Full paper formula §4.2:
    //   sim(P_i, v_{1:t}) = Σ_k I[(c_{k-1}→c_k) ∈ T_i] · δ_k / (1 + |c_k − v_k|)
    float compute_similarity_full(const FSMPattern& pat,
                                   const std::vector<int>& cluster_seq,
                                   const std::vector<std::vector<float>>& embedding_seq) const;

    // ---- inspection ----
    int   num_patterns()  const;
    const FSMPattern* get_pattern(int pattern_id) const;  // nullptr if not found
    FSMConfig config()    const { return cfg_; }

    int merge_patterns(int id_a, int id_b);
    void clear();

    // Replace the config (patterns are preserved).
    // Use this instead of reconstructing the table (mutex is non-movable).
    void set_config(FSMConfig cfg);

private:
    FSMPattern make_pattern_from_trajectory_(
        const RequestTrajectory& traj,
        const std::vector<float>& cluster_centroids,
        int dim,
        float cluster_delta_default) const;

    void evict_by_merging_();

    // Internal merge without re-acquiring fsm_mu_ (caller already holds it).
    void merge_nolock_(int id_a, int id_b);

    int next_id_ = 0;
    FSMConfig cfg_;
    mutable std::mutex fsm_mu_;
    std::unordered_map<int, FSMPattern> patterns_;
    std::vector<int> order_;
};

} // namespace m3
