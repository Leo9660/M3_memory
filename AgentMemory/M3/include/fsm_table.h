#pragma once
// FSM-based access-pattern learning for M3.
// All types live in namespace m3::fsm.
// Compiled only when -DM3_WITH_FSM is passed (see CMakeLists.txt option M3_WITH_FSM).

#ifndef M3_WITH_FSM
// Nothing exposed when FSM is disabled.
#else

#include <cstdint>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace m3 {
namespace fsm {

// ============================================================
// RequestTrajectory
//
// Accumulates (cluster_id, query_vector) pairs across all
// search steps of a single request.  One instance per live
// request; owned by the caller (Python backend).
// ============================================================
struct RequestTrajectory {
    std::string              request_id;
    std::vector<int>         steps;      // cluster IDs in order
    std::vector<std::vector<float>> step_vecs;  // raw query vectors [step][dim]

    RequestTrajectory() = default;
    explicit RequestTrajectory(std::string rid) : request_id(std::move(rid)) {}

    void append_step(int cid, const float* vec, int dim) {
        steps.push_back(cid);
        step_vecs.emplace_back(vec, vec + dim);
    }
    int  length() const { return static_cast<int>(steps.size()); }
    void clear() { steps.clear(); step_vecs.clear(); }
};

// ============================================================
// FSMState  —  one (c, δ) state in a pattern
// ============================================================
struct FSMState {
    int                cid;
    std::vector<float> centroid;  // [dim]
    float              delta;     // avg ‖v − c‖ for vectors in this cluster
};

// ============================================================
// FSMPattern  —  P = (S, T)
//   S  = ordered states (after intra-pattern merge)
//   T  = directed transition set {(from_cid, to_cid)}
// ============================================================
struct FSMPattern {
    std::vector<FSMState>              states;
    std::set<std::pair<int,int>>       transitions;
    int                                hit_count = 1;

    // cid → index into `states`  (O(1) centroid lookup during scoring)
    std::unordered_map<int, int>       cid_to_idx;

    void rebuild_cid_map() {
        cid_to_idx.clear();
        for (int i = 0; i < static_cast<int>(states.size()); ++i)
            cid_to_idx[states[i].cid] = i;
    }

    // Returns all cids reachable from last_cid via one transition.
    std::vector<int> predict_next(int last_cid) const {
        std::vector<int> out;
        for (const auto& [a, b] : transitions)
            if (a == last_cid) out.push_back(b);
        return out;
    }

    // Reinforce: union transitions; add centroids for any new cids (no state merge).
    void add_transitions(const std::set<std::pair<int,int>>& new_trans,
                         const std::vector<FSMState>&        new_states) {
        transitions.insert(new_trans.begin(), new_trans.end());
        for (const auto& s : new_states) {
            if (cid_to_idx.find(s.cid) == cid_to_idx.end()) {
                cid_to_idx[s.cid] = static_cast<int>(states.size());
                states.push_back(s);
            }
        }
        ++hit_count;
    }
};

// ============================================================
// FSMConfig
// ============================================================
struct FSMConfig {
    int   max_patterns        = 500;   // Np
    int   ns_max_states       = 8;     // NS
    float d_merge             = 0.3f;  // dmerge
    float reinforce_threshold = 0.5f;
    int   min_hits_to_predict = 2;
    int   min_traj_len        = 2;
};

// ============================================================
// FSMTable
//
// Thread-safe FSM pattern table.  All public methods are
// safe to call concurrently from multiple threads.
//
// Similarity formula (paper eq.):
//   sim(Pi, v1:t) = Σ_{k=1}^{t}  I[(c_{k-1}→c_k) ∈ T_i]
//                                · δ_k / (1 + ‖c_k − v_k‖)
// ============================================================
class FSMTable {
public:
    explicit FSMTable(FSMConfig cfg = {});

    // Score all patterns against the current in-flight trajectory
    // and return predicted next-cluster IDs from the best-matching pattern.
    // Returns empty vector when no pattern has sufficient hits or score.
    std::vector<int> match_and_predict(const RequestTrajectory& traj) const;

    // Called after a request completes.
    // Updates δ estimates, finds best match (reinforce or create new),
    // then prunes if #patterns > Np.
    // centroids: [nlist × dim] row-major; may be nullptr.
    void update_from_trajectory(const RequestTrajectory& traj,
                                const float* centroids,
                                int nlist, int dim);

    int num_patterns() const;
    int total_hits()   const;
    const FSMConfig& config() const { return cfg_; }

private:
    // sim(Pi, v1:t) — requires mu_ to be held (or called from const context
    // where patterns_ won't change — callers must hold mu_ appropriately).
    float score_against_(const FSMPattern& pat,
                         const RequestTrajectory& traj) const;

    // Merge consecutive states within d_merge, then cap at NS.
    std::vector<FSMState> merge_states_(std::vector<FSMState> states, int dim) const;

    // Merge the two globally most-similar patterns (Jaccard on transitions).
    void merge_two_most_similar_();

    float get_delta_(int cid) const;
    void  update_delta_(int cid, float dist);

    FSMConfig                       cfg_;
    std::vector<FSMPattern>         patterns_;
    std::unordered_map<int,float>   delta_sum_;
    std::unordered_map<int,int>     delta_cnt_;
    mutable std::mutex              mu_;
};

} // namespace fsm
} // namespace m3

#endif // M3_WITH_FSM
