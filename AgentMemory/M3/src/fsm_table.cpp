// fsm_table.cpp — FSM pattern table implementation.
// Compiled only when -DM3_WITH_FSM is defined.

#ifndef M3_WITH_FSM
// Nothing to compile.
#else

#include "fsm_table.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace m3 {
namespace fsm {

// ============================================================
// FSMTable
// ============================================================

FSMTable::FSMTable(FSMConfig cfg) : cfg_(std::move(cfg)) {}

// ------------------------------------------------------------------
// Private helpers
// ------------------------------------------------------------------

float FSMTable::get_delta_(int cid) const {
    auto it = delta_cnt_.find(cid);
    if (it == delta_cnt_.end() || it->second == 0) return 1.0f;
    return delta_sum_.at(cid) / static_cast<float>(it->second);
}

void FSMTable::update_delta_(int cid, float dist) {
    delta_sum_[cid] += dist;
    delta_cnt_[cid] += 1;
}

float FSMTable::score_against_(const FSMPattern& pat,
                                const RequestTrajectory& traj) const {
    const auto& steps = traj.steps;
    const auto& vecs  = traj.step_vecs;
    const int t = static_cast<int>(steps.size());
    float score = 0.0f;

    for (int k = 1; k < t; ++k) {
        int prev_cid = steps[k - 1];
        int curr_cid = steps[k];
        if (pat.transitions.find({prev_cid, curr_cid}) == pat.transitions.end())
            continue;

        float delta_k = get_delta_(curr_cid);

        // Find centroid for curr_cid in this pattern.
        float dist_k = 0.0f;
        auto cidx_it = pat.cid_to_idx.find(curr_cid);
        if (cidx_it != pat.cid_to_idx.end() && k < static_cast<int>(vecs.size())) {
            const auto& c    = pat.states[cidx_it->second].centroid;
            const auto& v    = vecs[k];
            const int   dim  = static_cast<int>(c.size());
            float sq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float e = c[d] - v[d];
                sq += e * e;
            }
            dist_k = std::sqrt(sq);
        }

        score += delta_k / (1.0f + dist_k);
    }
    return score;
}

// Pass 1: merge consecutive states within d_merge (keep smaller δ).
// Pass 2: cap at NS via closest-pair merge.
std::vector<FSMState> FSMTable::merge_states_(std::vector<FSMState> states,
                                               int dim) const {
    if (states.empty()) return states;

    // Pass 1
    std::vector<FSMState> merged;
    merged.reserve(states.size());
    merged.push_back(std::move(states[0]));
    for (int i = 1; i < static_cast<int>(states.size()); ++i) {
        auto& prev = merged.back();
        auto& cur  = states[i];
        // Squared-L2 between centroids (sqrt not needed since we compare to d_merge²).
        float sq = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float e = prev.centroid[d] - cur.centroid[d];
            sq += e * e;
        }
        if (std::sqrt(sq) < cfg_.d_merge) {
            // Keep the state with smaller δ (tighter cluster).
            if (cur.delta < prev.delta) prev = std::move(cur);
        } else {
            merged.push_back(std::move(cur));
        }
    }

    // Pass 2: cap at NS by merging closest consecutive pair repeatedly.
    while (static_cast<int>(merged.size()) > cfg_.ns_max_states) {
        float best_sq = std::numeric_limits<float>::infinity();
        int   best_i  = 0;
        for (int i = 0; i < static_cast<int>(merged.size()) - 1; ++i) {
            float sq = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float e = merged[i].centroid[d] - merged[i+1].centroid[d];
                sq += e * e;
            }
            if (sq < best_sq) { best_sq = sq; best_i = i; }
        }
        // Drop the state with larger δ.
        if (merged[best_i].delta <= merged[best_i + 1].delta)
            merged.erase(merged.begin() + best_i + 1);
        else
            merged.erase(merged.begin() + best_i);
    }

    return merged;
}

void FSMTable::merge_two_most_similar_() {
    if (patterns_.size() < 2) return;
    float best_sim = -1.0f;
    int   best_i = 0, best_j = 1;
    for (int i = 0; i < static_cast<int>(patterns_.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(patterns_.size()); ++j) {
            const auto& A = patterns_[i].transitions;
            const auto& B = patterns_[j].transitions;
            // Jaccard on transition sets.
            int inter = 0;
            for (const auto& t : A)
                if (B.count(t)) ++inter;
            int uni = static_cast<int>(A.size() + B.size()) - inter;
            float sim = (uni == 0) ? 1.0f : (static_cast<float>(inter) / uni);
            if (sim > best_sim) { best_sim = sim; best_i = i; best_j = j; }
        }
    }
    auto& pa = patterns_[best_i];
    auto& pb = patterns_[best_j];
    // Merge pb into pa: union transitions + keep longer state list.
    pa.add_transitions(pb.transitions, pb.states);
    if (pb.states.size() > pa.states.size())
        pa.states = pb.states;
    pa.rebuild_cid_map();
    patterns_.erase(patterns_.begin() + best_j);
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

std::vector<int> FSMTable::match_and_predict(const RequestTrajectory& traj) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (traj.steps.empty() || patterns_.empty()) return {};

    float best_score = 0.0f;  // only return if score > 0
    const FSMPattern* best_pat = nullptr;

    for (const auto& pat : patterns_) {
        if (pat.hit_count < cfg_.min_hits_to_predict) continue;
        float s = score_against_(pat, traj);
        if (s > best_score) { best_score = s; best_pat = &pat; }
    }

    if (best_pat == nullptr) return {};
    return best_pat->predict_next(traj.steps.back());
}

void FSMTable::update_from_trajectory(const RequestTrajectory& traj,
                                       const float* centroids,
                                       int nlist, int dim) {
    const auto& steps = traj.steps;
    const auto& vecs  = traj.step_vecs;
    if (static_cast<int>(steps.size()) < cfg_.min_traj_len) return;

    std::lock_guard<std::mutex> lk(mu_);

    // 1. Build raw states and update δ estimates.
    std::vector<FSMState> raw_states;
    raw_states.reserve(steps.size());
    for (int i = 0; i < static_cast<int>(steps.size()); ++i) {
        int cid = steps[i];
        FSMState s;
        s.cid = cid;

        // Centroid for this cluster.
        if (centroids && cid >= 0 && cid < nlist) {
            s.centroid.assign(centroids + cid * dim,
                              centroids + cid * dim + dim);
        } else {
            s.centroid.assign(dim, 0.0f);
        }

        // Distance from query to centroid → feed δ estimate.
        float dist = 0.0f;
        if (i < static_cast<int>(vecs.size()) && !s.centroid.empty()) {
            for (int d = 0; d < dim; ++d) {
                float e = s.centroid[d] - vecs[i][d];
                dist += e * e;
            }
            dist = std::sqrt(dist);
        }
        update_delta_(cid, dist);
        s.delta = get_delta_(cid);
        raw_states.push_back(std::move(s));
    }

    // 2. Find best-matching existing pattern (paper similarity formula).
    float best_score = -1.0f;
    int   best_idx   = -1;
    for (int i = 0; i < static_cast<int>(patterns_.size()); ++i) {
        float s = score_against_(patterns_[i], traj);
        if (s > best_score) { best_score = s; best_idx = i; }
    }

    if (best_idx >= 0 && best_score >= cfg_.reinforce_threshold) {
        // 3a. Reinforce: union transitions, keep existing states (NO merge).
        std::set<std::pair<int,int>> new_trans;
        for (int k = 1; k < static_cast<int>(raw_states.size()); ++k)
            new_trans.insert({raw_states[k-1].cid, raw_states[k].cid});
        patterns_[best_idx].add_transitions(new_trans, raw_states);
    } else {
        // 3b. Create new FSM: merge states, build transitions.
        auto merged = merge_states_(raw_states, dim);
        std::set<std::pair<int,int>> trans;
        for (int k = 1; k < static_cast<int>(merged.size()); ++k)
            trans.insert({merged[k-1].cid, merged[k].cid});
        FSMPattern pat;
        pat.states      = std::move(merged);
        pat.transitions = std::move(trans);
        pat.hit_count   = 1;
        pat.rebuild_cid_map();
        patterns_.push_back(std::move(pat));
    }

    // 4. Global prune.
    if (static_cast<int>(patterns_.size()) > cfg_.max_patterns)
        merge_two_most_similar_();
}

int FSMTable::num_patterns() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(patterns_.size());
}

int FSMTable::total_hits() const {
    std::lock_guard<std::mutex> lk(mu_);
    int total = 0;
    for (const auto& p : patterns_) total += p.hit_count;
    return total;
}

} // namespace fsm
} // namespace m3

#endif // M3_WITH_FSM
