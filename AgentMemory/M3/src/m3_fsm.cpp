#include "m3_fsm.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace m3 {

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

static uint64_t fsm_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
}

static float vec_l2(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) { float d = a[i] - b[i]; s += d * d; }
    return std::sqrt(s);
}

// Extract centroid for cluster cid from a flat [nlist*dim] array.
static std::vector<float> extract_centroid(const std::vector<float>& centroids,
                                           int cid, int dim) {
    if (cid < 0 || dim <= 0) return {};
    const size_t offset = static_cast<size_t>(cid) * static_cast<size_t>(dim);
    if (offset + static_cast<size_t>(dim) > centroids.size()) return {};
    return std::vector<float>(
        centroids.begin() + static_cast<std::ptrdiff_t>(offset),
        centroids.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(dim)));
}

// ======================================================================
// FSMPattern
// ======================================================================

void FSMPattern::add_or_update_state(int cid, float delta,
                                     const std::vector<float>& centroid) {
    auto it = states.find(cid);
    if (it == states.end()) {
        states.emplace(cid, FSMState(cid, delta, centroid));
    } else {
        it->second.delta = 0.9f * it->second.delta + 0.1f * delta;
        if (!centroid.empty() && centroid.size() == it->second.centroid.size()) {
            for (size_t i = 0; i < centroid.size(); ++i)
                it->second.centroid[i] = 0.9f * it->second.centroid[i] + 0.1f * centroid[i];
        }
    }
}

void FSMPattern::add_transition(int from, int to) { transitions[from].insert(to); }

bool FSMPattern::has_transition(int from, int to) const {
    auto it = transitions.find(from);
    return (it != transitions.end()) && it->second.count(to);
}

int FSMPattern::predict_next(int last_cluster) const {
    auto it = transitions.find(last_cluster);
    if (it == transitions.end() || it->second.empty()) return -1;
    int best_cid = -1;
    float best_val = std::numeric_limits<float>::infinity();
    for (int to : it->second) {
        auto sit = states.find(to);
        float score = (sit != states.end()) ? sit->second.delta : 1.0f;
        if (score < best_val) { best_val = score; best_cid = to; }
    }
    return best_cid;
}

// TODO-3: ranked prediction — all outgoing transitions sorted by delta asc.
std::vector<int> FSMPattern::predict_next_ranked(int last_cluster) const {
    auto it = transitions.find(last_cluster);
    if (it == transitions.end() || it->second.empty()) return {};

    std::vector<std::pair<float, int>> by_delta;
    by_delta.reserve(it->second.size());
    for (int to : it->second) {
        auto sit = states.find(to);
        float delta = (sit != states.end()) ? sit->second.delta : 1.0f;
        by_delta.emplace_back(delta, to);
    }
    std::sort(by_delta.begin(), by_delta.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    std::vector<int> result;
    result.reserve(by_delta.size());
    for (const auto& p : by_delta) result.push_back(p.second);
    return result;
}

size_t FSMPattern::num_transitions() const {
    size_t n = 0;
    for (const auto& kv : transitions) n += kv.second.size();
    return n;
}

// ======================================================================
// RequestTrajectory
// ======================================================================

void RequestTrajectory::append_step(std::vector<float> embedding,
                                    int cluster_id,
                                    FSMLayer layer,
                                    uint64_t time_ns) {
    embeddings.push_back(std::move(embedding));
    steps.push_back({cluster_id, layer});
    timestamps_ns.push_back(time_ns > 0 ? time_ns : fsm_now_ns());
}

void RequestTrajectory::finalize() { is_complete = true; }

std::vector<int> RequestTrajectory::cluster_id_sequence() const {
    std::vector<int> seq;
    seq.reserve(steps.size());
    for (const auto& s : steps) seq.push_back(s.cluster_id);
    return seq;
}

// ======================================================================
// FSMTable
// ======================================================================

FSMTable::FSMTable(FSMConfig cfg) : cfg_(std::move(cfg)) {}

float FSMTable::compute_similarity_transitions(
        const FSMPattern& pat, const std::vector<int>& cluster_seq) const {
    const int T = static_cast<int>(cluster_seq.size());
    if (T < 2) return 0.0f;
    float score = 0.0f;
    const float step = 1.0f / static_cast<float>(T - 1);
    for (int k = 1; k < T; ++k) {
        if (pat.has_transition(cluster_seq[k - 1], cluster_seq[k]))
            score += step;
        else
            score -= 0.5f * step;
    }
    return score;
}

float FSMTable::compute_similarity_full(
        const FSMPattern& pat,
        const std::vector<int>& cluster_seq,
        const std::vector<std::vector<float>>& embedding_seq) const {
    const int T = static_cast<int>(cluster_seq.size());
    if (T == 0) return 0.0f;
    assert(cluster_seq.size() == embedding_seq.size());
    float total = 0.0f;
    for (int k = 0; k < T; ++k) {
        int cid = cluster_seq[k];
        auto sit = pat.states.find(cid);
        float delta = (sit != pat.states.end()) ? sit->second.delta : 1.0f;
        float dist  = 0.0f;
        if (sit != pat.states.end() && !sit->second.centroid.empty()
                && k < static_cast<int>(embedding_seq.size()))
            dist = vec_l2(sit->second.centroid, embedding_seq[k]);
        float indicator = (k == 0)
            ? ((sit != pat.states.end()) ? 1.0f : 0.0f)
            : (pat.has_transition(cluster_seq[k - 1], cid) ? 1.0f : 0.0f);
        total += indicator * delta / (1.0f + dist);
    }
    return total / static_cast<float>(T);
}

// ======================================================================
// match_and_predict — returns a ranked cluster list (TODO-3)
// ======================================================================
FSMTable::PredictResult FSMTable::match_and_predict(
        const RequestTrajectory& traj,
        FSMLayer /*layer*/) const {

    std::lock_guard<std::mutex> lk(fsm_mu_);
    PredictResult result;
    if (patterns_.empty() || traj.length() == 0) return result;

    const auto cseq  = traj.cluster_id_sequence();
    const auto& eseq = traj.embeddings;

    float best_score = -std::numeric_limits<float>::infinity();
    int   best_id    = -1;
    for (int pid : order_) {
        auto it = patterns_.find(pid);
        if (it == patterns_.end()) continue;
        float score = (!eseq.empty() && eseq.size() == cseq.size())
            ? compute_similarity_full(it->second, cseq, eseq)
            : compute_similarity_transitions(it->second, cseq);
        if (score > best_score) { best_score = score; best_id = pid; }
    }
    if (best_id == -1 || best_score < cfg_.match_threshold) return result;

    const FSMPattern& winner = patterns_.at(best_id);
    result.ranked_clusters  = winner.predict_next_ranked(cseq.back());
    result.best_score       = best_score;
    result.best_pattern_id  = best_id;
    return result;
}

// ======================================================================
// update_from_trajectory — takes flat centroid array + dim (BUG-2 fix)
// ======================================================================
void FSMTable::update_from_trajectory(
        const RequestTrajectory& traj,
        const std::vector<float>& cluster_centroids,
        int dim,
        float cluster_delta_default) {

    if (traj.length() < 2) return;
    std::lock_guard<std::mutex> lk(fsm_mu_);

    const auto cseq  = traj.cluster_id_sequence();
    const auto& eseq = traj.embeddings;

    float best_score = -std::numeric_limits<float>::infinity();
    int   best_id    = -1;
    for (int pid : order_) {
        auto it = patterns_.find(pid);
        if (it == patterns_.end()) continue;
        float score = (!eseq.empty() && eseq.size() == cseq.size())
            ? compute_similarity_full(it->second, cseq, eseq)
            : compute_similarity_transitions(it->second, cseq);
        if (score > best_score) { best_score = score; best_id = pid; }
    }

    if (best_id != -1 && best_score >= cfg_.merge_threshold) {
        FSMPattern& pat = patterns_.at(best_id);
        pat.frequency++;
        pat.last_used = fsm_now_ns();
        for (int k = 0; k < static_cast<int>(cseq.size()); ++k) {
            int cid = cseq[k];
            std::vector<float> cent = extract_centroid(cluster_centroids, cid, dim);
            float delta = cluster_delta_default;
            if (!cent.empty() && k < static_cast<int>(eseq.size())) {
                delta = vec_l2(cent, eseq[k]);
                if (delta < 1e-6f) delta = 1e-6f;
            }
            pat.add_or_update_state(cid, delta, cent);
        }
        for (int k = 1; k < static_cast<int>(cseq.size()); ++k)
            pat.add_transition(cseq[k - 1], cseq[k]);
        return;
    }

    FSMPattern new_pat = make_pattern_from_trajectory_(
        traj, cluster_centroids, dim, cluster_delta_default);
    new_pat.last_used  = fsm_now_ns();
    int new_id = next_id_++;
    new_pat.pattern_id = new_id;
    patterns_.emplace(new_id, std::move(new_pat));
    order_.push_back(new_id);

    if (static_cast<int>(patterns_.size()) > cfg_.max_patterns)
        evict_by_merging_();
}

// ======================================================================
// invalidate_cluster  (BUG-3 / Q-2 fix — called on L0 topology changes)
// ======================================================================
void FSMTable::invalidate_cluster(int cluster_id) {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    for (auto& [pid, pat] : patterns_) {
        pat.states.erase(cluster_id);
        pat.transitions.erase(cluster_id);
        for (auto& [from, tos] : pat.transitions) tos.erase(cluster_id);
    }
}

// ======================================================================
// merge_patterns  (public)
// ======================================================================
int FSMTable::merge_patterns(int id_a, int id_b) {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    if (!patterns_.count(id_a) || !patterns_.count(id_b))
        throw std::invalid_argument("FSMTable::merge_patterns: unknown pattern id");
    int keep = (patterns_.at(id_a).frequency >= patterns_.at(id_b).frequency) ? id_a : id_b;
    merge_nolock_(id_a, id_b);
    return keep;
}

int FSMTable::num_patterns() const {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    return static_cast<int>(patterns_.size());
}

const FSMPattern* FSMTable::get_pattern(int pattern_id) const {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    auto it = patterns_.find(pattern_id);
    return (it != patterns_.end()) ? &it->second : nullptr;
}

void FSMTable::clear() {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    patterns_.clear();
    order_.clear();
}

void FSMTable::set_config(FSMConfig cfg) {
    std::lock_guard<std::mutex> lk(fsm_mu_);
    cfg_ = std::move(cfg);
}

// ======================================================================
// Private helpers
// ======================================================================

FSMPattern FSMTable::make_pattern_from_trajectory_(
        const RequestTrajectory& traj,
        const std::vector<float>& cluster_centroids,
        int dim,
        float cluster_delta_default) const {
    FSMPattern pat;
    pat.frequency = 1;
    const auto cseq  = traj.cluster_id_sequence();
    const auto& eseq = traj.embeddings;
    for (int k = 0; k < static_cast<int>(cseq.size()); ++k) {
        int cid = cseq[k];
        if (cid < 0) continue;
        std::vector<float> cent = extract_centroid(cluster_centroids, cid, dim);
        float delta = cluster_delta_default;
        if (!cent.empty() && k < static_cast<int>(eseq.size())) {
            delta = vec_l2(cent, eseq[k]);
            if (delta < 1e-6f) delta = 1e-6f;
        }
        pat.add_or_update_state(cid, delta, cent);
    }
    for (int k = 1; k < static_cast<int>(cseq.size()); ++k)
        if (cseq[k - 1] >= 0 && cseq[k] >= 0)
            pat.add_transition(cseq[k - 1], cseq[k]);
    return pat;
}

void FSMTable::merge_nolock_(int id_a, int id_b) {
    auto ita = patterns_.find(id_a);
    auto itb = patterns_.find(id_b);
    if (ita == patterns_.end() || itb == patterns_.end()) return;

    FSMPattern& dominant  = (ita->second.frequency >= itb->second.frequency)
                          ? ita->second : itb->second;
    FSMPattern& recessive = (ita->second.frequency >= itb->second.frequency)
                          ? itb->second : ita->second;
    int remove_id = recessive.pattern_id;

    for (auto& [cid, state] : recessive.states)
        if (!dominant.states.count(cid)) dominant.states.emplace(cid, state);
    for (auto& [from, tos] : recessive.transitions)
        for (int to : tos) dominant.transitions[from].insert(to);

    dominant.frequency += recessive.frequency;
    dominant.last_used  = std::max(dominant.last_used, recessive.last_used);

    if (static_cast<int>(dominant.states.size()) > cfg_.max_states_per_fsm) {
        std::vector<std::pair<float, int>> by_delta;
        by_delta.reserve(dominant.states.size());
        for (auto& [cid, st] : dominant.states) by_delta.emplace_back(st.delta, cid);
        std::sort(by_delta.begin(), by_delta.end(),
                  [](const auto& a, const auto& b){ return a.first > b.first; });
        int excess = static_cast<int>(dominant.states.size()) - cfg_.max_states_per_fsm;
        for (int i = 0; i < excess; ++i) {
            int drop = by_delta[static_cast<size_t>(i)].second;
            dominant.states.erase(drop);
            dominant.transitions.erase(drop);
            for (auto& [from, tos] : dominant.transitions) tos.erase(drop);
        }
    }
    patterns_.erase(remove_id);
    order_.erase(std::remove(order_.begin(), order_.end(), remove_id), order_.end());
}

void FSMTable::evict_by_merging_() {
    if (patterns_.size() < 2) return;
    float best_sim = -std::numeric_limits<float>::infinity();
    int best_a = -1, best_b = -1;
    for (size_t i = 0; i < order_.size(); ++i) {
        for (size_t j = i + 1; j < order_.size(); ++j) {
            int ia = order_[i], ib = order_[j];
            const FSMPattern& pa = patterns_.at(ia);
            const FSMPattern& pb = patterns_.at(ib);
            std::vector<int> sa, sb;
            for (auto& [cid, _] : pa.states) sa.push_back(cid);
            for (auto& [cid, _] : pb.states) sb.push_back(cid);
            float sim = 0.5f * (compute_similarity_transitions(pa, sb)
                              + compute_similarity_transitions(pb, sa));
            if (sim > best_sim) { best_sim = sim; best_a = ia; best_b = ib; }
        }
    }
    if (best_a != -1) merge_nolock_(best_a, best_b);
}

} // namespace m3
