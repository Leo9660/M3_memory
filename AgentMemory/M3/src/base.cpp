#include "base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace m3 {

// ---------------- Metric helpers ----------------

float l2_dist(const float* a, const float* b, int d) {
    // Squared L2 distance
    float s = 0.f;
    for (int i = 0; i < d; ++i) {
        float df = a[i] - b[i];
        s += df * df;
    }
    return s;
}

float ip_score(const float* a, const float* b, int d) {
    // Dot product
    float s = 0.f;
    for (int i = 0; i < d; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

float cos_dist(const float* a, const float* b, int d) {
    // 1 - cosine similarity
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < d; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.f || nb == 0.f) return 1.f; // treat zero vector as maximally distant
    return 1.f - dot / std::sqrt(na * nb);
}

// ---------------- Top-k helper ----------------

void topk_smallest(std::vector<Pair>& buf, int k) {
    if (k <= 0 || buf.empty()) {
        buf.clear();
        return;
    }
    if (static_cast<int>(buf.size()) <= k) {
        std::sort(buf.begin(), buf.end(),
                  [](const Pair& a, const Pair& b){ return a.score < b.score; });
        return;
    }

    // Partition so that elements < buf[k] are in front (unordered),
    // then shrink to k and sort the front region.
    std::nth_element(buf.begin(), buf.begin() + k, buf.end(),
                     [](const Pair& a, const Pair& b){ return a.score < b.score; });
    buf.resize(static_cast<size_t>(k));
    std::sort(buf.begin(), buf.end(),
              [](const Pair& a, const Pair& b){ return a.score < b.score; });
}

} // namespace m3
