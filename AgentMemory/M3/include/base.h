#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>

namespace m3 {

// ----- Common metric & id -----
enum class Metric : int { L2 = 0, IP = 1, COSINE = 2 };
using DocId = int64_t;

// ----- Low-level metric helpers (defined in base.cpp) -----
float l2_dist (const float* a, const float* b, int d);   // squared L2 distance
float ip_score(const float* a, const float* b, int d);   // dot product
float cos_dist(const float* a, const float* b, int d);   // 1 - cosine(a,b)

// ----- Unified scoring: "smaller is better" for all metrics -----
// L2      -> l2_dist
// IP      -> -ip_score
// COSINE  -> if normalized: 1 - dot; else: cos_dist
inline float unified_score(const float* q, const float* v, int dim,
                           Metric metric, bool normalized) {
    switch (metric) {
        case Metric::L2:     return l2_dist(q, v, dim);
        case Metric::IP:     return -ip_score(q, v, dim);
        case Metric::COSINE: return normalized ? (1.f - ip_score(q, v, dim))
                                               :  cos_dist(q, v, dim);
    }
    return std::numeric_limits<float>::infinity();
}

// ----- Generic pair + top-k helper (defined in base.cpp) -----
struct Pair { float score; DocId id; };

// Keep the smallest k items in-place (ascending by score).
// If buf.size() <= k, sorts ascending; otherwise nth_element + shrink + sort.
void topk_smallest(std::vector<Pair>& buf, int k);

} // namespace m3
