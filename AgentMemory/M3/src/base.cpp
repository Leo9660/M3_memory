#include "base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef __AVX2__
#  include <immintrin.h>
#endif

namespace m3 {

// ---------------- Metric helpers ----------------

float l2_dist(const float* a, const float* b, int d) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    // horizontal sum of 8-lane accumulator
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    // scalar tail for d not divisible by 8
    for (; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
#endif
}

float ip_score(const float* a, const float* b, int d) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= d; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float s = _mm_cvtss_f32(lo);
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
#endif
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
