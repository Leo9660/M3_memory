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

#ifdef __AVX2__
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}
#endif

float ip_score(const float* a, const float* b, int d) {
#ifdef __AVX2__
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= d; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i   ), _mm256_loadu_ps(b+i   ), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+ 8), _mm256_loadu_ps(b+i+ 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float s = hsum256(acc0);
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
#endif
}

float l2_dist(const float* a, const float* b, int d) {
#ifdef __AVX2__
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 32 <= d; i += 32) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a+i   ), _mm256_loadu_ps(b+i   ));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a+i+ 8), _mm256_loadu_ps(b+i+ 8));
        __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16));
        __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    acc2 = _mm256_add_ps(acc2, acc3);
    acc0 = _mm256_add_ps(acc0, acc2);
    float s = hsum256(acc0);
    for (; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
#endif
}

float cos_dist(const float* a, const float* b, int d) {
    float dot = 0.f, na = 0.f, nb = 0.f;
    for (int i = 0; i < d; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.f || nb == 0.f) return 1.f;
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

    std::nth_element(buf.begin(), buf.begin() + k, buf.end(),
                     [](const Pair& a, const Pair& b){ return a.score < b.score; });
    buf.resize(static_cast<size_t>(k));
    std::sort(buf.begin(), buf.end(),
              [](const Pair& a, const Pair& b){ return a.score < b.score; });
}

} // namespace m3
