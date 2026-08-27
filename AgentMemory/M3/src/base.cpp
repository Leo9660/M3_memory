#include "base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef __AVX512F__
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#endif

namespace m3 {

// ---------------- Metric helpers ----------------

#ifdef __AVX512F__
static inline float hsum512(__m512 v) {
    // reduce 16-wide to 8-wide, then use hsum256
    __m256 lo = _mm512_castps512_ps256(v);
    __m256 hi = _mm512_extractf32x8_ps(v, 1);
    lo = _mm256_add_ps(lo, hi);
    __m128 a = _mm256_castps256_ps128(lo);
    __m128 b = _mm256_extractf128_ps(lo, 1);
    a = _mm_add_ps(a, b);
    a = _mm_hadd_ps(a, a);
    a = _mm_hadd_ps(a, a);
    return _mm_cvtss_f32(a);
}
#elif defined(__AVX2__)
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
#ifdef __AVX512F__
    // 8 ZMM accumulators: 8 × 16 = 128 floats per iteration.
    // Zen 4 has 2 FMA units × 4-cycle latency → needs 8 in-flight chains to
    // saturate both units with no stalls.  4 accumulators left one unit idle.
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 128 <= d; i += 128) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i   ), _mm512_loadu_ps(b+i   ), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
        acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+64), _mm512_loadu_ps(b+i+64), acc4);
        acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+80), _mm512_loadu_ps(b+i+80), acc5);
        acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+96), _mm512_loadu_ps(b+i+96), acc6);
        acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+112),_mm512_loadu_ps(b+i+112),acc7);
    }
    // 64-float cleanup for dims that are multiples of 64 but not 128 (e.g. d=64)
    for (; i + 64 <= d; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i   ), _mm512_loadu_ps(b+i   ), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48), acc3);
    }
    // tree reduction
    acc0 = _mm512_add_ps(acc0, acc4);
    acc1 = _mm512_add_ps(acc1, acc5);
    acc2 = _mm512_add_ps(acc2, acc6);
    acc3 = _mm512_add_ps(acc3, acc7);
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float s = hsum512(acc0);
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
#elif defined(__AVX2__)
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
#ifdef __AVX512F__
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 128 <= d; i += 128) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a+i   ), _mm512_loadu_ps(b+i   ));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48));
        __m512 d4 = _mm512_sub_ps(_mm512_loadu_ps(a+i+64), _mm512_loadu_ps(b+i+64));
        __m512 d5 = _mm512_sub_ps(_mm512_loadu_ps(a+i+80), _mm512_loadu_ps(b+i+80));
        __m512 d6 = _mm512_sub_ps(_mm512_loadu_ps(a+i+96), _mm512_loadu_ps(b+i+96));
        __m512 d7 = _mm512_sub_ps(_mm512_loadu_ps(a+i+112),_mm512_loadu_ps(b+i+112));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
        acc4 = _mm512_fmadd_ps(d4, d4, acc4);
        acc5 = _mm512_fmadd_ps(d5, d5, acc5);
        acc6 = _mm512_fmadd_ps(d6, d6, acc6);
        acc7 = _mm512_fmadd_ps(d7, d7, acc7);
    }
    for (; i + 64 <= d; i += 64) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a+i   ), _mm512_loadu_ps(b+i   ));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a+i+16), _mm512_loadu_ps(b+i+16));
        __m512 d2 = _mm512_sub_ps(_mm512_loadu_ps(a+i+32), _mm512_loadu_ps(b+i+32));
        __m512 d3 = _mm512_sub_ps(_mm512_loadu_ps(a+i+48), _mm512_loadu_ps(b+i+48));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
        acc2 = _mm512_fmadd_ps(d2, d2, acc2);
        acc3 = _mm512_fmadd_ps(d3, d3, acc3);
    }
    acc0 = _mm512_add_ps(acc0, acc4);
    acc1 = _mm512_add_ps(acc1, acc5);
    acc2 = _mm512_add_ps(acc2, acc6);
    acc3 = _mm512_add_ps(acc3, acc7);
    acc0 = _mm512_add_ps(acc0, acc1);
    acc2 = _mm512_add_ps(acc2, acc3);
    acc0 = _mm512_add_ps(acc0, acc2);
    float s = hsum512(acc0);
    for (; i < d; ++i) { float df = a[i] - b[i]; s += df * df; }
    return s;
#elif defined(__AVX2__)
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
