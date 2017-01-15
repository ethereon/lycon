#pragma once

#include <cmath>

#include "lycon/defs.h"

namespace lycon
{
/** @brief Rounds floating-point number to the nearest integer

 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
static inline int fast_round(double value)
{
#if LYCON_SSE2
    __m128d t = _mm_set_sd(value);
    return _mm_cvtsd_si32(t);
#else
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
#endif
}

/** @brief Rounds floating-point number to the nearest integer not larger than the original.

 The function computes an integer i such that:
 \f[i \le \texttt{value} < i+1\f]
 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
static inline int fast_floor(double value)
{
#if LYCON_SSE2
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i - _mm_movemask_pd(_mm_cmplt_sd(t, _mm_cvtsi32_sd(t, i)));
#else
    int i = fast_round(value);
    float diff = (float)(value - i);
    return i - (diff < 0);
#endif
}

/** @brief Rounds floating-point number to the nearest integer not smaller than the original.

 The function computes an integer i such that:
 \f[i \le \texttt{value} < i+1\f]
 @param value floating-point number. If the value is outside of INT_MIN ... INT_MAX range, the
 result is not defined.
 */
static inline int fast_ceil(double value)
{
#if LYCON_SSE2
    __m128d t = _mm_set_sd(value);
    int i = _mm_cvtsd_si32(t);
    return i + _mm_movemask_pd(_mm_cmplt_sd(_mm_cvtsi32_sd(t, i), t));
#else
    int i = fast_round(value);
    float diff = (float)(i - value);
    return i + (diff < 0);
#endif
}

/** @overload */
static inline int fast_round(float value)
{
#if LYCON_SSE2
    __m128 t = _mm_set_ss(value);
    return _mm_cvtss_si32(t);
#else
    /* it's ok if round does not comply with IEEE754 standard;
     the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

/** @overload */
static inline int fast_round(int value) { return value; }

/** @overload */
static inline int fast_floor(float value)
{
#if LYCON_SSE2
    __m128 t = _mm_set_ss(value);
    int i = _mm_cvtss_si32(t);
    return i - _mm_movemask_ps(_mm_cmplt_ss(t, _mm_cvtsi32_ss(t, i)));
#else
    int i = fast_round(value);
    float diff = (float)(value - i);
    return i - (diff < 0);
#endif
}

/** @overload */
static inline int fast_floor(int value) { return value; }

/** @overload */
static inline int fast_ceil(float value)
{
#if LYCON_SSE2
    __m128 t = _mm_set_ss(value);
    int i = _mm_cvtss_si32(t);
    return i + _mm_movemask_ps(_mm_cmplt_ss(_mm_cvtsi32_ss(t, i), t));
#else
    int i = fast_round(value);
    float diff = (float)(i - value);
    return i + (diff < 0);
#endif
}

/** @overload */
static inline int fast_ceil(int value) { return value; }
}
