#include "lycon/mat/convert.h"

#include <cfloat>
#include <cstring>

#include "lycon/mat/mat.h"
#include "lycon/mat/shared.h"
#include "lycon/util/hardware.h"
#include "lycon/util/util.h"

namespace lycon
{
// Allows for potential tegra/hal optimizations later
#define GET_OPTIMIZED(func) (func)

template <typename T, typename DT, typename WT> struct cvtScaleAbs_SIMD
{
    int operator()(const T*, DT*, int, WT, WT) const
    {
        return 0;
    }
};

#if LYCON_SSE2

template <> struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator()(const uchar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i*)(src + x));
                __m128i v_src12 = _mm_unpacklo_epi8(v_src, v_zero_i), v_src_34 = _mm_unpackhi_epi8(v_src, v_zero_i);
                __m128 v_dst1 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src12, v_zero_i)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src12, v_zero_i)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);
                __m128 v_dst3 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src_34, v_zero_i)), v_scale), v_shift);
                v_dst3 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst3), v_dst3);
                __m128 v_dst4 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src_34, v_zero_i)), v_scale), v_shift);
                v_dst4 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst4), v_dst4);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)),
                                                   _mm_packs_epi32(_mm_cvtps_epi32(v_dst3), _mm_cvtps_epi32(v_dst4)));
                _mm_storeu_si128((__m128i*)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator()(const schar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i*)(src + x));
                __m128i v_src_12 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero_i, v_src), 8),
                        v_src_34 = _mm_srai_epi16(_mm_unpackhi_epi8(v_zero_i, v_src), 8);
                __m128 v_dst1 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero_i, v_src_12), 16)), v_scale),
                    v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero_i, v_src_12), 16)), v_scale),
                    v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);
                __m128 v_dst3 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero_i, v_src_34), 16)), v_scale),
                    v_shift);
                v_dst3 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst3), v_dst3);
                __m128 v_dst4 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero_i, v_src_34), 16)), v_scale),
                    v_shift);
                v_dst4 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst4), v_dst4);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)),
                                                   _mm_packs_epi32(_mm_cvtps_epi32(v_dst3), _mm_cvtps_epi32(v_dst4)));
                _mm_storeu_si128((__m128i*)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator()(const ushort* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 8; x += 8)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i*)(src + x));
                __m128 v_dst1 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero_i)), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 =
                    _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero_i)), v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i =
                    _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)), v_zero_i);
                _mm_storel_epi64((__m128i*)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator()(const short* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 8; x += 8)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i*)(src + x));
                __m128 v_dst1 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_src, v_src), 16)), v_scale),
                    v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);
                __m128 v_dst2 = _mm_add_ps(
                    _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_src, v_src), 16)), v_scale),
                    v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i =
                    _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2)), v_zero_i);
                _mm_storel_epi64((__m128i*)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator()(const int* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 8; x += 4)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i*)(src + x));
                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);

                __m128i v_dst_i = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(v_dst1), v_zero_i), v_zero_i);
                _mm_storel_epi64((__m128i*)(dst + x), v_dst_i);
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator()(const float* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 8; x += 4)
            {
                __m128 v_dst = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + x), v_scale), v_shift);
                v_dst = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst), v_dst);

                __m128i v_dst_i = _mm_packs_epi32(_mm_cvtps_epi32(v_dst), v_zero_i);
                _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst_i, v_zero_i));
            }
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<double, uchar, float>
{
    int operator()(const double* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (USE_SSE2)
        {
            __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift), v_zero_f = _mm_setzero_ps();
            __m128i v_zero_i = _mm_setzero_si128();

            for (; x <= width - 8; x += 8)
            {
                __m128 v_src1 =
                    _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
                __m128 v_src2 =
                    _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));

                __m128 v_dst1 = _mm_add_ps(_mm_mul_ps(v_src1, v_scale), v_shift);
                v_dst1 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst1), v_dst1);

                __m128 v_dst2 = _mm_add_ps(_mm_mul_ps(v_src2, v_scale), v_shift);
                v_dst2 = _mm_max_ps(_mm_sub_ps(v_zero_f, v_dst2), v_dst2);

                __m128i v_dst_i = _mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2));

                _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst_i, v_zero_i));
            }
        }

        return x;
    }
};

#elif LYCON_NEON

template <> struct cvtScaleAbs_SIMD<uchar, uchar, float>
{
    int operator()(const uchar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 16; x += 16)
        {
            uint8x16_t v_src = vld1q_u8(src + x);
            uint16x8_t v_half = vmovl_u8(vget_low_u8(v_src));

            uint32x4_t v_quat = vmovl_u16(vget_low_u16(v_half));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_quat = vmovl_u16(vget_high_u16(v_half));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            v_half = vmovl_u8(vget_high_u8(v_src));

            v_quat = vmovl_u16(vget_low_u16(v_half));
            float32x4_t v_dst_2 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_2 = vabsq_f32(vaddq_f32(v_dst_2, v_shift));

            v_quat = vmovl_u16(vget_high_u16(v_half));
            float32x4_t v_dst_3 = vmulq_n_f32(vcvtq_f32_u32(v_quat), scale);
            v_dst_3 = vabsq_f32(vaddq_f32(v_dst_3, v_shift));

            uint16x8_t v_dsti_0 =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));
            uint16x8_t v_dsti_1 =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_2)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_3)));

            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_dsti_0), vqmovn_u16(v_dsti_1)));
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<schar, uchar, float>
{
    int operator()(const schar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 16; x += 16)
        {
            int8x16_t v_src = vld1q_s8(src + x);
            int16x8_t v_half = vmovl_s8(vget_low_s8(v_src));

            int32x4_t v_quat = vmovl_s16(vget_low_s16(v_half));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_quat = vmovl_s16(vget_high_s16(v_half));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            v_half = vmovl_s8(vget_high_s8(v_src));

            v_quat = vmovl_s16(vget_low_s16(v_half));
            float32x4_t v_dst_2 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_2 = vabsq_f32(vaddq_f32(v_dst_2, v_shift));

            v_quat = vmovl_s16(vget_high_s16(v_half));
            float32x4_t v_dst_3 = vmulq_n_f32(vcvtq_f32_s32(v_quat), scale);
            v_dst_3 = vabsq_f32(vaddq_f32(v_dst_3, v_shift));

            uint16x8_t v_dsti_0 =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));
            uint16x8_t v_dsti_1 =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_2)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_3)));

            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_dsti_0), vqmovn_u16(v_dsti_1)));
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<ushort, uchar, float>
{
    int operator()(const ushort* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);

            uint32x4_t v_half = vmovl_u16(vget_low_u16(v_src));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_u32(v_half), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_half = vmovl_u16(vget_high_u16(v_src));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_u32(v_half), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            uint16x8_t v_dst =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));

            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<short, uchar, float>
{
    int operator()(const short* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);

            int32x4_t v_half = vmovl_s16(vget_low_s16(v_src));
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(v_half), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));

            v_half = vmovl_s16(vget_high_s16(v_src));
            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(v_half), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));

            uint16x8_t v_dst =
                vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst_0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst_1)));

            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<int, uchar, float>
{
    int operator()(const int* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst_0 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(src + x)), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));
            uint16x4_t v_dsti_0 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_0));

            float32x4_t v_dst_1 = vmulq_n_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));
            uint16x4_t v_dsti_1 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_1));

            uint16x8_t v_dst = vcombine_u16(v_dsti_0, v_dsti_1);
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScaleAbs_SIMD<float, uchar, float>
{
    int operator()(const float* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst_0 = vmulq_n_f32(vld1q_f32(src + x), scale);
            v_dst_0 = vabsq_f32(vaddq_f32(v_dst_0, v_shift));
            uint16x4_t v_dsti_0 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_0));

            float32x4_t v_dst_1 = vmulq_n_f32(vld1q_f32(src + x + 4), scale);
            v_dst_1 = vabsq_f32(vaddq_f32(v_dst_1, v_shift));
            uint16x4_t v_dsti_1 = vqmovn_u32(cv_vrndq_u32_f32(v_dst_1));

            uint16x8_t v_dst = vcombine_u16(v_dsti_0, v_dsti_1);
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

#endif

template <typename T, typename DT, typename WT>
static void cvtScaleAbs_(const T* src, size_t sstep, DT* dst, size_t dstep, Size size, WT scale, WT shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    cvtScaleAbs_SIMD<T, DT, WT> vop;

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = vop(src, dst, size.width, scale, shift);

        for (; x < size.width; x++)
            dst[x] = saturate_cast<DT>(std::abs(src[x] * scale + shift));
    }
}

template <typename T, typename DT, typename WT> struct cvtScale_SIMD
{
    int operator()(const T*, DT*, int, WT, WT) const
    {
        return 0;
    }
};

#if LYCON_SSE2

// from uchar

template <> struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator()(const uchar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, schar, float>
{
    int operator()(const uchar* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<uchar, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const uchar* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<uchar, short, float>
{
    int operator()(const uchar* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, int, float>
{
    int operator()(const uchar* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, float, float>
{
    int operator()(const uchar* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, double, double>
{
    int operator()(const uchar* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const*)(src + x)), v_zero);

            __m128i v_src_s32 = _mm_unpacklo_epi16(v_src, v_zero);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_unpackhi_epi16(v_src, v_zero);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from schar

template <> struct cvtScale_SIMD<schar, uchar, float>
{
    int operator()(const schar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, schar, float>
{
    int operator()(const schar* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<schar, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const schar* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<schar, short, float>
{
    int operator()(const schar* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, int, float>
{
    int operator()(const schar* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, float, float>
{
    int operator()(const schar* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x))), 8);
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, double, double>
{
    int operator()(const schar* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const*)(src + x)));
            v_src = _mm_srai_epi16(v_src, 8);

            __m128i v_src_s32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from ushort

template <> struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator()(const ushort* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, schar, float>
{
    int operator()(const ushort* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<ushort, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const ushort* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<ushort, short, float>
{
    int operator()(const ushort* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, int, float>
{
    int operator()(const ushort* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, float, float>
{
    int operator()(const ushort* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, double, double>
{
    int operator()(const ushort* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));

            __m128i v_src_s32 = _mm_unpacklo_epi16(v_src, v_zero);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_unpackhi_epi16(v_src, v_zero);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from short

template <> struct cvtScale_SIMD<short, uchar, float>
{
    int operator()(const short* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, schar, float>
{
    int operator()(const short* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<short, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const short* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<short, short, float>
{
    int operator()(const short* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, int, float>
{
    int operator()(const short* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, float, float>
{
    int operator()(const short* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

            _mm_storeu_ps(dst + x, v_dst_0);
            _mm_storeu_ps(dst + x + 4, v_dst_1);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, double, double>
{
    int operator()(const short* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));

            __m128i v_src_s32 = _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);

            v_src_s32 = _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16);
            v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src_s32), v_scale), v_shift);
            v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(_mm_srli_si128(v_src_s32, 8)), v_scale), v_shift);
            _mm_storeu_pd(dst + x + 4, v_dst_0);
            _mm_storeu_pd(dst + x + 6, v_dst_1);
        }

        return x;
    }
};

// from int

template <> struct cvtScale_SIMD<int, uchar, float>
{
    int operator()(const int* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const*)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, schar, float>
{
    int operator()(const int* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const*)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<int, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const int* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const*)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<int, short, float>
{
    int operator()(const int* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            v_src = _mm_loadu_si128((__m128i const*)(src + x + 4));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, int, double>
{
    int operator()(const int* src, int* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            __m128 v_dst =
                _mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_dst_0)), _mm_castsi128_ps(_mm_cvtpd_epi32(v_dst_1)));

            _mm_storeu_si128((__m128i*)(dst + x), _mm_castps_si128(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, float, double>
{
    int operator()(const int* src, float* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            _mm_storeu_ps(dst + x, _mm_movelh_ps(_mm_cvtpd_ps(v_dst_0), _mm_cvtpd_ps(v_dst_1)));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, double, double>
{
    int operator()(const int* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const*)(src + x));
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            v_src = _mm_srli_si128(v_src, 8);
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtepi32_pd(v_src), v_scale), v_shift);

            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);
        }

        return x;
    }
};

// from float

template <> struct cvtScale_SIMD<float, uchar, float>
{
    int operator()(const float* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, schar, float>
{
    int operator()(const float* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<float, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const float* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<float, short, float>
{
    int operator()(const float* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, int, float>
{
    int operator()(const float* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_loadu_ps(src + x + 4);
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_dst_0));
            _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(v_dst_1));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, float, float>
{
    int operator()(const float* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128 v_dst = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);
            _mm_storeu_ps(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, double, double>
{
    int operator()(const float* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src + x);
            __m128d v_dst_0 = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(v_src), v_scale), v_shift);
            v_src = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_src), 8));
            __m128d v_dst_1 = _mm_add_pd(_mm_mul_pd(_mm_cvtps_pd(v_src), v_scale), v_shift);

            _mm_storeu_pd(dst + x, v_dst_0);
            _mm_storeu_pd(dst + x + 2, v_dst_1);
        }

        return x;
    }
};

// from double

template <> struct cvtScale_SIMD<double, uchar, float>
{
    int operator()(const double* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_zero));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<double, schar, float>
{
    int operator()(const double* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_zero));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct cvtScale_SIMD<double, ushort, float>
{
    cvtScale_SIMD()
    {
        haveSSE = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const double* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <> struct cvtScale_SIMD<double, short, float>
{
    int operator()(const double* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
            __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)), _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
            __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_dst_0), _mm_cvtps_epi32(v_dst_1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<double, int, double>
{
    int operator()(const double* src, int* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst0 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            v_src = _mm_loadu_pd(src + x + 2);
            __m128d v_dst1 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            __m128 v_dst =
                _mm_movelh_ps(_mm_castsi128_ps(_mm_cvtpd_epi32(v_dst0)), _mm_castsi128_ps(_mm_cvtpd_epi32(v_dst1)));

            _mm_storeu_si128((__m128i*)(dst + x), _mm_castps_si128(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<double, float, double>
{
    int operator()(const double* src, float* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 4; x += 4)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst0 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            v_src = _mm_loadu_pd(src + x + 2);
            __m128d v_dst1 = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);

            __m128 v_dst = _mm_movelh_ps(_mm_cvtpd_ps(v_dst0), _mm_cvtpd_ps(v_dst1));

            _mm_storeu_ps(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<double, double, double>
{
    int operator()(const double* src, double* dst, int width, double scale, double shift) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        __m128d v_scale = _mm_set1_pd(scale), v_shift = _mm_set1_pd(shift);

        for (; x <= width - 2; x += 2)
        {
            __m128d v_src = _mm_loadu_pd(src + x);
            __m128d v_dst = _mm_add_pd(_mm_mul_pd(v_src, v_scale), v_shift);
            _mm_storeu_pd(dst + x, v_dst);
        }

        return x;
    }
};

#elif LYCON_NEON

// from uchar

template <> struct cvtScale_SIMD<uchar, uchar, float>
{
    int operator()(const uchar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, schar, float>
{
    int operator()(const uchar* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, ushort, float>
{
    int operator()(const uchar* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, short, float>
{
    int operator()(const uchar* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, int, float>
{
    int operator()(const uchar* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<uchar, float, float>
{
    int operator()(const uchar* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4,
                      vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from schar

template <> struct cvtScale_SIMD<schar, uchar, float>
{
    int operator()(const schar* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, schar, float>
{
    int operator()(const schar* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, ushort, float>
{
    int operator()(const schar* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, short, float>
{
    int operator()(const schar* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, int, float>
{
    int operator()(const schar* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<schar, float, float>
{
    int operator()(const schar* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4,
                      vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from ushort

template <> struct cvtScale_SIMD<ushort, uchar, float>
{
    int operator()(const ushort* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, schar, float>
{
    int operator()(const ushort* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, ushort, float>
{
    int operator()(const ushort* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, short, float>
{
    int operator()(const ushort* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, int, float>
{
    int operator()(const ushort* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_dst1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_dst2));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<ushort, float, float>
{
    int operator()(const ushort* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4,
                      vaddq_f32(vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from short

template <> struct cvtScale_SIMD<short, uchar, float>
{
    int operator()(const short* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, schar, float>
{
    int operator()(const short* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, ushort, float>
{
    int operator()(const short* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<short, float, float>
{
    int operator()(const short* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))), v_scale), v_shift));
            vst1q_f32(dst + x + 4,
                      vaddq_f32(vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))), v_scale), v_shift));
        }

        return x;
    }
};

// from int

template <> struct cvtScale_SIMD<int, uchar, float>
{
    int operator()(const int* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, schar, float>
{
    int operator()(const int* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, ushort, float>
{
    int operator()(const int* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<int, short, float>
{
    int operator()(const int* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x)), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + x + 4)), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

// from float

template <> struct cvtScale_SIMD<float, uchar, float>
{
    int operator()(const float* src, uchar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1_u8(dst + x, vqmovn_u16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, schar, float>
{
    int operator()(const float* src, schar* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1_s8(dst + x, vqmovn_s16(v_dst));
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, ushort, float>
{
    int operator()(const float* src, ushort* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)), vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
            vst1q_u16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, short, float>
{
    int operator()(const float* src, short* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst1 = vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift);
            float32x4_t v_dst2 = vaddq_f32(vmulq_f32(vld1q_f32(src + x + 4), v_scale), v_shift);

            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)), vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
            vst1q_s16(dst + x, v_dst);
        }

        return x;
    }
};

template <> struct cvtScale_SIMD<float, int, float>
{
    int operator()(const float* src, int* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 4; x += 4)
            vst1q_s32(dst + x, cv_vrndq_s32_f32(vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift)));

        return x;
    }
};

template <> struct cvtScale_SIMD<float, float, float>
{
    int operator()(const float* src, float* dst, int width, float scale, float shift) const
    {
        int x = 0;
        float32x4_t v_shift = vdupq_n_f32(shift), v_scale = vdupq_n_f32(scale);

        for (; x <= width - 4; x += 4)
            vst1q_f32(dst + x, vaddq_f32(vmulq_f32(vld1q_f32(src + x), v_scale), v_shift));

        return x;
    }
};

#endif

template <typename T, typename DT, typename WT>
static void cvtScale_(const T* src, size_t sstep, DT* dst, size_t dstep, Size size, WT scale, WT shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    cvtScale_SIMD<T, DT, WT> vop;

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = vop(src, dst, size.width, scale, shift);

#if LYCON_ENABLE_UNROLLED
        for (; x <= size.width - 4; x += 4)
        {
            DT t0, t1;
            t0 = saturate_cast<DT>(src[x] * scale + shift);
            t1 = saturate_cast<DT>(src[x + 1] * scale + shift);
            dst[x] = t0;
            dst[x + 1] = t1;
            t0 = saturate_cast<DT>(src[x + 2] * scale + shift);
            t1 = saturate_cast<DT>(src[x + 3] * scale + shift);
            dst[x + 2] = t0;
            dst[x + 3] = t1;
        }
#endif

        for (; x < size.width; x++)
            dst[x] = saturate_cast<DT>(src[x] * scale + shift);
    }
}

// vz optimized template specialization
template <>
void cvtScale_<short, short, float>(const short* src, size_t sstep, short* dst, size_t dstep, Size size, float scale,
                                    float shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = 0;
#if LYCON_SSE2
        if (USE_SSE2)
        {
            __m128 scale128 = _mm_set1_ps(scale);
            __m128 shift128 = _mm_set1_ps(shift);
            for (; x <= size.width - 8; x += 8)
            {
                __m128i r0 = _mm_loadl_epi64((const __m128i*)(src + x));
                __m128i r1 = _mm_loadl_epi64((const __m128i*)(src + x + 4));
                __m128 rf0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                __m128 rf1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r1, r1), 16));
                rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);
                r0 = _mm_cvtps_epi32(rf0);
                r1 = _mm_cvtps_epi32(rf1);
                r0 = _mm_packs_epi32(r0, r1);
                _mm_storeu_si128((__m128i*)(dst + x), r0);
            }
        }
#elif LYCON_NEON
        float32x4_t v_shift = vdupq_n_f32(shift);
        for (; x <= size.width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_tmp1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src)));
            float32x4_t v_tmp2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src)));

            v_tmp1 = vaddq_f32(vmulq_n_f32(v_tmp1, scale), v_shift);
            v_tmp2 = vaddq_f32(vmulq_n_f32(v_tmp2, scale), v_shift);

            vst1q_s16(dst + x,
                      vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_tmp1)), vqmovn_s32(cv_vrndq_s32_f32(v_tmp2))));
        }
#endif

        for (; x < size.width; x++)
            dst[x] = saturate_cast<short>(src[x] * scale + shift);
    }
}

template <>
void cvtScale_<short, int, float>(const short* src, size_t sstep, int* dst, size_t dstep, Size size, float scale,
                                  float shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = 0;

#if LYCON_AVX2
        if (USE_AVX2)
        {
            __m256 scale256 = _mm256_set1_ps(scale);
            __m256 shift256 = _mm256_set1_ps(shift);
            const int shuffle = 0xD8;

            for (; x <= size.width - 16; x += 16)
            {
                __m256i v_src = _mm256_loadu_si256((const __m256i*)(src + x));
                v_src = _mm256_permute4x64_epi64(v_src, shuffle);
                __m256i v_src_lo = _mm256_srai_epi32(_mm256_unpacklo_epi16(v_src, v_src), 16);
                __m256i v_src_hi = _mm256_srai_epi32(_mm256_unpackhi_epi16(v_src, v_src), 16);
                __m256 v_dst0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_lo), scale256), shift256);
                __m256 v_dst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_hi), scale256), shift256);
                _mm256_storeu_si256((__m256i*)(dst + x), _mm256_cvtps_epi32(v_dst0));
                _mm256_storeu_si256((__m256i*)(dst + x + 8), _mm256_cvtps_epi32(v_dst1));
            }
        }
#endif
#if LYCON_SSE2
        if (USE_SSE2) //~5X
        {
            __m128 scale128 = _mm_set1_ps(scale);
            __m128 shift128 = _mm_set1_ps(shift);
            for (; x <= size.width - 8; x += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)(src + x));

                __m128 rf0 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(r0, r0), 16));
                __m128 rf1 = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(r0, r0), 16));
                rf0 = _mm_add_ps(_mm_mul_ps(rf0, scale128), shift128);
                rf1 = _mm_add_ps(_mm_mul_ps(rf1, scale128), shift128);

                _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(rf0));
                _mm_storeu_si128((__m128i*)(dst + x + 4), _mm_cvtps_epi32(rf1));
            }
        }
#elif LYCON_NEON
        float32x4_t v_shift = vdupq_n_f32(shift);
        for (; x <= size.width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            float32x4_t v_tmp1 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src)));
            float32x4_t v_tmp2 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src)));

            v_tmp1 = vaddq_f32(vmulq_n_f32(v_tmp1, scale), v_shift);
            v_tmp2 = vaddq_f32(vmulq_n_f32(v_tmp2, scale), v_shift);

            vst1q_s32(dst + x, cv_vrndq_s32_f32(v_tmp1));
            vst1q_s32(dst + x + 4, cv_vrndq_s32_f32(v_tmp2));
        }
#endif

        for (; x < size.width; x++)
            dst[x] = saturate_cast<int>(src[x] * scale + shift);
    }
}

template <typename T, typename DT> struct Cvt_SIMD
{
    int operator()(const T*, DT*, int) const
    {
        return 0;
    }
};

#if LYCON_SSE2

// from double

template <> struct Cvt_SIMD<double, uchar>
{
    int operator()(const double* src, uchar* dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0), _mm_cvtps_epi32(v_src1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(v_dst, v_dst));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<double, schar>
{
    int operator()(const double* src, schar* dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0), _mm_cvtps_epi32(v_src1));
            _mm_storel_epi64((__m128i*)(dst + x), _mm_packs_epi16(v_dst, v_dst));
        }

        return x;
    }
};

#if LYCON_SSE4_1

template <> struct Cvt_SIMD<double, ushort>
{
    bool haveSIMD;
    Cvt_SIMD()
    {
        haveSIMD = checkHardwareSupport(LYCON_CPU_SSE4_1);
    }

    int operator()(const double* src, ushort* dst, int width) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_src0), _mm_cvtps_epi32(v_src1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

#endif // LYCON_SSE4_1

template <> struct Cvt_SIMD<double, short>
{
    int operator()(const double* src, short* dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for (; x <= width - 8; x += 8)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
            __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

            v_src0 = _mm_movelh_ps(v_src0, v_src1);
            v_src1 = _mm_movelh_ps(v_src2, v_src3);

            __m128i v_dst = _mm_packs_epi32(_mm_cvtps_epi32(v_src0), _mm_cvtps_epi32(v_src1));
            _mm_storeu_si128((__m128i*)(dst + x), v_dst);
        }

        return x;
    }
};

template <> struct Cvt_SIMD<double, int>
{
    int operator()(const double* src, int* dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for (; x <= width - 4; x += 4)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
            v_src0 = _mm_movelh_ps(v_src0, v_src1);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_cvtps_epi32(v_src0));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<double, float>
{
    int operator()(const double* src, float* dst, int width) const
    {
        int x = 0;

        if (!USE_SSE2)
            return x;

        for (; x <= width - 4; x += 4)
        {
            __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
            __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));

            _mm_storeu_ps(dst + x, _mm_movelh_ps(v_src0, v_src1));
        }

        return x;
    }
};

#elif LYCON_NEON

// from uchar

template <> struct Cvt_SIMD<uchar, schar>
{
    int operator()(const uchar* src, schar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
            vst1_s8(dst + x, vqmovn_s16(vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x)))));

        return x;
    }
};

template <> struct Cvt_SIMD<uchar, ushort>
{
    int operator()(const uchar* src, ushort* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
            vst1q_u16(dst + x, vmovl_u8(vld1_u8(src + x)));

        return x;
    }
};

template <> struct Cvt_SIMD<uchar, short>
{
    int operator()(const uchar* src, short* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
            vst1q_s16(dst + x, vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + x))));

        return x;
    }
};

template <> struct Cvt_SIMD<uchar, int>
{
    int operator()(const uchar* src, int* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_s32(dst + x, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_s32(dst + x + 4, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<uchar, float>
{
    int operator()(const uchar* src, float* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src + x));
            vst1q_f32(dst + x, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

// from schar

template <> struct Cvt_SIMD<schar, uchar>
{
    int operator()(const schar* src, uchar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
            vst1_u8(dst + x, vqmovun_s16(vmovl_s8(vld1_s8(src + x))));

        return x;
    }
};

template <> struct Cvt_SIMD<schar, short>
{
    int operator()(const schar* src, short* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
            vst1q_s16(dst + x, vmovl_s8(vld1_s8(src + x)));

        return x;
    }
};

template <> struct Cvt_SIMD<schar, ushort>
{
    int operator()(const schar* src, ushort* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_u16(dst + x, vcombine_u16(vqmovun_s32(vmovl_s16(vget_low_s16(v_src))),
                                            vqmovun_s32(vmovl_s16(vget_high_s16(v_src)))));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<schar, int>
{
    int operator()(const schar* src, int* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_s32(dst + x, vmovl_s16(vget_low_s16(v_src)));
            vst1q_s32(dst + x + 4, vmovl_s16(vget_high_s16(v_src)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<schar, float>
{
    int operator()(const schar* src, float* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src + x));
            vst1q_f32(dst + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))));
        }

        return x;
    }
};

// from ushort

template <> struct Cvt_SIMD<ushort, uchar>
{
    int operator()(const ushort* src, uchar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            uint16x8_t v_src1 = vld1q_u16(src + x), v_src2 = vld1q_u16(src + x + 8);
            vst1q_u8(dst + x, vcombine_u8(vqmovn_u16(v_src1), vqmovn_u16(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<ushort, schar>
{
    int operator()(const ushort* src, schar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            uint16x8_t v_src1 = vld1q_u16(src + x), v_src2 = vld1q_u16(src + x + 8);
            int32x4_t v_dst10 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src1)));
            int32x4_t v_dst11 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src1)));
            int32x4_t v_dst20 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src2)));
            int32x4_t v_dst21 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src2)));

            vst1q_s8(dst + x, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(v_dst10), vqmovn_s32(v_dst11))),
                                          vqmovn_s16(vcombine_s16(vqmovn_s32(v_dst20), vqmovn_s32(v_dst21)))));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<ushort, short>
{
    int operator()(const ushort* src, short* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            int32x4_t v_dst0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src)));
            int32x4_t v_dst1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src)));

            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(v_dst0), vqmovn_s32(v_dst1)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<ushort, int>
{
    int operator()(const ushort* src, int* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_s32(dst + x, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_s32(dst + x + 4, vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<ushort, float>
{
    int operator()(const ushort* src, float* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src + x);
            vst1q_f32(dst + x, vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src))));
        }

        return x;
    }
};

// from short

template <> struct Cvt_SIMD<short, uchar>
{
    int operator()(const short* src, uchar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            int16x8_t v_src1 = vld1q_s16(src + x), v_src2 = vld1q_s16(src + x + 8);
            vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(v_src1), vqmovun_s16(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<short, schar>
{
    int operator()(const short* src, schar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            int16x8_t v_src1 = vld1q_s16(src + x), v_src2 = vld1q_s16(src + x + 8);
            vst1q_s8(dst + x, vcombine_s8(vqmovn_s16(v_src1), vqmovn_s16(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<short, ushort>
{
    int operator()(const short* src, ushort* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            uint16x4_t v_dst1 = vqmovun_s32(vmovl_s16(vget_low_s16(v_src)));
            uint16x4_t v_dst2 = vqmovun_s32(vmovl_s16(vget_high_s16(v_src)));
            vst1q_u16(dst + x, vcombine_u16(v_dst1, v_dst2));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<short, int>
{
    int operator()(const short* src, int* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_s32(dst + x, vmovl_s16(vget_low_s16(v_src)));
            vst1q_s32(dst + x + 4, vmovl_s16(vget_high_s16(v_src)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<short, float>
{
    int operator()(const short* src, float* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src + x);
            vst1q_f32(dst + x, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src))));
            vst1q_f32(dst + x + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src))));
        }

        return x;
    }
};

// from int

template <> struct Cvt_SIMD<int, uchar>
{
    int operator()(const int* src, uchar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            int32x4_t v_src3 = vld1q_s32(src + x + 8), v_src4 = vld1q_s32(src + x + 12);
            uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovun_s32(v_src1), vqmovun_s32(v_src2)));
            uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovun_s32(v_src3), vqmovun_s32(v_src4)));
            vst1q_u8(dst + x, vcombine_u8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<int, schar>
{
    int operator()(const int* src, schar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            int32x4_t v_src3 = vld1q_s32(src + x + 8), v_src4 = vld1q_s32(src + x + 12);
            int8x8_t v_dst1 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
            int8x8_t v_dst2 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src3), vqmovn_s32(v_src4)));
            vst1q_s8(dst + x, vcombine_s8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<int, ushort>
{
    int operator()(const int* src, ushort* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            vst1q_u16(dst + x, vcombine_u16(vqmovun_s32(v_src1), vqmovun_s32(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<int, short>
{
    int operator()(const int* src, short* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            int32x4_t v_src1 = vld1q_s32(src + x), v_src2 = vld1q_s32(src + x + 4);
            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<int, float>
{
    int operator()(const int* src, float* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 4; x += 4)
            vst1q_f32(dst + x, vcvtq_f32_s32(vld1q_s32(src + x)));

        return x;
    }
};

// from float

template <> struct Cvt_SIMD<float, uchar>
{
    int operator()(const float* src, uchar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            uint32x4_t v_src1 = cv_vrndq_u32_f32(vld1q_f32(src + x));
            uint32x4_t v_src2 = cv_vrndq_u32_f32(vld1q_f32(src + x + 4));
            uint32x4_t v_src3 = cv_vrndq_u32_f32(vld1q_f32(src + x + 8));
            uint32x4_t v_src4 = cv_vrndq_u32_f32(vld1q_f32(src + x + 12));
            uint8x8_t v_dst1 = vqmovn_u16(vcombine_u16(vqmovn_u32(v_src1), vqmovn_u32(v_src2)));
            uint8x8_t v_dst2 = vqmovn_u16(vcombine_u16(vqmovn_u32(v_src3), vqmovn_u32(v_src4)));
            vst1q_u8(dst + x, vcombine_u8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<float, schar>
{
    int operator()(const float* src, schar* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 16; x += 16)
        {
            int32x4_t v_src1 = cv_vrndq_s32_f32(vld1q_f32(src + x));
            int32x4_t v_src2 = cv_vrndq_s32_f32(vld1q_f32(src + x + 4));
            int32x4_t v_src3 = cv_vrndq_s32_f32(vld1q_f32(src + x + 8));
            int32x4_t v_src4 = cv_vrndq_s32_f32(vld1q_f32(src + x + 12));
            int8x8_t v_dst1 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src1), vqmovn_s32(v_src2)));
            int8x8_t v_dst2 = vqmovn_s16(vcombine_s16(vqmovn_s32(v_src3), vqmovn_s32(v_src4)));
            vst1q_s8(dst + x, vcombine_s8(v_dst1, v_dst2));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<float, ushort>
{
    int operator()(const float* src, ushort* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 8; x += 8)
        {
            uint32x4_t v_src1 = cv_vrndq_u32_f32(vld1q_f32(src + x));
            uint32x4_t v_src2 = cv_vrndq_u32_f32(vld1q_f32(src + x + 4));
            vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(v_src1), vqmovn_u32(v_src2)));
        }

        return x;
    }
};

template <> struct Cvt_SIMD<float, int>
{
    int operator()(const float* src, int* dst, int width) const
    {
        int x = 0;

        for (; x <= width - 4; x += 4)
            vst1q_s32(dst + x, cv_vrndq_s32_f32(vld1q_f32(src + x)));

        return x;
    }
};

#endif

#if !LYCON_FP16_TYPE
// const numbers for floating points format
const unsigned int kShiftSignificand = 13;
const unsigned int kMaskFp16Significand = 0x3ff;
const unsigned int kBiasFp16Exponent = 15;
const unsigned int kBiasFp32Exponent = 127;
#endif

#if LYCON_FP16_TYPE
static float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Lycon16suf a;
    a.i = fp16;
    return (float)a.h;
}
#else
static float convertFp16SW(short fp16)
{
    // Fp16 -> Fp32
    Lycon16suf b;
    b.i = fp16;
    int exponent = b.fmt.exponent - kBiasFp16Exponent;
    int significand = b.fmt.significand;

    Lycon32suf a;
    a.i = 0;
    a.fmt.sign = b.fmt.sign; // sign bit
    if (exponent == 16)
    {
        // Inf or NaN
        a.i = a.i | 0x7F800000;
        if (significand != 0)
        {
// NaN
#if defined(__x86_64__) || defined(_M_X64)
            // 64bit
            a.i = a.i | 0x7FC00000;
#endif
            a.fmt.significand = a.fmt.significand | (significand << kShiftSignificand);
        }
        return a.f;
    }
    else if (exponent == -15)
    {
        // subnormal in Fp16
        if (significand == 0)
        {
            // zero
            return a.f;
        }
        else
        {
            int shift = -1;
            while ((significand & 0x400) == 0)
            {
                significand = significand << 1;
                shift++;
            }
            significand = significand & kMaskFp16Significand;
            exponent -= shift;
        }
    }

    a.fmt.exponent = (exponent + kBiasFp32Exponent);
    a.fmt.significand = significand << kShiftSignificand;
    return a.f;
}
#endif

#if LYCON_FP16_TYPE
static short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Lycon16suf a;
    a.h = (__fp16)fp32;
    return a.i;
}
#else
static short convertFp16SW(float fp32)
{
    // Fp32 -> Fp16
    Lycon32suf a;
    a.f = fp32;
    int exponent = a.fmt.exponent - kBiasFp32Exponent;
    int significand = a.fmt.significand;

    Lycon16suf result;
    result.i = 0;
    unsigned int absolute = a.i & 0x7fffffff;
    if (0x477ff000 <= absolute)
    {
        // Inf in Fp16
        result.i = result.i | 0x7C00;
        if (exponent == 128 && significand != 0)
        {
            // NaN
            result.i = (short)(result.i | 0x200 | (significand >> kShiftSignificand));
        }
    }
    else if (absolute < 0x33000001)
    {
        // too small for fp16
        result.i = 0;
    }
    else if (absolute < 0x33c00000)
    {
        result.i = 1;
    }
    else if (absolute < 0x34200001)
    {
        result.i = 2;
    }
    else if (absolute < 0x387fe000)
    {
        // subnormal in Fp16
        int fp16Significand = significand | 0x800000;
        int bitShift = (-exponent) - 1;
        fp16Significand = fp16Significand >> bitShift;

        // special cases to round up
        bitShift = exponent + 24;
        int threshold =
            ((0x400000 >> bitShift) | (((significand & (0x800000 >> bitShift)) >> (126 - a.fmt.exponent)) ^ 1));
        if (threshold <= (significand & (0xffffff >> (exponent + 25))))
        {
            fp16Significand++;
        }
        result.i = (short)fp16Significand;
    }
    else
    {
        // usual situation
        // exponent
        result.fmt.exponent = (exponent + kBiasFp16Exponent);

        // significand;
        short fp16Significand = (short)(significand >> kShiftSignificand);
        result.fmt.significand = fp16Significand;

        // special cases to round up
        short lsb10bitsFp32 = (significand & 0x1fff);
        short threshold = 0x1000 + ((fp16Significand & 0x1) ? 0 : 1);
        if (threshold <= lsb10bitsFp32)
        {
            result.i++;
        }
        else if (fp16Significand == 0x3ff && exponent == -15)
        {
            result.i++;
        }
    }

    // sign bit
    result.fmt.sign = a.fmt.sign;
    return result.i;
}
#endif

// template for FP16 HW conversion function
template <typename T, typename DT>
static void cvtScaleHalf_(const T* src, size_t sstep, DT* dst, size_t dstep, Size size);

template <> void cvtScaleHalf_<float, short>(const float* src, size_t sstep, short* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    if (checkHardwareSupport(LYCON_CPU_FP16))
    {
        for (; size.height--; src += sstep, dst += dstep)
        {
            int x = 0;

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(i386)
            if (((intptr_t)dst & 0xf) == 0)
#endif
            {
#if LYCON_FP16 && LYCON_SIMD128
                for (; x <= size.width - 4; x += 4)
                {
                    v_float32x4 v_src = v_load(src + x);

                    v_float16x4 v_dst = v_cvt_f16(v_src);

                    v_store_f16(dst + x, v_dst);
                }
#endif
            }
            for (; x < size.width; x++)
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
    else
    {
        for (; size.height--; src += sstep, dst += dstep)
        {
            int x = 0;
            for (; x < size.width; x++)
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
}

template <> void cvtScaleHalf_<short, float>(const short* src, size_t sstep, float* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    if (checkHardwareSupport(LYCON_CPU_FP16))
    {
        for (; size.height--; src += sstep, dst += dstep)
        {
            int x = 0;

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(i386)
            if (((intptr_t)src & 0xf) == 0)
#endif
            {
#if LYCON_FP16 && LYCON_SIMD128
                for (; x <= size.width - 4; x += 4)
                {
                    v_float16x4 v_src = v_load_f16(src + x);

                    v_float32x4 v_dst = v_cvt_f32(v_src);

                    v_store(dst + x, v_dst);
                }
#endif
            }
            for (; x < size.width; x++)
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
    else
    {
        for (; size.height--; src += sstep, dst += dstep)
        {
            int x = 0;
            for (; x < size.width; x++)
            {
                dst[x] = convertFp16SW(src[x]);
            }
        }
    }
}

template <typename T, typename DT> static void cvt_(const T* src, size_t sstep, DT* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);
    Cvt_SIMD<T, DT> vop;

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = vop(src, dst, size.width);
        for (; x < size.width; x++)
            dst[x] = saturate_cast<DT>(src[x]);
    }
}

// vz optimized template specialization, test Core_ConvertScale/ElemWiseTest
template <> void cvt_<float, short>(const float* src, size_t sstep, short* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = 0;
#if LYCON_SSE2
        if (USE_SSE2)
        {
            for (; x <= size.width - 8; x += 8)
            {
                __m128 src128 = _mm_loadu_ps(src + x);
                __m128i src_int128 = _mm_cvtps_epi32(src128);

                src128 = _mm_loadu_ps(src + x + 4);
                __m128i src1_int128 = _mm_cvtps_epi32(src128);

                src1_int128 = _mm_packs_epi32(src_int128, src1_int128);
                _mm_storeu_si128((__m128i*)(dst + x), src1_int128);
            }
        }
#elif LYCON_NEON
        for (; x <= size.width - 8; x += 8)
        {
            float32x4_t v_src1 = vld1q_f32(src + x), v_src2 = vld1q_f32(src + x + 4);
            int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_src1)), vqmovn_s32(cv_vrndq_s32_f32(v_src2)));
            vst1q_s16(dst + x, v_dst);
        }
#endif
        for (; x < size.width; x++)
            dst[x] = saturate_cast<short>(src[x]);
    }
}

template <typename T> static void cpy_(const T* src, size_t sstep, T* dst, size_t dstep, Size size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (; size.height--; src += sstep, dst += dstep)
        memcpy(dst, src, size.width * sizeof(src[0]));
}

#define DEF_CVT_SCALE_ABS_FUNC(suffix, tfunc, stype, dtype, wtype)                                                     \
    static void cvtScaleAbs##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep,    \
                                    Size size, double* scale)                                                          \
    {                                                                                                                  \
        tfunc(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]);                                         \
    }

#define DEF_CVT_SCALE_FP16_FUNC(suffix, stype, dtype)                                                                  \
    static void cvtScaleHalf##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep,   \
                                     Size size, double*)                                                               \
    {                                                                                                                  \
        cvtScaleHalf_<stype, dtype>(src, sstep, dst, dstep, size);                                                     \
    }

#define DEF_CVT_SCALE_FUNC(suffix, stype, dtype, wtype)                                                                \
    static void cvtScale##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep,       \
                                 Size size, double* scale)                                                             \
    {                                                                                                                  \
        cvtScale_(src, sstep, dst, dstep, size, (wtype)scale[0], (wtype)scale[1]);                                     \
    }

#if defined(HAVE_IPP)
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor)                                                                 \
    static void cvt##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep, Size size, \
                            double*)                                                                                   \
    {                                                                                                                  \
        cvt_(src, sstep, dst, dstep, size);                                                                            \
    }

#define DEF_CVT_FUNC_F2(suffix, stype, dtype, ippFavor)                                                                \
    static void cvt##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep, Size size, \
                            double*)                                                                                   \
    {                                                                                                                  \
        cvt_(src, sstep, dst, dstep, size);                                                                            \
    }
#else
#define DEF_CVT_FUNC_F(suffix, stype, dtype, ippFavor)                                                                 \
    static void cvt##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep, Size size, \
                            double*)                                                                                   \
    {                                                                                                                  \
        cvt_(src, sstep, dst, dstep, size);                                                                            \
    }
#define DEF_CVT_FUNC_F2 DEF_CVT_FUNC_F
#endif

#define DEF_CVT_FUNC(suffix, stype, dtype)                                                                             \
    static void cvt##suffix(const stype* src, size_t sstep, const uchar*, size_t, dtype* dst, size_t dstep, Size size, \
                            double*)                                                                                   \
    {                                                                                                                  \
        cvt_(src, sstep, dst, dstep, size);                                                                            \
    }

#define DEF_CPY_FUNC(suffix, stype)                                                                                    \
    static void cvt##suffix(const stype* src, size_t sstep, const uchar*, size_t, stype* dst, size_t dstep, Size size, \
                            double*)                                                                                   \
    {                                                                                                                  \
        cpy_(src, sstep, dst, dstep, size);                                                                            \
    }

DEF_CVT_SCALE_ABS_FUNC(8u, cvtScaleAbs_, uchar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(8s8u, cvtScaleAbs_, schar, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16u8u, cvtScaleAbs_, ushort, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(16s8u, cvtScaleAbs_, short, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32s8u, cvtScaleAbs_, int, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(32f8u, cvtScaleAbs_, float, uchar, float)
DEF_CVT_SCALE_ABS_FUNC(64f8u, cvtScaleAbs_, double, uchar, float)

DEF_CVT_SCALE_FP16_FUNC(32f16f, float, short)
DEF_CVT_SCALE_FP16_FUNC(16f32f, short, float)

DEF_CVT_SCALE_FUNC(8u, uchar, uchar, float)
DEF_CVT_SCALE_FUNC(8s8u, schar, uchar, float)
DEF_CVT_SCALE_FUNC(16u8u, ushort, uchar, float)
DEF_CVT_SCALE_FUNC(16s8u, short, uchar, float)
DEF_CVT_SCALE_FUNC(32s8u, int, uchar, float)
DEF_CVT_SCALE_FUNC(32f8u, float, uchar, float)
DEF_CVT_SCALE_FUNC(64f8u, double, uchar, float)

DEF_CVT_SCALE_FUNC(8u8s, uchar, schar, float)
DEF_CVT_SCALE_FUNC(8s, schar, schar, float)
DEF_CVT_SCALE_FUNC(16u8s, ushort, schar, float)
DEF_CVT_SCALE_FUNC(16s8s, short, schar, float)
DEF_CVT_SCALE_FUNC(32s8s, int, schar, float)
DEF_CVT_SCALE_FUNC(32f8s, float, schar, float)
DEF_CVT_SCALE_FUNC(64f8s, double, schar, float)

DEF_CVT_SCALE_FUNC(8u16u, uchar, ushort, float)
DEF_CVT_SCALE_FUNC(8s16u, schar, ushort, float)
DEF_CVT_SCALE_FUNC(16u, ushort, ushort, float)
DEF_CVT_SCALE_FUNC(16s16u, short, ushort, float)
DEF_CVT_SCALE_FUNC(32s16u, int, ushort, float)
DEF_CVT_SCALE_FUNC(32f16u, float, ushort, float)
DEF_CVT_SCALE_FUNC(64f16u, double, ushort, float)

DEF_CVT_SCALE_FUNC(8u16s, uchar, short, float)
DEF_CVT_SCALE_FUNC(8s16s, schar, short, float)
DEF_CVT_SCALE_FUNC(16u16s, ushort, short, float)
DEF_CVT_SCALE_FUNC(16s, short, short, float)
DEF_CVT_SCALE_FUNC(32s16s, int, short, float)
DEF_CVT_SCALE_FUNC(32f16s, float, short, float)
DEF_CVT_SCALE_FUNC(64f16s, double, short, float)

DEF_CVT_SCALE_FUNC(8u32s, uchar, int, float)
DEF_CVT_SCALE_FUNC(8s32s, schar, int, float)
DEF_CVT_SCALE_FUNC(16u32s, ushort, int, float)
DEF_CVT_SCALE_FUNC(16s32s, short, int, float)
DEF_CVT_SCALE_FUNC(32s, int, int, double)
DEF_CVT_SCALE_FUNC(32f32s, float, int, float)
DEF_CVT_SCALE_FUNC(64f32s, double, int, double)

DEF_CVT_SCALE_FUNC(8u32f, uchar, float, float)
DEF_CVT_SCALE_FUNC(8s32f, schar, float, float)
DEF_CVT_SCALE_FUNC(16u32f, ushort, float, float)
DEF_CVT_SCALE_FUNC(16s32f, short, float, float)
DEF_CVT_SCALE_FUNC(32s32f, int, float, double)
DEF_CVT_SCALE_FUNC(32f, float, float, float)
DEF_CVT_SCALE_FUNC(64f32f, double, float, double)

DEF_CVT_SCALE_FUNC(8u64f, uchar, double, double)
DEF_CVT_SCALE_FUNC(8s64f, schar, double, double)
DEF_CVT_SCALE_FUNC(16u64f, ushort, double, double)
DEF_CVT_SCALE_FUNC(16s64f, short, double, double)
DEF_CVT_SCALE_FUNC(32s64f, int, double, double)
DEF_CVT_SCALE_FUNC(32f64f, float, double, double)
DEF_CVT_SCALE_FUNC(64f, double, double, double)

DEF_CPY_FUNC(8u, uchar)
DEF_CVT_FUNC_F(8s8u, schar, uchar, 8s8u_C1Rs)
DEF_CVT_FUNC_F(16u8u, ushort, uchar, 16u8u_C1R)
DEF_CVT_FUNC_F(16s8u, short, uchar, 16s8u_C1R)
DEF_CVT_FUNC_F(32s8u, int, uchar, 32s8u_C1R)
DEF_CVT_FUNC_F2(32f8u, float, uchar, 32f8u_C1RSfs)
DEF_CVT_FUNC(64f8u, double, uchar)

DEF_CVT_FUNC_F2(8u8s, uchar, schar, 8u8s_C1RSfs)
DEF_CVT_FUNC_F2(16u8s, ushort, schar, 16u8s_C1RSfs)
DEF_CVT_FUNC_F2(16s8s, short, schar, 16s8s_C1RSfs)
DEF_CVT_FUNC_F(32s8s, int, schar, 32s8s_C1R)
DEF_CVT_FUNC_F2(32f8s, float, schar, 32f8s_C1RSfs)
DEF_CVT_FUNC(64f8s, double, schar)

DEF_CVT_FUNC_F(8u16u, uchar, ushort, 8u16u_C1R)
DEF_CVT_FUNC_F(8s16u, schar, ushort, 8s16u_C1Rs)
DEF_CPY_FUNC(16u, ushort)
DEF_CVT_FUNC_F(16s16u, short, ushort, 16s16u_C1Rs)
DEF_CVT_FUNC_F2(32s16u, int, ushort, 32s16u_C1RSfs)
DEF_CVT_FUNC_F2(32f16u, float, ushort, 32f16u_C1RSfs)
DEF_CVT_FUNC(64f16u, double, ushort)

DEF_CVT_FUNC_F(8u16s, uchar, short, 8u16s_C1R)
DEF_CVT_FUNC_F(8s16s, schar, short, 8s16s_C1R)
DEF_CVT_FUNC_F2(16u16s, ushort, short, 16u16s_C1RSfs)
DEF_CVT_FUNC_F2(32s16s, int, short, 32s16s_C1RSfs)
DEF_CVT_FUNC(32f16s, float, short)
DEF_CVT_FUNC(64f16s, double, short)

DEF_CVT_FUNC_F(8u32s, uchar, int, 8u32s_C1R)
DEF_CVT_FUNC_F(8s32s, schar, int, 8s32s_C1R)
DEF_CVT_FUNC_F(16u32s, ushort, int, 16u32s_C1R)
DEF_CVT_FUNC_F(16s32s, short, int, 16s32s_C1R)
DEF_CPY_FUNC(32s, int)
DEF_CVT_FUNC_F2(32f32s, float, int, 32f32s_C1RSfs)
DEF_CVT_FUNC(64f32s, double, int)

DEF_CVT_FUNC_F(8u32f, uchar, float, 8u32f_C1R)
DEF_CVT_FUNC_F(8s32f, schar, float, 8s32f_C1R)
DEF_CVT_FUNC_F(16u32f, ushort, float, 16u32f_C1R)
DEF_CVT_FUNC_F(16s32f, short, float, 16s32f_C1R)
DEF_CVT_FUNC_F(32s32f, int, float, 32s32f_C1R)
DEF_CVT_FUNC(64f32f, double, float)

DEF_CVT_FUNC(8u64f, uchar, double)
DEF_CVT_FUNC(8s64f, schar, double)
DEF_CVT_FUNC(16u64f, ushort, double)
DEF_CVT_FUNC(16s64f, short, double)
DEF_CVT_FUNC(32s64f, int, double)
DEF_CVT_FUNC(32f64f, float, double)
DEF_CPY_FUNC(64s, int64)

static BinaryFunc getCvtScaleAbsFunc(int depth)
{
    static BinaryFunc cvtScaleAbsTab[] = {(BinaryFunc)cvtScaleAbs8u,    (BinaryFunc)cvtScaleAbs8s8u,
                                          (BinaryFunc)cvtScaleAbs16u8u, (BinaryFunc)cvtScaleAbs16s8u,
                                          (BinaryFunc)cvtScaleAbs32s8u, (BinaryFunc)cvtScaleAbs32f8u,
                                          (BinaryFunc)cvtScaleAbs64f8u, 0};

    return cvtScaleAbsTab[depth];
}

BinaryFunc getConvertFuncFp16(int ddepth)
{
    static BinaryFunc cvtTab[] = {
        0, 0, 0, (BinaryFunc)(cvtScaleHalf32f16f), 0, (BinaryFunc)(cvtScaleHalf16f32f), 0, 0,
    };
    return cvtTab[LYCON_MAT_DEPTH(ddepth)];
}

BinaryFunc getConvertFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtTab[][8] = {
        {(BinaryFunc)(cvt8u), (BinaryFunc)GET_OPTIMIZED(cvt8s8u), (BinaryFunc)GET_OPTIMIZED(cvt16u8u),
         (BinaryFunc)GET_OPTIMIZED(cvt16s8u), (BinaryFunc)GET_OPTIMIZED(cvt32s8u), (BinaryFunc)GET_OPTIMIZED(cvt32f8u),
         (BinaryFunc)GET_OPTIMIZED(cvt64f8u), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u8s), (BinaryFunc)cvt8u, (BinaryFunc)GET_OPTIMIZED(cvt16u8s),
         (BinaryFunc)GET_OPTIMIZED(cvt16s8s), (BinaryFunc)GET_OPTIMIZED(cvt32s8s), (BinaryFunc)GET_OPTIMIZED(cvt32f8s),
         (BinaryFunc)GET_OPTIMIZED(cvt64f8s), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u16u), (BinaryFunc)GET_OPTIMIZED(cvt8s16u), (BinaryFunc)cvt16u,
         (BinaryFunc)GET_OPTIMIZED(cvt16s16u), (BinaryFunc)GET_OPTIMIZED(cvt32s16u),
         (BinaryFunc)GET_OPTIMIZED(cvt32f16u), (BinaryFunc)GET_OPTIMIZED(cvt64f16u), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u16s), (BinaryFunc)GET_OPTIMIZED(cvt8s16s), (BinaryFunc)GET_OPTIMIZED(cvt16u16s),
         (BinaryFunc)cvt16u, (BinaryFunc)GET_OPTIMIZED(cvt32s16s), (BinaryFunc)GET_OPTIMIZED(cvt32f16s),
         (BinaryFunc)GET_OPTIMIZED(cvt64f16s), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u32s), (BinaryFunc)GET_OPTIMIZED(cvt8s32s), (BinaryFunc)GET_OPTIMIZED(cvt16u32s),
         (BinaryFunc)GET_OPTIMIZED(cvt16s32s), (BinaryFunc)cvt32s, (BinaryFunc)GET_OPTIMIZED(cvt32f32s),
         (BinaryFunc)GET_OPTIMIZED(cvt64f32s), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u32f), (BinaryFunc)GET_OPTIMIZED(cvt8s32f), (BinaryFunc)GET_OPTIMIZED(cvt16u32f),
         (BinaryFunc)GET_OPTIMIZED(cvt16s32f), (BinaryFunc)GET_OPTIMIZED(cvt32s32f), (BinaryFunc)cvt32s,
         (BinaryFunc)GET_OPTIMIZED(cvt64f32f), 0},
        {(BinaryFunc)GET_OPTIMIZED(cvt8u64f), (BinaryFunc)GET_OPTIMIZED(cvt8s64f), (BinaryFunc)GET_OPTIMIZED(cvt16u64f),
         (BinaryFunc)GET_OPTIMIZED(cvt16s64f), (BinaryFunc)GET_OPTIMIZED(cvt32s64f),
         (BinaryFunc)GET_OPTIMIZED(cvt32f64f), (BinaryFunc)(cvt64s), 0},
        {0, 0, 0, 0, 0, 0, 0, 0}};

    return cvtTab[LYCON_MAT_DEPTH(ddepth)][LYCON_MAT_DEPTH(sdepth)];
}

static BinaryFunc getConvertScaleFunc(int sdepth, int ddepth)
{
    static BinaryFunc cvtScaleTab[][8] = {
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s8u),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u8u), (BinaryFunc)GET_OPTIMIZED(cvtScale16s8u),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s8u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8u), (BinaryFunc)cvtScale64f8u,
         0},
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u8s), (BinaryFunc)GET_OPTIMIZED(cvtScale16s8s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s8s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f8s), (BinaryFunc)cvtScale64f8s,
         0},
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u16u), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16u),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u), (BinaryFunc)GET_OPTIMIZED(cvtScale16s16u),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s16u), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16u),
         (BinaryFunc)cvtScale64f16u, 0},
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s16s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u16s), (BinaryFunc)GET_OPTIMIZED(cvtScale16s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s16s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f16s),
         (BinaryFunc)cvtScale64f16s, 0},
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u32s), (BinaryFunc)GET_OPTIMIZED(cvtScale16s32s),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s), (BinaryFunc)GET_OPTIMIZED(cvtScale32f32s), (BinaryFunc)cvtScale64f32s,
         0},
        {(BinaryFunc)GET_OPTIMIZED(cvtScale8u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale8s32f),
         (BinaryFunc)GET_OPTIMIZED(cvtScale16u32f), (BinaryFunc)GET_OPTIMIZED(cvtScale16s32f),
         (BinaryFunc)GET_OPTIMIZED(cvtScale32s32f), (BinaryFunc)GET_OPTIMIZED(cvtScale32f), (BinaryFunc)cvtScale64f32f,
         0},
        {(BinaryFunc)cvtScale8u64f, (BinaryFunc)cvtScale8s64f, (BinaryFunc)cvtScale16u64f, (BinaryFunc)cvtScale16s64f,
         (BinaryFunc)cvtScale32s64f, (BinaryFunc)cvtScale32f64f, (BinaryFunc)cvtScale64f, 0},
        {0, 0, 0, 0, 0, 0, 0, 0}};

    return cvtScaleTab[LYCON_MAT_DEPTH(ddepth)][LYCON_MAT_DEPTH(sdepth)];
}

void convertScaleAbs(InputArray _src, OutputArray _dst, double alpha, double beta)
{
    Mat src = _src.getMat();
    int cn = src.channels();
    double scale[] = {alpha, beta};
    _dst.create(src.dims, src.size, LYCON_8UC(cn));
    Mat dst = _dst.getMat();
    BinaryFunc func = getCvtScaleAbsFunc(src.depth());
    LYCON_ASSERT(func != 0);

    if (src.dims <= 2)
    {
        Size sz = getContinuousSize(src, dst, cn);
        func(src.ptr(), src.step, 0, 0, dst.ptr(), dst.step, sz, scale);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)it.size * cn, 1);

        for (size_t i = 0; i < it.nplanes; i++, ++it)
            func(ptrs[0], 0, 0, 0, ptrs[1], 0, sz, scale);
    }
}

void convertFp16(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    int ddepth = 0;

    switch (src.depth())
    {
    case LYCON_32F: ddepth = LYCON_16S; break;
    case LYCON_16S: ddepth = LYCON_32F; break;
    default: LYCON_ERROR("Unsupported input depth"); return;
    }

    int type = LYCON_MAKETYPE(ddepth, src.channels());
    _dst.create(src.dims, src.size, type);
    Mat dst = _dst.getMat();
    BinaryFunc func = getConvertFuncFp16(ddepth);
    int cn = src.channels();
    LYCON_ASSERT(func != 0);

    if (src.dims <= 2)
    {
        Size sz = getContinuousSize(src, dst, cn);
        func(src.data, src.step, 0, 0, dst.data, dst.step, sz, 0);
    }
    else
    {
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size * cn), 1);

        for (size_t i = 0; i < it.nplanes; i++, ++it)
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, 0);
    }
}

void Mat::convertTo(OutputArray _dst, int _type, double alpha, double beta) const
{
    bool noScale = fabs(alpha - 1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;

    if (_type < 0)
        _type = _dst.fixedType() ? _dst.type() : type();
    else
        _type = LYCON_MAKETYPE(LYCON_MAT_DEPTH(_type), channels());

    int sdepth = depth(), ddepth = LYCON_MAT_DEPTH(_type);
    if (sdepth == ddepth && noScale)
    {
        copyTo(_dst);
        return;
    }

    Mat src = *this;

    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);
    double scale[] = {alpha, beta};
    int cn = channels();
    LYCON_ASSERT(func != 0);

    if (dims <= 2)
    {
        _dst.create(size(), _type);
        Mat dst = _dst.getMat();
        Size sz = getContinuousSize(src, dst, cn);
        func(src.data, src.step, 0, 0, dst.data, dst.step, sz, scale);
    }
    else
    {
        _dst.create(dims, size, _type);
        Mat dst = _dst.getMat();
        const Mat* arrays[] = {&src, &dst, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        Size sz((int)(it.size * cn), 1);

        for (size_t i = 0; i < it.nplanes; i++, ++it)
            func(ptrs[0], 1, 0, 0, ptrs[1], 1, sz, scale);
    }
}

void convertAndUnrollScalar(const Mat& sc, int buftype, uchar* scbuf, size_t blocksize)
{
    int scn = (int)sc.total(), cn = LYCON_MAT_CN(buftype);
    size_t esz = LYCON_ELEM_SIZE(buftype);
    getConvertFunc(sc.depth(), buftype)(sc.ptr(), 1, 0, 1, scbuf, 1, Size(std::min(cn, scn), 1), 0);
    // unroll the scalar
    if (scn < cn)
    {
        LYCON_ASSERT(scn == 1);
        size_t esz1 = LYCON_ELEM_SIZE1(buftype);
        for (size_t i = esz1; i < esz; i++)
            scbuf[i] = scbuf[i - esz1];
    }
    for (size_t i = esz; i < blocksize * esz; i++)
        scbuf[i] = scbuf[i - esz];
}
}
