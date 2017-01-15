#if LYCON_SSE2
struct VResizeLinearVec_32s8u
{
    int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE2)) return 0;

        const int** src = (const int**)_src;
        const short* beta = (const short*)_beta;
        const int *S0 = src[0], *S1 = src[1];
        int x = 0;
        __m128i b0 = _mm_set1_epi16(beta[0]), b1 = _mm_set1_epi16(beta[1]);
        __m128i delta = _mm_set1_epi16(2);

        if ((((size_t)S0 | (size_t)S1) & 15) == 0)
            for (; x <= width - 16; x += 16)
            {
                __m128i x0, x1, x2, y0, y1, y2;
                x0 = _mm_load_si128((const __m128i*)(S0 + x));
                x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S1 + x));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));
                x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
                y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

                x1 = _mm_load_si128((const __m128i*)(S0 + x + 8));
                x2 = _mm_load_si128((const __m128i*)(S0 + x + 12));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 8));
                y2 = _mm_load_si128((const __m128i*)(S1 + x + 12));
                x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
                y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

                x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
                x1 = _mm_adds_epi16(_mm_mulhi_epi16(x1, b0), _mm_mulhi_epi16(y1, b1));

                x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
                x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
                _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
            }
        else
            for (; x <= width - 16; x += 16)
            {
                __m128i x0, x1, x2, y0, y1, y2;
                x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));
                x0 = _mm_packs_epi32(_mm_srai_epi32(x0, 4), _mm_srai_epi32(x1, 4));
                y0 = _mm_packs_epi32(_mm_srai_epi32(y0, 4), _mm_srai_epi32(y1, 4));

                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 8));
                x2 = _mm_loadu_si128((const __m128i*)(S0 + x + 12));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 8));
                y2 = _mm_loadu_si128((const __m128i*)(S1 + x + 12));
                x1 = _mm_packs_epi32(_mm_srai_epi32(x1, 4), _mm_srai_epi32(x2, 4));
                y1 = _mm_packs_epi32(_mm_srai_epi32(y1, 4), _mm_srai_epi32(y2, 4));

                x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
                x1 = _mm_adds_epi16(_mm_mulhi_epi16(x1, b0), _mm_mulhi_epi16(y1, b1));

                x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
                x1 = _mm_srai_epi16(_mm_adds_epi16(x1, delta), 2);
                _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi16(x0, x1));
            }

        for (; x < width - 4; x += 4)
        {
            __m128i x0, y0;
            x0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S0 + x)), 4);
            y0 = _mm_srai_epi32(_mm_loadu_si128((const __m128i*)(S1 + x)), 4);
            x0 = _mm_packs_epi32(x0, x0);
            y0 = _mm_packs_epi32(y0, y0);
            x0 = _mm_adds_epi16(_mm_mulhi_epi16(x0, b0), _mm_mulhi_epi16(y0, b1));
            x0 = _mm_srai_epi16(_mm_adds_epi16(x0, delta), 2);
            x0 = _mm_packus_epi16(x0, x0);
            *(int*)(dst + x) = _mm_cvtsi128_si32(x0);
        }

        return x;
    }
};

template <int shiftval>
struct VResizeLinearVec_32f16
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE2)) return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        ushort* dst = (ushort*)_dst;
        int x = 0;

        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);
        __m128i preshift = _mm_set1_epi32(shiftval);
        __m128i postshift = _mm_set1_epi16((short)shiftval);

        if ((((size_t)S0 | (size_t)S1) & 15) == 0)
            for (; x <= width - 16; x += 16)
            {
                __m128 x0, x1, y0, y1;
                __m128i t0, t1, t2;
                x0 = _mm_load_ps(S0 + x);
                x1 = _mm_load_ps(S0 + x + 4);
                y0 = _mm_load_ps(S1 + x);
                y1 = _mm_load_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

                x0 = _mm_load_ps(S0 + x + 8);
                x1 = _mm_load_ps(S0 + x + 12);
                y0 = _mm_load_ps(S1 + x + 8);
                y1 = _mm_load_ps(S1 + x + 12);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

                _mm_storeu_si128((__m128i*)(dst + x), t0);
                _mm_storeu_si128((__m128i*)(dst + x + 8), t1);
            }
        else
            for (; x <= width - 16; x += 16)
            {
                __m128 x0, x1, y0, y1;
                __m128i t0, t1, t2;
                x0 = _mm_loadu_ps(S0 + x);
                x1 = _mm_loadu_ps(S0 + x + 4);
                y0 = _mm_loadu_ps(S1 + x);
                y1 = _mm_loadu_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t0 = _mm_add_epi16(_mm_packs_epi32(t0, t2), postshift);

                x0 = _mm_loadu_ps(S0 + x + 8);
                x1 = _mm_loadu_ps(S0 + x + 12);
                y0 = _mm_loadu_ps(S1 + x + 8);
                y1 = _mm_loadu_ps(S1 + x + 12);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));
                t1 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
                t2 = _mm_add_epi32(_mm_cvtps_epi32(x1), preshift);
                t1 = _mm_add_epi16(_mm_packs_epi32(t1, t2), postshift);

                _mm_storeu_si128((__m128i*)(dst + x), t0);
                _mm_storeu_si128((__m128i*)(dst + x + 8), t1);
            }

        for (; x < width - 4; x += 4)
        {
            __m128 x0, y0;
            __m128i t0;
            x0 = _mm_loadu_ps(S0 + x);
            y0 = _mm_loadu_ps(S1 + x);

            x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
            t0 = _mm_add_epi32(_mm_cvtps_epi32(x0), preshift);
            t0 = _mm_add_epi16(_mm_packs_epi32(t0, t0), postshift);
            _mm_storel_epi64((__m128i*)(dst + x), t0);
        }

        return x;
    }
};

typedef VResizeLinearVec_32f16<SHRT_MIN> VResizeLinearVec_32f16u;
typedef VResizeLinearVec_32f16<0> VResizeLinearVec_32f16s;

struct VResizeLinearVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE)) return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        float* dst = (float*)_dst;
        int x = 0;

        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]);

        if ((((size_t)S0 | (size_t)S1) & 15) == 0)
            for (; x <= width - 8; x += 8)
            {
                __m128 x0, x1, y0, y1;
                x0 = _mm_load_ps(S0 + x);
                x1 = _mm_load_ps(S0 + x + 4);
                y0 = _mm_load_ps(S1 + x);
                y1 = _mm_load_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

                _mm_storeu_ps(dst + x, x0);
                _mm_storeu_ps(dst + x + 4, x1);
            }
        else
            for (; x <= width - 8; x += 8)
            {
                __m128 x0, x1, y0, y1;
                x0 = _mm_loadu_ps(S0 + x);
                x1 = _mm_loadu_ps(S0 + x + 4);
                y0 = _mm_loadu_ps(S1 + x);
                y1 = _mm_loadu_ps(S1 + x + 4);

                x0 = _mm_add_ps(_mm_mul_ps(x0, b0), _mm_mul_ps(y0, b1));
                x1 = _mm_add_ps(_mm_mul_ps(x1, b0), _mm_mul_ps(y1, b1));

                _mm_storeu_ps(dst + x, x0);
                _mm_storeu_ps(dst + x + 4, x1);
            }

        return x;
    }
};

#elif LYCON_NEON

struct VResizeLinearVec_32s8u
{
    int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
    {
        const int **src = (const int**)_src, *S0 = src[0], *S1 = src[1];
        const short* beta = (const short*)_beta;
        int x = 0;
        int16x8_t v_b0 = vdupq_n_s16(beta[0]), v_b1 = vdupq_n_s16(beta[1]), v_delta = vdupq_n_s16(2);

        for (; x <= width - 16; x += 16)
        {
            int32x4_t v_src00 = vshrq_n_s32(vld1q_s32(S0 + x), 4), v_src10 = vshrq_n_s32(vld1q_s32(S1 + x), 4);
            int32x4_t v_src01 = vshrq_n_s32(vld1q_s32(S0 + x + 4), 4), v_src11 = vshrq_n_s32(vld1q_s32(S1 + x + 4), 4);

            int16x8_t v_src0 = vcombine_s16(vmovn_s32(v_src00), vmovn_s32(v_src01));
            int16x8_t v_src1 = vcombine_s16(vmovn_s32(v_src10), vmovn_s32(v_src11));

            int16x8_t v_dst0 =
                vaddq_s16(vshrq_n_s16(vqdmulhq_s16(v_src0, v_b0), 1), vshrq_n_s16(vqdmulhq_s16(v_src1, v_b1), 1));
            v_dst0 = vshrq_n_s16(vaddq_s16(v_dst0, v_delta), 2);

            v_src00 = vshrq_n_s32(vld1q_s32(S0 + x + 8), 4);
            v_src10 = vshrq_n_s32(vld1q_s32(S1 + x + 8), 4);
            v_src01 = vshrq_n_s32(vld1q_s32(S0 + x + 12), 4);
            v_src11 = vshrq_n_s32(vld1q_s32(S1 + x + 12), 4);

            v_src0 = vcombine_s16(vmovn_s32(v_src00), vmovn_s32(v_src01));
            v_src1 = vcombine_s16(vmovn_s32(v_src10), vmovn_s32(v_src11));

            int16x8_t v_dst1 =
                vaddq_s16(vshrq_n_s16(vqdmulhq_s16(v_src0, v_b0), 1), vshrq_n_s16(vqdmulhq_s16(v_src1, v_b1), 1));
            v_dst1 = vshrq_n_s16(vaddq_s16(v_dst1, v_delta), 2);

            vst1q_u8(dst + x, vcombine_u8(vqmovun_s16(v_dst0), vqmovun_s16(v_dst1)));
        }

        return x;
    }
};

struct VResizeLinearVec_32f16u
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        ushort* dst = (ushort*)_dst;
        int x = 0;

        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
            float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

            float32x4_t v_dst0 = vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1);
            float32x4_t v_dst1 = vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1);

            vst1q_u16(dst + x,
                      vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeLinearVec_32f16s
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        short* dst = (short*)_dst;
        int x = 0;

        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
            float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

            float32x4_t v_dst0 = vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1);
            float32x4_t v_dst1 = vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1);

            vst1q_s16(dst + x,
                      vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst0)), vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeLinearVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1];
        float* dst = (float*)_dst;
        int x = 0;

        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_src00 = vld1q_f32(S0 + x), v_src01 = vld1q_f32(S0 + x + 4);
            float32x4_t v_src10 = vld1q_f32(S1 + x), v_src11 = vld1q_f32(S1 + x + 4);

            vst1q_f32(dst + x, vmlaq_f32(vmulq_f32(v_src00, v_b0), v_src10, v_b1));
            vst1q_f32(dst + x + 4, vmlaq_f32(vmulq_f32(v_src01, v_b0), v_src11, v_b1));
        }

        return x;
    }
};

typedef VResizeNoVec VResizeCubicVec_32s8u;

#else

typedef VResizeNoVec VResizeLinearVec_32s8u;
typedef VResizeNoVec VResizeLinearVec_32f16u;
typedef VResizeNoVec VResizeLinearVec_32f16s;
typedef VResizeNoVec VResizeLinearVec_32f;

#endif

typedef HResizeNoVec HResizeLinearVec_8u32s;
typedef HResizeNoVec HResizeLinearVec_16u32f;
typedef HResizeNoVec HResizeLinearVec_16s32f;
typedef HResizeNoVec HResizeLinearVec_32f;
typedef HResizeNoVec HResizeLinearVec_64f;

template <typename T, typename WT, typename AT, int ONE, class VecOp>
struct HResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count, const int* xofs, const AT* alpha, int swidth, int dwidth,
                    int cn, int xmin, int xmax) const
    {
        int dx, k;
        VecOp vecOp;

        int dx0 =
            vecOp((const uchar**)src, (uchar**)dst, count, xofs, (const uchar*)alpha, swidth, dwidth, cn, xmin, xmax);

        for (k = 0; k <= count - 2; k++)
        {
            const T *S0 = src[k], *S1 = src[k + 1];
            WT *D0 = dst[k], *D1 = dst[k + 1];
            for (dx = dx0; dx < xmax; dx++)
            {
                int sx = xofs[dx];
                WT a0 = alpha[dx * 2], a1 = alpha[dx * 2 + 1];
                WT t0 = S0[sx] * a0 + S0[sx + cn] * a1;
                WT t1 = S1[sx] * a0 + S1[sx + cn] * a1;
                D0[dx] = t0;
                D1[dx] = t1;
            }

            for (; dx < dwidth; dx++)
            {
                int sx = xofs[dx];
                D0[dx] = WT(S0[sx] * ONE);
                D1[dx] = WT(S1[sx] * ONE);
            }
        }

        for (; k < count; k++)
        {
            const T* S = src[k];
            WT* D = dst[k];
            for (dx = 0; dx < xmax; dx++)
            {
                int sx = xofs[dx];
                D[dx] = S[sx] * alpha[dx * 2] + S[sx + cn] * alpha[dx * 2 + 1];
            }

            for (; dx < dwidth; dx++) D[dx] = WT(S[xofs[dx]] * ONE);
        }
    }
};

template <typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLinear
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width) const
    {
        WT b0 = beta[0], b1 = beta[1];
        const WT *S0 = src[0], *S1 = src[1];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if LYCON_ENABLE_UNROLLED
        for (; x <= width - 4; x += 4)
        {
            WT t0, t1;
            t0 = S0[x] * b0 + S1[x] * b1;
            t1 = S0[x + 1] * b0 + S1[x + 1] * b1;
            dst[x] = castOp(t0);
            dst[x + 1] = castOp(t1);
            t0 = S0[x + 2] * b0 + S1[x + 2] * b1;
            t1 = S0[x + 3] * b0 + S1[x + 3] * b1;
            dst[x + 2] = castOp(t0);
            dst[x + 3] = castOp(t1);
        }
#endif
        for (; x < width; x++) dst[x] = castOp(S0[x] * b0 + S1[x] * b1);
    }
};

template <>
struct VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>, VResizeLinearVec_32s8u>
{
    typedef uchar value_type;
    typedef int buf_type;
    typedef short alpha_type;

    void operator()(const buf_type** src, value_type* dst, const alpha_type* beta, int width) const
    {
        alpha_type b0 = beta[0], b1 = beta[1];
        const buf_type *S0 = src[0], *S1 = src[1];
        VResizeLinearVec_32s8u vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if LYCON_ENABLE_UNROLLED
        for (; x <= width - 4; x += 4)
        {
            dst[x + 0] = uchar((((b0 * (S0[x + 0] >> 4)) >> 16) + ((b1 * (S1[x + 0] >> 4)) >> 16) + 2) >> 2);
            dst[x + 1] = uchar((((b0 * (S0[x + 1] >> 4)) >> 16) + ((b1 * (S1[x + 1] >> 4)) >> 16) + 2) >> 2);
            dst[x + 2] = uchar((((b0 * (S0[x + 2] >> 4)) >> 16) + ((b1 * (S1[x + 2] >> 4)) >> 16) + 2) >> 2);
            dst[x + 3] = uchar((((b0 * (S0[x + 3] >> 4)) >> 16) + ((b1 * (S1[x + 3] >> 4)) >> 16) + 2) >> 2);
        }
#endif
        for (; x < width; x++) dst[x] = uchar((((b0 * (S0[x] >> 4)) >> 16) + ((b1 * (S1[x] >> 4)) >> 16) + 2) >> 2);
    }
};
