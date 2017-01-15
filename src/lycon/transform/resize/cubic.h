#if LYCON_SSE2

struct VResizeCubicVec_32s8u
{
    int operator()(const uchar** _src, uchar* dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE2)) return 0;

        const int** src = (const int**)_src;
        const short* beta = (const short*)_beta;
        const int *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        int x = 0;
        float scale = 1.f / (INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE);
        __m128 b0 = _mm_set1_ps(beta[0] * scale), b1 = _mm_set1_ps(beta[1] * scale), b2 = _mm_set1_ps(beta[2] * scale),
               b3 = _mm_set1_ps(beta[3] * scale);

        if ((((size_t)S0 | (size_t)S1 | (size_t)S2 | (size_t)S3) & 15) == 0)
            for (; x <= width - 8; x += 8)
            {
                __m128i x0, x1, y0, y1;
                __m128 s0, s1, f0, f1;
                x0 = _mm_load_si128((const __m128i*)(S0 + x));
                x1 = _mm_load_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S1 + x));
                y1 = _mm_load_si128((const __m128i*)(S1 + x + 4));

                s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
                s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_load_si128((const __m128i*)(S2 + x));
                x1 = _mm_load_si128((const __m128i*)(S2 + x + 4));
                y0 = _mm_load_si128((const __m128i*)(S3 + x));
                y1 = _mm_load_si128((const __m128i*)(S3 + x + 4));

                f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_cvtps_epi32(s0);
                x1 = _mm_cvtps_epi32(s1);

                x0 = _mm_packs_epi32(x0, x1);
                _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
            }
        else
            for (; x <= width - 8; x += 8)
            {
                __m128i x0, x1, y0, y1;
                __m128 s0, s1, f0, f1;
                x0 = _mm_loadu_si128((const __m128i*)(S0 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S0 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S1 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S1 + x + 4));

                s0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b0);
                s1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b0);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b1);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b1);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_loadu_si128((const __m128i*)(S2 + x));
                x1 = _mm_loadu_si128((const __m128i*)(S2 + x + 4));
                y0 = _mm_loadu_si128((const __m128i*)(S3 + x));
                y1 = _mm_loadu_si128((const __m128i*)(S3 + x + 4));

                f0 = _mm_mul_ps(_mm_cvtepi32_ps(x0), b2);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(x1), b2);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);
                f0 = _mm_mul_ps(_mm_cvtepi32_ps(y0), b3);
                f1 = _mm_mul_ps(_mm_cvtepi32_ps(y1), b3);
                s0 = _mm_add_ps(s0, f0);
                s1 = _mm_add_ps(s1, f1);

                x0 = _mm_cvtps_epi32(s0);
                x1 = _mm_cvtps_epi32(s1);

                x0 = _mm_packs_epi32(x0, x1);
                _mm_storel_epi64((__m128i*)(dst + x), _mm_packus_epi16(x0, x0));
            }

        return x;
    }
};

template <int shiftval>
struct VResizeCubicVec_32f16
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE2)) return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        ushort* dst = (ushort*)_dst;
        int x = 0;
        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]), b2 = _mm_set1_ps(beta[2]),
               b3 = _mm_set1_ps(beta[3]);
        __m128i preshift = _mm_set1_epi32(shiftval);
        __m128i postshift = _mm_set1_epi16((short)shiftval);

        for (; x <= width - 8; x += 8)
        {
            __m128 x0, x1, y0, y1, s0, s1;
            __m128i t0, t1;
            x0 = _mm_loadu_ps(S0 + x);
            x1 = _mm_loadu_ps(S0 + x + 4);
            y0 = _mm_loadu_ps(S1 + x);
            y1 = _mm_loadu_ps(S1 + x + 4);

            s0 = _mm_mul_ps(x0, b0);
            s1 = _mm_mul_ps(x1, b0);
            y0 = _mm_mul_ps(y0, b1);
            y1 = _mm_mul_ps(y1, b1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            x0 = _mm_loadu_ps(S2 + x);
            x1 = _mm_loadu_ps(S2 + x + 4);
            y0 = _mm_loadu_ps(S3 + x);
            y1 = _mm_loadu_ps(S3 + x + 4);

            x0 = _mm_mul_ps(x0, b2);
            x1 = _mm_mul_ps(x1, b2);
            y0 = _mm_mul_ps(y0, b3);
            y1 = _mm_mul_ps(y1, b3);
            s0 = _mm_add_ps(s0, x0);
            s1 = _mm_add_ps(s1, x1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            t0 = _mm_add_epi32(_mm_cvtps_epi32(s0), preshift);
            t1 = _mm_add_epi32(_mm_cvtps_epi32(s1), preshift);

            t0 = _mm_add_epi16(_mm_packs_epi32(t0, t1), postshift);
            _mm_storeu_si128((__m128i*)(dst + x), t0);
        }

        return x;
    }
};

typedef VResizeCubicVec_32f16<SHRT_MIN> VResizeCubicVec_32f16u;
typedef VResizeCubicVec_32f16<0> VResizeCubicVec_32f16s;

struct VResizeCubicVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        if (!checkHardwareSupport(LYCON_CPU_SSE)) return 0;

        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        float* dst = (float*)_dst;
        int x = 0;
        __m128 b0 = _mm_set1_ps(beta[0]), b1 = _mm_set1_ps(beta[1]), b2 = _mm_set1_ps(beta[2]),
               b3 = _mm_set1_ps(beta[3]);

        for (; x <= width - 8; x += 8)
        {
            __m128 x0, x1, y0, y1, s0, s1;
            x0 = _mm_loadu_ps(S0 + x);
            x1 = _mm_loadu_ps(S0 + x + 4);
            y0 = _mm_loadu_ps(S1 + x);
            y1 = _mm_loadu_ps(S1 + x + 4);

            s0 = _mm_mul_ps(x0, b0);
            s1 = _mm_mul_ps(x1, b0);
            y0 = _mm_mul_ps(y0, b1);
            y1 = _mm_mul_ps(y1, b1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            x0 = _mm_loadu_ps(S2 + x);
            x1 = _mm_loadu_ps(S2 + x + 4);
            y0 = _mm_loadu_ps(S3 + x);
            y1 = _mm_loadu_ps(S3 + x + 4);

            x0 = _mm_mul_ps(x0, b2);
            x1 = _mm_mul_ps(x1, b2);
            y0 = _mm_mul_ps(y0, b3);
            y1 = _mm_mul_ps(y1, b3);
            s0 = _mm_add_ps(s0, x0);
            s1 = _mm_add_ps(s1, x1);
            s0 = _mm_add_ps(s0, y0);
            s1 = _mm_add_ps(s1, y1);

            _mm_storeu_ps(dst + x, s0);
            _mm_storeu_ps(dst + x + 4, s1);
        }

        return x;
    }
};

#elif LYCON_NEON

struct VResizeCubicVec_32f16u
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        ushort* dst = (ushort*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst0 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                    vld1q_f32(S2 + x)),
                          v_b3, vld1q_f32(S3 + x));
            float32x4_t v_dst1 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)), v_b1, vld1q_f32(S1 + x + 4)),
                                    v_b2, vld1q_f32(S2 + x + 4)),
                          v_b3, vld1q_f32(S3 + x + 4));

            vst1q_u16(dst + x,
                      vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst0)), vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeCubicVec_32f16s
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        short* dst = (short*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst0 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                    vld1q_f32(S2 + x)),
                          v_b3, vld1q_f32(S3 + x));
            float32x4_t v_dst1 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)), v_b1, vld1q_f32(S1 + x + 4)),
                                    v_b2, vld1q_f32(S2 + x + 4)),
                          v_b3, vld1q_f32(S3 + x + 4));

            vst1q_s16(dst + x,
                      vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst0)), vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeCubicVec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        float* dst = (float*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]);

        for (; x <= width - 8; x += 8)
        {
            vst1q_f32(dst + x,
                      vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                          vld1q_f32(S2 + x)),
                                v_b3, vld1q_f32(S3 + x)));
            vst1q_f32(
                dst + x + 4,
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)), v_b1, vld1q_f32(S1 + x + 4)),
                                    v_b2, vld1q_f32(S2 + x + 4)),
                          v_b3, vld1q_f32(S3 + x + 4)));
        }

        return x;
    }
};

#else

typedef VResizeNoVec VResizeCubicVec_32s8u;
typedef VResizeNoVec VResizeCubicVec_32f16u;
typedef VResizeNoVec VResizeCubicVec_32f16s;
typedef VResizeNoVec VResizeCubicVec_32f;

#endif

template <typename T, typename WT, typename AT>
struct HResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const T** src, WT** dst, int count, const int* xofs, const AT* alpha, int swidth, int dwidth,
                    int cn, int xmin, int xmax) const
    {
        for (int k = 0; k < count; k++)
        {
            const T* S = src[k];
            WT* D = dst[k];
            int dx = 0, limit = xmin;
            for (;;)
            {
                for (; dx < limit; dx++, alpha += 4)
                {
                    int j, sx = xofs[dx] - cn;
                    WT v = 0;
                    for (j = 0; j < 4; j++)
                    {
                        int sxj = sx + j * cn;
                        if ((unsigned)sxj >= (unsigned)swidth)
                        {
                            while (sxj < 0) sxj += cn;
                            while (sxj >= swidth) sxj -= cn;
                        }
                        v += S[sxj] * alpha[j];
                    }
                    D[dx] = v;
                }
                if (limit == dwidth) break;
                for (; dx < xmax; dx++, alpha += 4)
                {
                    int sx = xofs[dx];
                    D[dx] =
                        S[sx - cn] * alpha[0] + S[sx] * alpha[1] + S[sx + cn] * alpha[2] + S[sx + cn * 2] * alpha[3];
                }
                limit = dwidth;
            }
            alpha -= dwidth * 4;
        }
    }
};

template <typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeCubic
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width) const
    {
        WT b0 = beta[0], b1 = beta[1], b2 = beta[2], b3 = beta[3];
        const WT *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3];
        CastOp castOp;
        VecOp vecOp;

        int x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
        for (; x < width; x++) dst[x] = castOp(S0[x] * b0 + S1[x] * b1 + S2[x] * b2 + S3[x] * b3);
    }
};
