#if LYCON_SSE2
#if LYCON_SSE4_1

struct VResizeLanczos4Vec_32f16u
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        short* dst = (short*)_dst;
        int x = 0;
        __m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]), v_b2 = _mm_set1_ps(beta[2]),
               v_b3 = _mm_set1_ps(beta[3]), v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
               v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_dst0 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

            __m128 v_dst1 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x + 4));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x + 4)));

            __m128i v_dsti0 = _mm_cvtps_epi32(v_dst0);
            __m128i v_dsti1 = _mm_cvtps_epi32(v_dst1);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_packus_epi32(v_dsti0, v_dsti1));
        }

        return x;
    }
};

#else

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;

#endif

struct VResizeLanczos4Vec_32f16s
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        short* dst = (short*)_dst;
        int x = 0;
        __m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]), v_b2 = _mm_set1_ps(beta[2]),
               v_b3 = _mm_set1_ps(beta[3]), v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
               v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

        for (; x <= width - 8; x += 8)
        {
            __m128 v_dst0 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
            v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

            __m128 v_dst1 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x + 4));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x + 4)));
            v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x + 4)));

            __m128i v_dsti0 = _mm_cvtps_epi32(v_dst0);
            __m128i v_dsti1 = _mm_cvtps_epi32(v_dst1);

            _mm_storeu_si128((__m128i*)(dst + x), _mm_packs_epi32(v_dsti0, v_dsti1));
        }

        return x;
    }
};

struct VResizeLanczos4Vec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        float* dst = (float*)_dst;
        int x = 0;

        __m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]), v_b2 = _mm_set1_ps(beta[2]),
               v_b3 = _mm_set1_ps(beta[3]), v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
               v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

        for (; x <= width - 4; x += 4)
        {
            __m128 v_dst = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
            v_dst = _mm_add_ps(v_dst, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

            _mm_storeu_ps(dst + x, v_dst);
        }

        return x;
    }
};

#elif LYCON_NEON

struct VResizeLanczos4Vec_32f16u
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        ushort* dst = (ushort*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]), v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
                    v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst0 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                    vld1q_f32(S2 + x)),
                          v_b3, vld1q_f32(S3 + x));
            float32x4_t v_dst1 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)), v_b5, vld1q_f32(S5 + x)), v_b6,
                                    vld1q_f32(S6 + x)),
                          v_b7, vld1q_f32(S7 + x));
            float32x4_t v_dst = vaddq_f32(v_dst0, v_dst1);

            v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)), v_b1, vld1q_f32(S1 + x + 4)),
                                         v_b2, vld1q_f32(S2 + x + 4)),
                               v_b3, vld1q_f32(S3 + x + 4));
            v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x + 4)), v_b5, vld1q_f32(S5 + x + 4)),
                                         v_b6, vld1q_f32(S6 + x + 4)),
                               v_b7, vld1q_f32(S7 + x + 4));
            v_dst1 = vaddq_f32(v_dst0, v_dst1);

            vst1q_u16(dst + x, vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst)), vqmovn_u32(cv_vrndq_u32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeLanczos4Vec_32f16s
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        short* dst = (short*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]), v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
                    v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

        for (; x <= width - 8; x += 8)
        {
            float32x4_t v_dst0 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                    vld1q_f32(S2 + x)),
                          v_b3, vld1q_f32(S3 + x));
            float32x4_t v_dst1 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)), v_b5, vld1q_f32(S5 + x)), v_b6,
                                    vld1q_f32(S6 + x)),
                          v_b7, vld1q_f32(S7 + x));
            float32x4_t v_dst = vaddq_f32(v_dst0, v_dst1);

            v_dst0 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x + 4)), v_b1, vld1q_f32(S1 + x + 4)),
                                         v_b2, vld1q_f32(S2 + x + 4)),
                               v_b3, vld1q_f32(S3 + x + 4));
            v_dst1 = vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x + 4)), v_b5, vld1q_f32(S5 + x + 4)),
                                         v_b6, vld1q_f32(S6 + x + 4)),
                               v_b7, vld1q_f32(S7 + x + 4));
            v_dst1 = vaddq_f32(v_dst0, v_dst1);

            vst1q_s16(dst + x, vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst)), vqmovn_s32(cv_vrndq_s32_f32(v_dst1))));
        }

        return x;
    }
};

struct VResizeLanczos4Vec_32f
{
    int operator()(const uchar** _src, uchar* _dst, const uchar* _beta, int width) const
    {
        const float** src = (const float**)_src;
        const float* beta = (const float*)_beta;
        const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3], *S4 = src[4], *S5 = src[5], *S6 = src[6],
                    *S7 = src[7];
        float* dst = (float*)_dst;
        int x = 0;
        float32x4_t v_b0 = vdupq_n_f32(beta[0]), v_b1 = vdupq_n_f32(beta[1]), v_b2 = vdupq_n_f32(beta[2]),
                    v_b3 = vdupq_n_f32(beta[3]), v_b4 = vdupq_n_f32(beta[4]), v_b5 = vdupq_n_f32(beta[5]),
                    v_b6 = vdupq_n_f32(beta[6]), v_b7 = vdupq_n_f32(beta[7]);

        for (; x <= width - 4; x += 4)
        {
            float32x4_t v_dst0 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b0, vld1q_f32(S0 + x)), v_b1, vld1q_f32(S1 + x)), v_b2,
                                    vld1q_f32(S2 + x)),
                          v_b3, vld1q_f32(S3 + x));
            float32x4_t v_dst1 =
                vmlaq_f32(vmlaq_f32(vmlaq_f32(vmulq_f32(v_b4, vld1q_f32(S4 + x)), v_b5, vld1q_f32(S5 + x)), v_b6,
                                    vld1q_f32(S6 + x)),
                          v_b7, vld1q_f32(S7 + x));
            vst1q_f32(dst + x, vaddq_f32(v_dst0, v_dst1));
        }

        return x;
    }
};

#else

typedef VResizeNoVec VResizeLanczos4Vec_32f16u;
typedef VResizeNoVec VResizeLanczos4Vec_32f16s;
typedef VResizeNoVec VResizeLanczos4Vec_32f;

#endif

template <typename T, typename WT, typename AT>
struct HResizeLanczos4
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
                for (; dx < limit; dx++, alpha += 8)
                {
                    int j, sx = xofs[dx] - cn * 3;
                    WT v = 0;
                    for (j = 0; j < 8; j++)
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
                for (; dx < xmax; dx++, alpha += 8)
                {
                    int sx = xofs[dx];
                    D[dx] = S[sx - cn * 3] * alpha[0] + S[sx - cn * 2] * alpha[1] + S[sx - cn] * alpha[2] +
                            S[sx] * alpha[3] + S[sx + cn] * alpha[4] + S[sx + cn * 2] * alpha[5] +
                            S[sx + cn * 3] * alpha[6] + S[sx + cn * 4] * alpha[7];
                }
                limit = dwidth;
            }
            alpha -= dwidth * 8;
        }
    }
};

template <typename T, typename WT, typename AT, class CastOp, class VecOp>
struct VResizeLanczos4
{
    typedef T value_type;
    typedef WT buf_type;
    typedef AT alpha_type;

    void operator()(const WT** src, T* dst, const AT* beta, int width) const
    {
        CastOp castOp;
        VecOp vecOp;
        int k, x = vecOp((const uchar**)src, (uchar*)dst, (const uchar*)beta, width);
#if LYCON_ENABLE_UNROLLED
        for (; x <= width - 4; x += 4)
        {
            WT b = beta[0];
            const WT* S = src[0];
            WT s0 = S[x] * b, s1 = S[x + 1] * b, s2 = S[x + 2] * b, s3 = S[x + 3] * b;

            for (k = 1; k < 8; k++)
            {
                b = beta[k];
                S = src[k];
                s0 += S[x] * b;
                s1 += S[x + 1] * b;
                s2 += S[x + 2] * b;
                s3 += S[x + 3] * b;
            }

            dst[x] = castOp(s0);
            dst[x + 1] = castOp(s1);
            dst[x + 2] = castOp(s2);
            dst[x + 3] = castOp(s3);
        }
#endif
        for (; x < width; x++)
        {
            dst[x] = castOp(src[0][x] * beta[0] + src[1][x] * beta[1] + src[2][x] * beta[2] + src[3][x] * beta[3] +
                            src[4][x] * beta[4] + src[5][x] * beta[5] + src[6][x] * beta[6] + src[7][x] * beta[7]);
        }
    }
};
