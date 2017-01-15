template <typename T, typename WT>
struct ResizeAreaFastNoVec
{
    ResizeAreaFastNoVec(int, int) {}
    ResizeAreaFastNoVec(int, int, int, int) {}
    int operator()(const T*, T*, int) const { return 0; }
};

#if LYCON_NEON

class ResizeAreaFastVec_SIMD_8u
{
   public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) : cn(_cn), step(_step) {}

    int operator()(const uchar* S, uchar* D, int w) const
    {
        int dx = 0;
        const uchar *S0 = S, *S1 = S0 + step;

        uint16x8_t v_2 = vdupq_n_u16(2);

        if (cn == 1)
        {
            for (; dx <= w - 16; dx += 16, S0 += 32, S1 += 32, D += 16)
            {
                uint8x16x2_t v_row0 = vld2q_u8(S0), v_row1 = vld2q_u8(S1);

                uint16x8_t v_dst0 = vaddl_u8(vget_low_u8(v_row0.val[0]), vget_low_u8(v_row0.val[1]));
                v_dst0 = vaddq_u16(v_dst0, vaddl_u8(vget_low_u8(v_row1.val[0]), vget_low_u8(v_row1.val[1])));
                v_dst0 = vshrq_n_u16(vaddq_u16(v_dst0, v_2), 2);

                uint16x8_t v_dst1 = vaddl_u8(vget_high_u8(v_row0.val[0]), vget_high_u8(v_row0.val[1]));
                v_dst1 = vaddq_u16(v_dst1, vaddl_u8(vget_high_u8(v_row1.val[0]), vget_high_u8(v_row1.val[1])));
                v_dst1 = vshrq_n_u16(vaddq_u16(v_dst1, v_2), 2);

                vst1q_u8(D, vcombine_u8(vmovn_u16(v_dst0), vmovn_u16(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for (; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint8x16_t v_row0 = vld1q_u8(S0), v_row1 = vld1q_u8(S1);

                uint16x8_t v_row00 = vmovl_u8(vget_low_u8(v_row0));
                uint16x8_t v_row01 = vmovl_u8(vget_high_u8(v_row0));
                uint16x8_t v_row10 = vmovl_u8(vget_low_u8(v_row1));
                uint16x8_t v_row11 = vmovl_u8(vget_high_u8(v_row1));

                uint16x4_t v_p0 = vadd_u16(vadd_u16(vget_low_u16(v_row00), vget_high_u16(v_row00)),
                                           vadd_u16(vget_low_u16(v_row10), vget_high_u16(v_row10)));
                uint16x4_t v_p1 = vadd_u16(vadd_u16(vget_low_u16(v_row01), vget_high_u16(v_row01)),
                                           vadd_u16(vget_low_u16(v_row11), vget_high_u16(v_row11)));
                uint16x8_t v_dst = vshrq_n_u16(vaddq_u16(vcombine_u16(v_p0, v_p1), v_2), 2);

                vst1_u8(D, vmovn_u16(v_dst));
            }
        }

        return dx;
    }

   private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16u
{
   public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) : cn(_cn), step(_step) {}

    int operator()(const ushort* S, ushort* D, int w) const
    {
        int dx = 0;
        const ushort *S0 = S, *S1 = (const ushort*)((const uchar*)(S0) + step);

        uint32x4_t v_2 = vdupq_n_u32(2);

        if (cn == 1)
        {
            for (; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                uint16x8x2_t v_row0 = vld2q_u16(S0), v_row1 = vld2q_u16(S1);

                uint32x4_t v_dst0 = vaddl_u16(vget_low_u16(v_row0.val[0]), vget_low_u16(v_row0.val[1]));
                v_dst0 = vaddq_u32(v_dst0, vaddl_u16(vget_low_u16(v_row1.val[0]), vget_low_u16(v_row1.val[1])));
                v_dst0 = vshrq_n_u32(vaddq_u32(v_dst0, v_2), 2);

                uint32x4_t v_dst1 = vaddl_u16(vget_high_u16(v_row0.val[0]), vget_high_u16(v_row0.val[1]));
                v_dst1 = vaddq_u32(v_dst1, vaddl_u16(vget_high_u16(v_row1.val[0]), vget_high_u16(v_row1.val[1])));
                v_dst1 = vshrq_n_u32(vaddq_u32(v_dst1, v_2), 2);

                vst1q_u16(D, vcombine_u16(vmovn_u32(v_dst0), vmovn_u32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                uint16x8_t v_row0 = vld1q_u16(S0), v_row1 = vld1q_u16(S1);
                uint32x4_t v_dst = vaddq_u32(vaddl_u16(vget_low_u16(v_row0), vget_high_u16(v_row0)),
                                             vaddl_u16(vget_low_u16(v_row1), vget_high_u16(v_row1)));
                vst1_u16(D, vmovn_u32(vshrq_n_u32(vaddq_u32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

   private:
    int cn, step;
};

class ResizeAreaFastVec_SIMD_16s
{
   public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) : cn(_cn), step(_step) {}

    int operator()(const short* S, short* D, int w) const
    {
        int dx = 0;
        const short *S0 = S, *S1 = (const short*)((const uchar*)(S0) + step);

        int32x4_t v_2 = vdupq_n_s32(2);

        if (cn == 1)
        {
            for (; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                int16x8x2_t v_row0 = vld2q_s16(S0), v_row1 = vld2q_s16(S1);

                int32x4_t v_dst0 = vaddl_s16(vget_low_s16(v_row0.val[0]), vget_low_s16(v_row0.val[1]));
                v_dst0 = vaddq_s32(v_dst0, vaddl_s16(vget_low_s16(v_row1.val[0]), vget_low_s16(v_row1.val[1])));
                v_dst0 = vshrq_n_s32(vaddq_s32(v_dst0, v_2), 2);

                int32x4_t v_dst1 = vaddl_s16(vget_high_s16(v_row0.val[0]), vget_high_s16(v_row0.val[1]));
                v_dst1 = vaddq_s32(v_dst1, vaddl_s16(vget_high_s16(v_row1.val[0]), vget_high_s16(v_row1.val[1])));
                v_dst1 = vshrq_n_s32(vaddq_s32(v_dst1, v_2), 2);

                vst1q_s16(D, vcombine_s16(vmovn_s32(v_dst0), vmovn_s32(v_dst1)));
            }
        }
        else if (cn == 4)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                int16x8_t v_row0 = vld1q_s16(S0), v_row1 = vld1q_s16(S1);
                int32x4_t v_dst = vaddq_s32(vaddl_s16(vget_low_s16(v_row0), vget_high_s16(v_row0)),
                                            vaddl_s16(vget_low_s16(v_row1), vget_high_s16(v_row1)));
                vst1_s16(D, vmovn_s32(vshrq_n_s32(vaddq_s32(v_dst, v_2), 2)));
            }
        }

        return dx;
    }

   private:
    int cn, step;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) : cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
    }

    int operator()(const float* S, float* D, int w) const
    {
        if (!fast_mode) return 0;

        const float *S0 = S, *S1 = (const float*)((const uchar*)(S0) + step);
        int dx = 0;

        float32x4_t v_025 = vdupq_n_f32(0.25f);

        if (cn == 1)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4x2_t v_row0 = vld2q_f32(S0), v_row1 = vld2q_f32(S1);

                float32x4_t v_dst0 = vaddq_f32(v_row0.val[0], v_row0.val[1]);
                float32x4_t v_dst1 = vaddq_f32(v_row1.val[0], v_row1.val[1]);

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                float32x4_t v_dst0 = vaddq_f32(vld1q_f32(S0), vld1q_f32(S0 + 4));
                float32x4_t v_dst1 = vaddq_f32(vld1q_f32(S1), vld1q_f32(S1 + 4));

                vst1q_f32(D, vmulq_f32(vaddq_f32(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

   private:
    int cn;
    bool fast_mode;
    int step;
};

#elif LYCON_SSE2

class ResizeAreaFastVec_SIMD_8u
{
   public:
    ResizeAreaFastVec_SIMD_8u(int _cn, int _step) : cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(LYCON_CPU_SSE2);
    }

    int operator()(const uchar* S, uchar* D, int w) const
    {
        if (!use_simd) return 0;

        int dx = 0;
        const uchar* S0 = S;
        const uchar* S1 = S0 + step;
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi16(2);

        if (cn == 1)
        {
            __m128i masklow = _mm_set1_epi16(0x00ff);
            for (; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi16(_mm_srli_epi16(r0, 8), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi16(_mm_srli_epi16(r1, 8), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi16(_mm_add_epi16(s0, s1), delta2);
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for (; dx <= w - 11; dx += 6, S0 += 12, S1 += 12, D += 6)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi8(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi8(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 6));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 6));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 6));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                s0 = _mm_packus_epi16(_mm_srli_epi16(s0, 2), zero);
                _mm_storel_epi64((__m128i*)(D + 3), s0);
            }
        else
        {
            LYCON_ASSERT(cn == 4);
            int v[] = {0, 0, -1, -1};
            __m128i mask = _mm_loadu_si128((const __m128i*)v);

            for (; dx <= w - 8; dx += 8, S0 += 16, S1 += 16, D += 8)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi8(r0, zero);
                __m128i r0_16h = _mm_unpackhi_epi8(r0, zero);
                __m128i r1_16l = _mm_unpacklo_epi8(r1, zero);
                __m128i r1_16h = _mm_unpackhi_epi8(r1, zero);

                __m128i s0 = _mm_add_epi16(r0_16l, _mm_srli_si128(r0_16l, 8));
                __m128i s1 = _mm_add_epi16(r1_16l, _mm_srli_si128(r1_16l, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res0 = _mm_srli_epi16(s0, 2);

                s0 = _mm_add_epi16(r0_16h, _mm_srli_si128(r0_16h, 8));
                s1 = _mm_add_epi16(r1_16h, _mm_srli_si128(r1_16h, 8));
                s0 = _mm_add_epi16(s1, _mm_add_epi16(s0, delta2));
                __m128i res1 = _mm_srli_epi16(s0, 2);
                s0 = _mm_packus_epi16(
                    _mm_or_si128(_mm_andnot_si128(mask, res0), _mm_and_si128(mask, _mm_slli_si128(res1, 8))), zero);
                _mm_storel_epi64((__m128i*)(D), s0);
            }
        }

        return dx;
    }

   private:
    int cn;
    bool use_simd;
    int step;
};

class ResizeAreaFastVec_SIMD_16u
{
   public:
    ResizeAreaFastVec_SIMD_16u(int _cn, int _step) : cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(LYCON_CPU_SSE2);
    }

    int operator()(const ushort* S, ushort* D, int w) const
    {
        if (!use_simd) return 0;

        int dx = 0;
        const ushort* S0 = (const ushort*)S;
        const ushort* S1 = (const ushort*)((const uchar*)(S) + step);
        __m128i masklow = _mm_set1_epi32(0x0000ffff);
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi32(2);

#define _mm_packus_epi32(a, zero) _mm_packs_epi32(_mm_srai_epi32(_mm_slli_epi32(a, 16), 16), zero)

        if (cn == 1)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi32(_mm_srli_epi32(r0, 16), _mm_and_si128(r0, masklow));
                __m128i s1 = _mm_add_epi32(_mm_srli_epi32(r1, 16), _mm_and_si128(r1, masklow));
                s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), delta2);
                s0 = _mm_srli_epi32(s0, 2);
                s0 = _mm_packus_epi32(s0, zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for (; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_16h = _mm_unpacklo_epi16(_mm_srli_si128(r0, 6), zero);
                __m128i r1_16l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_16h = _mm_unpacklo_epi16(_mm_srli_si128(r1, 6), zero);

                __m128i s0 = _mm_add_epi32(r0_16l, r0_16h);
                __m128i s1 = _mm_add_epi32(r1_16l, r1_16h);
                s0 = _mm_add_epi32(delta2, _mm_add_epi32(s0, s1));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        else
        {
            LYCON_ASSERT(cn == 4);
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_32l = _mm_unpacklo_epi16(r0, zero);
                __m128i r0_32h = _mm_unpackhi_epi16(r0, zero);
                __m128i r1_32l = _mm_unpacklo_epi16(r1, zero);
                __m128i r1_32h = _mm_unpackhi_epi16(r1, zero);

                __m128i s0 = _mm_add_epi32(r0_32l, r0_32h);
                __m128i s1 = _mm_add_epi32(r1_32l, r1_32h);
                s0 = _mm_add_epi32(s1, _mm_add_epi32(s0, delta2));
                s0 = _mm_packus_epi32(_mm_srli_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        }

#undef _mm_packus_epi32

        return dx;
    }

   private:
    int cn;
    int step;
    bool use_simd;
};

class ResizeAreaFastVec_SIMD_16s
{
   public:
    ResizeAreaFastVec_SIMD_16s(int _cn, int _step) : cn(_cn), step(_step)
    {
        use_simd = checkHardwareSupport(LYCON_CPU_SSE2);
    }

    int operator()(const short* S, short* D, int w) const
    {
        if (!use_simd) return 0;

        int dx = 0;
        const short* S0 = (const short*)S;
        const short* S1 = (const short*)((const uchar*)(S) + step);
        __m128i masklow = _mm_set1_epi32(0x0000ffff);
        __m128i zero = _mm_setzero_si128();
        __m128i delta2 = _mm_set1_epi32(2);

        if (cn == 1)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i s0 = _mm_add_epi32(_mm_srai_epi32(r0, 16),
                                           _mm_srai_epi32(_mm_slli_epi32(_mm_and_si128(r0, masklow), 16), 16));
                __m128i s1 = _mm_add_epi32(_mm_srai_epi32(r1, 16),
                                           _mm_srai_epi32(_mm_slli_epi32(_mm_and_si128(r1, masklow), 16), 16));
                s0 = _mm_add_epi32(_mm_add_epi32(s0, s1), delta2);
                s0 = _mm_srai_epi32(s0, 2);
                s0 = _mm_packs_epi32(s0, zero);

                _mm_storel_epi64((__m128i*)D, s0);
            }
        }
        else if (cn == 3)
            for (; dx <= w - 4; dx += 3, S0 += 6, S1 += 6, D += 3)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_16l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r0), 16);
                __m128i r0_16h = _mm_srai_epi32(_mm_unpacklo_epi16(zero, _mm_srli_si128(r0, 6)), 16);
                __m128i r1_16l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r1), 16);
                __m128i r1_16h = _mm_srai_epi32(_mm_unpacklo_epi16(zero, _mm_srli_si128(r1, 6)), 16);

                __m128i s0 = _mm_add_epi32(r0_16l, r0_16h);
                __m128i s1 = _mm_add_epi32(r1_16l, r1_16h);
                s0 = _mm_add_epi32(delta2, _mm_add_epi32(s0, s1));
                s0 = _mm_packs_epi32(_mm_srai_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        else
        {
            LYCON_ASSERT(cn == 4);
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128i r0 = _mm_loadu_si128((const __m128i*)S0);
                __m128i r1 = _mm_loadu_si128((const __m128i*)S1);

                __m128i r0_32l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r0), 16);
                __m128i r0_32h = _mm_srai_epi32(_mm_unpackhi_epi16(zero, r0), 16);
                __m128i r1_32l = _mm_srai_epi32(_mm_unpacklo_epi16(zero, r1), 16);
                __m128i r1_32h = _mm_srai_epi32(_mm_unpackhi_epi16(zero, r1), 16);

                __m128i s0 = _mm_add_epi32(r0_32l, r0_32h);
                __m128i s1 = _mm_add_epi32(r1_32l, r1_32h);
                s0 = _mm_add_epi32(s1, _mm_add_epi32(s0, delta2));
                s0 = _mm_packs_epi32(_mm_srai_epi32(s0, 2), zero);
                _mm_storel_epi64((__m128i*)D, s0);
            }
        }

        return dx;
    }

   private:
    int cn;
    int step;
    bool use_simd;
};

struct ResizeAreaFastVec_SIMD_32f
{
    ResizeAreaFastVec_SIMD_32f(int _scale_x, int _scale_y, int _cn, int _step) : cn(_cn), step(_step)
    {
        fast_mode = _scale_x == 2 && _scale_y == 2 && (cn == 1 || cn == 4);
        fast_mode = fast_mode && checkHardwareSupport(LYCON_CPU_SSE2);
    }

    int operator()(const float* S, float* D, int w) const
    {
        if (!fast_mode) return 0;

        const float *S0 = S, *S1 = (const float*)((const uchar*)(S0) + step);
        int dx = 0;

        __m128 v_025 = _mm_set1_ps(0.25f);

        if (cn == 1)
        {
            const int shuffle_lo = _MM_SHUFFLE(2, 0, 2, 0), shuffle_hi = _MM_SHUFFLE(3, 1, 3, 1);
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128 v_row00 = _mm_loadu_ps(S0), v_row01 = _mm_loadu_ps(S0 + 4), v_row10 = _mm_loadu_ps(S1),
                       v_row11 = _mm_loadu_ps(S1 + 4);

                __m128 v_dst0 = _mm_add_ps(_mm_shuffle_ps(v_row00, v_row01, shuffle_lo),
                                           _mm_shuffle_ps(v_row00, v_row01, shuffle_hi));
                __m128 v_dst1 = _mm_add_ps(_mm_shuffle_ps(v_row10, v_row11, shuffle_lo),
                                           _mm_shuffle_ps(v_row10, v_row11, shuffle_hi));

                _mm_storeu_ps(D, _mm_mul_ps(_mm_add_ps(v_dst0, v_dst1), v_025));
            }
        }
        else if (cn == 4)
        {
            for (; dx <= w - 4; dx += 4, S0 += 8, S1 += 8, D += 4)
            {
                __m128 v_dst0 = _mm_add_ps(_mm_loadu_ps(S0), _mm_loadu_ps(S0 + 4));
                __m128 v_dst1 = _mm_add_ps(_mm_loadu_ps(S1), _mm_loadu_ps(S1 + 4));

                _mm_storeu_ps(D, _mm_mul_ps(_mm_add_ps(v_dst0, v_dst1), v_025));
            }
        }

        return dx;
    }

   private:
    int cn;
    bool fast_mode;
    int step;
};

#else

typedef ResizeAreaFastNoVec<uchar, uchar> ResizeAreaFastVec_SIMD_8u;
typedef ResizeAreaFastNoVec<ushort, ushort> ResizeAreaFastVec_SIMD_16u;
typedef ResizeAreaFastNoVec<short, short> ResizeAreaFastVec_SIMD_16s;
typedef ResizeAreaFastNoVec<float, float> ResizeAreaFastVec_SIMD_32f;

#endif

template <typename T, typename SIMDVecOp>
struct ResizeAreaFastVec
{
    ResizeAreaFastVec(int _scale_x, int _scale_y, int _cn, int _step)
        : scale_x(_scale_x), scale_y(_scale_y), cn(_cn), step(_step), vecOp(_cn, _step)
    {
        fast_mode = scale_x == 2 && scale_y == 2 && (cn == 1 || cn == 3 || cn == 4);
    }

    int operator()(const T* S, T* D, int w) const
    {
        if (!fast_mode) return 0;

        const T* nextS = (const T*)((const uchar*)S + step);
        int dx = vecOp(S, D, w);

        if (cn == 1)
            for (; dx < w; ++dx)
            {
                int index = dx * 2;
                D[dx] = (T)((S[index] + S[index + 1] + nextS[index] + nextS[index + 1] + 2) >> 2);
            }
        else if (cn == 3)
            for (; dx < w; dx += 3)
            {
                int index = dx * 2;
                D[dx] = (T)((S[index] + S[index + 3] + nextS[index] + nextS[index + 3] + 2) >> 2);
                D[dx + 1] = (T)((S[index + 1] + S[index + 4] + nextS[index + 1] + nextS[index + 4] + 2) >> 2);
                D[dx + 2] = (T)((S[index + 2] + S[index + 5] + nextS[index + 2] + nextS[index + 5] + 2) >> 2);
            }
        else
        {
            LYCON_ASSERT(cn == 4);
            for (; dx < w; dx += 4)
            {
                int index = dx * 2;
                D[dx] = (T)((S[index] + S[index + 4] + nextS[index] + nextS[index + 4] + 2) >> 2);
                D[dx + 1] = (T)((S[index + 1] + S[index + 5] + nextS[index + 1] + nextS[index + 5] + 2) >> 2);
                D[dx + 2] = (T)((S[index + 2] + S[index + 6] + nextS[index + 2] + nextS[index + 6] + 2) >> 2);
                D[dx + 3] = (T)((S[index + 3] + S[index + 7] + nextS[index + 3] + nextS[index + 7] + 2) >> 2);
            }
        }

        return dx;
    }

   private:
    int scale_x, scale_y;
    int cn;
    bool fast_mode;
    int step;
    SIMDVecOp vecOp;
};

template <typename T, typename WT, typename VecOp>
class resizeAreaFast_Invoker : public ParallelLoopBody
{
   public:
    resizeAreaFast_Invoker(const Mat& _src, Mat& _dst, int _scale_x, int _scale_y, const int* _ofs, const int* _xofs)
        : ParallelLoopBody(), src(_src), dst(_dst), scale_x(_scale_x), scale_y(_scale_y), ofs(_ofs), xofs(_xofs)
    {
    }

    virtual void operator()(const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int cn = src.channels();
        int area = scale_x * scale_y;
        float scale = 1.f / (area);
        int dwidth1 = (ssize.width / scale_x) * cn;
        dsize.width *= cn;
        ssize.width *= cn;
        int dy, dx, k = 0;

        VecOp vop(scale_x, scale_y, src.channels(), (int)src.step /*, area_ofs*/);

        for (dy = range.start; dy < range.end; dy++)
        {
            T* D = (T*)(dst.data + dst.step * dy);
            int sy0 = dy * scale_y;
            int w = sy0 + scale_y <= ssize.height ? dwidth1 : 0;

            if (sy0 >= ssize.height)
            {
                for (dx = 0; dx < dsize.width; dx++) D[dx] = 0;
                continue;
            }

            dx = vop(src.template ptr<T>(sy0), D, w);
            for (; dx < w; dx++)
            {
                const T* S = src.template ptr<T>(sy0) + xofs[dx];
                WT sum = 0;
                k = 0;
#if LYCON_ENABLE_UNROLLED
                for (; k <= area - 4; k += 4) sum += S[ofs[k]] + S[ofs[k + 1]] + S[ofs[k + 2]] + S[ofs[k + 3]];
#endif
                for (; k < area; k++) sum += S[ofs[k]];

                D[dx] = saturate_cast<T>(sum * scale);
            }

            for (; dx < dsize.width; dx++)
            {
                WT sum = 0;
                int count = 0, sx0 = xofs[dx];
                if (sx0 >= ssize.width) D[dx] = 0;

                for (int sy = 0; sy < scale_y; sy++)
                {
                    if (sy0 + sy >= ssize.height) break;
                    const T* S = src.template ptr<T>(sy0 + sy) + sx0;
                    for (int sx = 0; sx < scale_x * cn; sx += cn)
                    {
                        if (sx0 + sx >= ssize.width) break;
                        sum += S[sx];
                        count++;
                    }
                }

                D[dx] = saturate_cast<T>((float)sum / count);
            }
        }
    }

   private:
    Mat src;
    Mat dst;
    int scale_x, scale_y;
    const int *ofs, *xofs;
};

template <typename T, typename WT, typename VecOp>
static void resizeAreaFast_(const Mat& src, Mat& dst, const int* ofs, const int* xofs, int scale_x, int scale_y)
{
    Range range(0, dst.rows);
    resizeAreaFast_Invoker<T, WT, VecOp> invoker(src, dst, scale_x, scale_y, ofs, xofs);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

struct DecimateAlpha
{
    int si, di;
    float alpha;
};

template <typename T, typename WT>
class ResizeArea_Invoker : public ParallelLoopBody
{
   public:
    ResizeArea_Invoker(const Mat& _src, Mat& _dst, const DecimateAlpha* _xtab, int _xtab_size,
                       const DecimateAlpha* _ytab, int _ytab_size, const int* _tabofs)
    {
        src = &_src;
        dst = &_dst;
        xtab0 = _xtab;
        xtab_size0 = _xtab_size;
        ytab = _ytab;
        ytab_size = _ytab_size;
        tabofs = _tabofs;
    }

    virtual void operator()(const Range& range) const
    {
        Size dsize = dst->size();
        int cn = dst->channels();
        dsize.width *= cn;
        AutoBuffer<WT> _buffer(dsize.width * 2);
        const DecimateAlpha* xtab = xtab0;
        int xtab_size = xtab_size0;
        WT *buf = _buffer, *sum = buf + dsize.width;
        int j_start = tabofs[range.start], j_end = tabofs[range.end], j, k, dx, prev_dy = ytab[j_start].di;

        for (dx = 0; dx < dsize.width; dx++) sum[dx] = (WT)0;

        for (j = j_start; j < j_end; j++)
        {
            WT beta = ytab[j].alpha;
            int dy = ytab[j].di;
            int sy = ytab[j].si;

            {
                const T* S = src->template ptr<T>(sy);
                for (dx = 0; dx < dsize.width; dx++) buf[dx] = (WT)0;

                if (cn == 1)
                    for (k = 0; k < xtab_size; k++)
                    {
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        buf[dxn] += S[xtab[k].si] * alpha;
                    }
                else if (cn == 2)
                    for (k = 0; k < xtab_size; k++)
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn] * alpha;
                        WT t1 = buf[dxn + 1] + S[sxn + 1] * alpha;
                        buf[dxn] = t0;
                        buf[dxn + 1] = t1;
                    }
                else if (cn == 3)
                    for (k = 0; k < xtab_size; k++)
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn] * alpha;
                        WT t1 = buf[dxn + 1] + S[sxn + 1] * alpha;
                        WT t2 = buf[dxn + 2] + S[sxn + 2] * alpha;
                        buf[dxn] = t0;
                        buf[dxn + 1] = t1;
                        buf[dxn + 2] = t2;
                    }
                else if (cn == 4)
                {
                    for (k = 0; k < xtab_size; k++)
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        WT t0 = buf[dxn] + S[sxn] * alpha;
                        WT t1 = buf[dxn + 1] + S[sxn + 1] * alpha;
                        buf[dxn] = t0;
                        buf[dxn + 1] = t1;
                        t0 = buf[dxn + 2] + S[sxn + 2] * alpha;
                        t1 = buf[dxn + 3] + S[sxn + 3] * alpha;
                        buf[dxn + 2] = t0;
                        buf[dxn + 3] = t1;
                    }
                }
                else
                {
                    for (k = 0; k < xtab_size; k++)
                    {
                        int sxn = xtab[k].si;
                        int dxn = xtab[k].di;
                        WT alpha = xtab[k].alpha;
                        for (int c = 0; c < cn; c++) buf[dxn + c] += S[sxn + c] * alpha;
                    }
                }
            }

            if (dy != prev_dy)
            {
                T* D = dst->template ptr<T>(prev_dy);

                for (dx = 0; dx < dsize.width; dx++)
                {
                    D[dx] = saturate_cast<T>(sum[dx]);
                    sum[dx] = beta * buf[dx];
                }
                prev_dy = dy;
            }
            else
            {
                for (dx = 0; dx < dsize.width; dx++) sum[dx] += beta * buf[dx];
            }
        }

        {
            T* D = dst->template ptr<T>(prev_dy);
            for (dx = 0; dx < dsize.width; dx++) D[dx] = saturate_cast<T>(sum[dx]);
        }
    }

   private:
    const Mat* src;
    Mat* dst;
    const DecimateAlpha* xtab0;
    const DecimateAlpha* ytab;
    int xtab_size0, ytab_size;
    const int* tabofs;
};

template <typename T, typename WT>
static void resizeArea_(const Mat& src, Mat& dst, const DecimateAlpha* xtab, int xtab_size, const DecimateAlpha* ytab,
                        int ytab_size, const int* tabofs)
{
    parallel_for_(Range(0, dst.rows), ResizeArea_Invoker<T, WT>(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs),
                  dst.total() / ((double)(1 << 16)));
}

typedef void (*ResizeAreaFastFunc)(const Mat& src, Mat& dst, const int* ofs, const int* xofs, int scale_x, int scale_y);

typedef void (*ResizeAreaFunc)(const Mat& src, Mat& dst, const DecimateAlpha* xtab, int xtab_size,
                               const DecimateAlpha* ytab, int ytab_size, const int* yofs);

static int computeResizeAreaTab(int ssize, int dsize, int cn, double scale, DecimateAlpha* tab)
{
    int k = 0;
    for (int dx = 0; dx < dsize; dx++)
    {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cellWidth = std::min(scale, ssize - fsx1);

        int sx1 = fast_ceil(fsx1), sx2 = fast_floor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3)
        {
            assert(k < ssize * 2);
            tab[k].di = dx * cn;
            tab[k].si = (sx1 - 1) * cn;
            tab[k++].alpha = (float)((sx1 - fsx1) / cellWidth);
        }

        for (int sx = sx1; sx < sx2; sx++)
        {
            assert(k < ssize * 2);
            tab[k].di = dx * cn;
            tab[k].si = sx * cn;
            tab[k++].alpha = float(1.0 / cellWidth);
        }

        if (fsx2 - sx2 > 1e-3)
        {
            assert(k < ssize * 2);
            tab[k].di = dx * cn;
            tab[k].si = sx2 * cn;
            tab[k++].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cellWidth) / cellWidth);
        }
    }
    return k;
}
