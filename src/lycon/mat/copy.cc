#include <algorithm>
#include <cstring>

#include "lycon/mat/convert.h"
#include "lycon/mat/mat.h"
#include "lycon/mat/shared.h"
#include "lycon/util/auto_buffer.h"
#include "lycon/util/error.h"
#include "lycon/util/hardware.h"
#include "lycon/util/util.h"

namespace lycon
{
void scalarToRawData(const Scalar& s, void* _buf, int type, int unroll_to)
{
    int i, depth = LYCON_MAT_DEPTH(type), cn = LYCON_MAT_CN(type);
    LYCON_ASSERT(cn <= 4);
    switch (depth)
    {
    case LYCON_8U:
    {
        uchar* buf = (uchar*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<uchar>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_8S:
    {
        schar* buf = (schar*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<schar>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_16U:
    {
        ushort* buf = (ushort*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<ushort>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_16S:
    {
        short* buf = (short*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<short>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_32S:
    {
        int* buf = (int*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<int>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_32F:
    {
        float* buf = (float*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<float>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
    }
    break;
    case LYCON_64F:
    {
        double* buf = (double*)_buf;
        for (i = 0; i < cn; i++)
            buf[i] = saturate_cast<double>(s.val[i]);
        for (; i < unroll_to; i++)
            buf[i] = buf[i - cn];
        break;
    }
    default: LYCON_ERROR("Unsupported format");
    }
}

enum
{
    BLOCK_SIZE = 1024
};

inline bool checkScalar(const Mat& sc, int atype, int sckind, int akind)
{
    if (sc.dims > 2 || !sc.isContinuous())
        return false;
    Size sz = sc.size();
    if (sz.width != 1 && sz.height != 1)
        return false;
    int cn = LYCON_MAT_CN(atype);
    if (akind == _InputArray::MATX && sckind != _InputArray::MATX)
        return false;
    return sz == Size(1, 1) || sz == Size(1, cn) || sz == Size(cn, 1) ||
           (sz == Size(1, 4) && sc.type() == LYCON_64F && cn <= 4);
}

inline bool checkScalar(InputArray sc, int atype, int sckind, int akind)
{
    if (sc.dims() > 2 || !sc.isContinuous())
        return false;
    Size sz = sc.size();
    if (sz.width != 1 && sz.height != 1)
        return false;
    int cn = LYCON_MAT_CN(atype);
    if (akind == _InputArray::MATX && sckind != _InputArray::MATX)
        return false;
    return sz == Size(1, 1) || sz == Size(1, cn) || sz == Size(cn, 1) ||
           (sz == Size(1, 4) && sc.type() == LYCON_64F && cn <= 4);
}

template <typename T>
static void copyMask_(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep,
                      Size size)
{
    for (; size.height--; mask += mstep, _src += sstep, _dst += dstep)
    {
        const T* src = (const T*)_src;
        T* dst = (T*)_dst;
        int x = 0;
        for (; x < size.width; x++)
            if (mask[x])
                dst[x] = src[x];
    }
}

template <>
void copyMask_<uchar>(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep,
                      Size size)
{
    for (; size.height--; mask += mstep, _src += sstep, _dst += dstep)
    {
        const uchar* src = (const uchar*)_src;
        uchar* dst = (uchar*)_dst;
        int x = 0;
#if LYCON_SSE4_2
        if (USE_SSE4_2) //
        {
            __m128i zero = _mm_setzero_si128();

            for (; x <= size.width - 16; x += 16)
            {
                const __m128i rSrc = _mm_lddqu_si128((const __m128i*)(src + x));
                __m128i _mask = _mm_lddqu_si128((const __m128i*)(mask + x));
                __m128i rDst = _mm_lddqu_si128((__m128i*)(dst + x));
                __m128i _negMask = _mm_cmpeq_epi8(_mask, zero);
                rDst = _mm_blendv_epi8(rSrc, rDst, _negMask);
                _mm_storeu_si128((__m128i*)(dst + x), rDst);
            }
        }
#elif LYCON_NEON
        uint8x16_t v_one = vdupq_n_u8(1);
        for (; x <= size.width - 16; x += 16)
        {
            uint8x16_t v_mask = vcgeq_u8(vld1q_u8(mask + x), v_one);
            uint8x16_t v_dst = vld1q_u8(dst + x), v_src = vld1q_u8(src + x);
            vst1q_u8(dst + x, vbslq_u8(v_mask, v_src, v_dst));
        }
#endif
        for (; x < size.width; x++)
            if (mask[x])
                dst[x] = src[x];
    }
}

template <>
void copyMask_<ushort>(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep,
                       Size size)
{
    for (; size.height--; mask += mstep, _src += sstep, _dst += dstep)
    {
        const ushort* src = (const ushort*)_src;
        ushort* dst = (ushort*)_dst;
        int x = 0;
#if LYCON_SSE4_2
        if (USE_SSE4_2) //
        {
            __m128i zero = _mm_setzero_si128();
            for (; x <= size.width - 8; x += 8)
            {
                const __m128i rSrc = _mm_lddqu_si128((const __m128i*)(src + x));
                __m128i _mask = _mm_loadl_epi64((const __m128i*)(mask + x));
                _mask = _mm_unpacklo_epi8(_mask, _mask);
                __m128i rDst = _mm_lddqu_si128((const __m128i*)(dst + x));
                __m128i _negMask = _mm_cmpeq_epi8(_mask, zero);
                rDst = _mm_blendv_epi8(rSrc, rDst, _negMask);
                _mm_storeu_si128((__m128i*)(dst + x), rDst);
            }
        }
#elif LYCON_NEON
        uint8x8_t v_one = vdup_n_u8(1);
        for (; x <= size.width - 8; x += 8)
        {
            uint8x8_t v_mask = vcge_u8(vld1_u8(mask + x), v_one);
            uint8x8x2_t v_mask2 = vzip_u8(v_mask, v_mask);
            uint16x8_t v_mask_res = vreinterpretq_u16_u8(vcombine_u8(v_mask2.val[0], v_mask2.val[1]));

            uint16x8_t v_src = vld1q_u16(src + x), v_dst = vld1q_u16(dst + x);
            vst1q_u16(dst + x, vbslq_u16(v_mask_res, v_src, v_dst));
        }
#endif
        for (; x < size.width; x++)
            if (mask[x])
                dst[x] = src[x];
    }
}

static void copyMaskGeneric(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep,
                            Size size, void* _esz)
{
    size_t k, esz = *(size_t*)_esz;
    for (; size.height--; mask += mstep, _src += sstep, _dst += dstep)
    {
        const uchar* src = _src;
        uchar* dst = _dst;
        int x = 0;
        for (; x < size.width; x++, src += esz, dst += esz)
        {
            if (!mask[x])
                continue;
            for (k = 0; k < esz; k++)
                dst[k] = src[k];
        }
    }
}

#define DEF_COPY_MASK(suffix, type)                                                                                    \
    static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, uchar* dst,          \
                                 size_t dstep, Size size, void*)                                                       \
    {                                                                                                                  \
        copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size);                                                    \
    }

#define DEF_COPY_MASK_F(suffix, type, ippfavor, ipptype)                                                               \
    static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, uchar* dst,          \
                                 size_t dstep, Size size, void*)                                                       \
    {                                                                                                                  \
        copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size);                                                    \
    }

DEF_COPY_MASK_F(8uC3, Vec3b, 8u_C3MR, Ipp8u)
DEF_COPY_MASK_F(32sC3, Vec3i, 32s_C3MR, Ipp32s)
DEF_COPY_MASK(8u, uchar)
DEF_COPY_MASK(16u, ushort)
DEF_COPY_MASK_F(32s, int, 32s_C1MR, Ipp32s)
DEF_COPY_MASK_F(16uC3, Vec3s, 16u_C3MR, Ipp16u)
DEF_COPY_MASK(32sC2, Vec2i)
DEF_COPY_MASK_F(32sC4, Vec4i, 32s_C4MR, Ipp32s)
DEF_COPY_MASK(32sC6, Vec6i)
DEF_COPY_MASK(32sC8, Vec8i)

BinaryFunc copyMaskTab[] = {0,
                            copyMask8u,
                            copyMask16u,
                            copyMask8uC3,
                            copyMask32s,
                            0,
                            copyMask16uC3,
                            0,
                            copyMask32sC2,
                            0,
                            0,
                            0,
                            copyMask32sC3,
                            0,
                            0,
                            0,
                            copyMask32sC4,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            copyMask32sC6,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            copyMask32sC8};

BinaryFunc getCopyMaskFunc(size_t esz)
{
    return esz <= 32 && copyMaskTab[esz] ? copyMaskTab[esz] : copyMaskGeneric;
}

/* dst = src */
void Mat::copyTo(OutputArray _dst) const
{
    int dtype = _dst.type();
    if (_dst.fixedType() && dtype != type())
    {
        LYCON_ASSERT(channels() == LYCON_MAT_CN(dtype));
        convertTo(_dst, dtype);
        return;
    }

    if (dims <= 2)
    {
        _dst.create(rows, cols, type());
        Mat dst = _dst.getMat();
        if (data == dst.data)
            return;

        if (rows > 0 && cols > 0)
        {
            // For some cases (with vector) dst.size != src.size, so force to column-based form
            // It prevents memory corruption in case of column-based src
            if (_dst.isVector())
                dst = dst.reshape(0, (int)dst.total());

            const uchar* sptr = data;
            uchar* dptr = dst.data;

            Size sz = getContinuousSize(*this, dst);
            size_t len = sz.width * elemSize();

            for (; sz.height--; sptr += step, dptr += dst.step)
                memcpy(dptr, sptr, len);
        }
        return;
    }

    _dst.create(dims, size, type());
    Mat dst = _dst.getMat();
    if (data == dst.data)
        return;

    if (total() != 0)
    {
        const Mat* arrays[] = {this, &dst};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs, 2);
        size_t sz = it.size * elemSize();

        for (size_t i = 0; i < it.nplanes; i++, ++it)
            memcpy(ptrs[1], ptrs[0], sz);
    }
}

void Mat::copyTo(OutputArray _dst, InputArray _mask) const
{
    Mat mask = _mask.getMat();
    if (!mask.data)
    {
        copyTo(_dst);
        return;
    }

    int cn = channels(), mcn = mask.channels();
    LYCON_ASSERT(mask.depth() == LYCON_8U && (mcn == 1 || mcn == cn));
    bool colorMask = mcn > 1;

    size_t esz = colorMask ? elemSize1() : elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);

    uchar* data0 = _dst.getMat().data;
    _dst.create(dims, size, type());
    Mat dst = _dst.getMat();

    if (dst.data != data0) // do not leave dst uninitialized
        dst = Scalar(0);

    if (dims <= 2)
    {
        LYCON_ASSERT(size() == mask.size());
        Size sz = getContinuousSize(*this, dst, mask, mcn);
        copymask(data, step, mask.data, mask.step, dst.data, dst.step, sz, &esz);
        return;
    }

    const Mat* arrays[] = {this, &dst, &mask, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    Size sz((int)(it.size * mcn), 1);

    for (size_t i = 0; i < it.nplanes; i++, ++it)
        copymask(ptrs[0], 0, ptrs[2], 0, ptrs[1], 0, sz, &esz);
}

Mat& Mat::operator=(const Scalar& s)
{
    const Mat* arrays[] = {this};
    uchar* dptr;
    NAryMatIterator it(arrays, &dptr, 1);
    size_t elsize = it.size * elemSize();
    const int64* is = (const int64*)&s.val[0];

    if (is[0] == 0 && is[1] == 0 && is[2] == 0 && is[3] == 0)
    {
        for (size_t i = 0; i < it.nplanes; i++, ++it)
            memset(dptr, 0, elsize);
    }
    else
    {
        if (it.nplanes > 0)
        {
            double scalar[12];
            scalarToRawData(s, scalar, type(), 12);
            size_t blockSize = 12 * elemSize1();

            for (size_t j = 0; j < elsize; j += blockSize)
            {
                size_t sz = std::min(blockSize, elsize - j);
                memcpy(dptr + j, scalar, sz);
            }
        }

        for (size_t i = 1; i < it.nplanes; i++)
        {
            ++it;
            memcpy(dptr, data, elsize);
        }
    }
    return *this;
}

Mat& Mat::setTo(InputArray _value, InputArray _mask)
{
    if (empty())
        return *this;

    Mat value = _value.getMat(), mask = _mask.getMat();

    LYCON_ASSERT(checkScalar(value, type(), _value.kind(), _InputArray::MAT));
    LYCON_ASSERT(mask.empty() || (mask.type() == LYCON_8U && size == mask.size));

    size_t esz = elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);

    const Mat* arrays[] = {this, !mask.empty() ? &mask : 0, 0};
    uchar* ptrs[2] = {0, 0};
    NAryMatIterator it(arrays, ptrs);
    int totalsz = (int)it.size, blockSize0 = std::min(totalsz, (int)((BLOCK_SIZE + esz - 1) / esz));
    AutoBuffer<uchar> _scbuf(blockSize0 * esz + 32);
    uchar* scbuf = alignPtr((uchar*)_scbuf, (int)sizeof(double));
    convertAndUnrollScalar(value, type(), scbuf, blockSize0);

    for (size_t i = 0; i < it.nplanes; i++, ++it)
    {
        for (int j = 0; j < totalsz; j += blockSize0)
        {
            Size sz(std::min(blockSize0, totalsz - j), 1);
            size_t blockSize = sz.width * esz;
            if (ptrs[1])
            {
                copymask(scbuf, 0, ptrs[1], 0, ptrs[0], 0, sz, &esz);
                ptrs[1] += sz.width;
            }
            else
                memcpy(ptrs[0], scbuf, blockSize);
            ptrs[0] += blockSize;
        }
    }
    return *this;
}
}
