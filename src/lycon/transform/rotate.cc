#include "lycon/transform/rotate.h"

#include "lycon/types.h"
#include "lycon/util/auto_buffer.h"
#include "lycon/util/error.h"

namespace lycon
{

static void flipHoriz(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz)
{
    int i, j, limit = (int)(((size.width + 1) / 2) * esz);
    AutoBuffer<int> _tab(size.width * esz);
    int* tab = _tab;

    for (i = 0; i < size.width; i++)
        for (size_t k = 0; k < esz; k++)
            tab[i * esz + k] = (int)((size.width - i - 1) * esz + k);

    for (; size.height--; src += sstep, dst += dstep)
    {
        for (i = 0; i < limit; i++)
        {
            j = tab[i];
            uchar t0 = src[i], t1 = src[j];
            dst[i] = t1;
            dst[j] = t0;
        }
    }
}

static void flipVert(const uchar* src0, size_t sstep, uchar* dst0, size_t dstep, Size size, size_t esz)
{
    const uchar* src1 = src0 + (size.height - 1) * sstep;
    uchar* dst1 = dst0 + (size.height - 1) * dstep;
    size.width *= (int)esz;

    for (int y = 0; y < (size.height + 1) / 2; y++, src0 += sstep, src1 -= sstep, dst0 += dstep, dst1 -= dstep)
    {
        int i = 0;
        if (((size_t)src0 | (size_t)dst0 | (size_t)src1 | (size_t)dst1) % sizeof(int) == 0)
        {
            for (; i <= size.width - 16; i += 16)
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;

                t0 = ((int*)(src0 + i))[1];
                t1 = ((int*)(src1 + i))[1];

                ((int*)(dst0 + i))[1] = t1;
                ((int*)(dst1 + i))[1] = t0;

                t0 = ((int*)(src0 + i))[2];
                t1 = ((int*)(src1 + i))[2];

                ((int*)(dst0 + i))[2] = t1;
                ((int*)(dst1 + i))[2] = t0;

                t0 = ((int*)(src0 + i))[3];
                t1 = ((int*)(src1 + i))[3];

                ((int*)(dst0 + i))[3] = t1;
                ((int*)(dst1 + i))[3] = t0;
            }

            for (; i <= size.width - 4; i += 4)
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;
            }
        }

        for (; i < size.width; i++)
        {
            uchar t0 = src0[i];
            uchar t1 = src1[i];

            dst0[i] = t1;
            dst1[i] = t0;
        }
    }
}

void flip(InputArray _src, OutputArray _dst, int flip_mode)
{
    LYCON_ASSERT(_src.dims() <= 2);
    Size size = _src.size();

    if (flip_mode < 0)
    {
        if (size.width == 1)
            flip_mode = 0;
        if (size.height == 1)
            flip_mode = 1;
    }

    if ((size.width == 1 && flip_mode > 0) || (size.height == 1 && flip_mode == 0) ||
        (size.height == 1 && size.width == 1 && flip_mode < 0))
    {
        return _src.copyTo(_dst);
    }

    Mat src = _src.getMat();
    int type = src.type();
    _dst.create(size, type);
    Mat dst = _dst.getMat();

    size_t esz = LYCON_ELEM_SIZE(type);

    if (flip_mode <= 0)
        flipVert(src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz);
    else
        flipHoriz(src.ptr(), src.step, dst.ptr(), dst.step, src.size(), esz);

    if (flip_mode < 0)
        flipHoriz(dst.ptr(), dst.step, dst.ptr(), dst.step, dst.size(), esz);
}

template <typename T> static void transpose_(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)
{
    int i = 0, j, m = sz.width, n = sz.height;

    for (; i <= m - 4; i += 4)
    {
        T* d0 = (T*)(dst + dstep * i);
        T* d1 = (T*)(dst + dstep * (i + 1));
        T* d2 = (T*)(dst + dstep * (i + 2));
        T* d3 = (T*)(dst + dstep * (i + 3));

        for (j = 0; j <= n - 4; j += 4)
        {
            const T* s0 = (const T*)(src + i * sizeof(T) + sstep * j);
            const T* s1 = (const T*)(src + i * sizeof(T) + sstep * (j + 1));
            const T* s2 = (const T*)(src + i * sizeof(T) + sstep * (j + 2));
            const T* s3 = (const T*)(src + i * sizeof(T) + sstep * (j + 3));

            d0[j] = s0[0];
            d0[j + 1] = s1[0];
            d0[j + 2] = s2[0];
            d0[j + 3] = s3[0];
            d1[j] = s0[1];
            d1[j + 1] = s1[1];
            d1[j + 2] = s2[1];
            d1[j + 3] = s3[1];
            d2[j] = s0[2];
            d2[j + 1] = s1[2];
            d2[j + 2] = s2[2];
            d2[j + 3] = s3[2];
            d3[j] = s0[3];
            d3[j + 1] = s1[3];
            d3[j + 2] = s2[3];
            d3[j + 3] = s3[3];
        }

        for (; j < n; j++)
        {
            const T* s0 = (const T*)(src + i * sizeof(T) + j * sstep);
            d0[j] = s0[0];
            d1[j] = s0[1];
            d2[j] = s0[2];
            d3[j] = s0[3];
        }
    }

    for (; i < m; i++)
    {
        T* d0 = (T*)(dst + dstep * i);
        j = 0;
        for (; j <= n - 4; j += 4)
        {
            const T* s0 = (const T*)(src + i * sizeof(T) + sstep * j);
            const T* s1 = (const T*)(src + i * sizeof(T) + sstep * (j + 1));
            const T* s2 = (const T*)(src + i * sizeof(T) + sstep * (j + 2));
            const T* s3 = (const T*)(src + i * sizeof(T) + sstep * (j + 3));

            d0[j] = s0[0];
            d0[j + 1] = s1[0];
            d0[j + 2] = s2[0];
            d0[j + 3] = s3[0];
        }
        for (; j < n; j++)
        {
            const T* s0 = (const T*)(src + i * sizeof(T) + j * sstep);
            d0[j] = s0[0];
        }
    }
}

template <typename T> static void transposeI_(uchar* data, size_t step, int n)
{
    for (int i = 0; i < n; i++)
    {
        T* row = (T*)(data + step * i);
        uchar* data1 = data + i * sizeof(T);
        for (int j = i + 1; j < n; j++)
            std::swap(row[j], *(T*)(data1 + step * j));
    }
}

typedef void (*TransposeFunc)(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz);
typedef void (*TransposeInplaceFunc)(uchar* data, size_t step, int n);

#define DEF_TRANSPOSE_FUNC(suffix, type)                                                                               \
    static void transpose_##suffix(const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size sz)                  \
    {                                                                                                                  \
        transpose_<type>(src, sstep, dst, dstep, sz);                                                                  \
    }                                                                                                                  \
                                                                                                                       \
    static void transposeI_##suffix(uchar* data, size_t step, int n)                                                   \
    {                                                                                                                  \
        transposeI_<type>(data, step, n);                                                                              \
    }

DEF_TRANSPOSE_FUNC(8u, uchar)
DEF_TRANSPOSE_FUNC(16u, ushort)
DEF_TRANSPOSE_FUNC(8uC3, Vec3b)
DEF_TRANSPOSE_FUNC(32s, int)
DEF_TRANSPOSE_FUNC(16uC3, Vec3s)
DEF_TRANSPOSE_FUNC(32sC2, Vec2i)
DEF_TRANSPOSE_FUNC(32sC3, Vec3i)
DEF_TRANSPOSE_FUNC(32sC4, Vec4i)
DEF_TRANSPOSE_FUNC(32sC6, Vec6i)
DEF_TRANSPOSE_FUNC(32sC8, Vec8i)

static TransposeFunc transposeTab[] = {0,
                                       transpose_8u,
                                       transpose_16u,
                                       transpose_8uC3,
                                       transpose_32s,
                                       0,
                                       transpose_16uC3,
                                       0,
                                       transpose_32sC2,
                                       0,
                                       0,
                                       0,
                                       transpose_32sC3,
                                       0,
                                       0,
                                       0,
                                       transpose_32sC4,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       transpose_32sC6,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,
                                       transpose_32sC8};

static TransposeInplaceFunc transposeInplaceTab[] = {0,
                                                     transposeI_8u,
                                                     transposeI_16u,
                                                     transposeI_8uC3,
                                                     transposeI_32s,
                                                     0,
                                                     transposeI_16uC3,
                                                     0,
                                                     transposeI_32sC2,
                                                     0,
                                                     0,
                                                     0,
                                                     transposeI_32sC3,
                                                     0,
                                                     0,
                                                     0,
                                                     transposeI_32sC4,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     transposeI_32sC6,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     0,
                                                     transposeI_32sC8};

void transpose(InputArray _src, OutputArray _dst)
{

    int type = _src.type(), esz = LYCON_ELEM_SIZE(type);
    LYCON_ASSERT(_src.dims() <= 2 && esz <= 32);

    Mat src = _src.getMat();
    if (src.empty())
    {
        _dst.release();
        return;
    }

    _dst.create(src.cols, src.rows, src.type());
    Mat dst = _dst.getMat();

    // handle the case of single-column/single-row matrices, stored in STL vectors.
    if (src.rows != dst.cols || src.cols != dst.rows)
    {
        LYCON_ASSERT(src.size() == dst.size() && (src.cols == 1 || src.rows == 1));
        src.copyTo(dst);
        return;
    }

    if (dst.data == src.data)
    {
        TransposeInplaceFunc func = transposeInplaceTab[esz];
        LYCON_ASSERT(func != 0);
        LYCON_ASSERT(dst.cols == dst.rows);
        func(dst.ptr(), dst.step, dst.rows);
    }
    else
    {
        TransposeFunc func = transposeTab[esz];
        LYCON_ASSERT(func != 0);
        func(src.ptr(), src.step, dst.ptr(), dst.step, src.size());
    }
}

void rotate(InputArray _src, OutputArray _dst, int rotateMode)
{
    LYCON_ASSERT(_src.dims() <= 2);

    switch (rotateMode)
    {
    case ROTATE_90_CLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 1);
        break;
    case ROTATE_180: flip(_src, _dst, -1); break;
    case ROTATE_90_COUNTERCLOCKWISE:
        transpose(_src, _dst);
        flip(_dst, _dst, 0);
        break;
    default: break;
    }
}
}
