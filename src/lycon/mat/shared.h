#pragma once

#include "lycon/defs.h"
#include "lycon/mat/mat.h"
#include "lycon/types.h"

namespace lycon
{

typedef void (*BinaryFunc)(const uchar *src1, size_t step1, const uchar *src2, size_t step2, uchar *dst, size_t step,
                           Size sz, void *);

inline Size getContinuousSize_(int flags, int cols, int rows, int widthScale)
{
    int64 sz = (int64)cols * rows * widthScale;
    return (flags & Mat::CONTINUOUS_FLAG) != 0 && (int)sz == sz ? Size((int)sz, 1) : Size(cols * widthScale, rows);
}

inline Size getContinuousSize(const Mat &m1, int widthScale = 1)
{
    return getContinuousSize_(m1.flags, m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize(const Mat &m1, const Mat &m2, int widthScale = 1)
{
    return getContinuousSize_(m1.flags & m2.flags, m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize(const Mat &m1, const Mat &m2, const Mat &m3, int widthScale = 1)
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags, m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize(const Mat &m1, const Mat &m2, const Mat &m3, const Mat &m4, int widthScale = 1)
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags & m4.flags, m1.cols, m1.rows, widthScale);
}

inline Size getContinuousSize(const Mat &m1, const Mat &m2, const Mat &m3, const Mat &m4, const Mat &m5,
                              int widthScale = 1)
{
    return getContinuousSize_(m1.flags & m2.flags & m3.flags & m4.flags & m5.flags, m1.cols, m1.rows, widthScale);
}
}
