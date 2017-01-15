#pragma once

#include <cstddef>

#include "lycon/util/error.h"

namespace lycon
{

/** @brief Aligns a pointer to the specified number of bytes.

The function returns the aligned pointer of the same type as the input pointer:
\f[\texttt{(_Tp*)(((size_t)ptr + n-1) & -n)}\f]
@param ptr Aligned pointer.
@param n Alignment size that must be a power of two.
 */
template <typename _Tp> static inline _Tp *alignPtr(_Tp *ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp *)(((size_t)ptr + n - 1) & -n);
}

/** @brief Aligns a buffer size to the specified number of bytes.

The function returns the minimum number that is greater or equal to sz and is divisible by n :
\f[\texttt{(sz + n-1) & -n}\f]
@param sz Buffer size to align.
@param n Alignment size that must be a power of two.
 */
static inline size_t alignSize(size_t sz, int n)
{
    LYCON_DbgAssert((n & (n - 1)) == 0); // n is a power of 2
    return (sz + n - 1) & -n;
}

static inline bool isBigEndian(void)
{
    return (((const int *)"\0\x1\x2\x3\x4\x5\x6\x7")[0] & 255) != 0;
}
}
