#pragma once

#include <cstddef>
#include <cstdint>

#include "lycon/arch.h"

namespace lycon
{
#define LYCON_VERSION_STRING "0.2.0"

// Type aliases
using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using int64 = int64_t;
using uint64 = uint64_t;

// LYCON_EXPORTS
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined LYCONAPI_EXPORTS
#define LYCON_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#define LYCON_EXPORTS __attribute__((visibility("default")))
#else
#define LYCON_EXPORTS
#endif

#define LYCON_CN_MAX 512
#define LYCON_CN_SHIFT 3
#define LYCON_DEPTH_MAX (1 << LYCON_CN_SHIFT)

#define LYCON_8U 0
#define LYCON_8S 1
#define LYCON_16U 2
#define LYCON_16S 3
#define LYCON_32S 4
#define LYCON_32F 5
#define LYCON_64F 6
#define LYCON_USRTYPE1 7

#define LYCON_MAT_DEPTH_MASK (LYCON_DEPTH_MAX - 1)
#define LYCON_MAT_DEPTH(flags) ((flags)&LYCON_MAT_DEPTH_MASK)

#define LYCON_MAKETYPE(depth, cn) (LYCON_MAT_DEPTH(depth) + (((cn)-1) << LYCON_CN_SHIFT))
#define LYCON_MAKE_TYPE LYCON_MAKETYPE

#define LYCON_8UC1 LYCON_MAKETYPE(LYCON_8U, 1)
#define LYCON_8UC2 LYCON_MAKETYPE(LYCON_8U, 2)
#define LYCON_8UC3 LYCON_MAKETYPE(LYCON_8U, 3)
#define LYCON_8UC4 LYCON_MAKETYPE(LYCON_8U, 4)
#define LYCON_8UC(n) LYCON_MAKETYPE(LYCON_8U, (n))

#define LYCON_8SC1 LYCON_MAKETYPE(LYCON_8S, 1)
#define LYCON_8SC2 LYCON_MAKETYPE(LYCON_8S, 2)
#define LYCON_8SC3 LYCON_MAKETYPE(LYCON_8S, 3)
#define LYCON_8SC4 LYCON_MAKETYPE(LYCON_8S, 4)
#define LYCON_8SC(n) LYCON_MAKETYPE(LYCON_8S, (n))

#define LYCON_16UC1 LYCON_MAKETYPE(LYCON_16U, 1)
#define LYCON_16UC2 LYCON_MAKETYPE(LYCON_16U, 2)
#define LYCON_16UC3 LYCON_MAKETYPE(LYCON_16U, 3)
#define LYCON_16UC4 LYCON_MAKETYPE(LYCON_16U, 4)
#define LYCON_16UC(n) LYCON_MAKETYPE(LYCON_16U, (n))

#define LYCON_16SC1 LYCON_MAKETYPE(LYCON_16S, 1)
#define LYCON_16SC2 LYCON_MAKETYPE(LYCON_16S, 2)
#define LYCON_16SC3 LYCON_MAKETYPE(LYCON_16S, 3)
#define LYCON_16SC4 LYCON_MAKETYPE(LYCON_16S, 4)
#define LYCON_16SC(n) LYCON_MAKETYPE(LYCON_16S, (n))

#define LYCON_32SC1 LYCON_MAKETYPE(LYCON_32S, 1)
#define LYCON_32SC2 LYCON_MAKETYPE(LYCON_32S, 2)
#define LYCON_32SC3 LYCON_MAKETYPE(LYCON_32S, 3)
#define LYCON_32SC4 LYCON_MAKETYPE(LYCON_32S, 4)
#define LYCON_32SC(n) LYCON_MAKETYPE(LYCON_32S, (n))

#define LYCON_32FC1 LYCON_MAKETYPE(LYCON_32F, 1)
#define LYCON_32FC2 LYCON_MAKETYPE(LYCON_32F, 2)
#define LYCON_32FC3 LYCON_MAKETYPE(LYCON_32F, 3)
#define LYCON_32FC4 LYCON_MAKETYPE(LYCON_32F, 4)
#define LYCON_32FC(n) LYCON_MAKETYPE(LYCON_32F, (n))

#define LYCON_64FC1 LYCON_MAKETYPE(LYCON_64F, 1)
#define LYCON_64FC2 LYCON_MAKETYPE(LYCON_64F, 2)
#define LYCON_64FC3 LYCON_MAKETYPE(LYCON_64F, 3)
#define LYCON_64FC4 LYCON_MAKETYPE(LYCON_64F, 4)
#define LYCON_64FC(n) LYCON_MAKETYPE(LYCON_64F, (n))

#define LYCON_MAT_CN_MASK ((LYCON_CN_MAX - 1) << LYCON_CN_SHIFT)
#define LYCON_MAT_CN(flags) ((((flags)&LYCON_MAT_CN_MASK) >> LYCON_CN_SHIFT) + 1)
#define LYCON_MAT_TYPE_MASK (LYCON_DEPTH_MAX * LYCON_CN_MAX - 1)
#define LYCON_MAT_TYPE(flags) ((flags)&LYCON_MAT_TYPE_MASK)
#define LYCON_MAT_CONT_FLAG_SHIFT 14
#define LYCON_MAT_CONT_FLAG (1 << LYCON_MAT_CONT_FLAG_SHIFT)
#define LYCON_IS_MAT_CONT(flags) ((flags)&LYCON_MAT_CONT_FLAG)
#define LYCON_IS_CONT_MAT LYCON_IS_MAT_CONT
#define LYCON_SUBMAT_FLAG_SHIFT 15
#define LYCON_SUBMAT_FLAG (1 << LYCON_SUBMAT_FLAG_SHIFT)

/** Size of each channel item,
   0x124489 = 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define LYCON_ELEM_SIZE1(type) ((((sizeof(size_t) << 28) | 0x8442211) >> LYCON_MAT_DEPTH(type) * 4) & 15)

/** 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define LYCON_ELEM_SIZE(type)                                                                                          \
    (LYCON_MAT_CN(type) << ((((sizeof(size_t) / 4 + 1) * 16384 | 0x3a50) >> LYCON_MAT_DEPTH(type) * 2) & 3))

#define LYCON_MAX_DIM 32
#define LYCON_MAX_DIM_HEAP 1024

#define HAVE_PTHREADS

/* parallel_for with pthreads */
#define HAVE_PTHREADS_PF

// Mat forward declarations
class LYCON_EXPORTS Mat;
typedef Mat MatND;
template <typename _Tp> class Mat_;
}
