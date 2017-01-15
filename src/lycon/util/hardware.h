#pragma once

#include "lycon/defs.h"

namespace lycon
{
/* CPU features and intrinsics support */
#define LYCON_CPU_NONE 0
#define LYCON_CPU_MMX 1
#define LYCON_CPU_SSE 2
#define LYCON_CPU_SSE2 3
#define LYCON_CPU_SSE3 4
#define LYCON_CPU_SSSE3 5
#define LYCON_CPU_SSE4_1 6
#define LYCON_CPU_SSE4_2 7
#define LYCON_CPU_POPCNT 8
#define LYCON_CPU_FP16 9
#define LYCON_CPU_AVX 10
#define LYCON_CPU_AVX2 11
#define LYCON_CPU_FMA3 12

#define LYCON_CPU_AVX_512F 13
#define LYCON_CPU_AVX_512BW 14
#define LYCON_CPU_AVX_512CD 15
#define LYCON_CPU_AVX_512DQ 16
#define LYCON_CPU_AVX_512ER 17
#define LYCON_CPU_AVX_512IFMA512 18
#define LYCON_CPU_AVX_512PF 19
#define LYCON_CPU_AVX_512VBMI 20
#define LYCON_CPU_AVX_512VL 21

#define LYCON_CPU_NEON 100

// when adding to this list remember to update the following enum
#define LYCON_HARDWARE_MAX_FEATURE 255

/** @brief Returns true if the specified feature is supported by the host hardware.

The function returns true if the host hardware supports the specified feature. When user calls
setUseOptimized(false), the subsequent calls to checkHardwareSupport() will return false until
setUseOptimized(true) is called. This way user can dynamically switch on and off the optimized code
in OpenCV.
@param feature The feature of interest, one of cv::CpuFeatures
*/
LYCON_EXPORTS bool checkHardwareSupport(int feature);

#define USE_SSE2 (checkHardwareSupport(LYCON_CPU_SSE))
#define USE_SSE4_2 (checkHardwareSupport(LYCON_CPU_SSE4_2))
#define USE_AVX (checkHardwareSupport(LYCON_CPU_AVX))
#define USE_AVX2 (checkHardwareSupport(LYCON_CPU_AVX2))
}
