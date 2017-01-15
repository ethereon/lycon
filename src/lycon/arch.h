#pragma once

#if defined __SSE2__ || defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#include <emmintrin.h>
#define LYCON_MMX 1
#define LYCON_SSE 1
#define LYCON_SSE2 1
#if defined __SSE3__ || (defined _MSC_VER && _MSC_VER >= 1500)
#include <pmmintrin.h>
#define LYCON_SSE3 1
#endif
#if defined __SSSE3__ || (defined _MSC_VER && _MSC_VER >= 1500)
#include <tmmintrin.h>
#define LYCON_SSSE3 1
#endif
#if defined __SSE4_1__ || (defined _MSC_VER && _MSC_VER >= 1500)
#include <smmintrin.h>
#define LYCON_SSE4_1 1
#endif
#if defined __SSE4_2__ || (defined _MSC_VER && _MSC_VER >= 1500)
#include <nmmintrin.h>
#define LYCON_SSE4_2 1
#endif
#if defined __POPCNT__ || (defined _MSC_VER && _MSC_VER >= 1500)
#ifdef _MSC_VER
#include <nmmintrin.h>
#else
#include <popcntintrin.h>
#endif
#define LYCON_POPCNT 1
#endif
#if defined __AVX__ || (defined _MSC_VER && _MSC_VER >= 1600 && 0)
// MS Visual Studio 2010 (2012?) has no macro pre-defined to identify the use of /arch:AVX
// See:
// http://connect.microsoft.com/VisualStudio/feedback/details/605858/arch-avx-should-define-a-predefined-macro-in-x64-and-set-a-unique-value-for-m-ix86-fp-in-win32
#include <immintrin.h>
#define LYCON_AVX 1
#if defined(_XCR_XFEATURE_ENABLED_MASK)
#define __xgetbv() _xgetbv(_XCR_XFEATURE_ENABLED_MASK)
#else
#define __xgetbv() 0
#endif
#endif
#if defined __AVX2__ || (defined _MSC_VER && _MSC_VER >= 1800 && 0)
#include <immintrin.h>
#define LYCON_AVX2 1
#if defined __FMA__
#define LYCON_FMA3 1
#endif
#endif
#endif

#if (defined WIN32 || defined _WIN32) && defined(_M_ARM)
#include <Intrin.h>
#include <arm_neon.h>
#define LYCON_NEON 1
#define CPU_HAS_NEON_FEATURE (true)
#elif defined(__ARM_NEON__) || (defined(__ARM_NEON) && defined(__aarch64__))
#include <arm_neon.h>
#define LYCON_NEON 1
#endif

#if defined __GNUC__ && defined __arm__ && (defined __ARM_PCS_VFP || defined __ARM_VFPV3__ || defined __ARM_NEON__) && \
    !defined __SOFTFP__
#define LYCON_VFP 1
#endif

#ifndef LYCON_POPCNT
#define LYCON_POPCNT 0
#endif
#ifndef LYCON_MMX
#define LYCON_MMX 0
#endif
#ifndef LYCON_SSE
#define LYCON_SSE 0
#endif
#ifndef LYCON_SSE2
#define LYCON_SSE2 0
#endif
#ifndef LYCON_SSE3
#define LYCON_SSE3 0
#endif
#ifndef LYCON_SSSE3
#define LYCON_SSSE3 0
#endif
#ifndef LYCON_SSE4_1
#define LYCON_SSE4_1 0
#endif
#ifndef LYCON_SSE4_2
#define LYCON_SSE4_2 0
#endif
#ifndef LYCON_AVX
#define LYCON_AVX 0
#endif
#ifndef LYCON_AVX2
#define LYCON_AVX2 0
#endif
#ifndef LYCON_FMA3
#define LYCON_FMA3 0
#endif
#ifndef LYCON_AVX_512F
#define LYCON_AVX_512F 0
#endif
#ifndef LYCON_AVX_512BW
#define LYCON_AVX_512BW 0
#endif
#ifndef LYCON_AVX_512CD
#define LYCON_AVX_512CD 0
#endif
#ifndef LYCON_AVX_512DQ
#define LYCON_AVX_512DQ 0
#endif
#ifndef LYCON_AVX_512ER
#define LYCON_AVX_512ER 0
#endif
#ifndef LYCON_AVX_512IFMA512
#define LYCON_AVX_512IFMA512 0
#endif
#ifndef LYCON_AVX_512PF
#define LYCON_AVX_512PF 0
#endif
#ifndef LYCON_AVX_512VBMI
#define LYCON_AVX_512VBMI 0
#endif
#ifndef LYCON_AVX_512VL
#define LYCON_AVX_512VL 0
#endif

#ifndef LYCON_NEON
#define LYCON_NEON 0
#endif

#ifndef LYCON_VFP
#define LYCON_VFP 0
#endif
