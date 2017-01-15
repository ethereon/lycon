#pragma once

// --- LYCON_XADD ---
#if defined __GNUC__
#if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#ifdef __ATOMIC_ACQ_REL
#define LYCON_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int) *)(addr), delta, __ATOMIC_ACQ_REL)
#else
#define LYCON_XADD(addr, delta) __atomic_fetch_add((_Atomic(int) *)(addr), delta, 4)
#endif
#else
#if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#define LYCON_XADD(addr, delta) (int)__atomic_fetch_add((unsigned *)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#else
#define LYCON_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned *)(addr), (unsigned)(delta))
#endif
#endif
#elif defined _MSC_VER && !defined RC_INVOKED
#include <intrin.h>
#define LYCON_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile *)addr, delta)
#else
LYCON_INLINE LYCON_XADD(int *addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}
#endif
