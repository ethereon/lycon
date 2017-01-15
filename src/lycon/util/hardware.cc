#include "lycon/util/hardware.h"

#include <cstring>

#include "lycon/util/error.h"

namespace lycon
{

struct HWFeatures
{
    enum
    {
        MAX_FEATURE = LYCON_HARDWARE_MAX_FEATURE
    };

    HWFeatures(void)
    {
        memset(have, 0, sizeof(have));
        x86_family = 0;
    }

    static HWFeatures initialize(void)
    {
        HWFeatures f;
        int cpuid_data[4] = {0, 0, 0, 0};

#if defined _MSC_VER && (defined _M_IX86 || defined _M_X64)
        __cpuid(cpuid_data, 1);
#elif defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#ifdef __x86_64__
        asm __volatile__("movl $1, %%eax\n\t"
                         "cpuid\n\t"
                         : [eax] "=a"(cpuid_data[0]), [ebx] "=b"(cpuid_data[1]), [ecx] "=c"(cpuid_data[2]),
                           [edx] "=d"(cpuid_data[3])
                         :
                         : "cc");
#else
        asm volatile("pushl %%ebx\n\t"
                     "movl $1,%%eax\n\t"
                     "cpuid\n\t"
                     "popl %%ebx\n\t"
                     : "=a"(cpuid_data[0]), "=c"(cpuid_data[2]), "=d"(cpuid_data[3])
                     :
                     : "cc");
#endif
#endif

        f.x86_family = (cpuid_data[0] >> 8) & 15;
        if (f.x86_family >= 6)
        {
            f.have[LYCON_CPU_MMX] = (cpuid_data[3] & (1 << 23)) != 0;
            f.have[LYCON_CPU_SSE] = (cpuid_data[3] & (1 << 25)) != 0;
            f.have[LYCON_CPU_SSE2] = (cpuid_data[3] & (1 << 26)) != 0;
            f.have[LYCON_CPU_SSE3] = (cpuid_data[2] & (1 << 0)) != 0;
            f.have[LYCON_CPU_SSSE3] = (cpuid_data[2] & (1 << 9)) != 0;
            f.have[LYCON_CPU_FMA3] = (cpuid_data[2] & (1 << 12)) != 0;
            f.have[LYCON_CPU_SSE4_1] = (cpuid_data[2] & (1 << 19)) != 0;
            f.have[LYCON_CPU_SSE4_2] = (cpuid_data[2] & (1 << 20)) != 0;
            f.have[LYCON_CPU_POPCNT] = (cpuid_data[2] & (1 << 23)) != 0;
            f.have[LYCON_CPU_AVX] = (((cpuid_data[2] & (1 << 28)) != 0) &&
                                     ((cpuid_data[2] & (1 << 27)) != 0)); // OS uses XSAVE_XRSTORE and CPU support AVX
            f.have[LYCON_CPU_FP16] = (cpuid_data[2] & (1 << 29)) != 0;

// make the second call to the cpuid command in order to get
// information about extended features like AVX2
#if defined _MSC_VER && (defined _M_IX86 || defined _M_X64)
            __cpuidex(cpuid_data, 7, 0);
#elif defined __GNUC__ && (defined __i386__ || defined __x86_64__)
#ifdef __x86_64__
            asm __volatile__("movl $7, %%eax\n\t"
                             "movl $0, %%ecx\n\t"
                             "cpuid\n\t"
                             : [eax] "=a"(cpuid_data[0]), [ebx] "=b"(cpuid_data[1]), [ecx] "=c"(cpuid_data[2]),
                               [edx] "=d"(cpuid_data[3])
                             :
                             : "cc");
#else
            asm volatile("pushl %%ebx\n\t"
                         "movl $7,%%eax\n\t"
                         "movl $0,%%ecx\n\t"
                         "cpuid\n\t"
                         "movl %%ebx, %0\n\t"
                         "popl %%ebx\n\t"
                         : "=r"(cpuid_data[1]), "=c"(cpuid_data[2])
                         :
                         : "cc");
#endif
#endif
            f.have[LYCON_CPU_AVX2] = (cpuid_data[1] & (1 << 5)) != 0;

            f.have[LYCON_CPU_AVX_512F] = (cpuid_data[1] & (1 << 16)) != 0;
            f.have[LYCON_CPU_AVX_512DQ] = (cpuid_data[1] & (1 << 17)) != 0;
            f.have[LYCON_CPU_AVX_512IFMA512] = (cpuid_data[1] & (1 << 21)) != 0;
            f.have[LYCON_CPU_AVX_512PF] = (cpuid_data[1] & (1 << 26)) != 0;
            f.have[LYCON_CPU_AVX_512ER] = (cpuid_data[1] & (1 << 27)) != 0;
            f.have[LYCON_CPU_AVX_512CD] = (cpuid_data[1] & (1 << 28)) != 0;
            f.have[LYCON_CPU_AVX_512BW] = (cpuid_data[1] & (1 << 30)) != 0;
            f.have[LYCON_CPU_AVX_512VL] = (cpuid_data[1] & (1 << 31)) != 0;
            f.have[LYCON_CPU_AVX_512VBMI] = (cpuid_data[2] & (1 << 1)) != 0;
        }

#if defined ANDROID || defined __linux__
#ifdef __aarch64__
        f.have[LYCON_CPU_NEON] = true;
        f.have[LYCON_CPU_FP16] = true;
#elif defined __arm__
        int cpufile = open("/proc/self/auxv", O_RDONLY);

        if (cpufile >= 0)
        {
            Elf32_auxv_t auxv;
            const size_t size_auxv_t = sizeof(auxv);

            while ((size_t)read(cpufile, &auxv, size_auxv_t) == size_auxv_t)
            {
                if (auxv.a_type == AT_HWCAP)
                {
                    f.have[LYCON_CPU_NEON] = (auxv.a_un.a_val & 4096) != 0;
                    f.have[LYCON_CPU_FP16] = (auxv.a_un.a_val & 2) != 0;
                    break;
                }
            }

            close(cpufile);
        }
#endif
#elif (defined __clang__ || defined __APPLE__)
#if (defined __ARM_NEON__ || (defined __ARM_NEON && defined __aarch64__))
        f.have[LYCON_CPU_NEON] = true;
#endif
#if (defined __ARM_FP && (((__ARM_FP & 0x2) != 0) && defined __ARM_NEON__))
        f.have[LYCON_CPU_FP16] = true;
#endif
#endif

        return f;
    }

    int x86_family;
    bool have[MAX_FEATURE + 1];
};

static HWFeatures featuresEnabled = HWFeatures::initialize(), featuresDisabled = HWFeatures();
static HWFeatures* currentFeatures = &featuresEnabled;

bool checkHardwareSupport(int feature)
{
    LYCON_DbgAssert(0 <= feature && feature <= LYCON_HARDWARE_MAX_FEATURE);
    return currentFeatures->have[feature];
}
}
