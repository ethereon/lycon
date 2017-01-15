#pragma once

namespace lycon
{
#if defined __ARM_FP16_FORMAT_IEEE && !defined __CUDACC__
#define LYCON_FP16_TYPE 1
#else
#define LYCON_FP16_TYPE 0
#endif

typedef union Lycon16suf {
    short i;
#if LYCON_FP16_TYPE
    __fp16 h;
#endif
    struct _fp16Format
    {
        unsigned int significand : 10;
        unsigned int exponent : 5;
        unsigned int sign : 1;
    } fmt;
} Lycon16suf;

typedef union Lycon32suf {
    int i;
    unsigned u;
    float f;
    struct _fp32Format
    {
        unsigned int significand : 23;
        unsigned int exponent : 8;
        unsigned int sign : 1;
    } fmt;
} Lycon32suf;

typedef union Lycon64suf {
    int64 i;
    uint64 u;
    double f;
} Lycon64suf;
}
