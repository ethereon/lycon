#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "lycon/defs.h"

namespace lycon
{
class RuntimeError : public std::runtime_error
{
   public:
    RuntimeError(const char *what_arg) : std::runtime_error(what_arg) {}
};

#if defined __GNUC__
#define LYCON_Func __func__
#elif defined _MSC_VER
#define LYCON_Func __FUNCTION__
#else
#define LYCON_Func ""
#endif

#define LYCON_ERROR(...)                      \
    {                                         \
        char err_msg[2048];                   \
        snprintf(err_msg, 2048, __VA_ARGS__); \
        throw RuntimeError(err_msg);          \
    }

#define LYCON_StaticAssert static_assert
#define LYCON_DbgAssert assert
#define LYCON_ASSERT(expr) \
    if (!!(expr))          \
        ;                  \
    else                   \
        LYCON_ERROR("Assertion Failure: `%s` evaluated to false in `%s` (%s:%d)", #expr, LYCON_Func, __FILE__, __LINE__)
}
