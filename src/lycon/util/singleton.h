#pragma once

#include <mutex>

namespace lycon
{
std::recursive_mutex &getInitializationMutex();

#define LYCON_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, RET_VALUE)                     \
    static TYPE *volatile instance = NULL;                                           \
    if (instance == NULL)                                                            \
    {                                                                                \
        std::lock_guard<std::recursive_mutex> lock(lycon::getInitializationMutex()); \
        if (instance == NULL) instance = INITIALIZER;                                \
    }                                                                                \
    return RET_VALUE;

#define LYCON_SINGLETON_LAZY_INIT(TYPE, INITIALIZER) LYCON_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, instance)
#define LYCON_SINGLETON_LAZY_INIT_REF(TYPE, INITIALIZER) LYCON_SINGLETON_LAZY_INIT_(TYPE, INITIALIZER, *instance)
}
