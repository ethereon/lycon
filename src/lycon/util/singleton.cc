#include "lycon/util/singleton.h"

namespace lycon
{
std::recursive_mutex& getInitializationMutex()
{
    static std::recursive_mutex g_init_mutex;
    return g_init_mutex;
}
}
