#include "lycon/util/singleton.h"

namespace lycon
{
std::recursive_mutex g_init_mutex;
std::recursive_mutex& getInitializationMutex()
{
    return g_init_mutex;
}
}
