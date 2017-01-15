#include "lycon/mat/umat_data.h"

#include <mutex>

#include "lycon/mat/mat.h"

namespace lycon
{
enum
{
    UMAT_NLOCKS = 31
};

static std::mutex umatLocks[UMAT_NLOCKS];

UMatData::UMatData(const MatAllocator* allocator)
{
    prevAllocator = currAllocator = allocator;
    urefcount = refcount = mapcount = 0;
    data = origdata = 0;
    size = 0;
    flags = 0;
    handle = 0;
    userdata = 0;
    allocatorFlags_ = 0;
    originalUMatData = NULL;
}

UMatData::~UMatData()
{
    prevAllocator = currAllocator = 0;
    urefcount = refcount = 0;
    LYCON_ASSERT(mapcount == 0);
    data = origdata = 0;
    size = 0;
    flags = 0;
    handle = 0;
    userdata = 0;
    allocatorFlags_ = 0;
    if (originalUMatData)
    {
        UMatData* u = originalUMatData;
        LYCON_XADD(&(u->urefcount), -1);
        LYCON_XADD(&(u->refcount), -1);
        bool showWarn = false;
        if (u->refcount == 0)
        {
            if (u->urefcount > 0)
                showWarn = true;
            // simulate Mat::deallocate
            if (u->mapcount != 0)
            {
                (u->currAllocator ? u->currAllocator : /* TODO allocator ? allocator :*/ Mat::getDefaultAllocator())
                    ->unmap(u);
            }
            else
            {
                // we don't do "map", so we can't do "unmap"
            }
        }
        if (u->refcount == 0 && u->urefcount == 0) // oops, we need to free resources
        {
            showWarn = true;
            // simulate UMat::deallocate
            u->currAllocator->deallocate(u);
        }
#ifndef NDEBUG
        if (showWarn)
        {
            static int warn_message_showed = 0;
            if (warn_message_showed++ < 100)
            {
                fflush(stdout);
                fprintf(stderr, "\n! getUMat()/getMat() call chain possible problem."
                                "\n! Base object is dead, while nested/derived object is still alive or processed."
                                "\n! Please check lifetime of UMat/Mat objects!\n");
                fflush(stderr);
            }
        }
#else
        (void)showWarn;
#endif
        originalUMatData = NULL;
    }
}

void UMatData::lock()
{
    umatLocks[(size_t)(void*)this % UMAT_NLOCKS].lock();
}

void UMatData::unlock()
{
    umatLocks[(size_t)(void*)this % UMAT_NLOCKS].unlock();
}
}
