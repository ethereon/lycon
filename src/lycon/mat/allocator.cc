#include "lycon/mat/allocator.h"

#include <cstring>

#include "lycon/mat/iterator.h"
#include "lycon/mat/mat.h"
#include "lycon/mat/umat_data.h"
#include "lycon/util/singleton.h"

#ifdef LYCON_USE_NUMPY_ALLOCATOR_BY_DEFAULT
#include "lycon/python/interop.h"
#endif

#define LYCON_AUTOSTEP 0x7fffffff

namespace lycon
{
void MatAllocator::map(UMatData*, int) const
{
}

void MatAllocator::unmap(UMatData* u) const
{
    if (u->urefcount == 0 && u->refcount == 0)
    {
        deallocate(u);
        u = NULL;
    }
}

void MatAllocator::download(UMatData* u, void* dstptr, int dims, const size_t sz[], const size_t srcofs[],
                            const size_t srcstep[], const size_t dststep[]) const
{
    if (!u)
        return;
    int isz[LYCON_MAX_DIM];
    uchar* srcptr = u->data;
    for (int i = 0; i < dims; i++)
    {
        LYCON_ASSERT(sz[i] <= (size_t)INT_MAX);
        if (sz[i] == 0)
            return;
        if (srcofs)
            srcptr += srcofs[i] * (i <= dims - 2 ? srcstep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, LYCON_8U, srcptr, srcstep);
    Mat dst(dims, isz, LYCON_8U, dstptr, dststep);

    const Mat* arrays[] = {&src, &dst};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for (size_t j = 0; j < it.nplanes; j++, ++it)
        memcpy(ptrs[1], ptrs[0], planesz);
}

void MatAllocator::upload(UMatData* u, const void* srcptr, int dims, const size_t sz[], const size_t dstofs[],
                          const size_t dststep[], const size_t srcstep[]) const
{
    if (!u)
        return;
    int isz[LYCON_MAX_DIM];
    uchar* dstptr = u->data;
    for (int i = 0; i < dims; i++)
    {
        LYCON_ASSERT(sz[i] <= (size_t)INT_MAX);
        if (sz[i] == 0)
            return;
        if (dstofs)
            dstptr += dstofs[i] * (i <= dims - 2 ? dststep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, LYCON_8U, (void*)srcptr, srcstep);
    Mat dst(dims, isz, LYCON_8U, dstptr, dststep);

    const Mat* arrays[] = {&src, &dst};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for (size_t j = 0; j < it.nplanes; j++, ++it)
        memcpy(ptrs[1], ptrs[0], planesz);
}

void MatAllocator::copy(UMatData* usrc, UMatData* udst, int dims, const size_t sz[], const size_t srcofs[],
                        const size_t srcstep[], const size_t dstofs[], const size_t dststep[], bool /*sync*/) const
{
    if (!usrc || !udst)
        return;
    int isz[LYCON_MAX_DIM];
    uchar* srcptr = usrc->data;
    uchar* dstptr = udst->data;
    for (int i = 0; i < dims; i++)
    {
        LYCON_ASSERT(sz[i] <= (size_t)INT_MAX);
        if (sz[i] == 0)
            return;
        if (srcofs)
            srcptr += srcofs[i] * (i <= dims - 2 ? srcstep[i] : 1);
        if (dstofs)
            dstptr += dstofs[i] * (i <= dims - 2 ? dststep[i] : 1);
        isz[i] = (int)sz[i];
    }

    Mat src(dims, isz, LYCON_8U, srcptr, srcstep);
    Mat dst(dims, isz, LYCON_8U, dstptr, dststep);

    const Mat* arrays[] = {&src, &dst};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs, 2);
    size_t planesz = it.size;

    for (size_t j = 0; j < it.nplanes; j++, ++it)
        memcpy(ptrs[1], ptrs[0], planesz);
}

BufferPoolController* MatAllocator::getBufferPoolController(const char* id) const
{
    return nullptr;
}

class StdMatAllocator : public MatAllocator
{
  public:
    UMatData* allocate(int dims, const int* sizes, int type, void* data0, size_t* step, int /*flags*/,
                       UMatUsageFlags /*usageFlags*/) const
    {
        size_t total = LYCON_ELEM_SIZE(type);
        for (int i = dims - 1; i >= 0; i--)
        {
            if (step)
            {
                if (data0 && step[i] != LYCON_AUTOSTEP)
                {
                    LYCON_ASSERT(total <= step[i]);
                    total = step[i];
                }
                else
                    step[i] = total;
            }
            total *= sizes[i];
        }
        uchar* data = data0 ? (uchar*)data0 : (uchar*)fastMalloc(total);
        UMatData* u = new UMatData(this);
        u->data = u->origdata = data;
        u->size = total;
        if (data0)
            u->flags |= UMatData::USER_ALLOCATED;

        return u;
    }

    bool allocate(UMatData* u, int /*accessFlags*/, UMatUsageFlags /*usageFlags*/) const
    {
        if (!u)
            return false;
        return true;
    }

    void deallocate(UMatData* u) const
    {
        if (!u)
            return;

        LYCON_ASSERT(u->urefcount == 0);
        LYCON_ASSERT(u->refcount == 0);
        if (!(u->flags & UMatData::USER_ALLOCATED))
        {
            fastFree(u->origdata);
            u->origdata = 0;
        }
        delete u;
    }
};
namespace
{
MatAllocator* g_matAllocator = NULL;
}

MatAllocator* Mat::getDefaultAllocator()
{
    if (g_matAllocator == NULL)
    {
#ifdef LYCON_USE_NUMPY_ALLOCATOR_BY_DEFAULT
        g_matAllocator = &(NumpyAllocator::getNumpyAllocator());
#else
        g_matAllocator = getStdAllocator();
#endif
    }
    return g_matAllocator;
}
void Mat::setDefaultAllocator(MatAllocator* allocator)
{
    g_matAllocator = allocator;
}
MatAllocator* Mat::getStdAllocator()
{
    LYCON_SINGLETON_LAZY_INIT(MatAllocator, new StdMatAllocator())
}
}
