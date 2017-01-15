#pragma once

#include "lycon/defs.h"
#include "lycon/mat/buffer_pool.h"

namespace lycon
{
struct UMatData;

enum UMatUsageFlags
{
    USAGE_DEFAULT = 0,

    // buffer allocation policy is platform and usage specific
    USAGE_ALLOCATE_HOST_MEMORY = 1 << 0,
    USAGE_ALLOCATE_DEVICE_MEMORY = 1 << 1,
    USAGE_ALLOCATE_SHARED_MEMORY =
        1 << 2,  // It is not equal to: USAGE_ALLOCATE_HOST_MEMORY | USAGE_ALLOCATE_DEVICE_MEMORY

    __UMAT_USAGE_FLAGS_32BIT = 0x7fffffff  // Binary compatibility hint
};

class LYCON_EXPORTS MatAllocator
{
   public:
    MatAllocator() {}
    virtual ~MatAllocator() {}

    virtual UMatData *allocate(int dims, const int *sizes, int type, void *data, size_t *step, int flags,
                               UMatUsageFlags usageFlags) const = 0;
    virtual bool allocate(UMatData *data, int accessflags, UMatUsageFlags usageFlags) const = 0;
    virtual void deallocate(UMatData *data) const = 0;
    virtual void map(UMatData *data, int accessflags) const;
    virtual void unmap(UMatData *data) const;
    virtual void download(UMatData *data, void *dst, int dims, const size_t sz[], const size_t srcofs[],
                          const size_t srcstep[], const size_t dststep[]) const;
    virtual void upload(UMatData *data, const void *src, int dims, const size_t sz[], const size_t dstofs[],
                        const size_t dststep[], const size_t srcstep[]) const;
    virtual void copy(UMatData *srcdata, UMatData *dstdata, int dims, const size_t sz[], const size_t srcofs[],
                      const size_t srcstep[], const size_t dstofs[], const size_t dststep[], bool sync) const;

    // default implementation returns DummyBufferPoolController
    virtual BufferPoolController *getBufferPoolController(const char *id = NULL) const;
};
}
