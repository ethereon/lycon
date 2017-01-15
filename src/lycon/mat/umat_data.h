#pragma once

#include "lycon/defs.h"
#include "lycon/util/error.h"
#include "lycon/util/macros.h"

namespace lycon
{

class MatAllocator;

struct LYCON_EXPORTS UMatData
{
    enum
    {
        COPY_ON_MAP = 1,
        HOST_COPY_OBSOLETE = 2,
        DEVICE_COPY_OBSOLETE = 4,
        TEMP_UMAT = 8,
        TEMP_COPIED_UMAT = 24,
        USER_ALLOCATED = 32,
        DEVICE_MEM_MAPPED = 64
    };
    UMatData(const MatAllocator *allocator);
    ~UMatData();

    // provide atomic access to the structure
    void lock();
    void unlock();

    bool hostCopyObsolete() const;
    bool deviceCopyObsolete() const;
    bool deviceMemMapped() const;
    bool copyOnMap() const;
    bool tempUMat() const;
    bool tempCopiedUMat() const;
    void markHostCopyObsolete(bool flag);
    void markDeviceCopyObsolete(bool flag);
    void markDeviceMemMapped(bool flag);

    const MatAllocator *prevAllocator;
    const MatAllocator *currAllocator;
    int urefcount;
    int refcount;
    uchar *data;
    uchar *origdata;
    size_t size;

    int flags;
    void *handle;
    void *userdata;
    int allocatorFlags_;
    int mapcount;
    UMatData *originalUMatData;
};

inline bool UMatData::hostCopyObsolete() const
{
    return (flags & HOST_COPY_OBSOLETE) != 0;
}
inline bool UMatData::deviceCopyObsolete() const
{
    return (flags & DEVICE_COPY_OBSOLETE) != 0;
}
inline bool UMatData::deviceMemMapped() const
{
    return (flags & DEVICE_MEM_MAPPED) != 0;
}
inline bool UMatData::copyOnMap() const
{
    return (flags & COPY_ON_MAP) != 0;
}
inline bool UMatData::tempUMat() const
{
    return (flags & TEMP_UMAT) != 0;
}
inline bool UMatData::tempCopiedUMat() const
{
    return (flags & TEMP_COPIED_UMAT) == TEMP_COPIED_UMAT;
}

inline void UMatData::markDeviceMemMapped(bool flag)
{
    if (flag)
        flags |= DEVICE_MEM_MAPPED;
    else
        flags &= ~DEVICE_MEM_MAPPED;
}

inline void UMatData::markHostCopyObsolete(bool flag)
{
    if (flag)
        flags |= HOST_COPY_OBSOLETE;
    else
        flags &= ~HOST_COPY_OBSOLETE;
}
inline void UMatData::markDeviceCopyObsolete(bool flag)
{
    if (flag)
        flags |= DEVICE_COPY_OBSOLETE;
    else
        flags &= ~DEVICE_COPY_OBSOLETE;
}
}
