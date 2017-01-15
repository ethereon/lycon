#include "lycon/util/alloc.h"
#include "lycon/util/error.h"
#include "lycon/util/util.h"

#define LYCON_MALLOC_ALIGN 16

namespace lycon
{
void* fastMalloc(size_t size)
{
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + LYCON_MALLOC_ALIGN);
    if (!udata)
        LYCON_ERROR("Failed to allocate %lu bytes", (unsigned long)size);
    uchar** adata = alignPtr((uchar**)udata + 1, LYCON_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if (ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        LYCON_DbgAssert(udata < (uchar*)ptr &&
                        ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*) + LYCON_MALLOC_ALIGN));
        free(udata);
    }
}
}
