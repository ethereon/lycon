#pragma once

#include "lycon/defs.h"

namespace lycon
{
/** @brief Allocates an aligned memory buffer.

The function allocates the buffer of the specified size and returns it. When the buffer size is 16
bytes or more, the returned buffer is aligned to 16 bytes.
@param bufSize Allocated buffer size.
*/
LYCON_EXPORTS void *fastMalloc(size_t bufSize);

/** @brief Deallocates a memory buffer.

The function deallocates the buffer allocated with fastMalloc . If NULL pointer is passed, the
function does nothing. C version of the function clears the pointer *pptr* to avoid problems with
double memory deallocation.
@param ptr Pointer to the allocated buffer.
 */
LYCON_EXPORTS void fastFree(void *ptr);
}
