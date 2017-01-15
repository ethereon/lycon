#pragma once

namespace lycon
{
class BufferPoolController
{
   protected:
    ~BufferPoolController() {}

   public:
    virtual size_t getReservedSize() const = 0;
    virtual size_t getMaxReservedSize() const = 0;
    virtual void setMaxReservedSize(size_t size) = 0;
    virtual void freeAllReservedBuffers() = 0;
};
}
