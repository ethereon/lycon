#pragma once

#include <vector>

#include "lycon/defs.h"

namespace lycon
{
class LYCON_EXPORTS TLSDataContainer
{
   protected:
    TLSDataContainer();
    virtual ~TLSDataContainer();

    void gatherData(std::vector<void*>& data) const;

    void* getData() const;
    void release();

   private:
    virtual void* createDataInstance() const = 0;
    virtual void deleteDataInstance(void* pData) const = 0;

    int key_;
};

// Main TLS data class
template <typename T>
class TLSData : protected TLSDataContainer
{
   public:
    inline TLSData() {}
    inline ~TLSData() { release(); }                 // Release key and delete associated data
    inline T* get() const { return (T*)getData(); }  // Get data assosiated with key

    // Get data from all threads
    inline void gather(std::vector<T*>& data) const
    {
        std::vector<void*>& dataVoid = reinterpret_cast<std::vector<void*>&>(data);
        gatherData(dataVoid);
    }

   private:
    virtual void* createDataInstance() const { return new T; }                // Wrapper to allocate data by template
    virtual void deleteDataInstance(void* pData) const { delete (T*)pData; }  // Wrapper to release data by template

    // Disable TLS copy operations
    TLSData(TLSData&) {}
    TLSData& operator=(const TLSData&) { return *this; }
};
}
