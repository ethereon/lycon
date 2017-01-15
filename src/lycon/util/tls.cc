#include "lycon/util/tls.h"

#include <mutex>

#include "lycon/util/error.h"
#include "lycon/util/singleton.h"

namespace lycon
{
#ifdef WIN32
#ifdef _MSC_VER
#pragma warning(disable : 4505) // unreferenced local function has been removed
#endif
#ifndef TLS_OUT_OF_INDEXES
#define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
#endif

// TLS platform abstraction layer
class TlsAbstraction
{
  public:
    TlsAbstraction();
    ~TlsAbstraction();
    void* GetData() const;
    void SetData(void* pData);

  private:
#ifdef WIN32
#ifndef WINRT
    DWORD tlsKey;
#endif
#else // WIN32
    pthread_key_t tlsKey;
#endif
};

#ifdef WIN32
#ifdef WINRT
static __declspec(thread) void* tlsData = NULL; // using C++11 thread attribute for local thread data
TlsAbstraction::TlsAbstraction()
{
}
TlsAbstraction::~TlsAbstraction()
{
}
void* TlsAbstraction::GetData() const
{
    return tlsData;
}
void TlsAbstraction::SetData(void* pData)
{
    tlsData = pData;
}
#else // WINRT
TlsAbstraction::TlsAbstraction()
{
    tlsKey = TlsAlloc();
    LYCON_ASSERT(tlsKey != TLS_OUT_OF_INDEXES);
}
TlsAbstraction::~TlsAbstraction()
{
    TlsFree(tlsKey);
}
void* TlsAbstraction::GetData() const
{
    return TlsGetValue(tlsKey);
}
void TlsAbstraction::SetData(void* pData)
{
    LYCON_ASSERT(TlsSetValue(tlsKey, pData) == TRUE);
}
#endif
#else // WIN32
TlsAbstraction::TlsAbstraction()
{
    LYCON_ASSERT(pthread_key_create(&tlsKey, NULL) == 0);
}
TlsAbstraction::~TlsAbstraction()
{
    LYCON_ASSERT(pthread_key_delete(tlsKey) == 0);
}
void* TlsAbstraction::GetData() const
{
    return pthread_getspecific(tlsKey);
}
void TlsAbstraction::SetData(void* pData)
{
    LYCON_ASSERT(pthread_setspecific(tlsKey, pData) == 0);
}
#endif

// Per-thread data structure
struct ThreadData
{
    ThreadData()
    {
        idx = 0;
        slots.reserve(32);
    }

    std::vector<void*> slots; // Data array for a thread
    size_t idx;               // Thread index in TLS storage. This is not OS thread ID!
};

// Main TLS storage class
class TlsStorage
{
  public:
    TlsStorage()
    {
        tlsSlots.reserve(32);
        threads.reserve(32);
    }
    ~TlsStorage()
    {
        for (size_t i = 0; i < threads.size(); i++)
        {
            if (threads[i])
            {
                /* Current architecture doesn't allow proper global objects relase, so this check can cause crashes

                // Check if all slots were properly cleared
                for(size_t j = 0; j < threads[i]->slots.size(); j++)
                {
                    LYCON_ASSERT(threads[i]->slots[j] == 0);
                }
                */
                delete threads[i];
            }
        }
        threads.clear();
    }

    void releaseThread()
    {
        // std::lock_guard<std::mutex> guard(mtxGlobalAccess);
        ThreadData* pTD = (ThreadData*)tls.GetData();
        for (size_t i = 0; i < threads.size(); i++)
        {
            if (pTD == threads[i])
            {
                threads[i] = 0;
                break;
            }
        }
        tls.SetData(0);
        delete pTD;
    }

    // Reserve TLS storage index
    size_t reserveSlot()
    {
        // std::lock_guard<std::mutex> guard(mtxGlobalAccess);

        // Find unused slots
        for (size_t slot = 0; slot < tlsSlots.size(); slot++)
        {
            if (!tlsSlots[slot])
            {
                tlsSlots[slot] = 1;
                return slot;
            }
        }

        // Create new slot
        tlsSlots.push_back(1);
        return (tlsSlots.size() - 1);
    }

    // Release TLS storage index and pass assosiated data to caller
    void releaseSlot(size_t slotIdx, std::vector<void*>& dataVec)
    {
        // std::lock_guard<std::mutex> guard(mtxGlobalAccess);
        LYCON_ASSERT(tlsSlots.size() > slotIdx);

        for (size_t i = 0; i < threads.size(); i++)
        {
            if (threads[i])
            {
                std::vector<void*>& thread_slots = threads[i]->slots;
                if (thread_slots.size() > slotIdx && thread_slots[slotIdx])
                {
                    dataVec.push_back(thread_slots[slotIdx]);
                    threads[i]->slots[slotIdx] = 0;
                }
            }
        }

        tlsSlots[slotIdx] = 0;
    }

    // Get data by TLS storage index
    void* getData(size_t slotIdx) const
    {
        LYCON_ASSERT(tlsSlots.size() > slotIdx);

        ThreadData* threadData = (ThreadData*)tls.GetData();
        if (threadData && threadData->slots.size() > slotIdx)
            return threadData->slots[slotIdx];

        return NULL;
    }

    // Gather data from threads by TLS storage index
    void gather(size_t slotIdx, std::vector<void*>& dataVec)
    {
        // std::lock_guard<std::mutex> guard(mtxGlobalAccess);
        LYCON_ASSERT(tlsSlots.size() > slotIdx);

        for (size_t i = 0; i < threads.size(); i++)
        {
            if (threads[i])
            {
                std::vector<void*>& thread_slots = threads[i]->slots;
                if (thread_slots.size() > slotIdx && thread_slots[slotIdx])
                    dataVec.push_back(thread_slots[slotIdx]);
            }
        }
    }

    // Set data to storage index
    void setData(size_t slotIdx, void* pData)
    {
        LYCON_ASSERT(tlsSlots.size() > slotIdx && pData != NULL);

        ThreadData* threadData = (ThreadData*)tls.GetData();
        if (!threadData)
        {
            threadData = new ThreadData;
            tls.SetData((void*)threadData);
            {
                // std::lock_guard<std::mutex> guard(mtxGlobalAccess);
                threadData->idx = threads.size();
                threads.push_back(threadData);
            }
        }

        if (slotIdx >= threadData->slots.size())
        {
            // std::lock_guard<std::mutex> guard(mtxGlobalAccess);
            while (slotIdx >= threadData->slots.size())
                threadData->slots.push_back(NULL);
        }
        threadData->slots[slotIdx] = pData;
    }

  private:
    // TLS abstraction layer instance
    TlsAbstraction tls;

    // Shared objects operation guard
    std::mutex mtxGlobalAccess;

    // TLS keys state
    std::vector<int> tlsSlots;

    // Array for all allocated data. Thread data pointers are placed here to allow data cleanup
    std::vector<ThreadData*> threads;
};

// Create global TLS storage object
static TlsStorage& getTlsStorage()
{
    LYCON_SINGLETON_LAZY_INIT_REF(TlsStorage, new TlsStorage())
}

TLSDataContainer::TLSDataContainer()
{
    key_ = (int)getTlsStorage().reserveSlot(); // Reserve key from TLS storage
}

TLSDataContainer::~TLSDataContainer()
{
    LYCON_ASSERT(key_ == -1); // Key must be released in child object
}

void TLSDataContainer::gatherData(std::vector<void*>& data) const
{
    getTlsStorage().gather(key_, data);
}

void TLSDataContainer::release()
{
    std::vector<void*> data;
    data.reserve(32);
    getTlsStorage().releaseSlot(key_, data); // Release key and get stored data for proper destruction
    for (size_t i = 0; i < data.size(); i++) // Delete all assosiated data
        deleteDataInstance(data[i]);
    key_ = -1;
}

void* TLSDataContainer::getData() const
{
    void* pData = getTlsStorage().getData(key_); // Check if data was already allocated
    if (!pData)
    {
        // Create new data instance and save it to TLS storage
        pData = createDataInstance();
        getTlsStorage().setData(key_, pData);
    }
    return pData;
}
}
