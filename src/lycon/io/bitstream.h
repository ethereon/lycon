#pragma once

#include <cstdio>
#include <vector>

#include "lycon/defs.h"
#include "lycon/util/string.h"

namespace lycon
{
enum
{
    RBS_THROW_EOS = -123,   // <end of stream> exception code
    RBS_THROW_FORB = -124,  // <forrbidden huffman code> exception code
    RBS_HUFF_FORB = 2047,   // forrbidden huffman code "value"
    RBS_BAD_HEADER = -125   // invalid header
};

typedef unsigned long ulong;

// class RBaseStream - base class for other reading streams.
class RBaseStream
{
   public:
    // methods
    RBaseStream();
    virtual ~RBaseStream();

    virtual bool open(const String &filename);
    virtual void close();
    bool isOpened();
    void setPos(int pos);
    int getPos();
    void skip(int bytes);

   protected:
    bool m_allocated;
    uchar *m_start;
    uchar *m_end;
    uchar *m_current;
    FILE *m_file;
    int m_block_size;
    int m_block_pos;
    bool m_is_opened;

    virtual void readBlock();
    virtual void release();
    virtual void allocate();
};

// class RLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-uchar value goes first
class RLByteStream : public RBaseStream
{
   public:
    virtual ~RLByteStream();

    int getByte();
    int getBytes(void *buffer, int count);
    int getWord();
    int getDWord();
};

// class RMBitStream - uchar-oriented stream.
// m in prefix means that the most significant uchar of a multi-uchar value go first
class RMByteStream : public RLByteStream
{
   public:
    virtual ~RMByteStream();

    int getWord();
    int getDWord();
};

// WBaseStream - base class for output streams
class WBaseStream
{
   public:
    // methods
    WBaseStream();
    virtual ~WBaseStream();

    virtual bool open(const String &filename);
    virtual bool open(std::vector<uchar> &buf);
    virtual void close();
    bool isOpened();
    int getPos();

   protected:
    uchar *m_start;
    uchar *m_end;
    uchar *m_current;
    int m_block_size;
    int m_block_pos;
    FILE *m_file;
    bool m_is_opened;
    std::vector<uchar> *m_buf;

    virtual void writeBlock();
    virtual void release();
    virtual void allocate();
};

// class WLByteStream - uchar-oriented stream.
// l in prefix means that the least significant uchar of a multi-byte value goes first
class WLByteStream : public WBaseStream
{
   public:
    virtual ~WLByteStream();

    void putByte(int val);
    void putBytes(const void *buffer, int count);
    void putWord(int val);
    void putDWord(int val);
};

// class WLByteStream - uchar-oriented stream.
// m in prefix means that the least significant uchar of a multi-byte value goes last
class WMByteStream : public WLByteStream
{
   public:
    virtual ~WMByteStream();
    void putWord(int val);
    void putDWord(int val);
};

inline unsigned BSWAP(unsigned v)
{
    return (v << 24) | ((v & 0xff00) << 8) | ((v >> 8) & 0xff00) | ((unsigned)v >> 24);
}

bool bsIsBigEndian(void);
}
