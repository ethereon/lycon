#pragma once

#include "lycon/io/base.h"

namespace lycon
{
class JpegDecoder : public BaseImageDecoder
{
   public:
    JpegDecoder();
    virtual ~JpegDecoder();

    bool readData(Mat &img);
    bool readHeader();
    void close();

    ImageDecoder newDecoder() const;

   protected:
    FILE *m_f;
    void *m_state;
};

class JpegEncoder : public BaseImageEncoder
{
   public:
    JpegEncoder();
    virtual ~JpegEncoder();

    bool write(const Mat &img, const std::vector<int> &params);
    ImageEncoder newEncoder() const;
};
}
