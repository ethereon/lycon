#pragma once

#include "lycon/io/base.h"

namespace lycon
{
class PngDecoder : public BaseImageDecoder
{
   public:
    PngDecoder();
    virtual ~PngDecoder();

    bool readData(Mat &img);
    bool readHeader();
    void close();

    ImageDecoder newDecoder() const;

   protected:
    static void readDataFromBuf(void *png_ptr, uchar *dst, size_t size);

    int m_bit_depth;
    void *m_png_ptr;   // pointer to decompression structure
    void *m_info_ptr;  // pointer to image information structure
    void *m_end_info;  // pointer to one more image information structure
    FILE *m_f;
    int m_color_type;
    size_t m_buf_pos;
};

class PngEncoder : public BaseImageEncoder
{
   public:
    PngEncoder();
    virtual ~PngEncoder();

    bool isFormatSupported(int depth) const;
    bool write(const Mat &img, const std::vector<int> &params);

    ImageEncoder newEncoder() const;

   protected:
    static void writeDataToBuf(void *png_ptr, uchar *src, size_t size);
    static void flushBuf(void *png_ptr);
};
}
