#pragma once

#include <memory>

#include "lycon/defs.h"
#include "lycon/io/bitstream.h"
#include "lycon/mat/mat.h"

namespace lycon
{
class BaseImageDecoder;
class BaseImageEncoder;
typedef std::shared_ptr<BaseImageEncoder> ImageEncoder;
typedef std::shared_ptr<BaseImageDecoder> ImageDecoder;

///////////////////////////////// base class for decoders ////////////////////////
class BaseImageDecoder
{
   public:
    BaseImageDecoder();
    virtual ~BaseImageDecoder() {}

    int width() const { return m_width; }
    int height() const { return m_height; }
    virtual int type() const { return m_type; }

    virtual bool setSource(const String &filename);
    virtual bool setSource(const Mat &buf);
    virtual int setScale(const int &scale_denom);
    virtual bool readHeader() = 0;
    virtual bool readData(Mat &img) = 0;

    /// Called after readData to advance to the next page, if any.
    virtual bool nextPage() { return false; }

    virtual size_t signatureLength() const;
    virtual bool checkSignature(const String &signature) const;
    virtual ImageDecoder newDecoder() const;

   protected:
    int m_width;   // width  of the image ( filled by readHeader )
    int m_height;  // height of the image ( filled by readHeader )
    int m_type;
    int m_scale_denom;
    String m_filename;
    String m_signature;
    Mat m_buf;
    bool m_buf_supported;
};

///////////////////////////// base class for encoders ////////////////////////////
class BaseImageEncoder
{
   public:
    BaseImageEncoder();
    virtual ~BaseImageEncoder() {}
    virtual bool isFormatSupported(int depth) const;

    virtual bool setDestination(const String &filename);
    virtual bool setDestination(std::vector<uchar> &buf);
    virtual bool write(const Mat &img, const std::vector<int> &params) = 0;

    virtual String getDescription() const;
    virtual ImageEncoder newEncoder() const;

    virtual void throwOnEror() const;

   protected:
    String m_description;

    String m_filename;
    std::vector<uchar> *m_buf;
    bool m_buf_supported;

    String m_last_error;
};
}
