#include "lycon/io/io.h"

#include <iostream>

#include "lycon/io/base.h"
#include "lycon/io/exif.h"
#include "lycon/io/options.h"
#include "lycon/transform/rotate.h"
#include "lycon/util/error.h"
#include "lycon/util/file.h"

#include "lycon/io/jpeg.h"
#include "lycon/io/png.h"

namespace lycon
{
/**
 * @struct ImageCodecInitializer
 *
 * Container which stores the registered codecs to be used by OpenCV
*/
struct ImageCodecInitializer
{
    /**
     * Default Constructor for the ImageCodeInitializer
    */
    ImageCodecInitializer()
    {
        decoders.push_back(std::make_shared<JpegDecoder>());
        encoders.push_back(std::make_shared<JpegEncoder>());
        // #ifdef HAVE_TIFF
        //         decoders.push_back(std::make_shared<TiffDecoder>());
        // #endif
        //         encoders.push_back(std::make_shared<TiffEncoder>());
        decoders.push_back(std::make_shared<PngDecoder>());
        encoders.push_back(std::make_shared<PngEncoder>());
    }

    std::vector<ImageDecoder> decoders;
    std::vector<ImageEncoder> encoders;
};

static ImageCodecInitializer codecs;

/**
 * Find the decoders
 *
 * @param[in] filename File to search
 *
 * @return Image decoder to parse image file.
*/
static ImageDecoder findDecoder(const String& filename)
{
    size_t i, maxlen = 0;

    /// iterate through list of registered codecs
    for (i = 0; i < codecs.decoders.size(); i++)
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    /// Open the file
    FILE* f = fopen(filename.c_str(), "rb");

    /// in the event of a failure, return an empty image decoder
    if (!f)
        return ImageDecoder();

    // read the file signature
    String signature(maxlen, ' ');
    maxlen = fread((void*)signature.c_str(), 1, maxlen, f);
    fclose(f);
    signature = signature.substr(0, maxlen);

    /// compare signature against all decoders
    for (i = 0; i < codecs.decoders.size(); i++)
    {
        if (codecs.decoders[i]->checkSignature(signature))
            return codecs.decoders[i]->newDecoder();
    }

    /// If no decoder was found, return base type
    return ImageDecoder();
}

static ImageDecoder findDecoder(const Mat& buf)
{
    size_t i, maxlen = 0;

    if (buf.rows * buf.cols < 1 || !buf.isContinuous())
        return ImageDecoder();

    for (i = 0; i < codecs.decoders.size(); i++)
    {
        size_t len = codecs.decoders[i]->signatureLength();
        maxlen = std::max(maxlen, len);
    }

    String signature(maxlen, ' ');
    size_t bufSize = buf.rows * buf.cols * buf.elemSize();
    maxlen = std::min(maxlen, bufSize);
    memcpy((void*)signature.c_str(), buf.data, maxlen);

    for (i = 0; i < codecs.decoders.size(); i++)
    {
        if (codecs.decoders[i]->checkSignature(signature))
            return codecs.decoders[i]->newDecoder();
    }

    return ImageDecoder();
}

static ImageEncoder findEncoder(const String& _ext)
{
    if (_ext.size() <= 1)
        return ImageEncoder();

    const char* ext = strrchr(_ext.c_str(), '.');
    if (!ext)
        return ImageEncoder();
    int len = 0;
    for (ext++; len < 128 && isalnum(ext[len]); len++)
        ;

    for (size_t i = 0; i < codecs.encoders.size(); i++)
    {
        String description = codecs.encoders[i]->getDescription();
        const char* descr = strchr(description.c_str(), '(');

        while (descr)
        {
            descr = strchr(descr + 1, '.');
            if (!descr)
                break;
            int j = 0;
            for (descr++; j < len && isalnum(descr[j]); j++)
            {
                int c1 = tolower(ext[j]);
                int c2 = tolower(descr[j]);
                if (c1 != c2)
                    break;
            }
            if (j == len && !isalnum(descr[j]))
                return codecs.encoders[i]->newEncoder();
            descr += j;
        }
    }

    return ImageEncoder();
}

enum
{
    LOAD_CVMAT = 0,
    LOAD_IMAGE = 1,
    LOAD_MAT = 2
};

static void ApplyExifOrientation(const String& filename, Mat& img)
{
    int orientation = IMAGE_ORIENTATION_TL;

    if (filename.size() > 0)
    {
        ExifReader reader(filename);
        if (reader.parse())
        {
            ExifEntry_t entry = reader.getTag(ORIENTATION);
            if (entry.tag != INVALID_TAG)
            {
                orientation = entry.field_u16; // orientation is unsigned short, so check field_u16
            }
        }
    }

    switch (orientation)
    {
    case IMAGE_ORIENTATION_TL: // 0th row == visual top, 0th column == visual left-hand side
        // do nothing, the image already has proper orientation
        break;
    case IMAGE_ORIENTATION_TR: // 0th row == visual top, 0th column == visual right-hand side
        flip(img, img, 1);     // flip horizontally
        break;
    case IMAGE_ORIENTATION_BR: // 0th row == visual bottom, 0th column == visual right-hand side
        flip(img, img, -1);    // flip both horizontally and vertically
        break;
    case IMAGE_ORIENTATION_BL: // 0th row == visual bottom, 0th column == visual left-hand side
        flip(img, img, 0);     // flip vertically
        break;
    case IMAGE_ORIENTATION_LT: // 0th row == visual left-hand side, 0th column == visual top
        transpose(img, img);
        break;
    case IMAGE_ORIENTATION_RT: // 0th row == visual right-hand side, 0th column == visual top
        transpose(img, img);
        flip(img, img, 1); // flip horizontally
        break;
    case IMAGE_ORIENTATION_RB: // 0th row == visual right-hand side, 0th column == visual bottom
        transpose(img, img);
        flip(img, img, -1); // flip both horizontally and vertically
        break;
    case IMAGE_ORIENTATION_LB: // 0th row == visual left-hand side, 0th column == visual bottom
        transpose(img, img);
        flip(img, img, 0); // flip vertically
        break;
    default:
        // by default the image read has normal (JPEG_ORIENTATION_TL) orientation
        break;
    }
}

/**
 * Read an image into memory and return the information
 *
 * @param[in] filename File to load
 * @param[in] flags Flags
 * @param[in] hdrtype { LOAD_CVMAT=0,
 *                      LOAD_IMAGE=1,
 *                      LOAD_MAT=2
 *                    }
 * @param[in] mat Reference to C++ Mat object (If LOAD_MAT)
 * @param[in] scale_denom Scale value
 *
*/
static void* imread_(const String& filename, int flags, Mat* mat = 0)
{
    Mat temp, *data = &temp;

    /// Search for the relevant decoder to handle the imagery
    ImageDecoder decoder = findDecoder(filename);

    /// if no decoder was found, return nothing.
    if (!decoder)
    {
        return 0;
    }

    int scale_denom = 1;
    if (flags > IMREAD_LOAD_GDAL)
    {
        if (flags & IMREAD_REDUCED_GRAYSCALE_2)
            scale_denom = 2;
        else if (flags & IMREAD_REDUCED_GRAYSCALE_4)
            scale_denom = 4;
        else if (flags & IMREAD_REDUCED_GRAYSCALE_8)
            scale_denom = 8;
    }

    /// set the scale_denom in the driver
    decoder->setScale(scale_denom);

    /// set the filename in the driver
    decoder->setSource(filename);

    // read the header to make sure it succeeds
    if (!decoder->readHeader())
        return 0;

    // established the required input image size
    Size size;
    size.width = decoder->width();
    size.height = decoder->height();

    // grab the decoded type
    int type = decoder->type();
    if ((flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED)
    {
        if ((flags & LYCON_LOAD_IMAGE_ANYDEPTH) == 0)
            type = LYCON_MAKETYPE(LYCON_8U, LYCON_MAT_CN(type));

        if ((flags & LYCON_LOAD_IMAGE_COLOR) != 0 ||
            ((flags & LYCON_LOAD_IMAGE_ANYCOLOR) != 0 && LYCON_MAT_CN(type) > 1))
            type = LYCON_MAKETYPE(LYCON_MAT_DEPTH(type), 3);
        else
            type = LYCON_MAKETYPE(LYCON_MAT_DEPTH(type), 1);
    }

    mat->create(size.height, size.width, type);
    data = mat;

    // read the image data
    if (!decoder->readData(*data))
    {
        if (mat)
            mat->release();
        return 0;
    }

    if (decoder->setScale(scale_denom) > 1) // if decoder is JpegDecoder then decoder->setScale always returns 1
    {
        // TODO(saumitro)
        //        resize(*mat, *mat, Size(size.width / scale_denom, size.height / scale_denom));
    }

    return (void*)mat;
}

/**
 * Read an image
 *
 *  This function merely calls the actual implementation above and returns itself.
 *
 * @param[in] filename File to load
 * @param[in] flags Flags you wish to set.
*/
Mat imread(const String& filename, int flags)
{
    /// create the basic container
    Mat img;

    /// load the data
    imread_(filename, flags, &img);

    /// optionally rotate the data if EXIF' orientation flag says so
    if ((flags & IMREAD_IGNORE_ORIENTATION) == 0 && flags != IMREAD_UNCHANGED)
    {
        ApplyExifOrientation(filename, img);
    }

    /// return a reference to the data
    return img;
}

static bool imwrite_(const String& filename, const Mat& image, const std::vector<int>& params, bool flipv)
{
    Mat temp;
    const Mat* pimage = &image;

    LYCON_ASSERT(image.channels() == 1 || image.channels() == 3 || image.channels() == 4);

    ImageEncoder encoder = findEncoder(filename);
    if (!encoder)
        LYCON_ERROR("could not find a writer for the specified extension");
    if (!encoder->isFormatSupported(image.depth()))
    {
        LYCON_ASSERT(encoder->isFormatSupported(LYCON_8U));
        image.convertTo(temp, LYCON_8U);
        pimage = &temp;
    }

    if (flipv)
    {
        flip(*pimage, temp, 0);
        pimage = &temp;
    }

    encoder->setDestination(filename);
    bool code = encoder->write(*pimage, params);

    //    LYCON_ASSERT( code );
    return code;
}

bool imwrite(const String& filename, InputArray _img, const std::vector<int>& params)
{
    Mat img = _img.getMat();
    return imwrite_(filename, img, params, false);
}

static void* imdecode_(const Mat& buf, int flags, Mat* mat = 0)
{
    LYCON_ASSERT(!buf.empty() && buf.isContinuous());
    Mat temp, *data = &temp;
    String filename;

    ImageDecoder decoder = findDecoder(buf);
    if (!decoder)
        return 0;

    if (!decoder->setSource(buf))
    {
        filename = tempfile();
        FILE* f = fopen(filename.c_str(), "wb");
        if (!f)
            return 0;
        size_t bufSize = buf.cols * buf.rows * buf.elemSize();
        fwrite(buf.ptr(), 1, bufSize, f);
        fclose(f);
        decoder->setSource(filename);
    }

    if (!decoder->readHeader())
    {
        decoder.reset();
        if (!filename.empty())
        {
            if (remove(filename.c_str()) != 0)
            {
                LYCON_ERROR("unable to remove temporary file");
            }
        }
        return 0;
    }

    Size size;
    size.width = decoder->width();
    size.height = decoder->height();

    int type = decoder->type();
    if ((flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED)
    {
        if ((flags & LYCON_LOAD_IMAGE_ANYDEPTH) == 0)
            type = LYCON_MAKETYPE(LYCON_8U, LYCON_MAT_CN(type));

        if ((flags & LYCON_LOAD_IMAGE_COLOR) != 0 ||
            ((flags & LYCON_LOAD_IMAGE_ANYCOLOR) != 0 && LYCON_MAT_CN(type) > 1))
            type = LYCON_MAKETYPE(LYCON_MAT_DEPTH(type), 3);
        else
            type = LYCON_MAKETYPE(LYCON_MAT_DEPTH(type), 1);
    }

    mat->create(size.height, size.width, type);
    data = mat;

    bool code = decoder->readData(*data);
    decoder.reset();
    if (!filename.empty())
    {
        if (remove(filename.c_str()) != 0)
        {
            LYCON_ERROR("unable to remove temporary file");
        }
    }

    if (!code)
    {
        if (mat)
            mat->release();
        return 0;
    }

    return (void*)mat;
}

Mat imdecode(InputArray _buf, int flags)
{
    Mat buf = _buf.getMat(), img;
    imdecode_(buf, flags, &img);
    return img;
}

Mat imdecode(InputArray _buf, int flags, Mat* dst)
{
    Mat buf = _buf.getMat(), img;
    dst = dst ? dst : &img;
    imdecode_(buf, flags, dst);
    return *dst;
}

bool imencode(const String& ext, InputArray _image, std::vector<uchar>& buf, const std::vector<int>& params)
{
    Mat image = _image.getMat();

    int channels = image.channels();
    LYCON_ASSERT(channels == 1 || channels == 3 || channels == 4);

    ImageEncoder encoder = findEncoder(ext);
    if (!encoder)
        LYCON_ERROR("could not find encoder for the specified extension");

    if (!encoder->isFormatSupported(image.depth()))
    {
        LYCON_ASSERT(encoder->isFormatSupported(LYCON_8U));
        Mat temp;
        image.convertTo(temp, LYCON_8U);
        image = temp;
    }

    bool code;
    if (encoder->setDestination(buf))
    {
        code = encoder->write(image, params);
        encoder->throwOnEror();
        LYCON_ASSERT(code);
    }
    else
    {
        String filename = tempfile();
        code = encoder->setDestination(filename);
        LYCON_ASSERT(code);

        code = encoder->write(image, params);
        encoder->throwOnEror();
        LYCON_ASSERT(code);

        FILE* f = fopen(filename.c_str(), "rb");
        LYCON_ASSERT(f != 0);
        fseek(f, 0, SEEK_END);
        long pos = ftell(f);
        buf.resize((size_t)pos);
        fseek(f, 0, SEEK_SET);
        buf.resize(fread(&buf[0], 1, buf.size(), f));
        fclose(f);
        remove(filename.c_str());
    }
    return code;
}
}
