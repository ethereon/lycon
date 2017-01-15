#pragma once

namespace lycon
{
//! Imread flags
enum ImreadModes
{
    IMREAD_UNCHANGED = -1,  //!< If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    IMREAD_GRAYSCALE = 0,   //!< If set, always convert image to the single channel grayscale image.
    IMREAD_COLOR = 1,       //!< If set, always convert image to the 3 channel BGR color image.
    IMREAD_ANYDEPTH = 2,   //!< If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise
                           //! convert it to 8-bit.
    IMREAD_ANYCOLOR = 4,   //!< If set, the image is read in any possible color format.
    IMREAD_LOAD_GDAL = 8,  //!< If set, use the gdal driver for loading the image.
    IMREAD_REDUCED_GRAYSCALE_2 =
        16,  //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
    IMREAD_REDUCED_COLOR_2 =
        17,  //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
    IMREAD_REDUCED_GRAYSCALE_4 =
        32,  //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
    IMREAD_REDUCED_COLOR_4 =
        33,  //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
    IMREAD_REDUCED_GRAYSCALE_8 =
        64,  //!< If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
    IMREAD_REDUCED_COLOR_8 =
        65,  //!< If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
    IMREAD_IGNORE_ORIENTATION = 128  //!< If set, do not rotate the image according to EXIF's orientation flag.
};

//! Imwrite flags
enum ImwriteFlags
{
    IMWRITE_JPEG_QUALITY =
        1,  //!< For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
    IMWRITE_JPEG_PROGRESSIVE = 2,     //!< Enable JPEG features, 0 or 1, default is False.
    IMWRITE_JPEG_OPTIMIZE = 3,        //!< Enable JPEG features, 0 or 1, default is False.
    IMWRITE_JPEG_RST_INTERVAL = 4,    //!< JPEG restart interval, 0 - 65535, default is 0 - no restart.
    IMWRITE_JPEG_LUMA_QUALITY = 5,    //!< Separate luma quality level, 0 - 100, default is 0 - don't use.
    IMWRITE_JPEG_CHROMA_QUALITY = 6,  //!< Separate chroma quality level, 0 - 100, default is 0 - don't use.
    IMWRITE_PNG_COMPRESSION = 16,     //!< For PNG, it can be the compression level from 0 to 9. A higher value means a
                                      //! smaller size and longer compression time. Default value is 3. Also strategy is
    //! changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY).
    IMWRITE_PNG_STRATEGY = 17,  //!< One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_DEFAULT.
    IMWRITE_PNG_BILEVEL = 18,   //!< Binary level PNG, 0 or 1, default is 0.
    IMWRITE_PXM_BINARY = 32,    //!< For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
    IMWRITE_WEBP_QUALITY = 64,  //!< For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default
                                //!(without any parameter) and for quality above 100 the lossless compression is used.
    IMWRITE_PAM_TUPLETYPE =
        128,  //!< For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
};

//! Imwrite PNG specific flags used to tune the compression algorithm.
/** These flags will be modify the way of PNG image compression and will be passed to the underlying zlib processing
stage.

-   The effect of IMWRITE_PNG_STRATEGY_FILTERED is to force more Huffman coding and less string matching; it is somewhat
    intermediate between IMWRITE_PNG_STRATEGY_DEFAULT and IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY.
-   IMWRITE_PNG_STRATEGY_RLE is designed to be almost as fast as IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY, but give better
    compression for PNG image data.
-   The strategy parameter only affects the compression ratio but not the correctness of the compressed output even if
    it is not set appropriately.
-   IMWRITE_PNG_STRATEGY_FIXED prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special
    applications.
*/
enum ImwritePNGFlags
{
    IMWRITE_PNG_STRATEGY_DEFAULT = 0,   //!< Use this value for normal data.
    IMWRITE_PNG_STRATEGY_FILTERED = 1,  //!< Use this value for data produced by a filter (or predictor).Filtered data
                                        //! consists mostly of small values with a somewhat random distribution. In this
                                        //! case, the compression algorithm is tuned to compress them better.
    IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2,  //!< Use this value to force Huffman encoding only (no string match).
    IMWRITE_PNG_STRATEGY_RLE = 3,           //!< Use this value to limit match distances to one (run-length encoding).
    IMWRITE_PNG_STRATEGY_FIXED = 4  //!< Using this value prevents the use of dynamic Huffman codes, allowing for a
                                    //! simpler decoder for special applications.
};

//! Imwrite PAM specific tupletype flags used to define the 'TUPETYPE' field of a PAM file.
enum ImwritePAMFlags
{
    IMWRITE_PAM_FORMAT_NULL = 0,
    IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1,
    IMWRITE_PAM_FORMAT_GRAYSCALE = 2,
    IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3,
    IMWRITE_PAM_FORMAT_RGB = 4,
    IMWRITE_PAM_FORMAT_RGB_ALPHA = 5,
};

enum
{
    /* 8bit, color or not */
    LYCON_LOAD_IMAGE_UNCHANGED = -1,
    /* 8bit, gray */
    LYCON_LOAD_IMAGE_GRAYSCALE = 0,
    /* ?, color */
    LYCON_LOAD_IMAGE_COLOR = 1,
    /* any depth, ? */
    LYCON_LOAD_IMAGE_ANYDEPTH = 2,
    /* ?, any color */
    LYCON_LOAD_IMAGE_ANYCOLOR = 4,
    /* ?, no rotate */
    LYCON_LOAD_IMAGE_IGNORE_ORIENTATION = 128
};
}
