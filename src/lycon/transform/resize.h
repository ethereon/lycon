#pragma once

#include "lycon/mat/mat.h"

namespace lycon
{
enum InterpolationFlags
{
    /** nearest neighbor interpolation */
    INTER_NEAREST = 0,
    /** bilinear interpolation */
    INTER_LINEAR = 1,
    /** bicubic interpolation */
    INTER_CUBIC = 2,
    /** resampling using pixel area relation. It may be a preferred method for
    image decimation, as
    it gives moire'-free results. But when the image is zoomed, it is similar to
    the INTER_NEAREST
    method. */
    INTER_AREA = 3,
    /** Lanczos interpolation over 8x8 neighborhood */
    INTER_LANCZOS4 = 4,
    /** mask for interpolation codes */
    INTER_MAX = 7,
    /** flag, fills all of the destination image pixels. If some of them
    correspond to outliers in the
    source image, they are set to zero */
    WARP_FILL_OUTLIERS = 8,
    /** flag, inverse transformation

    For example, @ref cv::linearPolar or @ref cv::logPolar transforms:
    - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    */
    WARP_INVERSE_MAP = 16
};

enum InterpolationMasks
{
    INTER_BITS = 5,
    INTER_BITS2 = INTER_BITS * 2,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

void resize(InputArray _src, OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y, int interpolation);
}
