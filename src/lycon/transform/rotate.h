#pragma once

#include "lycon/mat/mat.h"

namespace lycon
{

enum RotateFlags
{
    ROTATE_90_CLOCKWISE = 0,        // Rotate 90 degrees clockwise
    ROTATE_180 = 1,                 // Rotate 180 degrees clockwise
    ROTATE_90_COUNTERCLOCKWISE = 2, // Rotate 270 degrees clockwise
};

void flip(InputArray _src, OutputArray _dst, int flip_mode);

void transpose(InputArray _src, OutputArray _dst);

void rotate(InputArray _src, OutputArray _dst, int rotateMode);
}
