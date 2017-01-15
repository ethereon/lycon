#pragma once

#include "lycon/mat/shared.h"

namespace lycon
{
void convertAndUnrollScalar(const Mat &sc, int buftype, uchar *scbuf, size_t blocksize);

BinaryFunc getConvertFunc(int sdepth, int ddepth);
}
