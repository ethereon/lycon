#pragma once

#include "lycon/defs.h"
#include "lycon/io/options.h"
#include "lycon/mat/mat.h"
#include "lycon/util/string.h"

namespace lycon
{
LYCON_EXPORTS Mat imread(const String &filename, int flags = IMREAD_COLOR);

LYCON_EXPORTS bool imwrite(const String &filename, InputArray img, const std::vector<int> &params = std::vector<int>());

LYCON_EXPORTS Mat imdecode(InputArray buf, int flags);

LYCON_EXPORTS Mat imdecode(InputArray buf, int flags, Mat *dst);

LYCON_EXPORTS bool imencode(const String &ext, InputArray img, std::vector<uchar> &buf,
                            const std::vector<int> &params = std::vector<int>());
}
