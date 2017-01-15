#include "lycon/defs.h"
#include "lycon/types.h"

namespace lycon
{
class LYCON_EXPORTS ParallelLoopBody
{
   public:
    virtual ~ParallelLoopBody();
    virtual void operator()(const Range& range) const = 0;
};

LYCON_EXPORTS void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes = -1.);
}
