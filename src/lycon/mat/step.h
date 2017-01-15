#pragma once

#include "lycon/types.h"
#include "lycon/util/error.h"

namespace lycon
{

struct LYCON_EXPORTS MatStep
{
    MatStep();
    explicit MatStep(size_t s);
    const size_t &operator[](int i) const;
    size_t &operator[](int i);
    operator size_t() const;
    MatStep &operator=(size_t s);

    size_t *p;
    size_t buf[2];

  protected:
    MatStep &operator=(const MatStep &);
};

inline MatStep::MatStep()
{
    p = buf;
    p[0] = p[1] = 0;
}

inline MatStep::MatStep(size_t s)
{
    p = buf;
    p[0] = s;
    p[1] = 0;
}

inline const size_t &MatStep::operator[](int i) const
{
    return p[i];
}

inline size_t &MatStep::operator[](int i)
{
    return p[i];
}

inline MatStep::operator size_t() const
{
    LYCON_DbgAssert(p == buf);
    return buf[0];
}

inline MatStep &MatStep::operator=(size_t s)
{
    LYCON_DbgAssert(p == buf);
    buf[0] = s;
    return *this;
}
}
