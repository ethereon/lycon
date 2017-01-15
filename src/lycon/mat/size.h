#pragma once

#include "lycon/types.h"
#include "lycon/util/error.h"

namespace lycon
{
struct LYCON_EXPORTS MatSize
{
    explicit MatSize(int *_p);
    Size operator()() const;
    const int &operator[](int i) const;
    int &operator[](int i);
    operator const int *() const;
    bool operator==(const MatSize &sz) const;
    bool operator!=(const MatSize &sz) const;

    int *p;
};

inline MatSize::MatSize(int *_p) : p(_p)
{
}

inline Size MatSize::operator()() const
{
    LYCON_DbgAssert(p[-1] <= 2);
    return Size(p[1], p[0]);
}

inline const int &MatSize::operator[](int i) const
{
    return p[i];
}

inline int &MatSize::operator[](int i)
{
    return p[i];
}

inline MatSize::operator const int *() const
{
    return p;
}

inline bool MatSize::operator==(const MatSize &sz) const
{
    int d = p[-1];
    int dsz = sz.p[-1];
    if (d != dsz)
        return false;
    if (d == 2)
        return p[0] == sz.p[0] && p[1] == sz.p[1];

    for (int i = 0; i < d; i++)
        if (p[i] != sz.p[i])
            return false;
    return true;
}

inline bool MatSize::operator!=(const MatSize &sz) const
{
    return !(*this == sz);
}
}
