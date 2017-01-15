#pragma once

#include "lycon/defs.h"

namespace lycon
{

class LYCON_EXPORTS Range
{
  public:
    Range();
    Range(int _start, int _end);
    int size() const;
    bool empty() const;
    static Range all();

    int start, end;
};

inline Range::Range() : start(0), end(0)
{
}

inline Range::Range(int _start, int _end) : start(_start), end(_end)
{
}

inline int Range::size() const
{
    return end - start;
}

inline bool Range::empty() const
{
    return start == end;
}

inline Range Range::all()
{
    return Range(INT_MIN, INT_MAX);
}

static inline bool operator==(const Range &r1, const Range &r2)
{
    return r1.start == r2.start && r1.end == r2.end;
}

static inline bool operator!=(const Range &r1, const Range &r2)
{
    return !(r1 == r2);
}

static inline bool operator!(const Range &r)
{
    return r.start == r.end;
}

static inline Range operator&(const Range &r1, const Range &r2)
{
    Range r(std::max(r1.start, r2.start), std::min(r1.end, r2.end));
    r.end = std::max(r.end, r.start);
    return r;
}

static inline Range &operator&=(Range &r1, const Range &r2)
{
    r1 = r1 & r2;
    return r1;
}

static inline Range operator+(const Range &r1, int delta)
{
    return Range(r1.start + delta, r1.end + delta);
}

static inline Range operator+(int delta, const Range &r1)
{
    return Range(r1.start + delta, r1.end + delta);
}

static inline Range operator-(const Range &r1, int delta)
{
    return r1 + (-delta);
}
}
