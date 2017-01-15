#pragma once

#include "lycon/types/point.h"
#include "lycon/types/size.h"
#include "lycon/util/saturate_cast.h"

namespace lycon
{

template <typename _Tp> class Rect_
{
  public:
    typedef _Tp value_type;

    //! various constructors
    Rect_();
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
    Rect_(const Rect_ &r);
    Rect_(const Point_<_Tp> &org, const Size_<_Tp> &sz);
    Rect_(const Point_<_Tp> &pt1, const Point_<_Tp> &pt2);

    Rect_ &operator=(const Rect_ &r);
    //! the top-left corner
    Point_<_Tp> tl() const;
    //! the bottom-right corner
    Point_<_Tp> br() const;

    //! size (width, height) of the rectangle
    Size_<_Tp> size() const;
    //! area (width*height) of the rectangle
    _Tp area() const;

    //! conversion to another data type
    template <typename _Tp2> operator Rect_<_Tp2>() const;

    //! checks whether the rectangle contains the point
    bool contains(const Point_<_Tp> &pt) const;

    _Tp x, y, width, height; //< the top-left corner, as well as width and height of the rectangle
};

typedef Rect_<int> Rect2i;
typedef Rect_<float> Rect2f;
typedef Rect_<double> Rect2d;
typedef Rect2i Rect;

template <typename _Tp> inline Rect_<_Tp>::Rect_() : x(0), y(0), width(0), height(0)
{
}

template <typename _Tp> inline Rect_<_Tp>::Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height) : x(_x), y(_y), width(_width), height(_height)
{
}

template <typename _Tp> inline Rect_<_Tp>::Rect_(const Rect_<_Tp> &r) : x(r.x), y(r.y), width(r.width), height(r.height)
{
}

template <typename _Tp> inline Rect_<_Tp>::Rect_(const Point_<_Tp> &org, const Size_<_Tp> &sz) : x(org.x), y(org.y), width(sz.width), height(sz.height)
{
}

template <typename _Tp> inline Rect_<_Tp>::Rect_(const Point_<_Tp> &pt1, const Point_<_Tp> &pt2)
{
    x = std::min(pt1.x, pt2.x);
    y = std::min(pt1.y, pt2.y);
    width = std::max(pt1.x, pt2.x) - x;
    height = std::max(pt1.y, pt2.y) - y;
}

template <typename _Tp> inline Rect_<_Tp> &Rect_<_Tp>::operator=(const Rect_<_Tp> &r)
{
    x = r.x;
    y = r.y;
    width = r.width;
    height = r.height;
    return *this;
}

template <typename _Tp> inline Point_<_Tp> Rect_<_Tp>::tl() const
{
    return Point_<_Tp>(x, y);
}

template <typename _Tp> inline Point_<_Tp> Rect_<_Tp>::br() const
{
    return Point_<_Tp>(x + width, y + height);
}

template <typename _Tp> inline Size_<_Tp> Rect_<_Tp>::size() const
{
    return Size_<_Tp>(width, height);
}

template <typename _Tp> inline _Tp Rect_<_Tp>::area() const
{
    return width * height;
}

template <typename _Tp> template <typename _Tp2> inline Rect_<_Tp>::operator Rect_<_Tp2>() const
{
    return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
}

template <typename _Tp> inline bool Rect_<_Tp>::contains(const Point_<_Tp> &pt) const
{
    return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
}

template <typename _Tp> static inline Rect_<_Tp> &operator+=(Rect_<_Tp> &a, const Point_<_Tp> &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename _Tp> static inline Rect_<_Tp> &operator-=(Rect_<_Tp> &a, const Point_<_Tp> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <typename _Tp> static inline Rect_<_Tp> &operator+=(Rect_<_Tp> &a, const Size_<_Tp> &b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template <typename _Tp> static inline Rect_<_Tp> &operator-=(Rect_<_Tp> &a, const Size_<_Tp> &b)
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template <typename _Tp> static inline Rect_<_Tp> &operator&=(Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    _Tp x1 = std::max(a.x, b.x);
    _Tp y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect();
    return a;
}

template <typename _Tp> static inline Rect_<_Tp> &operator|=(Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    _Tp x1 = std::min(a.x, b.x);
    _Tp y1 = std::min(a.y, b.y);
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    return a;
}

template <typename _Tp> static inline bool operator==(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

template <typename _Tp> static inline bool operator!=(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    return a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height;
}

template <typename _Tp> static inline Rect_<_Tp> operator+(const Rect_<_Tp> &a, const Point_<_Tp> &b)
{
    return Rect_<_Tp>(a.x + b.x, a.y + b.y, a.width, a.height);
}

template <typename _Tp> static inline Rect_<_Tp> operator-(const Rect_<_Tp> &a, const Point_<_Tp> &b)
{
    return Rect_<_Tp>(a.x - b.x, a.y - b.y, a.width, a.height);
}

template <typename _Tp> static inline Rect_<_Tp> operator+(const Rect_<_Tp> &a, const Size_<_Tp> &b)
{
    return Rect_<_Tp>(a.x, a.y, a.width + b.width, a.height + b.height);
}

template <typename _Tp> static inline Rect_<_Tp> operator&(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

template <typename _Tp> static inline Rect_<_Tp> operator|(const Rect_<_Tp> &a, const Rect_<_Tp> &b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}
}
