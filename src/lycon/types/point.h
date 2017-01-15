#pragma once

#include "lycon/util/saturate_cast.h"

namespace lycon
{

template <typename _Tp> class Point_
{
  public:
    typedef _Tp value_type;

    // various constructors
    Point_();
    Point_(_Tp _x, _Tp _y);
    Point_(const Point_ &pt);

    Point_ &operator=(const Point_ &pt);
    //! conversion to another data type
    template <typename _Tp2> operator Point_<_Tp2>() const;

    //! dot product
    _Tp dot(const Point_ &pt) const;
    //! dot product computed in double-precision arithmetics
    double ddot(const Point_ &pt) const;
    //! cross-product
    double cross(const Point_ &pt) const;

    _Tp x, y; //< the point coordinates
};

typedef Point_<int> Point2i;
typedef Point_<int64> Point2l;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;
typedef Point2i Point;

template <typename _Tp> inline Point_<_Tp>::Point_() : x(0), y(0)
{
}

template <typename _Tp> inline Point_<_Tp>::Point_(_Tp _x, _Tp _y) : x(_x), y(_y)
{
}

template <typename _Tp> inline Point_<_Tp>::Point_(const Point_ &pt) : x(pt.x), y(pt.y)
{
}

template <typename _Tp> inline Point_<_Tp> &Point_<_Tp>::operator=(const Point_ &pt)
{
    x = pt.x;
    y = pt.y;
    return *this;
}

template <typename _Tp> template <typename _Tp2> inline Point_<_Tp>::operator Point_<_Tp2>() const
{
    return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
}

template <typename _Tp> inline _Tp Point_<_Tp>::dot(const Point_ &pt) const
{
    return saturate_cast<_Tp>(x * pt.x + y * pt.y);
}

template <typename _Tp> inline double Point_<_Tp>::ddot(const Point_ &pt) const
{
    return (double)x * pt.x + (double)y * pt.y;
}

template <typename _Tp> inline double Point_<_Tp>::cross(const Point_ &pt) const
{
    return (double)x * pt.y - (double)y * pt.x;
}

template <typename _Tp> static inline Point_<_Tp> &operator+=(Point_<_Tp> &a, const Point_<_Tp> &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator-=(Point_<_Tp> &a, const Point_<_Tp> &b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator*=(Point_<_Tp> &a, int b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator*=(Point_<_Tp> &a, float b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator*=(Point_<_Tp> &a, double b)
{
    a.x = saturate_cast<_Tp>(a.x * b);
    a.y = saturate_cast<_Tp>(a.y * b);
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator/=(Point_<_Tp> &a, int b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator/=(Point_<_Tp> &a, float b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template <typename _Tp> static inline Point_<_Tp> &operator/=(Point_<_Tp> &a, double b)
{
    a.x = saturate_cast<_Tp>(a.x / b);
    a.y = saturate_cast<_Tp>(a.y / b);
    return a;
}

template <typename _Tp> static inline double norm(const Point_<_Tp> &pt)
{
    return std::sqrt((double)pt.x * pt.x + (double)pt.y * pt.y);
}

template <typename _Tp> static inline bool operator==(const Point_<_Tp> &a, const Point_<_Tp> &b)
{
    return a.x == b.x && a.y == b.y;
}

template <typename _Tp> static inline bool operator!=(const Point_<_Tp> &a, const Point_<_Tp> &b)
{
    return a.x != b.x || a.y != b.y;
}

template <typename _Tp> static inline Point_<_Tp> operator+(const Point_<_Tp> &a, const Point_<_Tp> &b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(a.x + b.x), saturate_cast<_Tp>(a.y + b.y));
}

template <typename _Tp> static inline Point_<_Tp> operator-(const Point_<_Tp> &a, const Point_<_Tp> &b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(a.x - b.x), saturate_cast<_Tp>(a.y - b.y));
}

template <typename _Tp> static inline Point_<_Tp> operator-(const Point_<_Tp> &a)
{
    return Point_<_Tp>(saturate_cast<_Tp>(-a.x), saturate_cast<_Tp>(-a.y));
}

template <typename _Tp> static inline Point_<_Tp> operator*(const Point_<_Tp> &a, int b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(a.x * b), saturate_cast<_Tp>(a.y * b));
}

template <typename _Tp> static inline Point_<_Tp> operator*(int a, const Point_<_Tp> &b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a));
}

template <typename _Tp> static inline Point_<_Tp> operator*(const Point_<_Tp> &a, float b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(a.x * b), saturate_cast<_Tp>(a.y * b));
}

template <typename _Tp> static inline Point_<_Tp> operator*(float a, const Point_<_Tp> &b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a));
}

template <typename _Tp> static inline Point_<_Tp> operator*(const Point_<_Tp> &a, double b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(a.x * b), saturate_cast<_Tp>(a.y * b));
}

template <typename _Tp> static inline Point_<_Tp> operator*(double a, const Point_<_Tp> &b)
{
    return Point_<_Tp>(saturate_cast<_Tp>(b.x * a), saturate_cast<_Tp>(b.y * a));
}

template <typename _Tp> static inline Point_<_Tp> operator/(const Point_<_Tp> &a, int b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template <typename _Tp> static inline Point_<_Tp> operator/(const Point_<_Tp> &a, float b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template <typename _Tp> static inline Point_<_Tp> operator/(const Point_<_Tp> &a, double b)
{
    Point_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}
}
