#pragma once

namespace lycon
{

template <typename _Tp> class Size_
{
  public:
    typedef _Tp value_type;

    //! various constructors
    Size_();
    Size_(_Tp _width, _Tp _height);
    Size_(const Size_ &sz);

    Size_ &operator=(const Size_ &sz);
    //! the area (width*height)
    _Tp area() const;

    //! conversion of another data type.
    template <typename _Tp2> operator Size_<_Tp2>() const;

    _Tp width, height; // the width and the height
};

typedef Size_<int> Size2i;
typedef Size_<int64> Size2l;
typedef Size_<float> Size2f;
typedef Size_<double> Size2d;
typedef Size2i Size;

template <typename _Tp> inline Size_<_Tp>::Size_() : width(0), height(0)
{
}

template <typename _Tp> inline Size_<_Tp>::Size_(_Tp _width, _Tp _height) : width(_width), height(_height)
{
}

template <typename _Tp> inline Size_<_Tp>::Size_(const Size_ &sz) : width(sz.width), height(sz.height)
{
}

template <typename _Tp> template <typename _Tp2> inline Size_<_Tp>::operator Size_<_Tp2>() const
{
    return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
}

template <typename _Tp> inline Size_<_Tp> &Size_<_Tp>::operator=(const Size_<_Tp> &sz)
{
    width = sz.width;
    height = sz.height;
    return *this;
}

template <typename _Tp> inline _Tp Size_<_Tp>::area() const
{
    return width * height;
}

template <typename _Tp> static inline Size_<_Tp> &operator*=(Size_<_Tp> &a, _Tp b)
{
    a.width *= b;
    a.height *= b;
    return a;
}

template <typename _Tp> static inline Size_<_Tp> operator*(const Size_<_Tp> &a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp *= b;
    return tmp;
}

template <typename _Tp> static inline Size_<_Tp> &operator/=(Size_<_Tp> &a, _Tp b)
{
    a.width /= b;
    a.height /= b;
    return a;
}

template <typename _Tp> static inline Size_<_Tp> operator/(const Size_<_Tp> &a, _Tp b)
{
    Size_<_Tp> tmp(a);
    tmp /= b;
    return tmp;
}

template <typename _Tp> static inline Size_<_Tp> &operator+=(Size_<_Tp> &a, const Size_<_Tp> &b)
{
    a.width += b.width;
    a.height += b.height;
    return a;
}

template <typename _Tp> static inline Size_<_Tp> operator+(const Size_<_Tp> &a, const Size_<_Tp> &b)
{
    Size_<_Tp> tmp(a);
    tmp += b;
    return tmp;
}

template <typename _Tp> static inline Size_<_Tp> &operator-=(Size_<_Tp> &a, const Size_<_Tp> &b)
{
    a.width -= b.width;
    a.height -= b.height;
    return a;
}

template <typename _Tp> static inline Size_<_Tp> operator-(const Size_<_Tp> &a, const Size_<_Tp> &b)
{
    Size_<_Tp> tmp(a);
    tmp -= b;
    return tmp;
}

template <typename _Tp> static inline bool operator==(const Size_<_Tp> &a, const Size_<_Tp> &b)
{
    return a.width == b.width && a.height == b.height;
}

template <typename _Tp> static inline bool operator!=(const Size_<_Tp> &a, const Size_<_Tp> &b)
{
    return !(a == b);
}
}
