#pragma once

#include "lycon/mat/matx.h"

namespace lycon
{

template <typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1>
{
  public:
    typedef _Tp value_type;
    enum
    {
        depth = Matx<_Tp, cn, 1>::depth,
        channels = cn,
        type = LYCON_MAKETYPE(depth, channels)
    };

    //! default constructor
    Vec();

    Vec(_Tp v0);                                                                                                             //!< 1-element vector constructor
    Vec(_Tp v0, _Tp v1);                                                                                                     //!< 2-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2);                                                                                             //!< 3-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3);                                                                                     //!< 4-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4);                                                                             //!< 5-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5);                                                                     //!< 6-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6);                                                             //!< 7-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7);                                                     //!< 8-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8);                                             //!< 9-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9);                                     //!< 10-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13); //!< 14-element vector constructor
    explicit Vec(const _Tp *values);

    Vec(const Vec<_Tp, cn> &v);

    static Vec all(_Tp alpha);

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn> &v) const;

    /*!
      cross product of the two 3D vectors.

      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec &v) const;
    //! conversion to another data type
    template <typename T2> operator Vec<T2, cn>() const;

    /*! element access */
    const _Tp &operator[](int i) const;
    _Tp &operator[](int i);
    const _Tp &operator()(int i) const;
    _Tp &operator()(int i);

    Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_AddOp);
    Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_SubOp);
    template <typename _T2> Vec(const Matx<_Tp, cn, 1> &a, _T2 alpha, Matx_ScaleOp);
};

typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;

template <typename _Tp, int cn> class DataType<Vec<_Tp, cn>>
{
  public:
    typedef Vec<_Tp, cn> value_type;
    typedef Vec<typename DataType<_Tp>::work_type, cn> work_type;
    typedef _Tp channel_type;
    typedef value_type vec_type;

    enum
    {
        generic_type = 0,
        depth = DataType<channel_type>::depth,
        channels = cn,
        fmt = DataType<channel_type>::fmt + ((channels - 1) << 8),
        type = LYCON_MAKETYPE(depth, channels)
    };
};

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec()
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0) : Matx<_Tp, cn, 1>(v0)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1) : Matx<_Tp, cn, 1>(v0, v1)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2) : Matx<_Tp, cn, 1>(v0, v1, v2)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3) : Matx<_Tp, cn, 1>(v0, v1, v2, v3)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9) : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9)
{
}

template <typename _Tp, int cn>
inline Vec<_Tp, cn>::Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13)
    : Matx<_Tp, cn, 1>(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(const _Tp *values) : Matx<_Tp, cn, 1>(values)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(const Vec<_Tp, cn> &m) : Matx<_Tp, cn, 1>(m.val)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_AddOp op) : Matx<_Tp, cn, 1>(a, b, op)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1> &a, const Matx<_Tp, cn, 1> &b, Matx_SubOp op) : Matx<_Tp, cn, 1>(a, b, op)
{
}

template <typename _Tp, int cn> template <typename _T2> inline Vec<_Tp, cn>::Vec(const Matx<_Tp, cn, 1> &a, _T2 alpha, Matx_ScaleOp op) : Matx<_Tp, cn, 1>(a, alpha, op)
{
}

template <typename _Tp, int cn> inline Vec<_Tp, cn> Vec<_Tp, cn>::all(_Tp alpha)
{
    Vec v;
    for (int i = 0; i < cn; i++)
        v.val[i] = alpha;
    return v;
}

template <typename _Tp, int cn> inline Vec<_Tp, cn> Vec<_Tp, cn>::mul(const Vec<_Tp, cn> &v) const
{
    Vec<_Tp, cn> w;
    for (int i = 0; i < cn; i++)
        w.val[i] = saturate_cast<_Tp>(this->val[i] * v.val[i]);
    return w;
}

template <typename _Tp, int cn> inline Vec<_Tp, cn> Vec<_Tp, cn>::cross(const Vec<_Tp, cn> &) const
{
    LYCON_StaticAssert(cn == 3, "for arbitrary-size vector there is no cross-product defined");
    return Vec<_Tp, cn>();
}

template <> inline Vec<float, 3> Vec<float, 3>::cross(const Vec<float, 3> &v) const
{
    return Vec<float, 3>(this->val[1] * v.val[2] - this->val[2] * v.val[1], this->val[2] * v.val[0] - this->val[0] * v.val[2], this->val[0] * v.val[1] - this->val[1] * v.val[0]);
}

template <> inline Vec<double, 3> Vec<double, 3>::cross(const Vec<double, 3> &v) const
{
    return Vec<double, 3>(this->val[1] * v.val[2] - this->val[2] * v.val[1], this->val[2] * v.val[0] - this->val[0] * v.val[2], this->val[0] * v.val[1] - this->val[1] * v.val[0]);
}

template <typename _Tp, int cn> template <typename T2> inline Vec<_Tp, cn>::operator Vec<T2, cn>() const
{
    Vec<T2, cn> v;
    for (int i = 0; i < cn; i++)
        v.val[i] = saturate_cast<T2>(this->val[i]);
    return v;
}

template <typename _Tp, int cn> inline const _Tp &Vec<_Tp, cn>::operator[](int i) const
{
    LYCON_DbgAssert((unsigned)i < (unsigned)cn);
    return this->val[i];
}

template <typename _Tp, int cn> inline _Tp &Vec<_Tp, cn>::operator[](int i)
{
    LYCON_DbgAssert((unsigned)i < (unsigned)cn);
    return this->val[i];
}

template <typename _Tp, int cn> inline const _Tp &Vec<_Tp, cn>::operator()(int i) const
{
    LYCON_DbgAssert((unsigned)i < (unsigned)cn);
    return this->val[i];
}

template <typename _Tp, int cn> inline _Tp &Vec<_Tp, cn>::operator()(int i)
{
    LYCON_DbgAssert((unsigned)i < (unsigned)cn);
    return this->val[i];
}

template <typename _Tp1, typename _Tp2, int cn> static inline Vec<_Tp1, cn> &operator+=(Vec<_Tp1, cn> &a, const Vec<_Tp2, cn> &b)
{
    for (int i = 0; i < cn; i++)
        a.val[i] = saturate_cast<_Tp1>(a.val[i] + b.val[i]);
    return a;
}

template <typename _Tp1, typename _Tp2, int cn> static inline Vec<_Tp1, cn> &operator-=(Vec<_Tp1, cn> &a, const Vec<_Tp2, cn> &b)
{
    for (int i = 0; i < cn; i++)
        a.val[i] = saturate_cast<_Tp1>(a.val[i] - b.val[i]);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator+(const Vec<_Tp, cn> &a, const Vec<_Tp, cn> &b)
{
    return Vec<_Tp, cn>(a, b, Matx_AddOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator-(const Vec<_Tp, cn> &a, const Vec<_Tp, cn> &b)
{
    return Vec<_Tp, cn>(a, b, Matx_SubOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator*=(Vec<_Tp, cn> &a, int alpha)
{
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * alpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator*=(Vec<_Tp, cn> &a, float alpha)
{
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * alpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator*=(Vec<_Tp, cn> &a, double alpha)
{
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * alpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator/=(Vec<_Tp, cn> &a, int alpha)
{
    double ialpha = 1. / alpha;
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * ialpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator/=(Vec<_Tp, cn> &a, float alpha)
{
    float ialpha = 1.f / alpha;
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * ialpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> &operator/=(Vec<_Tp, cn> &a, double alpha)
{
    double ialpha = 1. / alpha;
    for (int i = 0; i < cn; i++)
        a[i] = saturate_cast<_Tp>(a[i] * ialpha);
    return a;
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(const Vec<_Tp, cn> &a, int alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(int alpha, const Vec<_Tp, cn> &a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(const Vec<_Tp, cn> &a, float alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(float alpha, const Vec<_Tp, cn> &a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(const Vec<_Tp, cn> &a, double alpha)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator*(double alpha, const Vec<_Tp, cn> &a)
{
    return Vec<_Tp, cn>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator/(const Vec<_Tp, cn> &a, int alpha)
{
    return Vec<_Tp, cn>(a, 1. / alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator/(const Vec<_Tp, cn> &a, float alpha)
{
    return Vec<_Tp, cn>(a, 1.f / alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator/(const Vec<_Tp, cn> &a, double alpha)
{
    return Vec<_Tp, cn>(a, 1. / alpha, Matx_ScaleOp());
}

template <typename _Tp, int cn> static inline Vec<_Tp, cn> operator-(const Vec<_Tp, cn> &a)
{
    Vec<_Tp, cn> t;
    for (int i = 0; i < cn; i++)
        t.val[i] = saturate_cast<_Tp>(-a.val[i]);
    return t;
}

template <typename _Tp> inline Vec<_Tp, 4> operator*(const Vec<_Tp, 4> &v1, const Vec<_Tp, 4> &v2)
{
    return Vec<_Tp, 4>(saturate_cast<_Tp>(v1[0] * v2[0] - v1[1] * v2[1] - v1[2] * v2[2] - v1[3] * v2[3]), saturate_cast<_Tp>(v1[0] * v2[1] + v1[1] * v2[0] + v1[2] * v2[3] - v1[3] * v2[2]),
                       saturate_cast<_Tp>(v1[0] * v2[2] - v1[1] * v2[3] + v1[2] * v2[0] + v1[3] * v2[1]), saturate_cast<_Tp>(v1[0] * v2[3] + v1[1] * v2[2] - v1[2] * v2[1] + v1[3] * v2[0]));
}

template <typename _Tp> inline Vec<_Tp, 4> &operator*=(Vec<_Tp, 4> &v1, const Vec<_Tp, 4> &v2)
{
    v1 = v1 * v2;
    return v1;
}

template <typename _Tp, int m, int n> static inline Vec<_Tp, m> operator*(const Matx<_Tp, m, n> &a, const Vec<_Tp, n> &b)
{
    Matx<_Tp, m, 1> c(a, b, Matx_MatMulOp());
    return (const Vec<_Tp, m> &)(c);
}
}
