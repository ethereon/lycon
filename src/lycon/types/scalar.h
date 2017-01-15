#pragma once

#include "lycon/defs.h"
#include "lycon/mat/vec.h"
#include "lycon/util/saturate_cast.h"

namespace lycon
{

template <typename _Tp> class Scalar_ : public Vec<_Tp, 4>
{
  public:
    //! various constructors
    Scalar_();
    Scalar_(_Tp v0, _Tp v1, _Tp v2 = 0, _Tp v3 = 0);
    Scalar_(_Tp v0);

    template <typename _Tp2, int cn> Scalar_(const Vec<_Tp2, cn> &v);

    //! returns a scalar with all elements set to v0
    static Scalar_<_Tp> all(_Tp v0);

    //! conversion to another data type
    template <typename T2> operator Scalar_<T2>() const;

    //! per-element product
    Scalar_<_Tp> mul(const Scalar_<_Tp> &a, double scale = 1) const;

    // returns (v0, -v1, -v2, -v3)
    Scalar_<_Tp> conj() const;

    // returns true iff v1 == v2 == v3 == 0
    bool isReal() const;
};

typedef Scalar_<double> Scalar;

template <typename _Tp> class DataType<Scalar_<_Tp>>
{
  public:
    typedef Scalar_<_Tp> value_type;
    typedef Scalar_<typename DataType<_Tp>::work_type> work_type;
    typedef _Tp channel_type;

    enum
    {
        generic_type = 0,
        depth = DataType<channel_type>::depth,
        channels = 4,
        fmt = DataType<channel_type>::fmt + ((channels - 1) << 8),
        type = LYCON_MAKETYPE(depth, channels)
    };

    typedef Vec<channel_type, channels> vec_type;
};

template <typename _Tp> inline Scalar_<_Tp>::Scalar_()
{
    this->val[0] = this->val[1] = this->val[2] = this->val[3] = 0;
}

template <typename _Tp> inline Scalar_<_Tp>::Scalar_(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    this->val[0] = v0;
    this->val[1] = v1;
    this->val[2] = v2;
    this->val[3] = v3;
}

template <typename _Tp> template <typename _Tp2, int cn> inline Scalar_<_Tp>::Scalar_(const Vec<_Tp2, cn> &v)
{
    int i;
    for (i = 0; i < (cn < 4 ? cn : 4); i++)
        this->val[i] = saturate_cast<_Tp>(v.val[i]);
    for (; i < 4; i++)
        this->val[i] = 0;
}

template <typename _Tp> inline Scalar_<_Tp>::Scalar_(_Tp v0)
{
    this->val[0] = v0;
    this->val[1] = this->val[2] = this->val[3] = 0;
}

template <typename _Tp> inline Scalar_<_Tp> Scalar_<_Tp>::all(_Tp v0)
{
    return Scalar_<_Tp>(v0, v0, v0, v0);
}

template <typename _Tp> inline Scalar_<_Tp> Scalar_<_Tp>::mul(const Scalar_<_Tp> &a, double scale) const
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(this->val[0] * a.val[0] * scale), saturate_cast<_Tp>(this->val[1] * a.val[1] * scale), saturate_cast<_Tp>(this->val[2] * a.val[2] * scale),
                        saturate_cast<_Tp>(this->val[3] * a.val[3] * scale));
}

template <typename _Tp> inline Scalar_<_Tp> Scalar_<_Tp>::conj() const
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(this->val[0]), saturate_cast<_Tp>(-this->val[1]), saturate_cast<_Tp>(-this->val[2]), saturate_cast<_Tp>(-this->val[3]));
}

template <typename _Tp> inline bool Scalar_<_Tp>::isReal() const
{
    return this->val[1] == 0 && this->val[2] == 0 && this->val[3] == 0;
}

template <typename _Tp> template <typename T2> inline Scalar_<_Tp>::operator Scalar_<T2>() const
{
    return Scalar_<T2>(saturate_cast<T2>(this->val[0]), saturate_cast<T2>(this->val[1]), saturate_cast<T2>(this->val[2]), saturate_cast<T2>(this->val[3]));
}

template <typename _Tp> static inline Scalar_<_Tp> &operator+=(Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    a.val[0] += b.val[0];
    a.val[1] += b.val[1];
    a.val[2] += b.val[2];
    a.val[3] += b.val[3];
    return a;
}

template <typename _Tp> static inline Scalar_<_Tp> &operator-=(Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    a.val[0] -= b.val[0];
    a.val[1] -= b.val[1];
    a.val[2] -= b.val[2];
    a.val[3] -= b.val[3];
    return a;
}

template <typename _Tp> static inline Scalar_<_Tp> &operator*=(Scalar_<_Tp> &a, _Tp v)
{
    a.val[0] *= v;
    a.val[1] *= v;
    a.val[2] *= v;
    a.val[3] *= v;
    return a;
}

template <typename _Tp> static inline bool operator==(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return a.val[0] == b.val[0] && a.val[1] == b.val[1] && a.val[2] == b.val[2] && a.val[3] == b.val[3];
}

template <typename _Tp> static inline bool operator!=(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return a.val[0] != b.val[0] || a.val[1] != b.val[1] || a.val[2] != b.val[2] || a.val[3] != b.val[3];
}

template <typename _Tp> static inline Scalar_<_Tp> operator+(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return Scalar_<_Tp>(a.val[0] + b.val[0], a.val[1] + b.val[1], a.val[2] + b.val[2], a.val[3] + b.val[3]);
}

template <typename _Tp> static inline Scalar_<_Tp> operator-(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(a.val[0] - b.val[0]), saturate_cast<_Tp>(a.val[1] - b.val[1]), saturate_cast<_Tp>(a.val[2] - b.val[2]), saturate_cast<_Tp>(a.val[3] - b.val[3]));
}

template <typename _Tp> static inline Scalar_<_Tp> operator*(const Scalar_<_Tp> &a, _Tp alpha)
{
    return Scalar_<_Tp>(a.val[0] * alpha, a.val[1] * alpha, a.val[2] * alpha, a.val[3] * alpha);
}

template <typename _Tp> static inline Scalar_<_Tp> operator*(_Tp alpha, const Scalar_<_Tp> &a)
{
    return a * alpha;
}

template <typename _Tp> static inline Scalar_<_Tp> operator-(const Scalar_<_Tp> &a)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(-a.val[0]), saturate_cast<_Tp>(-a.val[1]), saturate_cast<_Tp>(-a.val[2]), saturate_cast<_Tp>(-a.val[3]));
}

template <typename _Tp> static inline Scalar_<_Tp> operator*(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return Scalar_<_Tp>(saturate_cast<_Tp>(a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]), saturate_cast<_Tp>(a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]),
                        saturate_cast<_Tp>(a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]), saturate_cast<_Tp>(a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]));
}

template <typename _Tp> static inline Scalar_<_Tp> &operator*=(Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    a = a * b;
    return a;
}

template <typename _Tp> static inline Scalar_<_Tp> operator/(const Scalar_<_Tp> &a, _Tp alpha)
{
    return Scalar_<_Tp>(a.val[0] / alpha, a.val[1] / alpha, a.val[2] / alpha, a.val[3] / alpha);
}

template <typename _Tp> static inline Scalar_<float> operator/(const Scalar_<float> &a, float alpha)
{
    float s = 1 / alpha;
    return Scalar_<float>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template <typename _Tp> static inline Scalar_<double> operator/(const Scalar_<double> &a, double alpha)
{
    double s = 1 / alpha;
    return Scalar_<double>(a.val[0] * s, a.val[1] * s, a.val[2] * s, a.val[3] * s);
}

template <typename _Tp> static inline Scalar_<_Tp> &operator/=(Scalar_<_Tp> &a, _Tp alpha)
{
    a = a / alpha;
    return a;
}

template <typename _Tp> static inline Scalar_<_Tp> operator/(_Tp a, const Scalar_<_Tp> &b)
{
    _Tp s = a / (b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]);
    return b.conj() * s;
}

template <typename _Tp> static inline Scalar_<_Tp> operator/(const Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    return a * ((_Tp)1 / b);
}

template <typename _Tp> static inline Scalar_<_Tp> &operator/=(Scalar_<_Tp> &a, const Scalar_<_Tp> &b)
{
    a = a / b;
    return a;
}

template <typename _Tp> static inline Scalar operator*(const Matx<_Tp, 4, 4> &a, const Scalar &b)
{
    Matx<double, 4, 1> c((Matx<double, 4, 4>)a, b, Matx_MatMulOp());
    return reinterpret_cast<const Scalar &>(c);
}

template <> inline Scalar operator*(const Matx<double, 4, 4> &a, const Scalar &b)
{
    Matx<double, 4, 1> c(a, b, Matx_MatMulOp());
    return reinterpret_cast<const Scalar &>(c);
}
}
