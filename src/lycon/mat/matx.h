#pragma once

#include "lycon/types/traits.h"
#include "lycon/util/error.h"

namespace lycon
{
struct LYCON_EXPORTS Matx_AddOp
{
};
struct LYCON_EXPORTS Matx_SubOp
{
};
struct LYCON_EXPORTS Matx_ScaleOp
{
};
struct LYCON_EXPORTS Matx_MulOp
{
};
struct LYCON_EXPORTS Matx_DivOp
{
};
struct LYCON_EXPORTS Matx_MatMulOp
{
};
struct LYCON_EXPORTS Matx_TOp
{
};

template <typename _Tp, int m, int n>
class Matx
{
   public:
    enum
    {
        depth = DataType<_Tp>::depth,
        rows = m,
        cols = n,
        channels = rows * cols,
        type = LYCON_MAKETYPE(depth, channels),
        shortdim = (m < n ? m : n)
    };

    typedef _Tp value_type;
    typedef Matx<_Tp, m, n> mat_type;
    typedef Matx<_Tp, shortdim, 1> diag_type;

    //! default constructor
    Matx();

    Matx(_Tp v0);                                                                  //!< 1x1 matrix
    Matx(_Tp v0, _Tp v1);                                                          //!< 1x2 or 2x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2);                                                  //!< 1x3 or 3x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3);                                          //!< 1x4, 2x2 or 4x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4);                                  //!< 1x5 or 5x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5);                          //!< 1x6, 2x3, 3x2 or 6x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6);                  //!< 1x7 or 7x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7);          //!< 1x8, 2x4, 4x2 or 8x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8);  //!< 1x9, 3x3 or 9x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8,
         _Tp v9);  //!< 1x10, 2x5 or 5x2 or 10x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10,
         _Tp v11);  //!< 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12,
         _Tp v13);  //!< 1x14, 2x7, 7x2 or 14x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12,
         _Tp v13, _Tp v14, _Tp v15);  //!< 1x16, 4x4 or 16x1 matrix
    explicit Matx(const _Tp *vals);   //!< initialize from a plain array

    static Matx all(_Tp alpha);
    static Matx zeros();
    static Matx ones();
    static Matx eye();
    static Matx diag(const diag_type &d);
    static Matx randu(_Tp a, _Tp b);
    static Matx randn(_Tp a, _Tp b);

    //! dot product computed with the default precision
    _Tp dot(const Matx<_Tp, m, n> &v) const;

    //! dot product computed in double-precision arithmetics
    double ddot(const Matx<_Tp, m, n> &v) const;

    //! conversion to another data type
    template <typename T2>
    operator Matx<T2, m, n>() const;

    //! change the matrix shape
    template <int m1, int n1>
    Matx<_Tp, m1, n1> reshape() const;

    //! extract part of the matrix
    template <int m1, int n1>
    Matx<_Tp, m1, n1> get_minor(int i, int j) const;

    //! extract the matrix row
    Matx<_Tp, 1, n> row(int i) const;

    //! extract the matrix column
    Matx<_Tp, m, 1> col(int i) const;

    //! extract the matrix diagonal
    diag_type diag() const;

    //! transpose the matrix
    Matx<_Tp, n, m> t() const;

    //! multiply two matrices element-wise
    Matx<_Tp, m, n> mul(const Matx<_Tp, m, n> &a) const;

    //! divide two matrices element-wise
    Matx<_Tp, m, n> div(const Matx<_Tp, m, n> &a) const;

    //! element access
    const _Tp &operator()(int i, int j) const;
    _Tp &operator()(int i, int j);

    //! 1D element access
    const _Tp &operator()(int i) const;
    _Tp &operator()(int i);

    Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_AddOp);
    Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_SubOp);
    template <typename _T2>
    Matx(const Matx<_Tp, m, n> &a, _T2 alpha, Matx_ScaleOp);
    Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_MulOp);
    Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_DivOp);
    template <int l>
    Matx(const Matx<_Tp, m, l> &a, const Matx<_Tp, l, n> &b, Matx_MatMulOp);
    Matx(const Matx<_Tp, n, m> &a, Matx_TOp);

    _Tp val[m * n];  //< matrix elements
};

typedef Matx<float, 1, 2> Matx12f;
typedef Matx<double, 1, 2> Matx12d;
typedef Matx<float, 1, 3> Matx13f;
typedef Matx<double, 1, 3> Matx13d;
typedef Matx<float, 1, 4> Matx14f;
typedef Matx<double, 1, 4> Matx14d;
typedef Matx<float, 1, 6> Matx16f;
typedef Matx<double, 1, 6> Matx16d;

typedef Matx<float, 2, 1> Matx21f;
typedef Matx<double, 2, 1> Matx21d;
typedef Matx<float, 3, 1> Matx31f;
typedef Matx<double, 3, 1> Matx31d;
typedef Matx<float, 4, 1> Matx41f;
typedef Matx<double, 4, 1> Matx41d;
typedef Matx<float, 6, 1> Matx61f;
typedef Matx<double, 6, 1> Matx61d;

typedef Matx<float, 2, 2> Matx22f;
typedef Matx<double, 2, 2> Matx22d;
typedef Matx<float, 2, 3> Matx23f;
typedef Matx<double, 2, 3> Matx23d;
typedef Matx<float, 3, 2> Matx32f;
typedef Matx<double, 3, 2> Matx32d;

typedef Matx<float, 3, 3> Matx33f;
typedef Matx<double, 3, 3> Matx33d;

typedef Matx<float, 3, 4> Matx34f;
typedef Matx<double, 3, 4> Matx34d;
typedef Matx<float, 4, 3> Matx43f;
typedef Matx<double, 4, 3> Matx43d;

typedef Matx<float, 4, 4> Matx44f;
typedef Matx<double, 4, 4> Matx44d;
typedef Matx<float, 6, 6> Matx66f;
typedef Matx<double, 6, 6> Matx66d;

/*!
  traits
*/
template <typename _Tp, int m, int n>
class DataType<Matx<_Tp, m, n>>
{
   public:
    typedef Matx<_Tp, m, n> value_type;
    typedef Matx<typename DataType<_Tp>::work_type, m, n> work_type;
    typedef _Tp channel_type;
    typedef value_type vec_type;

    enum
    {
        generic_type = 0,
        depth = DataType<channel_type>::depth,
        channels = m * n,
        fmt = DataType<channel_type>::fmt + ((channels - 1) << 8),
        type = LYCON_MAKETYPE(depth, channels)
    };
};

/*
 Utility methods
*/
template <typename _Tp, int m>
static double determinant(const Matx<_Tp, m, m> &a);
template <typename _Tp, int m, int n>
static double trace(const Matx<_Tp, m, n> &a);
template <typename _Tp, int m, int n>
static double norm(const Matx<_Tp, m, n> &M);
template <typename _Tp, int m, int n>
static double norm(const Matx<_Tp, m, n> &M, int normType);

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx()
{
    for (int i = 0; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0)
{
    val[0] = v0;
    for (int i = 1; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1)
{
    LYCON_StaticAssert(channels >= 2, "Matx should have at least 2 elements.");
    val[0] = v0;
    val[1] = v1;
    for (int i = 2; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2)
{
    LYCON_StaticAssert(channels >= 3, "Matx should have at least 3 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    for (int i = 3; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3)
{
    LYCON_StaticAssert(channels >= 4, "Matx should have at least 4 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    for (int i = 4; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4)
{
    LYCON_StaticAssert(channels >= 5, "Matx should have at least 5 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    for (int i = 5; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5)
{
    LYCON_StaticAssert(channels >= 6, "Matx should have at least 6 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    for (int i = 6; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6)
{
    LYCON_StaticAssert(channels >= 7, "Matx should have at least 7 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    for (int i = 7; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7)
{
    LYCON_StaticAssert(channels >= 8, "Matx should have at least 8 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    for (int i = 8; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8)
{
    LYCON_StaticAssert(channels >= 9, "Matx should have at least 9 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    val[8] = v8;
    for (int i = 9; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9)
{
    LYCON_StaticAssert(channels >= 10, "Matx should have at least 10 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    val[8] = v8;
    val[9] = v9;
    for (int i = 10; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10,
                             _Tp v11)
{
    LYCON_StaticAssert(channels >= 12, "Matx should have at least 12 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    val[8] = v8;
    val[9] = v9;
    val[10] = v10;
    val[11] = v11;
    for (int i = 12; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10,
                             _Tp v11, _Tp v12, _Tp v13)
{
    LYCON_StaticAssert(channels == 14, "Matx should have at least 14 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    val[8] = v8;
    val[9] = v9;
    val[10] = v10;
    val[11] = v11;
    val[12] = v12;
    val[13] = v13;
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10,
                             _Tp v11, _Tp v12, _Tp v13, _Tp v14, _Tp v15)
{
    LYCON_StaticAssert(channels >= 16, "Matx should have at least 16 elements.");
    val[0] = v0;
    val[1] = v1;
    val[2] = v2;
    val[3] = v3;
    val[4] = v4;
    val[5] = v5;
    val[6] = v6;
    val[7] = v7;
    val[8] = v8;
    val[9] = v9;
    val[10] = v10;
    val[11] = v11;
    val[12] = v12;
    val[13] = v13;
    val[14] = v14;
    val[15] = v15;
    for (int i = 16; i < channels; i++) val[i] = _Tp(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const _Tp *values)
{
    for (int i = 0; i < channels; i++) val[i] = values[i];
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::all(_Tp alpha)
{
    Matx<_Tp, m, n> M;
    for (int i = 0; i < m * n; i++) M.val[i] = alpha;
    return M;
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::zeros()
{
    return all(0);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::ones()
{
    return all(1);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::eye()
{
    Matx<_Tp, m, n> M;
    for (int i = 0; i < shortdim; i++) M(i, i) = 1;
    return M;
}

template <typename _Tp, int m, int n>
inline _Tp Matx<_Tp, m, n>::dot(const Matx<_Tp, m, n> &M) const
{
    _Tp s = 0;
    for (int i = 0; i < channels; i++) s += val[i] * M.val[i];
    return s;
}

template <typename _Tp, int m, int n>
inline double Matx<_Tp, m, n>::ddot(const Matx<_Tp, m, n> &M) const
{
    double s = 0;
    for (int i = 0; i < channels; i++) s += (double)val[i] * M.val[i];
    return s;
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::diag(const typename Matx<_Tp, m, n>::diag_type &d)
{
    Matx<_Tp, m, n> M;
    for (int i = 0; i < shortdim; i++) M(i, i) = d(i, 0);
    return M;
}

template <typename _Tp, int m, int n>
template <typename T2>
inline Matx<_Tp, m, n>::operator Matx<T2, m, n>() const
{
    Matx<T2, m, n> M;
    for (int i = 0; i < m * n; i++) M.val[i] = saturate_cast<T2>(val[i]);
    return M;
}

template <typename _Tp, int m, int n>
template <int m1, int n1>
inline Matx<_Tp, m1, n1> Matx<_Tp, m, n>::reshape() const
{
    LYCON_StaticAssert(m1 * n1 == m * n, "Input and destnarion matrices must have the same number of elements");
    return (const Matx<_Tp, m1, n1> &)*this;
}

template <typename _Tp, int m, int n>
template <int m1, int n1>
inline Matx<_Tp, m1, n1> Matx<_Tp, m, n>::get_minor(int i, int j) const
{
    LYCON_DbgAssert(0 <= i && i + m1 <= m && 0 <= j && j + n1 <= n);
    Matx<_Tp, m1, n1> s;
    for (int di = 0; di < m1; di++)
        for (int dj = 0; dj < n1; dj++) s(di, dj) = (*this)(i + di, j + dj);
    return s;
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, 1, n> Matx<_Tp, m, n>::row(int i) const
{
    LYCON_DbgAssert((unsigned)i < (unsigned)m);
    return Matx<_Tp, 1, n>(&val[i * n]);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, 1> Matx<_Tp, m, n>::col(int j) const
{
    LYCON_DbgAssert((unsigned)j < (unsigned)n);
    Matx<_Tp, m, 1> v;
    for (int i = 0; i < m; i++) v.val[i] = val[i * n + j];
    return v;
}

template <typename _Tp, int m, int n>
inline typename Matx<_Tp, m, n>::diag_type Matx<_Tp, m, n>::diag() const
{
    diag_type d;
    for (int i = 0; i < shortdim; i++) d.val[i] = val[i * n + i];
    return d;
}

template <typename _Tp, int m, int n>
inline const _Tp &Matx<_Tp, m, n>::operator()(int i, int j) const
{
    LYCON_DbgAssert((unsigned)i < (unsigned)m && (unsigned)j < (unsigned)n);
    return this->val[i * n + j];
}

template <typename _Tp, int m, int n>
inline _Tp &Matx<_Tp, m, n>::operator()(int i, int j)
{
    LYCON_DbgAssert((unsigned)i < (unsigned)m && (unsigned)j < (unsigned)n);
    return val[i * n + j];
}

template <typename _Tp, int m, int n>
inline const _Tp &Matx<_Tp, m, n>::operator()(int i) const
{
    LYCON_StaticAssert(m == 1 || n == 1, "Single index indexation requires matrix to be a column or a row");
    LYCON_DbgAssert((unsigned)i < (unsigned)(m + n - 1));
    return val[i];
}

template <typename _Tp, int m, int n>
inline _Tp &Matx<_Tp, m, n>::operator()(int i)
{
    LYCON_StaticAssert(m == 1 || n == 1, "Single index indexation requires matrix to be a column or a row");
    LYCON_DbgAssert((unsigned)i < (unsigned)(m + n - 1));
    return val[i];
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_AddOp)
{
    for (int i = 0; i < channels; i++) val[i] = saturate_cast<_Tp>(a.val[i] + b.val[i]);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_SubOp)
{
    for (int i = 0; i < channels; i++) val[i] = saturate_cast<_Tp>(a.val[i] - b.val[i]);
}

template <typename _Tp, int m, int n>
template <typename _T2>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, n> &a, _T2 alpha, Matx_ScaleOp)
{
    for (int i = 0; i < channels; i++) val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_MulOp)
{
    for (int i = 0; i < channels; i++) val[i] = saturate_cast<_Tp>(a.val[i] * b.val[i]);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b, Matx_DivOp)
{
    for (int i = 0; i < channels; i++) val[i] = saturate_cast<_Tp>(a.val[i] / b.val[i]);
}

template <typename _Tp, int m, int n>
template <int l>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, m, l> &a, const Matx<_Tp, l, n> &b, Matx_MatMulOp)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            _Tp s = 0;
            for (int k = 0; k < l; k++) s += a(i, k) * b(k, j);
            val[i * n + j] = s;
        }
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n>::Matx(const Matx<_Tp, n, m> &a, Matx_TOp)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) val[i * n + j] = a(j, i);
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::mul(const Matx<_Tp, m, n> &a) const
{
    return Matx<_Tp, m, n>(*this, a, Matx_MulOp());
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, m, n> Matx<_Tp, m, n>::div(const Matx<_Tp, m, n> &a) const
{
    return Matx<_Tp, m, n>(*this, a, Matx_DivOp());
}

template <typename _Tp, int m, int n>
inline Matx<_Tp, n, m> Matx<_Tp, m, n>::t() const
{
    return Matx<_Tp, n, m>(*this, Matx_TOp());
}

template <typename _Tp1, typename _Tp2, int m, int n>
static inline Matx<_Tp1, m, n> &operator+=(Matx<_Tp1, m, n> &a, const Matx<_Tp2, m, n> &b)
{
    for (int i = 0; i < m * n; i++) a.val[i] = saturate_cast<_Tp1>(a.val[i] + b.val[i]);
    return a;
}

template <typename _Tp1, typename _Tp2, int m, int n>
static inline Matx<_Tp1, m, n> &operator-=(Matx<_Tp1, m, n> &a, const Matx<_Tp2, m, n> &b)
{
    for (int i = 0; i < m * n; i++) a.val[i] = saturate_cast<_Tp1>(a.val[i] - b.val[i]);
    return a;
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator+(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b)
{
    return Matx<_Tp, m, n>(a, b, Matx_AddOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator-(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b)
{
    return Matx<_Tp, m, n>(a, b, Matx_SubOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> &operator*=(Matx<_Tp, m, n> &a, int alpha)
{
    for (int i = 0; i < m * n; i++) a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> &operator*=(Matx<_Tp, m, n> &a, float alpha)
{
    for (int i = 0; i < m * n; i++) a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> &operator*=(Matx<_Tp, m, n> &a, double alpha)
{
    for (int i = 0; i < m * n; i++) a.val[i] = saturate_cast<_Tp>(a.val[i] * alpha);
    return a;
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(const Matx<_Tp, m, n> &a, int alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(const Matx<_Tp, m, n> &a, float alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(const Matx<_Tp, m, n> &a, double alpha)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(int alpha, const Matx<_Tp, m, n> &a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(float alpha, const Matx<_Tp, m, n> &a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator*(double alpha, const Matx<_Tp, m, n> &a)
{
    return Matx<_Tp, m, n>(a, alpha, Matx_ScaleOp());
}

template <typename _Tp, int m, int n>
static inline Matx<_Tp, m, n> operator-(const Matx<_Tp, m, n> &a)
{
    return Matx<_Tp, m, n>(a, -1, Matx_ScaleOp());
}

template <typename _Tp, int m, int n, int l>
static inline Matx<_Tp, m, n> operator*(const Matx<_Tp, m, l> &a, const Matx<_Tp, l, n> &b)
{
    return Matx<_Tp, m, n>(a, b, Matx_MatMulOp());
}

template <typename _Tp, int m, int n>
static inline bool operator==(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b)
{
    for (int i = 0; i < m * n; i++)
        if (a.val[i] != b.val[i]) return false;
    return true;
}

template <typename _Tp, int m, int n>
static inline bool operator!=(const Matx<_Tp, m, n> &a, const Matx<_Tp, m, n> &b)
{
    return !(a == b);
}
}
