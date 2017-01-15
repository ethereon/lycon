#include "lycon/mat/io_array.h"
#include "lycon/mat/mat.h"

namespace lycon
{
Mat _InputArray::getMat_(int i) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if (k == MAT)
    {
        const Mat* m = (const Mat*)obj;
        if (i < 0)
            return *m;
        return m->row(i);
    }

    if (k == MATX)
    {
        LYCON_ASSERT(i < 0);
        return Mat(sz, flags, obj);
    }

    if (k == STD_VECTOR)
    {
        LYCON_ASSERT(i < 0);
        int t = LYCON_MAT_TYPE(flags);
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;

        return !v.empty() ? Mat(size(), t, (void*)&v[0]) : Mat();
    }

    if (k == STD_BOOL_VECTOR)
    {
        LYCON_ASSERT(i < 0);
        int t = LYCON_8U;
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        int j, n = (int)v.size();
        if (n == 0)
            return Mat();
        Mat m(1, n, t);
        uchar* dst = m.data;
        for (j = 0; j < n; j++)
            dst[j] = (uchar)v[j];
        return m;
    }

    if (k == NONE)
        return Mat();

    if (k == STD_VECTOR_VECTOR)
    {
        int t = type(i);
        const std::vector<std::vector<uchar>>& vv = *(const std::vector<std::vector<uchar>>*)obj;
        LYCON_ASSERT(0 <= i && i < (int)vv.size());
        const std::vector<uchar>& v = vv[i];

        return !v.empty() ? Mat(size(i), t, (void*)&v[0]) : Mat();
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& v = *(const std::vector<Mat>*)obj;
        LYCON_ASSERT(0 <= i && i < (int)v.size());

        return v[i];
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return Mat();
}

void _InputArray::getMatVector(std::vector<Mat>& mv) const
{
    int k = kind();
    int accessFlags = flags & ACCESS_MASK;

    if (k == MAT)
    {
        const Mat& m = *(const Mat*)obj;
        int n = (int)m.size[0];
        mv.resize(n);

        for (int i = 0; i < n; i++)
            mv[i] = m.dims == 2 ? Mat(1, m.cols, m.type(), (void*)m.ptr(i))
                                : Mat(m.dims - 1, &m.size[1], m.type(), (void*)m.ptr(i), &m.step[1]);
        return;
    }

    if (k == MATX)
    {
        size_t n = sz.height, esz = LYCON_ELEM_SIZE(flags);
        mv.resize(n);

        for (size_t i = 0; i < n; i++)
            mv[i] = Mat(1, sz.width, LYCON_MAT_TYPE(flags), (uchar*)obj + esz * sz.width * i);
        return;
    }

    if (k == STD_VECTOR)
    {
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;

        size_t n = v.size(), esz = LYCON_ELEM_SIZE(flags);
        int t = LYCON_MAT_DEPTH(flags), cn = LYCON_MAT_CN(flags);
        mv.resize(n);

        for (size_t i = 0; i < n; i++)
            mv[i] = Mat(1, cn, t, (void*)(&v[0] + esz * i));
        return;
    }

    if (k == NONE)
    {
        mv.clear();
        return;
    }

    if (k == STD_VECTOR_VECTOR)
    {
        const std::vector<std::vector<uchar>>& vv = *(const std::vector<std::vector<uchar>>*)obj;
        int n = (int)vv.size();
        int t = LYCON_MAT_TYPE(flags);
        mv.resize(n);

        for (int i = 0; i < n; i++)
        {
            const std::vector<uchar>& v = vv[i];
            mv[i] = Mat(size(i), t, (void*)&v[0]);
        }
        return;
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& v = *(const std::vector<Mat>*)obj;
        size_t n = v.size();
        mv.resize(n);

        for (size_t i = 0; i < n; i++)
            mv[i] = v[i];
        return;
    }

    LYCON_ERROR("Unknown/unsupported array type");
}

int _InputArray::kind() const
{
    return flags & KIND_MASK;
}

int _InputArray::rows(int i) const
{
    return size(i).height;
}

int _InputArray::cols(int i) const
{
    return size(i).width;
}

Size _InputArray::size(int i) const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        return ((const Mat*)obj)->size();
    }

    if (k == MATX)
    {
        LYCON_ASSERT(i < 0);
        return sz;
    }

    if (k == STD_VECTOR)
    {
        LYCON_ASSERT(i < 0);
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;
        const std::vector<int>& iv = *(const std::vector<int>*)obj;
        size_t szb = v.size(), szi = iv.size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb / LYCON_ELEM_SIZE(flags)), 1);
    }

    if (k == STD_BOOL_VECTOR)
    {
        LYCON_ASSERT(i < 0);
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        return Size((int)v.size(), 1);
    }

    if (k == NONE)
        return Size();

    if (k == STD_VECTOR_VECTOR)
    {
        const std::vector<std::vector<uchar>>& vv = *(const std::vector<std::vector<uchar>>*)obj;
        if (i < 0)
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        LYCON_ASSERT(i < (int)vv.size());
        const std::vector<std::vector<int>>& ivv = *(const std::vector<std::vector<int>>*)obj;

        size_t szb = vv[i].size(), szi = ivv[i].size();
        return szb == szi ? Size((int)szb, 1) : Size((int)(szb / LYCON_ELEM_SIZE(flags)), 1);
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (i < 0)
            return vv.empty() ? Size() : Size((int)vv.size(), 1);
        LYCON_ASSERT(i < (int)vv.size());

        return vv[i].size();
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return Size();
}

int _InputArray::sizend(int* arrsz, int i) const
{
    int j, d = 0, k = kind();

    if (k == NONE)
        ;
    else if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        const Mat& m = *(const Mat*)obj;
        d = m.dims;
        if (arrsz)
            for (j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else if (k == STD_VECTOR_MAT && i >= 0)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        LYCON_ASSERT(i < (int)vv.size());
        const Mat& m = vv[i];
        d = m.dims;
        if (arrsz)
            for (j = 0; j < d; j++)
                arrsz[j] = m.size.p[j];
    }
    else
    {
        Size sz2d = size(i);
        d = 2;
        if (arrsz)
        {
            arrsz[0] = sz2d.height;
            arrsz[1] = sz2d.width;
        }
    }

    return d;
}

bool _InputArray::sameSize(const _InputArray& arr) const
{
    int k1 = kind(), k2 = arr.kind();
    Size sz1;

    if (k1 == MAT)
    {
        const Mat* m = ((const Mat*)obj);
        if (k2 == MAT)
            return m->size == ((const Mat*)arr.obj)->size;
        if (m->dims > 2)
            return false;
        sz1 = m->size();
    }
    else
        sz1 = size();
    if (arr.dims() > 2)
        return false;
    return sz1 == arr.size();
}

int _InputArray::dims(int i) const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        return ((const Mat*)obj)->dims;
    }

    if (k == MATX)
    {
        LYCON_ASSERT(i < 0);
        return 2;
    }

    if (k == STD_VECTOR || k == STD_BOOL_VECTOR)
    {
        LYCON_ASSERT(i < 0);
        return 2;
    }

    if (k == NONE)
        return 0;

    if (k == STD_VECTOR_VECTOR)
    {
        const std::vector<std::vector<uchar>>& vv = *(const std::vector<std::vector<uchar>>*)obj;
        if (i < 0)
            return 1;
        LYCON_ASSERT(i < (int)vv.size());
        return 2;
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (i < 0)
            return 1;
        LYCON_ASSERT(i < (int)vv.size());

        return vv[i].dims;
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return 0;
}

size_t _InputArray::total(int i) const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        return ((const Mat*)obj)->total();
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (i < 0)
            return vv.size();

        LYCON_ASSERT(i < (int)vv.size());
        return vv[i].total();
    }

    return size(i).area();
}

int _InputArray::type(int i) const
{
    int k = kind();

    if (k == MAT)
        return ((const Mat*)obj)->type();

    if (k == MATX || k == STD_VECTOR || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR)
        return LYCON_MAT_TYPE(flags);

    if (k == NONE)
        return -1;

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (vv.empty())
        {
            LYCON_ASSERT((flags & FIXED_TYPE) != 0);
            return LYCON_MAT_TYPE(flags);
        }
        LYCON_ASSERT(i < (int)vv.size());
        return vv[i >= 0 ? i : 0].type();
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return 0;
}

int _InputArray::depth(int i) const
{
    return LYCON_MAT_DEPTH(type(i));
}

int _InputArray::channels(int i) const
{
    return LYCON_MAT_CN(type(i));
}

bool _InputArray::empty() const
{
    int k = kind();

    if (k == MAT)
        return ((const Mat*)obj)->empty();

    if (k == MATX)
        return false;

    if (k == STD_VECTOR)
    {
        const std::vector<uchar>& v = *(const std::vector<uchar>*)obj;
        return v.empty();
    }

    if (k == STD_BOOL_VECTOR)
    {
        const std::vector<bool>& v = *(const std::vector<bool>*)obj;
        return v.empty();
    }

    if (k == NONE)
        return true;

    if (k == STD_VECTOR_VECTOR)
    {
        const std::vector<std::vector<uchar>>& vv = *(const std::vector<std::vector<uchar>>*)obj;
        return vv.empty();
    }

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        return vv.empty();
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return true;
}

bool _InputArray::isContinuous(int i) const
{
    int k = kind();

    if (k == MAT)
        return i < 0 ? ((const Mat*)obj)->isContinuous() : true;

    if (k == MATX || k == STD_VECTOR || k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR)
        return true;

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        LYCON_ASSERT((size_t)i < vv.size());
        return vv[i].isContinuous();
    }

    LYCON_ERROR("Unknown/unsupported array type");
    return false;
}

bool _InputArray::isSubmatrix(int i) const
{
    int k = kind();

    if (k == MAT)
        return i < 0 ? ((const Mat*)obj)->isSubmatrix() : false;

    if (k == MATX || k == STD_VECTOR || k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR)
        return false;

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        LYCON_ASSERT((size_t)i < vv.size());
        return vv[i].isSubmatrix();
    }

    LYCON_ERROR("Not Implemented");
    return false;
}

size_t _InputArray::offset(int i) const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        const Mat* const m = ((const Mat*)obj);
        return (size_t)(m->ptr() - m->datastart);
    }

    if (k == EXPR || k == MATX || k == STD_VECTOR || k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR)
        return 0;

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (i < 0)
            return 1;
        LYCON_ASSERT(i < (int)vv.size());

        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }

    LYCON_ERROR("Not Implemented");
    return 0;
}

size_t _InputArray::step(int i) const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        return ((const Mat*)obj)->step;
    }

    if (k == EXPR || k == MATX || k == STD_VECTOR || k == NONE || k == STD_VECTOR_VECTOR || k == STD_BOOL_VECTOR)
        return 0;

    if (k == STD_VECTOR_MAT)
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        if (i < 0)
            return 1;
        LYCON_ASSERT(i < (int)vv.size());
        return vv[i].step;
    }

    LYCON_ERROR("Not Implemented");
    return 0;
}

void _InputArray::copyTo(const _OutputArray& arr) const
{
    int k = kind();

    if (k == NONE)
        arr.release();
    else if (k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR)
    {
        Mat m = getMat();
        m.copyTo(arr);
    }
    else
        LYCON_ERROR("Not Implemented");
}

void _InputArray::copyTo(const _OutputArray& arr, const _InputArray& mask) const
{
    int k = kind();

    if (k == NONE)
        arr.release();
    else if (k == MAT || k == MATX || k == STD_VECTOR || k == STD_BOOL_VECTOR)
    {
        Mat m = getMat();
        m.copyTo(arr, mask);
    }
    else
        LYCON_ERROR("Not Implemented");
}

bool _OutputArray::fixedSize() const
{
    return (flags & FIXED_SIZE) == FIXED_SIZE;
}

bool _OutputArray::fixedType() const
{
    return (flags & FIXED_TYPE) == FIXED_TYPE;
}

void _OutputArray::create(Size _sz, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if (k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0)
    {
        LYCON_ASSERT(!fixedSize() || ((Mat*)obj)->size.operator()() == _sz);
        LYCON_ASSERT(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_sz, mtype);
        return;
    }
    int sizes[] = {_sz.height, _sz.width};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int _rows, int _cols, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    if (k == MAT && i < 0 && !allowTransposed && fixedDepthMask == 0)
    {
        LYCON_ASSERT(!fixedSize() || ((Mat*)obj)->size.operator()() == Size(_cols, _rows));
        LYCON_ASSERT(!fixedType() || ((Mat*)obj)->type() == mtype);
        ((Mat*)obj)->create(_rows, _cols, mtype);
        return;
    }
    int sizes[] = {_rows, _cols};
    create(2, sizes, mtype, i, allowTransposed, fixedDepthMask);
}

void _OutputArray::create(int d, const int* sizes, int mtype, int i, bool allowTransposed, int fixedDepthMask) const
{
    int k = kind();
    mtype = LYCON_MAT_TYPE(mtype);

    if (k == MAT)
    {
        LYCON_ASSERT(i < 0);
        Mat& m = *(Mat*)obj;
        if (allowTransposed)
        {
            if (!m.isContinuous())
            {
                LYCON_ASSERT(!fixedType() && !fixedSize());
                m.release();
            }

            if (d == 2 && m.dims == 2 && m.data && m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0])
                return;
        }

        if (fixedType())
        {
            if (LYCON_MAT_CN(mtype) == m.channels() && ((1 << LYCON_MAT_TYPE(flags)) & fixedDepthMask) != 0)
                mtype = m.type();
            else
                LYCON_ASSERT(LYCON_MAT_TYPE(mtype) == m.type());
        }
        if (fixedSize())
        {
            LYCON_ASSERT(m.dims == d);
            for (int j = 0; j < d; ++j)
                LYCON_ASSERT(m.size[j] == sizes[j]);
        }
        m.create(d, sizes, mtype);
        return;
    }

    if (k == MATX)
    {
        LYCON_ASSERT(i < 0);
        int type0 = LYCON_MAT_TYPE(flags);
        LYCON_ASSERT(mtype == type0 || (LYCON_MAT_CN(mtype) == 1 && ((1 << type0) & fixedDepthMask) != 0));
        LYCON_ASSERT(d == 2 && ((sizes[0] == sz.height && sizes[1] == sz.width) ||
                                (allowTransposed && sizes[0] == sz.width && sizes[1] == sz.height)));
        return;
    }

    if (k == STD_VECTOR || k == STD_VECTOR_VECTOR)
    {
        LYCON_ASSERT(d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0] * sizes[1] == 0));
        size_t len = sizes[0] * sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0;
        std::vector<uchar>* v = (std::vector<uchar>*)obj;

        if (k == STD_VECTOR_VECTOR)
        {
            std::vector<std::vector<uchar>>& vv = *(std::vector<std::vector<uchar>>*)obj;
            if (i < 0)
            {
                LYCON_ASSERT(!fixedSize() || len == vv.size());
                vv.resize(len);
                return;
            }
            LYCON_ASSERT(i < (int)vv.size());
            v = &vv[i];
        }
        else
            LYCON_ASSERT(i < 0);

        int type0 = LYCON_MAT_TYPE(flags);
        LYCON_ASSERT(mtype == type0 ||
                     (LYCON_MAT_CN(mtype) == LYCON_MAT_CN(type0) && ((1 << type0) & fixedDepthMask) != 0));

        int esz = LYCON_ELEM_SIZE(type0);
        LYCON_ASSERT(!fixedSize() || len == ((std::vector<uchar>*)v)->size() / esz);
        switch (esz)
        {
        case 1: ((std::vector<uchar>*)v)->resize(len); break;
        case 2: ((std::vector<Vec2b>*)v)->resize(len); break;
        case 3: ((std::vector<Vec3b>*)v)->resize(len); break;
        case 4: ((std::vector<int>*)v)->resize(len); break;
        case 6: ((std::vector<Vec3s>*)v)->resize(len); break;
        case 8: ((std::vector<Vec2i>*)v)->resize(len); break;
        case 12: ((std::vector<Vec3i>*)v)->resize(len); break;
        case 16: ((std::vector<Vec4i>*)v)->resize(len); break;
        case 24: ((std::vector<Vec6i>*)v)->resize(len); break;
        case 32: ((std::vector<Vec8i>*)v)->resize(len); break;
        case 36: ((std::vector<Vec<int, 9>>*)v)->resize(len); break;
        case 48: ((std::vector<Vec<int, 12>>*)v)->resize(len); break;
        case 64: ((std::vector<Vec<int, 16>>*)v)->resize(len); break;
        case 128: ((std::vector<Vec<int, 32>>*)v)->resize(len); break;
        case 256: ((std::vector<Vec<int, 64>>*)v)->resize(len); break;
        case 512: ((std::vector<Vec<int, 128>>*)v)->resize(len); break;
        default:
            LYCON_ERROR("Vectors with element size %d are not supported. Please, modify OutputArray::create()\n", esz);
        }
        return;
    }

    if (k == NONE)
    {
        LYCON_ERROR("create() called for the missing output array");
        return;
    }

    if (k == STD_VECTOR_MAT)
    {
        std::vector<Mat>& v = *(std::vector<Mat>*)obj;

        if (i < 0)
        {
            LYCON_ASSERT(d == 2 && (sizes[0] == 1 || sizes[1] == 1 || sizes[0] * sizes[1] == 0));
            size_t len = sizes[0] * sizes[1] > 0 ? sizes[0] + sizes[1] - 1 : 0, len0 = v.size();

            LYCON_ASSERT(!fixedSize() || len == len0);
            v.resize(len);
            if (fixedType())
            {
                int _type = LYCON_MAT_TYPE(flags);
                for (size_t j = len0; j < len; j++)
                {
                    if (v[j].type() == _type)
                        continue;
                    LYCON_ASSERT(v[j].empty());
                    v[j].flags = (v[j].flags & ~LYCON_MAT_TYPE_MASK) | _type;
                }
            }
            return;
        }

        LYCON_ASSERT(i < (int)v.size());
        Mat& m = v[i];

        if (allowTransposed)
        {
            if (!m.isContinuous())
            {
                LYCON_ASSERT(!fixedType() && !fixedSize());
                m.release();
            }

            if (d == 2 && m.dims == 2 && m.data && m.type() == mtype && m.rows == sizes[1] && m.cols == sizes[0])
                return;
        }

        if (fixedType())
        {
            if (LYCON_MAT_CN(mtype) == m.channels() && ((1 << LYCON_MAT_TYPE(flags)) & fixedDepthMask) != 0)
                mtype = m.type();
            else
                LYCON_ASSERT(LYCON_MAT_TYPE(mtype) == m.type());
        }
        if (fixedSize())
        {
            LYCON_ASSERT(m.dims == d);
            for (int j = 0; j < d; ++j)
                LYCON_ASSERT(m.size[j] == sizes[j]);
        }

        m.create(d, sizes, mtype);
        return;
    }

    LYCON_ERROR("Unknown/unsupported array type");
}

void _OutputArray::createSameSize(const _InputArray& arr, int mtype) const
{
    int arrsz[LYCON_MAX_DIM], d = arr.sizend(arrsz);
    create(d, arrsz, mtype);
}

void _OutputArray::release() const
{
    LYCON_ASSERT(!fixedSize());

    int k = kind();

    if (k == MAT)
    {
        ((Mat*)obj)->release();
        return;
    }

    if (k == NONE)
        return;

    if (k == STD_VECTOR)
    {
        create(Size(), LYCON_MAT_TYPE(flags));
        return;
    }

    if (k == STD_VECTOR_VECTOR)
    {
        ((std::vector<std::vector<uchar>>*)obj)->clear();
        return;
    }

    if (k == STD_VECTOR_MAT)
    {
        ((std::vector<Mat>*)obj)->clear();
        return;
    }
    LYCON_ERROR("Unknown/unsupported array type");
}

void _OutputArray::clear() const
{
    int k = kind();

    if (k == MAT)
    {
        LYCON_ASSERT(!fixedSize());
        ((Mat*)obj)->resize(0);
        return;
    }

    release();
}

bool _OutputArray::needed() const
{
    return kind() != NONE;
}

Mat& _OutputArray::getMatRef(int i) const
{
    int k = kind();
    if (i < 0)
    {
        LYCON_ASSERT(k == MAT);
        return *(Mat*)obj;
    }
    else
    {
        LYCON_ASSERT(k == STD_VECTOR_MAT);
        std::vector<Mat>& v = *(std::vector<Mat>*)obj;
        LYCON_ASSERT(i < (int)v.size());
        return v[i];
    }
}

void _OutputArray::setTo(const _InputArray& arr, const _InputArray& mask) const
{
    int k = kind();

    if (k == NONE)
        ;
    else if (k == MAT || k == MATX || k == STD_VECTOR)
    {
        Mat m = getMat();
        m.setTo(arr, mask);
    }
    else
        LYCON_ERROR("Not Implemented");
}

void _OutputArray::assign(const Mat& m) const
{
    int k = kind();
    if (k == MAT)
    {
        *(Mat*)obj = m;
    }
    else if (k == MATX)
    {
        m.copyTo(getMat());
    }
    else
    {
        LYCON_ERROR("Not Implemented");
    }
}

static _InputOutputArray _none;
InputOutputArray noArray()
{
    return _none;
}
}
