#include "lycon/mat/mat.h"

#include <cstring>

#include "lycon/transform/rotate.h"
#include "lycon/types.h"
#include "lycon/util/auto_buffer.h"
#include "lycon/util/error.h"

namespace lycon
{
static inline void setSize(Mat& m, int _dims, const int* _sz, const size_t* _steps, bool autoSteps = false)
{
    LYCON_ASSERT(0 <= _dims && _dims <= LYCON_MAX_DIM);
    if (m.dims != _dims)
    {
        if (m.step.p != m.step.buf)
        {
            fastFree(m.step.p);
            m.step.p = m.step.buf;
            m.size.p = &m.rows;
        }
        if (_dims > 2)
        {
            m.step.p = (size_t*)fastMalloc(_dims * sizeof(m.step.p[0]) + (_dims + 1) * sizeof(m.size.p[0]));
            m.size.p = (int*)(m.step.p + _dims) + 1;
            m.size.p[-1] = _dims;
            m.rows = m.cols = -1;
        }
    }

    m.dims = _dims;
    if (!_sz)
        return;

    size_t esz = LYCON_ELEM_SIZE(m.flags), esz1 = LYCON_ELEM_SIZE1(m.flags), total = esz;
    for (int i = _dims - 1; i >= 0; i--)
    {
        int s = _sz[i];
        LYCON_ASSERT(s >= 0);
        m.size.p[i] = s;

        if (_steps)
        {
            if (_steps[i] % esz1 != 0)
            {
                LYCON_ERROR("Step must be a multiple of esz1");
            }

            m.step.p[i] = i < _dims - 1 ? _steps[i] : esz;
        }
        else if (autoSteps)
        {
            m.step.p[i] = total;
            int64 total1 = (int64)total * s;
            if ((uint64)total1 != (size_t)total1)
                LYCON_ERROR("The total matrix size does not fit to \"size_t\" type");
            total = (size_t)total1;
        }
    }

    if (_dims == 1)
    {
        m.dims = 2;
        m.cols = 1;
        m.step[1] = esz;
    }
}

static void updateContinuityFlag(Mat& m)
{
    int i, j;
    for (i = 0; i < m.dims; i++)
    {
        if (m.size[i] > 1)
            break;
    }

    for (j = m.dims - 1; j > i; j--)
    {
        if (m.step[j] * m.size[j] < m.step[j - 1])
            break;
    }

    uint64 t = (uint64)m.step[0] * m.size[0];
    if (j <= i && t == (size_t)t)
        m.flags |= Mat::CONTINUOUS_FLAG;
    else
        m.flags &= ~Mat::CONTINUOUS_FLAG;
}

static void finalizeHdr(Mat& m)
{
    updateContinuityFlag(m);
    int d = m.dims;
    if (d > 2)
        m.rows = m.cols = -1;
    if (m.u)
        m.datastart = m.data = m.u->data;
    if (m.data)
    {
        m.datalimit = m.datastart + m.size[0] * m.step[0];
        if (m.size[0] > 0)
        {
            m.dataend = m.ptr() + m.size[d - 1] * m.step[d - 1];
            for (int i = 0; i < d - 1; i++)
                m.dataend += (m.size[i] - 1) * m.step[i];
        }
        else
            m.dataend = m.datalimit;
    }
    else
        m.dataend = m.datalimit = 0;
}

void Mat::create(int d, const int* _sizes, int _type)
{
    int i;
    LYCON_ASSERT(0 <= d && d <= LYCON_MAX_DIM && _sizes);
    _type = LYCON_MAT_TYPE(_type);

    if (data && (d == dims || (d == 1 && dims <= 2)) && _type == type())
    {
        if (d == 2 && rows == _sizes[0] && cols == _sizes[1])
            return;
        for (i = 0; i < d; i++)
            if (size[i] != _sizes[i])
                break;
        if (i == d && (d > 1 || size[1] == 1))
            return;
    }

    int _sizes_backup[LYCON_MAX_DIM]; // #5991
    if (_sizes == (this->size.p))
    {
        for (i = 0; i < d; i++)
            _sizes_backup[i] = _sizes[i];
        _sizes = _sizes_backup;
    }

    release();
    if (d == 0)
        return;
    flags = (_type & LYCON_MAT_TYPE_MASK) | MAGIC_VAL;
    setSize(*this, d, _sizes, 0, true);

    if (total() > 0)
    {
        MatAllocator *a = allocator, *a0 = getDefaultAllocator();
        if (!a)
            a = a0;

#ifdef LYCON_USE_NUMPY_ALLOCATOR_BY_DEFAULT
        // The mat to ndarray conversion expects the custom allocator to be set.
        allocator = a;
#endif
        try
        {
            u = a->allocate(dims, size, _type, 0, step.p, 0, USAGE_DEFAULT);
            LYCON_ASSERT(u != 0);
        }
        catch (...)
        {
            if (a != a0)
                u = a0->allocate(dims, size, _type, 0, step.p, 0, USAGE_DEFAULT);
            LYCON_ASSERT(u != 0);
        }
        LYCON_ASSERT(step[dims - 1] == (size_t)LYCON_ELEM_SIZE(flags));
    }
    addref();
    finalizeHdr(*this);
}

void Mat::create(const std::vector<int>& _sizes, int _type)
{
    create((int)_sizes.size(), _sizes.data(), _type);
}

void Mat::copySize(const Mat& m)
{
    setSize(*this, m.dims, 0, 0);
    for (int i = 0; i < dims; i++)
    {
        size[i] = m.size[i];
        step[i] = m.step[i];
    }
}

void Mat::deallocate()
{
    if (u)
        (u->currAllocator ? u->currAllocator : allocator ? allocator : getDefaultAllocator())->unmap(u);
    u = NULL;
}

Mat::Mat(const Mat& m, const Range& _rowRange, const Range& _colRange)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0),
      size(&rows)
{
    LYCON_ASSERT(m.dims >= 2);
    if (m.dims > 2)
    {
        AutoBuffer<Range> rs(m.dims);
        rs[0] = _rowRange;
        rs[1] = _colRange;
        for (int i = 2; i < m.dims; i++)
            rs[i] = Range::all();
        *this = m(rs);
        return;
    }

    *this = m;
    if (_rowRange != Range::all() && _rowRange != Range(0, rows))
    {
        LYCON_ASSERT(0 <= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows);
        rows = _rowRange.size();
        data += step * _rowRange.start;
        flags |= SUBMATRIX_FLAG;
    }

    if (_colRange != Range::all() && _colRange != Range(0, cols))
    {
        LYCON_ASSERT(0 <= _colRange.start && _colRange.start <= _colRange.end && _colRange.end <= m.cols);
        cols = _colRange.size();
        data += _colRange.start * elemSize();
        flags &= cols < m.cols ? ~CONTINUOUS_FLAG : -1;
        flags |= SUBMATRIX_FLAG;
    }

    if (rows == 1)
        flags |= CONTINUOUS_FLAG;

    if (rows <= 0 || cols <= 0)
    {
        release();
        rows = cols = 0;
    }
}

Mat::Mat(const Mat& m, const Rect& roi)
    : flags(m.flags), dims(2), rows(roi.height), cols(roi.width), data(m.data + roi.y * m.step[0]),
      datastart(m.datastart), dataend(m.dataend), datalimit(m.datalimit), allocator(m.allocator), u(m.u), size(&rows)
{
    LYCON_ASSERT(m.dims <= 2);
    flags &= roi.width < m.cols ? ~CONTINUOUS_FLAG : -1;
    flags |= roi.height == 1 ? CONTINUOUS_FLAG : 0;

    size_t esz = LYCON_ELEM_SIZE(flags);
    data += roi.x * esz;
    LYCON_ASSERT(0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height &&
                 roi.y + roi.height <= m.rows);
    if (u)
        LYCON_XADD(&u->refcount, 1);
    if (roi.width < m.cols || roi.height < m.rows)
        flags |= SUBMATRIX_FLAG;

    step[0] = m.step[0];
    step[1] = esz;

    if (rows <= 0 || cols <= 0)
    {
        release();
        rows = cols = 0;
    }
}

Mat::Mat(int _dims, const int* _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0),
      size(&rows)
{
    flags |= LYCON_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    setSize(*this, _dims, _sizes, _steps, true);
    finalizeHdr(*this);
}

Mat::Mat(const std::vector<int>& _sizes, int _type, void* _data, const size_t* _steps)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0),
      size(&rows)
{
    flags |= LYCON_MAT_TYPE(_type);
    datastart = data = (uchar*)_data;
    setSize(*this, (int)_sizes.size(), _sizes.data(), _steps, true);
    finalizeHdr(*this);
}

Mat::Mat(const Mat& m, const Range* ranges)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0),
      size(&rows)
{
    int d = m.dims;

    LYCON_ASSERT(ranges);
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        LYCON_ASSERT(r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]));
    }
    *this = m;
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        if (r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            data += r.start * step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}

Mat::Mat(const Mat& m, const std::vector<Range>& ranges)
    : flags(MAGIC_VAL), dims(0), rows(0), cols(0), data(0), datastart(0), dataend(0), datalimit(0), allocator(0), u(0),
      size(&rows)
{
    int d = m.dims;

    LYCON_ASSERT((int)ranges.size() == d);
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        LYCON_ASSERT(r == Range::all() || (0 <= r.start && r.start < r.end && r.end <= m.size[i]));
    }
    *this = m;
    for (int i = 0; i < d; i++)
    {
        Range r = ranges[i];
        if (r != Range::all() && r != Range(0, size.p[i]))
        {
            size.p[i] = r.end - r.start;
            data += r.start * step.p[i];
            flags |= SUBMATRIX_FLAG;
        }
    }
    updateContinuityFlag(*this);
}

Mat Mat::diag(int d) const
{
    LYCON_ASSERT(dims <= 2);
    Mat m = *this;
    size_t esz = elemSize();
    int len;

    if (d >= 0)
    {
        len = std::min(cols - d, rows);
        m.data += esz * d;
    }
    else
    {
        len = std::min(rows + d, cols);
        m.data -= step[0] * d;
    }
    LYCON_DbgAssert(len > 0);

    m.size[0] = m.rows = len;
    m.size[1] = m.cols = 1;
    m.step[0] += (len > 1 ? esz : 0);

    if (m.rows > 1)
        m.flags &= ~CONTINUOUS_FLAG;
    else
        m.flags |= CONTINUOUS_FLAG;

    if (size() != Size(1, 1))
        m.flags |= SUBMATRIX_FLAG;

    return m;
}

void Mat::pop_back(size_t nelems)
{
    LYCON_ASSERT(nelems <= (size_t)size.p[0]);

    if (isSubmatrix())
        *this = rowRange(0, size.p[0] - (int)nelems);
    else
    {
        size.p[0] -= (int)nelems;
        dataend -= nelems * step.p[0];
        /*if( size.p[0] <= 1 )
        {
            if( dims <= 2 )
                flags |= CONTINUOUS_FLAG;
            else
                updateContinuityFlag(*this);
        }*/
    }
}

void Mat::push_back_(const void* elem)
{
    int r = size.p[0];
    if (isSubmatrix() || dataend + step.p[0] > datalimit)
        reserve(std::max(r + 1, (r * 3 + 1) / 2));

    size_t esz = elemSize();
    memcpy(data + r * step.p[0], elem, esz);
    size.p[0] = r + 1;
    dataend += step.p[0];
    if (esz < step.p[0])
        flags &= ~CONTINUOUS_FLAG;
}

void Mat::reserve(size_t nelems)
{
    const size_t MIN_SIZE = 64;

    LYCON_ASSERT((int)nelems >= 0);
    if (!isSubmatrix() && data + step.p[0] * nelems <= datalimit)
        return;

    int r = size.p[0];

    if ((size_t)r >= nelems)
        return;

    size.p[0] = std::max((int)nelems, 1);
    size_t newsize = total() * elemSize();

    if (newsize < MIN_SIZE)
        size.p[0] = (int)((MIN_SIZE + newsize - 1) * nelems / newsize);

    Mat m(dims, size.p, type());
    size.p[0] = r;
    if (r > 0)
    {
        Mat mpart = m.rowRange(0, r);
        copyTo(mpart);
    }

    *this = m;
    size.p[0] = r;
    dataend = data + step.p[0] * r;
}

void Mat::resize(size_t nelems)
{
    int saveRows = size.p[0];
    if (saveRows == (int)nelems)
        return;
    LYCON_ASSERT((int)nelems >= 0);

    if (isSubmatrix() || data + step.p[0] * nelems > datalimit)
        reserve(nelems);

    size.p[0] = (int)nelems;
    dataend += (size.p[0] - saveRows) * step.p[0];

    // updateContinuityFlag(*this);
}

void Mat::resize(size_t nelems, const Scalar& s)
{
    int saveRows = size.p[0];
    resize(nelems);

    if (size.p[0] > saveRows)
    {
        Mat part = rowRange(saveRows, size.p[0]);
        part = s;
    }
}

void Mat::push_back(const Mat& elems)
{
    int r = size.p[0], delta = elems.size.p[0];
    if (delta == 0)
        return;
    if (this == &elems)
    {
        Mat tmp = elems;
        push_back(tmp);
        return;
    }
    if (!data)
    {
        *this = elems.clone();
        return;
    }

    size.p[0] = elems.size.p[0];
    bool eq = size == elems.size;
    size.p[0] = r;
    if (!eq)
        LYCON_ERROR("Pushed vector length is not equal to matrix row length");
    if (type() != elems.type())
        LYCON_ERROR("Pushed vector type is not the same as matrix type");

    if (isSubmatrix() || dataend + step.p[0] * delta > datalimit)
        reserve(std::max(r + delta, (r * 3 + 1) / 2));

    size.p[0] += delta;
    dataend += step.p[0] * delta;

    // updateContinuityFlag(*this);

    if (isContinuous() && elems.isContinuous())
        memcpy(data + r * step.p[0], elems.data, elems.total() * elems.elemSize());
    else
    {
        Mat part = rowRange(r, r + delta);
        elems.copyTo(part);
    }
}

void Mat::locateROI(Size& wholeSize, Point& ofs) const
{
    LYCON_ASSERT(dims <= 2 && step[0] > 0);
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = data - datastart, delta2 = dataend - datastart;

    if (delta1 == 0)
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1 / step[0]);
        ofs.x = (int)((delta1 - step[0] * ofs.y) / esz);
        LYCON_DbgAssert(data == datastart + ofs.y * step[0] + ofs.x * esz);
    }
    minstep = (ofs.x + cols) * esz;
    wholeSize.height = (int)((delta2 - minstep) / step[0] + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step * (wholeSize.height - 1)) / esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}

Mat& Mat::adjustROI(int dtop, int dbottom, int dleft, int dright)
{
    LYCON_ASSERT(dims <= 2 && step[0] > 0);
    Size wholeSize;
    Point ofs;
    size_t esz = elemSize();
    locateROI(wholeSize, ofs);
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    data += (row1 - ofs.y) * step + (col1 - ofs.x) * esz;
    rows = row2 - row1;
    cols = col2 - col1;
    size.p[0] = rows;
    size.p[1] = cols;
    if (esz * cols == step[0] || rows == 1)
        flags |= CONTINUOUS_FLAG;
    else
        flags &= ~CONTINUOUS_FLAG;
    return *this;
}

Mat Mat::reshape(int new_cn, int new_rows) const
{
    int cn = channels();
    Mat hdr = *this;

    if (dims > 2 && new_rows == 0 && new_cn != 0 && size[dims - 1] * cn % new_cn == 0)
    {
        hdr.flags = (hdr.flags & ~LYCON_MAT_CN_MASK) | ((new_cn - 1) << LYCON_CN_SHIFT);
        hdr.step[dims - 1] = LYCON_ELEM_SIZE(hdr.flags);
        hdr.size[dims - 1] = hdr.size[dims - 1] * cn / new_cn;
        return hdr;
    }

    LYCON_ASSERT(dims <= 2);

    if (new_cn == 0)
        new_cn = cn;

    int total_width = cols * cn;

    if ((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = rows * total_width / new_cn;

    if (new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * rows;
        if (!isContinuous())
            LYCON_ERROR("The matrix is not continuous, thus its number of rows can not be changed");

        if ((unsigned)new_rows > (unsigned)total_size)
            LYCON_ERROR("Bad new number of rows");

        total_width = total_size / new_rows;

        if (total_width * new_rows != total_size)
            LYCON_ERROR("The total number of matrix elements is not divisible by the new number of rows");

        hdr.rows = new_rows;
        hdr.step[0] = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if (new_width * new_cn != total_width)
        LYCON_ERROR("The total width is not divisible by the new number of channels");

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~LYCON_MAT_CN_MASK) | ((new_cn - 1) << LYCON_CN_SHIFT);
    hdr.step[1] = LYCON_ELEM_SIZE(hdr.flags);
    return hdr;
}

Mat Mat::diag(const Mat& d)
{
    LYCON_ASSERT(d.cols == 1 || d.rows == 1);
    int len = d.rows + d.cols - 1;
    Mat m(len, len, d.type(), Scalar(0));
    Mat md = m.diag();
    if (d.cols == 1)
        d.copyTo(md);
    else
        transpose(d, md);
    return m;
}

int Mat::checkVector(int _elemChannels, int _depth, bool _requireContinuous) const
{
    return (depth() == _depth || _depth <= 0) && (isContinuous() || !_requireContinuous) &&
                   ((dims == 2 && (((rows == 1 || cols == 1) && channels() == _elemChannels) ||
                                   (cols == _elemChannels && channels() == 1))) ||
                    (dims == 3 && channels() == 1 && size.p[2] == _elemChannels && (size.p[0] == 1 || size.p[1] == 1) &&
                     (isContinuous() || step.p[1] == step.p[2] * size.p[2])))
               ? (int)(total() * channels() / _elemChannels)
               : -1;
}

Mat Mat::reshape(int _cn, int _newndims, const int* _newsz) const
{
    if (_newndims == dims)
    {
        if (_newsz == 0)
            return reshape(_cn);
        if (_newndims == 2)
            return reshape(_cn, _newsz[0]);
    }

    if (isContinuous())
    {
        LYCON_ASSERT(_cn >= 0 && _newndims > 0 && _newndims <= LYCON_MAX_DIM && _newsz);

        if (_cn == 0)
            _cn = this->channels();
        else
            LYCON_ASSERT(_cn <= LYCON_CN_MAX);

        size_t total_elem1_ref = this->total() * this->channels();
        size_t total_elem1 = _cn;

        AutoBuffer<int, 4> newsz_buf((size_t)_newndims);

        for (int i = 0; i < _newndims; i++)
        {
            LYCON_ASSERT(_newsz[i] >= 0);

            if (_newsz[i] > 0)
                newsz_buf[i] = _newsz[i];
            else if (i < dims)
                newsz_buf[i] = this->size[i];
            else
                LYCON_ERROR("Copy dimension (which has zero size) is not present in source matrix");

            total_elem1 *= (size_t)newsz_buf[i];
        }

        if (total_elem1 != total_elem1_ref)
            LYCON_ERROR("Requested and source matrices have different count of elements");

        Mat hdr = *this;
        hdr.flags = (hdr.flags & ~LYCON_MAT_CN_MASK) | ((_cn - 1) << LYCON_CN_SHIFT);
        setSize(hdr, _newndims, (int*)newsz_buf, NULL, true);

        return hdr;
    }

    LYCON_ERROR("Reshaping of n-dimensional non-continuous matrices is not supported yet");
    // TBD
    return Mat();
}
}
