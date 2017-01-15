inline Mat::Mat()
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
}

inline Mat::Mat(int _rows, int _cols, int _type)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_rows, _cols, _type);
}

inline Mat::Mat(int _rows, int _cols, int _type, const Scalar &_s)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_rows, _cols, _type);
    *this = _s;
}

inline Mat::Mat(Size _sz, int _type)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_sz.height, _sz.width, _type);
}

inline Mat::Mat(Size _sz, int _type, const Scalar &_s)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_sz.height, _sz.width, _type);
    *this = _s;
}

inline Mat::Mat(int _dims, const int *_sz, int _type)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_dims, _sz, _type);
}

inline Mat::Mat(int _dims, const int *_sz, int _type, const Scalar &_s)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_dims, _sz, _type);
    *this = _s;
}

inline Mat::Mat(const std::vector<int> &_sz, int _type)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_sz, _type);
}

inline Mat::Mat(const std::vector<int> &_sz, int _type, const Scalar &_s)
    : flags(MAGIC_VAL),
      dims(0),
      rows(0),
      cols(0),
      data(0),
      datastart(0),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    create(_sz, _type);
    *this = _s;
}

inline Mat::Mat(const Mat &m)
    : flags(m.flags),
      dims(m.dims),
      rows(m.rows),
      cols(m.cols),
      data(m.data),
      datastart(m.datastart),
      dataend(m.dataend),
      datalimit(m.datalimit),
      allocator(m.allocator),
      u(m.u),
      size(&rows)
{
    if (u) LYCON_XADD(&u->refcount, 1);
    if (m.dims <= 2)
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        dims = 0;
        copySize(m);
    }
}

inline Mat::Mat(int _rows, int _cols, int _type, void *_data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)),
      dims(2),
      rows(_rows),
      cols(_cols),
      data((uchar *)_data),
      datastart((uchar *)_data),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    LYCON_ASSERT(total() == 0 || data != NULL);

    size_t esz = LYCON_ELEM_SIZE(_type), esz1 = LYCON_ELEM_SIZE1(_type);
    size_t minstep = cols * esz;
    if (_step == AUTO_STEP)
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1) _step = minstep;
        LYCON_DbgAssert(_step >= minstep);

        if (_step % esz1 != 0)
        {
            LYCON_ERROR("Step must be a multiple of esz1");
        }

        flags |= _step == minstep ? CONTINUOUS_FLAG : 0;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;
}

inline Mat::Mat(Size _sz, int _type, void *_data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)),
      dims(2),
      rows(_sz.height),
      cols(_sz.width),
      data((uchar *)_data),
      datastart((uchar *)_data),
      dataend(0),
      datalimit(0),
      allocator(0),
      u(0),
      size(&rows)
{
    LYCON_ASSERT(total() == 0 || data != NULL);

    size_t esz = LYCON_ELEM_SIZE(_type), esz1 = LYCON_ELEM_SIZE1(_type);
    size_t minstep = cols * esz;
    if (_step == AUTO_STEP)
    {
        _step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if (rows == 1) _step = minstep;
        LYCON_DbgAssert(_step >= minstep);

        if (_step % esz1 != 0)
        {
            LYCON_ERROR("Step must be a multiple of esz1");
        }

        flags |= _step == minstep ? CONTINUOUS_FLAG : 0;
    }
    step[0] = _step;
    step[1] = esz;
    datalimit = datastart + _step * rows;
    dataend = datalimit - _step + minstep;
}

template <typename _Tp>
inline Mat::Mat(const std::vector<_Tp> &vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | LYCON_MAT_CONT_FLAG),
      dims(2),
      rows((int)vec.size()),
      cols(1),
      data(0),
      datastart(0),
      dataend(0),
      allocator(0),
      u(0),
      size(&rows)
{
    if (vec.empty()) return;
    if (!copyData)
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar *)&vec[0];
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat((int)vec.size(), 1, DataType<_Tp>::type, (uchar *)&vec[0]).copyTo(*this);
}

template <typename _Tp, int n>
inline Mat::Mat(const Vec<_Tp, n> &vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | LYCON_MAT_CONT_FLAG),
      dims(2),
      rows(n),
      cols(1),
      data(0),
      datastart(0),
      dataend(0),
      allocator(0),
      u(0),
      size(&rows)
{
    if (!copyData)
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar *)vec.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(n, 1, DataType<_Tp>::type, (void *)vec.val).copyTo(*this);
}

template <typename _Tp, int m, int n>
inline Mat::Mat(const Matx<_Tp, m, n> &M, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | LYCON_MAT_CONT_FLAG),
      dims(2),
      rows(m),
      cols(n),
      data(0),
      datastart(0),
      dataend(0),
      allocator(0),
      u(0),
      size(&rows)
{
    if (!copyData)
    {
        step[0] = cols * sizeof(_Tp);
        step[1] = sizeof(_Tp);
        datastart = data = (uchar *)M.val;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
        Mat(m, n, DataType<_Tp>::type, (uchar *)M.val).copyTo(*this);
}

template <typename _Tp>
inline Mat::Mat(const Point_<_Tp> &pt, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | LYCON_MAT_CONT_FLAG),
      dims(2),
      rows(2),
      cols(1),
      data(0),
      datastart(0),
      dataend(0),
      allocator(0),
      u(0),
      size(&rows)
{
    if (!copyData)
    {
        step[0] = step[1] = sizeof(_Tp);
        datastart = data = (uchar *)&pt.x;
        datalimit = dataend = datastart + rows * step[0];
    }
    else
    {
        create(2, 1, DataType<_Tp>::type);
        ((_Tp *)data)[0] = pt.x;
        ((_Tp *)data)[1] = pt.y;
    }
}

inline Mat::~Mat()
{
    release();
    if (step.p != step.buf) fastFree(step.p);
}

inline Mat &Mat::operator=(const Mat &m)
{
    if (this != &m)
    {
        if (m.u) LYCON_XADD(&m.u->refcount, 1);
        release();
        flags = m.flags;
        if (dims <= 2 && m.dims <= 2)
        {
            dims = m.dims;
            rows = m.rows;
            cols = m.cols;
            step[0] = m.step[0];
            step[1] = m.step[1];
        }
        else
            copySize(m);
        data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        datalimit = m.datalimit;
        allocator = m.allocator;
        u = m.u;
    }
    return *this;
}

inline Mat Mat::row(int y) const { return Mat(*this, Range(y, y + 1), Range::all()); }

inline Mat Mat::col(int x) const { return Mat(*this, Range::all(), Range(x, x + 1)); }

inline Mat Mat::rowRange(int startrow, int endrow) const { return Mat(*this, Range(startrow, endrow), Range::all()); }

inline Mat Mat::rowRange(const Range &r) const { return Mat(*this, r, Range::all()); }

inline Mat Mat::colRange(int startcol, int endcol) const { return Mat(*this, Range::all(), Range(startcol, endcol)); }

inline Mat Mat::colRange(const Range &r) const { return Mat(*this, Range::all(), r); }

inline Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

inline void Mat::assignTo(Mat &m, int _type) const
{
    if (_type < 0)
        m = *this;
    else
        convertTo(m, _type);
}

inline void Mat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if (dims <= 2 && rows == _rows && cols == _cols && type() == _type && data) return;
    int sz[] = {_rows, _cols};
    create(2, sz, _type);
}

inline void Mat::create(Size _sz, int _type) { create(_sz.height, _sz.width, _type); }

inline void Mat::addref()
{
    if (u) LYCON_XADD(&u->refcount, 1);
}

inline void Mat::release()
{
    if (u && LYCON_XADD(&u->refcount, -1) == 1) deallocate();
    u = NULL;
    datastart = dataend = datalimit = data = 0;
    for (int i = 0; i < dims; i++) size.p[i] = 0;
#ifdef _DEBUG
    flags = MAGIC_VAL;
    dims = rows = cols = 0;
    if (step.p != step.buf)
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
#endif
}

inline Mat Mat::operator()(Range _rowRange, Range _colRange) const { return Mat(*this, _rowRange, _colRange); }

inline Mat Mat::operator()(const Rect &roi) const { return Mat(*this, roi); }

inline Mat Mat::operator()(const Range *ranges) const { return Mat(*this, ranges); }

inline Mat Mat::operator()(const std::vector<Range> &ranges) const { return Mat(*this, ranges); }

inline bool Mat::isContinuous() const { return (flags & CONTINUOUS_FLAG) != 0; }

inline bool Mat::isSubmatrix() const { return (flags & SUBMATRIX_FLAG) != 0; }

inline size_t Mat::elemSize() const { return dims > 0 ? step.p[dims - 1] : 0; }

inline size_t Mat::elemSize1() const { return LYCON_ELEM_SIZE1(flags); }

inline int Mat::type() const { return LYCON_MAT_TYPE(flags); }

inline int Mat::depth() const { return LYCON_MAT_DEPTH(flags); }

inline int Mat::channels() const { return LYCON_MAT_CN(flags); }

inline size_t Mat::step1(int i) const { return step.p[i] / elemSize1(); }

inline bool Mat::empty() const { return data == 0 || total() == 0; }

inline size_t Mat::total() const
{
    if (dims <= 2) return (size_t)rows * cols;
    size_t p = 1;
    for (int i = 0; i < dims; i++) p *= size[i];
    return p;
}

inline uchar *Mat::ptr(int y)
{
    LYCON_DbgAssert(y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]));
    return data + step.p[0] * y;
}

inline const uchar *Mat::ptr(int y) const
{
    LYCON_DbgAssert(y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]));
    return data + step.p[0] * y;
}

template <typename _Tp>
inline _Tp *Mat::ptr(int y)
{
    LYCON_DbgAssert(y == 0 || (data && dims >= 1 && (unsigned)y < (unsigned)size.p[0]));
    return (_Tp *)(data + step.p[0] * y);
}

template <typename _Tp>
inline const _Tp *Mat::ptr(int y) const
{
    LYCON_DbgAssert(y == 0 || (data && dims >= 1 && data && (unsigned)y < (unsigned)size.p[0]));
    return (const _Tp *)(data + step.p[0] * y);
}

inline uchar *Mat::ptr(int i0, int i1)
{
    LYCON_DbgAssert(dims >= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

inline const uchar *Mat::ptr(int i0, int i1) const
{
    LYCON_DbgAssert(dims >= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return data + i0 * step.p[0] + i1 * step.p[1];
}

template <typename _Tp>
inline _Tp *Mat::ptr(int i0, int i1)
{
    LYCON_DbgAssert(dims >= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (_Tp *)(data + i0 * step.p[0] + i1 * step.p[1]);
}

template <typename _Tp>
inline const _Tp *Mat::ptr(int i0, int i1) const
{
    LYCON_DbgAssert(dims >= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    return (const _Tp *)(data + i0 * step.p[0] + i1 * step.p[1]);
}

inline uchar *Mat::ptr(int i0, int i1, int i2)
{
    LYCON_DbgAssert(dims >= 3);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    LYCON_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

inline const uchar *Mat::ptr(int i0, int i1, int i2) const
{
    LYCON_DbgAssert(dims >= 3);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    LYCON_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2];
}

template <typename _Tp>
inline _Tp *Mat::ptr(int i0, int i1, int i2)
{
    LYCON_DbgAssert(dims >= 3);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    LYCON_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (_Tp *)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

template <typename _Tp>
inline const _Tp *Mat::ptr(int i0, int i1, int i2) const
{
    LYCON_DbgAssert(dims >= 3);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)i1 < (unsigned)size.p[1]);
    LYCON_DbgAssert((unsigned)i2 < (unsigned)size.p[2]);
    return (const _Tp *)(data + i0 * step.p[0] + i1 * step.p[1] + i2 * step.p[2]);
}

inline uchar *Mat::ptr(const int *idx)
{
    int i, d = dims;
    uchar *p = data;
    LYCON_DbgAssert(d >= 1 && p);
    for (i = 0; i < d; i++)
    {
        LYCON_DbgAssert((unsigned)idx[i] < (unsigned)size.p[i]);
        p += idx[i] * step.p[i];
    }
    return p;
}

inline const uchar *Mat::ptr(const int *idx) const
{
    int i, d = dims;
    uchar *p = data;
    LYCON_DbgAssert(d >= 1 && p);
    for (i = 0; i < d; i++)
    {
        LYCON_DbgAssert((unsigned)idx[i] < (unsigned)size.p[i]);
        p += idx[i] * step.p[i];
    }
    return p;
}

template <typename _Tp>
inline _Tp &Mat::at(int i0, int i1)
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    LYCON_DbgAssert(LYCON_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp *)(data + step.p[0] * i0))[i1];
}

template <typename _Tp>
inline const _Tp &Mat::at(int i0, int i1) const
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)(i1 * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    LYCON_DbgAssert(LYCON_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp *)(data + step.p[0] * i0))[i1];
}

template <typename _Tp>
inline _Tp &Mat::at(Point pt)
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    LYCON_DbgAssert(LYCON_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp *)(data + step.p[0] * pt.y))[pt.x];
}

template <typename _Tp>
inline const _Tp &Mat::at(Point pt) const
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)pt.y < (unsigned)size.p[0]);
    LYCON_DbgAssert((unsigned)(pt.x * DataType<_Tp>::channels) < (unsigned)(size.p[1] * channels()));
    LYCON_DbgAssert(LYCON_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp *)(data + step.p[0] * pt.y))[pt.x];
}

template <typename _Tp>
inline _Tp &Mat::at(int i0)
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    if (isContinuous() || size.p[0] == 1) return ((_Tp *)data)[i0];
    if (size.p[1] == 1) return *(_Tp *)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((_Tp *)(data + step.p[0] * i))[j];
}

template <typename _Tp>
inline const _Tp &Mat::at(int i0) const
{
    LYCON_DbgAssert(dims <= 2);
    LYCON_DbgAssert(data);
    LYCON_DbgAssert((unsigned)i0 < (unsigned)(size.p[0] * size.p[1]));
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    if (isContinuous() || size.p[0] == 1) return ((const _Tp *)data)[i0];
    if (size.p[1] == 1) return *(const _Tp *)(data + step.p[0] * i0);
    int i = i0 / cols, j = i0 - i * cols;
    return ((const _Tp *)(data + step.p[0] * i))[j];
}

template <typename _Tp>
inline _Tp &Mat::at(int i0, int i1, int i2)
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(_Tp *)ptr(i0, i1, i2);
}

template <typename _Tp>
inline const _Tp &Mat::at(int i0, int i1, int i2) const
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(const _Tp *)ptr(i0, i1, i2);
}

template <typename _Tp>
inline _Tp &Mat::at(const int *idx)
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(_Tp *)ptr(idx);
}

template <typename _Tp>
inline const _Tp &Mat::at(const int *idx) const
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(const _Tp *)ptr(idx);
}

template <typename _Tp, int n>
inline _Tp &Mat::at(const Vec<int, n> &idx)
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(_Tp *)ptr(idx.val);
}

template <typename _Tp, int n>
inline const _Tp &Mat::at(const Vec<int, n> &idx) const
{
    LYCON_DbgAssert(elemSize() == LYCON_ELEM_SIZE(DataType<_Tp>::type));
    return *(const _Tp *)ptr(idx.val);
}

template <typename _Tp>
inline MatConstIterator_<_Tp> Mat::begin() const
{
    LYCON_DbgAssert(elemSize() == sizeof(_Tp));
    return MatConstIterator_<_Tp>((const Mat_<_Tp> *)this);
}

template <typename _Tp>
inline MatConstIterator_<_Tp> Mat::end() const
{
    LYCON_DbgAssert(elemSize() == sizeof(_Tp));
    MatConstIterator_<_Tp> it((const Mat_<_Tp> *)this);
    it += total();
    return it;
}

template <typename _Tp>
inline MatIterator_<_Tp> Mat::begin()
{
    LYCON_DbgAssert(elemSize() == sizeof(_Tp));
    return MatIterator_<_Tp>((Mat_<_Tp> *)this);
}

template <typename _Tp>
inline MatIterator_<_Tp> Mat::end()
{
    LYCON_DbgAssert(elemSize() == sizeof(_Tp));
    MatIterator_<_Tp> it((Mat_<_Tp> *)this);
    it += total();
    return it;
}

template <typename _Tp, typename Functor>
inline void Mat::forEach(const Functor &operation)
{
    this->forEach_impl<_Tp>(operation);
}

template <typename _Tp, typename Functor>
inline void Mat::forEach(const Functor &operation) const
{
    // call as not const
    (const_cast<Mat *>(this))->forEach<const _Tp>(operation);
}

template <typename _Tp>
inline Mat::operator std::vector<_Tp>() const
{
    std::vector<_Tp> v;
    copyTo(v);
    return v;
}

template <typename _Tp, int n>
inline Mat::operator Vec<_Tp, n>() const
{
    LYCON_ASSERT(data && dims <= 2 && (rows == 1 || cols == 1) && rows + cols - 1 == n && channels() == 1);

    if (isContinuous() && type() == DataType<_Tp>::type) return Vec<_Tp, n>((_Tp *)data);
    Vec<_Tp, n> v;
    Mat tmp(rows, cols, DataType<_Tp>::type, v.val);
    convertTo(tmp, tmp.type());
    return v;
}

template <typename _Tp, int m, int n>
inline Mat::operator Matx<_Tp, m, n>() const
{
    LYCON_ASSERT(data && dims <= 2 && rows == m && cols == n && channels() == 1);

    if (isContinuous() && type() == DataType<_Tp>::type) return Matx<_Tp, m, n>((_Tp *)data);
    Matx<_Tp, m, n> mtx;
    Mat tmp(rows, cols, DataType<_Tp>::type, mtx.val);
    convertTo(tmp, tmp.type());
    return mtx;
}

template <typename _Tp>
inline void Mat::push_back(const _Tp &elem)
{
    if (!data)
    {
        *this = Mat(1, 1, DataType<_Tp>::type, (void *)&elem).clone();
        return;
    }
    LYCON_ASSERT(DataType<_Tp>::type == type() && cols == 1
                 /* && dims == 2 (cols == 1 implies dims == 2) */);
    const uchar *tmp = dataend + step[0];
    if (!isSubmatrix() && isContinuous() && tmp <= datalimit)
    {
        *(_Tp *)(data + (size.p[0]++) * step.p[0]) = elem;
        dataend = tmp;
    }
    else
        push_back_(&elem);
}

template <typename _Tp>
inline void Mat::push_back(const Mat_<_Tp> &m)
{
    push_back((const Mat &)m);
}

inline Mat::Mat(Mat &&m)
    : flags(m.flags),
      dims(m.dims),
      rows(m.rows),
      cols(m.cols),
      data(m.data),
      datastart(m.datastart),
      dataend(m.dataend),
      datalimit(m.datalimit),
      allocator(m.allocator),
      u(m.u),
      size(&rows)
{
    if (m.dims <= 2)  // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        LYCON_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL;
    m.dims = m.rows = m.cols = 0;
    m.data = NULL;
    m.datastart = NULL;
    m.dataend = NULL;
    m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
}

inline Mat &Mat::operator=(Mat &&m)
{
    if (this == &m) return *this;

    release();
    flags = m.flags;
    dims = m.dims;
    rows = m.rows;
    cols = m.cols;
    data = m.data;
    datastart = m.datastart;
    dataend = m.dataend;
    datalimit = m.datalimit;
    allocator = m.allocator;
    u = m.u;
    if (step.p != step.buf)  // release self step/size
    {
        fastFree(step.p);
        step.p = step.buf;
        size.p = &rows;
    }
    if (m.dims <= 2)  // move new step/size info
    {
        step[0] = m.step[0];
        step[1] = m.step[1];
    }
    else
    {
        LYCON_DbgAssert(m.step.p != m.step.buf);
        step.p = m.step.p;
        size.p = m.size.p;
        m.step.p = m.step.buf;
        m.size.p = &m.rows;
    }
    m.flags = MAGIC_VAL;
    m.dims = m.rows = m.cols = 0;
    m.data = NULL;
    m.datastart = NULL;
    m.dataend = NULL;
    m.datalimit = NULL;
    m.allocator = NULL;
    m.u = NULL;
    return *this;
}
