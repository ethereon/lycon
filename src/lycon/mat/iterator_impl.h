inline MatConstIterator::MatConstIterator() : m(0), elemSize(0), ptr(0), sliceStart(0), sliceEnd(0) {}

inline MatConstIterator::MatConstIterator(const Mat *_m)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    if (m && m->isContinuous())
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total() * elemSize;
    }
    seek((const int *)0);
}

inline MatConstIterator::MatConstIterator(const Mat *_m, int _row, int _col)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    LYCON_ASSERT(m && m->dims <= 2);
    if (m->isContinuous())
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total() * elemSize;
    }
    int idx[] = {_row, _col};
    seek(idx);
}

inline MatConstIterator::MatConstIterator(const Mat *_m, Point _pt)
    : m(_m), elemSize(_m->elemSize()), ptr(0), sliceStart(0), sliceEnd(0)
{
    LYCON_ASSERT(m && m->dims <= 2);
    if (m->isContinuous())
    {
        sliceStart = m->ptr();
        sliceEnd = sliceStart + m->total() * elemSize;
    }
    int idx[] = {_pt.y, _pt.x};
    seek(idx);
}

inline MatConstIterator::MatConstIterator(const MatConstIterator &it)
    : m(it.m), elemSize(it.elemSize), ptr(it.ptr), sliceStart(it.sliceStart), sliceEnd(it.sliceEnd)
{
}

inline MatConstIterator &MatConstIterator::operator=(const MatConstIterator &it)
{
    m = it.m;
    elemSize = it.elemSize;
    ptr = it.ptr;
    sliceStart = it.sliceStart;
    sliceEnd = it.sliceEnd;
    return *this;
}

inline const uchar *MatConstIterator::operator*() const { return ptr; }

inline MatConstIterator &MatConstIterator::operator+=(ptrdiff_t ofs)
{
    if (!m || ofs == 0) return *this;
    ptrdiff_t ofsb = ofs * elemSize;
    ptr += ofsb;
    if (ptr < sliceStart || sliceEnd <= ptr)
    {
        ptr -= ofsb;
        seek(ofs, true);
    }
    return *this;
}

inline MatConstIterator &MatConstIterator::operator-=(ptrdiff_t ofs) { return (*this += -ofs); }

inline MatConstIterator &MatConstIterator::operator--()
{
    if (m && (ptr -= elemSize) < sliceStart)
    {
        ptr += elemSize;
        seek(-1, true);
    }
    return *this;
}

inline MatConstIterator MatConstIterator::operator--(int)
{
    MatConstIterator b = *this;
    *this += -1;
    return b;
}

inline MatConstIterator &MatConstIterator::operator++()
{
    if (m && (ptr += elemSize) >= sliceEnd)
    {
        ptr -= elemSize;
        seek(1, true);
    }
    return *this;
}

inline MatConstIterator MatConstIterator::operator++(int)
{
    MatConstIterator b = *this;
    *this += 1;
    return b;
}

static inline bool operator==(const MatConstIterator &a, const MatConstIterator &b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

static inline bool operator!=(const MatConstIterator &a, const MatConstIterator &b) { return !(a == b); }

static inline bool operator<(const MatConstIterator &a, const MatConstIterator &b) { return a.ptr < b.ptr; }

static inline bool operator>(const MatConstIterator &a, const MatConstIterator &b) { return a.ptr > b.ptr; }

static inline bool operator<=(const MatConstIterator &a, const MatConstIterator &b) { return a.ptr <= b.ptr; }

static inline bool operator>=(const MatConstIterator &a, const MatConstIterator &b) { return a.ptr >= b.ptr; }

static inline ptrdiff_t operator-(const MatConstIterator &b, const MatConstIterator &a)
{
    if (a.m != b.m) return ((size_t)(-1) >> 1);
    if (a.sliceEnd == b.sliceEnd) return (b.ptr - a.ptr) / static_cast<ptrdiff_t>(b.elemSize);

    return b.lpos() - a.lpos();
}

static inline MatConstIterator operator+(const MatConstIterator &a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline MatConstIterator operator+(ptrdiff_t ofs, const MatConstIterator &a)
{
    MatConstIterator b = a;
    return b += ofs;
}

static inline MatConstIterator operator-(const MatConstIterator &a, ptrdiff_t ofs)
{
    MatConstIterator b = a;
    return b += -ofs;
}

inline const uchar *MatConstIterator::operator[](ptrdiff_t i) const { return *(*this + i); }

///////////////////////// MatConstIterator_ /////////////////////////

template <typename _Tp>
inline MatConstIterator_<_Tp>::MatConstIterator_()
{
}

template <typename _Tp>
inline MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp> *_m) : MatConstIterator(_m)
{
}

template <typename _Tp>
inline MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp> *_m, int _row, int _col)
    : MatConstIterator(_m, _row, _col)
{
}

template <typename _Tp>
inline MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp> *_m, Point _pt) : MatConstIterator(_m, _pt)
{
}

template <typename _Tp>
inline MatConstIterator_<_Tp>::MatConstIterator_(const MatConstIterator_ &it) : MatConstIterator(it)
{
}

template <typename _Tp>
inline MatConstIterator_<_Tp> &MatConstIterator_<_Tp>::operator=(const MatConstIterator_ &it)
{
    MatConstIterator::operator=(it);
    return *this;
}

template <typename _Tp>
inline const _Tp &MatConstIterator_<_Tp>::operator*() const
{
    return *(_Tp *)(this->ptr);
}

template <typename _Tp>
inline MatConstIterator_<_Tp> &MatConstIterator_<_Tp>::operator+=(ptrdiff_t ofs)
{
    MatConstIterator::operator+=(ofs);
    return *this;
}

template <typename _Tp>
inline MatConstIterator_<_Tp> &MatConstIterator_<_Tp>::operator-=(ptrdiff_t ofs)
{
    return (*this += -ofs);
}

template <typename _Tp>
inline MatConstIterator_<_Tp> &MatConstIterator_<_Tp>::operator--()
{
    MatConstIterator::operator--();
    return *this;
}

template <typename _Tp>
inline MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator--(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator--();
    return b;
}

template <typename _Tp>
inline MatConstIterator_<_Tp> &MatConstIterator_<_Tp>::operator++()
{
    MatConstIterator::operator++();
    return *this;
}

template <typename _Tp>
inline MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator++(int)
{
    MatConstIterator_ b = *this;
    MatConstIterator::operator++();
    return b;
}

template <typename _Tp>
inline Point MatConstIterator_<_Tp>::pos() const
{
    if (!m) return Point();
    LYCON_DbgAssert(m->dims <= 2);
    if (m->isContinuous())
    {
        ptrdiff_t ofs = (const _Tp *)ptr - (const _Tp *)m->data;
        int y = (int)(ofs / m->cols);
        int x = (int)(ofs - (ptrdiff_t)y * m->cols);
        return Point(x, y);
    }
    else
    {
        ptrdiff_t ofs = (uchar *)ptr - m->data;
        int y = (int)(ofs / m->step);
        int x = (int)((ofs - y * m->step) / sizeof(_Tp));
        return Point(x, y);
    }
}

template <typename _Tp>
static inline bool operator==(const MatConstIterator_<_Tp> &a, const MatConstIterator_<_Tp> &b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template <typename _Tp>
static inline bool operator!=(const MatConstIterator_<_Tp> &a, const MatConstIterator_<_Tp> &b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template <typename _Tp>
static inline MatConstIterator_<_Tp> operator+(const MatConstIterator_<_Tp> &a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator &)a + ofs;
    return (MatConstIterator_<_Tp> &)t;
}

template <typename _Tp>
static inline MatConstIterator_<_Tp> operator+(ptrdiff_t ofs, const MatConstIterator_<_Tp> &a)
{
    MatConstIterator t = (const MatConstIterator &)a + ofs;
    return (MatConstIterator_<_Tp> &)t;
}

template <typename _Tp>
static inline MatConstIterator_<_Tp> operator-(const MatConstIterator_<_Tp> &a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator &)a - ofs;
    return (MatConstIterator_<_Tp> &)t;
}

template <typename _Tp>
inline const _Tp &MatConstIterator_<_Tp>::operator[](ptrdiff_t i) const
{
    return *(_Tp *)MatConstIterator::operator[](i);
}

//////////////////////////// MatIterator_ ///////////////////////////

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_() : MatConstIterator_<_Tp>()
{
}

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp> *_m) : MatConstIterator_<_Tp>(_m)
{
}

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp> *_m, int _row, int _col) : MatConstIterator_<_Tp>(_m, _row, _col)
{
}

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp> *_m, Point _pt) : MatConstIterator_<_Tp>(_m, _pt)
{
}

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp> *_m, const int *_idx) : MatConstIterator_<_Tp>(_m, _idx)
{
}

template <typename _Tp>
inline MatIterator_<_Tp>::MatIterator_(const MatIterator_ &it) : MatConstIterator_<_Tp>(it)
{
}

template <typename _Tp>
inline MatIterator_<_Tp> &MatIterator_<_Tp>::operator=(const MatIterator_<_Tp> &it)
{
    MatConstIterator::operator=(it);
    return *this;
}

template <typename _Tp>
inline _Tp &MatIterator_<_Tp>::operator*() const
{
    return *(_Tp *)(this->ptr);
}

template <typename _Tp>
inline MatIterator_<_Tp> &MatIterator_<_Tp>::operator+=(ptrdiff_t ofs)
{
    MatConstIterator::operator+=(ofs);
    return *this;
}

template <typename _Tp>
inline MatIterator_<_Tp> &MatIterator_<_Tp>::operator-=(ptrdiff_t ofs)
{
    MatConstIterator::operator+=(-ofs);
    return *this;
}

template <typename _Tp>
inline MatIterator_<_Tp> &MatIterator_<_Tp>::operator--()
{
    MatConstIterator::operator--();
    return *this;
}

template <typename _Tp>
inline MatIterator_<_Tp> MatIterator_<_Tp>::operator--(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator--();
    return b;
}

template <typename _Tp>
inline MatIterator_<_Tp> &MatIterator_<_Tp>::operator++()
{
    MatConstIterator::operator++();
    return *this;
}

template <typename _Tp>
inline MatIterator_<_Tp> MatIterator_<_Tp>::operator++(int)
{
    MatIterator_ b = *this;
    MatConstIterator::operator++();
    return b;
}

template <typename _Tp>
inline _Tp &MatIterator_<_Tp>::operator[](ptrdiff_t i) const
{
    return *(*this + i);
}

template <typename _Tp>
static inline bool operator==(const MatIterator_<_Tp> &a, const MatIterator_<_Tp> &b)
{
    return a.m == b.m && a.ptr == b.ptr;
}

template <typename _Tp>
static inline bool operator!=(const MatIterator_<_Tp> &a, const MatIterator_<_Tp> &b)
{
    return a.m != b.m || a.ptr != b.ptr;
}

template <typename _Tp>
static inline MatIterator_<_Tp> operator+(const MatIterator_<_Tp> &a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator &)a + ofs;
    return (MatIterator_<_Tp> &)t;
}

template <typename _Tp>
static inline MatIterator_<_Tp> operator+(ptrdiff_t ofs, const MatIterator_<_Tp> &a)
{
    MatConstIterator t = (const MatConstIterator &)a + ofs;
    return (MatIterator_<_Tp> &)t;
}

template <typename _Tp>
static inline MatIterator_<_Tp> operator-(const MatIterator_<_Tp> &a, ptrdiff_t ofs)
{
    MatConstIterator t = (const MatConstIterator &)a - ofs;
    return (MatIterator_<_Tp> &)t;
}
