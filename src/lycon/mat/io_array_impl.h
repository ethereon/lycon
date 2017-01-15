inline void _InputArray::init(int _flags, const void *_obj)
{
    flags = _flags;
    obj = (void *)_obj;
}

inline void _InputArray::init(int _flags, const void *_obj, Size _sz)
{
    flags = _flags;
    obj = (void *)_obj;
    sz = _sz;
}

inline void *_InputArray::getObj() const { return obj; }
inline int _InputArray::getFlags() const { return flags; }
inline Size _InputArray::getSz() const { return sz; }

inline _InputArray::_InputArray() { init(NONE, 0); }
inline _InputArray::_InputArray(int _flags, void *_obj) { init(_flags, _obj); }
inline _InputArray::_InputArray(const Mat &m) { init(MAT + ACCESS_READ, &m); }
inline _InputArray::_InputArray(const std::vector<Mat> &vec) { init(STD_VECTOR_MAT + ACCESS_READ, &vec); }

template <typename _Tp>
inline _InputArray::_InputArray(const std::vector<_Tp> &vec)
{
    init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_READ, &vec);
}

inline _InputArray::_InputArray(const std::vector<bool> &vec)
{
    init(FIXED_TYPE + STD_BOOL_VECTOR + DataType<bool>::type + ACCESS_READ, &vec);
}

template <typename _Tp>
inline _InputArray::_InputArray(const std::vector<std::vector<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_READ, &vec);
}

template <typename _Tp>
inline _InputArray::_InputArray(const std::vector<Mat_<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_READ, &vec);
}

template <typename _Tp, int m, int n>
inline _InputArray::_InputArray(const Matx<_Tp, m, n> &mtx)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_READ, &mtx, Size(n, m));
}

template <typename _Tp>
inline _InputArray::_InputArray(const _Tp *vec, int n)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_READ, vec, Size(n, 1));
}

template <typename _Tp>
inline _InputArray::_InputArray(const Mat_<_Tp> &m)
{
    init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_READ, &m);
}

inline _InputArray::_InputArray(const double &val)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + LYCON_64F + ACCESS_READ, &val, Size(1, 1));
}

inline _InputArray::~_InputArray() {}

inline Mat _InputArray::getMat(int i) const
{
    if (kind() == MAT && i < 0) return *(const Mat *)obj;
    return getMat_(i);
}

inline bool _InputArray::isMat() const { return kind() == _InputArray::MAT; }

inline bool _InputArray::isMatVector() const { return kind() == _InputArray::STD_VECTOR_MAT; }

inline bool _InputArray::isMatx() const { return kind() == _InputArray::MATX; }
inline bool _InputArray::isVector() const
{
    return kind() == _InputArray::STD_VECTOR || kind() == _InputArray::STD_BOOL_VECTOR;
}

////////////////////////////////////////////////////////////////////////////////////////

inline _OutputArray::_OutputArray() { init(ACCESS_WRITE, 0); }
inline _OutputArray::_OutputArray(int _flags, void *_obj) { init(_flags | ACCESS_WRITE, _obj); }
inline _OutputArray::_OutputArray(Mat &m) { init(MAT + ACCESS_WRITE, &m); }
inline _OutputArray::_OutputArray(std::vector<Mat> &vec) { init(STD_VECTOR_MAT + ACCESS_WRITE, &vec); }

template <typename _Tp>
inline _OutputArray::_OutputArray(std::vector<_Tp> &vec)
{
    init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

inline _OutputArray::_OutputArray(std::vector<bool> &) { LYCON_ERROR("std::vector<bool> cannot be an output array\n"); }

template <typename _Tp>
inline _OutputArray::_OutputArray(std::vector<std::vector<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

template <typename _Tp>
inline _OutputArray::_OutputArray(std::vector<Mat_<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

template <typename _Tp>
inline _OutputArray::_OutputArray(Mat_<_Tp> &m)
{
    init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_WRITE, &m);
}

template <typename _Tp, int m, int n>
inline _OutputArray::_OutputArray(Matx<_Tp, m, n> &mtx)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, &mtx, Size(n, m));
}

template <typename _Tp>
inline _OutputArray::_OutputArray(_Tp *vec, int n)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, vec, Size(n, 1));
}

template <typename _Tp>
inline _OutputArray::_OutputArray(const std::vector<_Tp> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

template <typename _Tp>
inline _OutputArray::_OutputArray(const std::vector<std::vector<_Tp>> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

template <typename _Tp>
inline _OutputArray::_OutputArray(const std::vector<Mat_<_Tp>> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_WRITE, &vec);
}

template <typename _Tp>
inline _OutputArray::_OutputArray(const Mat_<_Tp> &m)
{
    init(FIXED_TYPE + FIXED_SIZE + MAT + DataType<_Tp>::type + ACCESS_WRITE, &m);
}

template <typename _Tp, int m, int n>
inline _OutputArray::_OutputArray(const Matx<_Tp, m, n> &mtx)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, &mtx, Size(n, m));
}

template <typename _Tp>
inline _OutputArray::_OutputArray(const _Tp *vec, int n)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_WRITE, vec, Size(n, 1));
}

inline _OutputArray::_OutputArray(const Mat &m) { init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_WRITE, &m); }

inline _OutputArray::_OutputArray(const std::vector<Mat> &vec)
{
    init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_WRITE, &vec);
}

///////////////////////////////////////////////////////////////////////////////////////////

inline _InputOutputArray::_InputOutputArray() { init(ACCESS_RW, 0); }
inline _InputOutputArray::_InputOutputArray(int _flags, void *_obj) { init(_flags | ACCESS_RW, _obj); }
inline _InputOutputArray::_InputOutputArray(Mat &m) { init(MAT + ACCESS_RW, &m); }
inline _InputOutputArray::_InputOutputArray(std::vector<Mat> &vec) { init(STD_VECTOR_MAT + ACCESS_RW, &vec); }
template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(std::vector<_Tp> &vec)
{
    init(FIXED_TYPE + STD_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec);
}

inline _InputOutputArray::_InputOutputArray(std::vector<bool> &)
{
    LYCON_ERROR("std::vector<bool> cannot be an input/output array\n");
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(std::vector<std::vector<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec);
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(std::vector<Mat_<_Tp>> &vec)
{
    init(FIXED_TYPE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_RW, &vec);
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(Mat_<_Tp> &m)
{
    init(FIXED_TYPE + MAT + DataType<_Tp>::type + ACCESS_RW, &m);
}

template <typename _Tp, int m, int n>
inline _InputOutputArray::_InputOutputArray(Matx<_Tp, m, n> &mtx)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, &mtx, Size(n, m));
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(_Tp *vec, int n)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, vec, Size(n, 1));
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(const std::vector<_Tp> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec);
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(const std::vector<std::vector<_Tp>> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_VECTOR + DataType<_Tp>::type + ACCESS_RW, &vec);
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(const std::vector<Mat_<_Tp>> &vec)
{
    init(FIXED_TYPE + FIXED_SIZE + STD_VECTOR_MAT + DataType<_Tp>::type + ACCESS_RW, &vec);
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(const Mat_<_Tp> &m)
{
    init(FIXED_TYPE + FIXED_SIZE + MAT + DataType<_Tp>::type + ACCESS_RW, &m);
}

template <typename _Tp, int m, int n>
inline _InputOutputArray::_InputOutputArray(const Matx<_Tp, m, n> &mtx)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, &mtx, Size(n, m));
}

template <typename _Tp>
inline _InputOutputArray::_InputOutputArray(const _Tp *vec, int n)
{
    init(FIXED_TYPE + FIXED_SIZE + MATX + DataType<_Tp>::type + ACCESS_RW, vec, Size(n, 1));
}

inline _InputOutputArray::_InputOutputArray(const Mat &m) { init(FIXED_TYPE + FIXED_SIZE + MAT + ACCESS_RW, &m); }

inline _InputOutputArray::_InputOutputArray(const std::vector<Mat> &vec)
{
    init(FIXED_SIZE + STD_VECTOR_MAT + ACCESS_RW, &vec);
}
