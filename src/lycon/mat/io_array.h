#pragma once

#include <vector>

#include "lycon/defs.h"
#include "lycon/types.h"

namespace lycon
{
enum
{
    ACCESS_READ = 1 << 24,
    ACCESS_WRITE = 1 << 25,
    ACCESS_RW = 3 << 24,
    ACCESS_MASK = ACCESS_RW,
    ACCESS_FAST = 1 << 26
};

class LYCON_EXPORTS _OutputArray;

class LYCON_EXPORTS _InputArray
{
   public:
    enum
    {
        KIND_SHIFT = 16,
        FIXED_TYPE = 0x8000 << KIND_SHIFT,
        FIXED_SIZE = 0x4000 << KIND_SHIFT,
        KIND_MASK = 31 << KIND_SHIFT,

        NONE = 0 << KIND_SHIFT,
        MAT = 1 << KIND_SHIFT,
        MATX = 2 << KIND_SHIFT,
        STD_VECTOR = 3 << KIND_SHIFT,
        STD_VECTOR_VECTOR = 4 << KIND_SHIFT,
        STD_VECTOR_MAT = 5 << KIND_SHIFT,
        EXPR = 6 << KIND_SHIFT,
        STD_BOOL_VECTOR = 12 << KIND_SHIFT,
    };

    _InputArray();
    _InputArray(int _flags, void *_obj);
    _InputArray(const Mat &m);
    _InputArray(const std::vector<Mat> &vec);
    template <typename _Tp>
    _InputArray(const Mat_<_Tp> &m);
    template <typename _Tp>
    _InputArray(const std::vector<_Tp> &vec);
    _InputArray(const std::vector<bool> &vec);
    template <typename _Tp>
    _InputArray(const std::vector<std::vector<_Tp>> &vec);
    template <typename _Tp>
    _InputArray(const std::vector<Mat_<_Tp>> &vec);
    template <typename _Tp>
    _InputArray(const _Tp *vec, int n);
    template <typename _Tp, int m, int n>
    _InputArray(const Matx<_Tp, m, n> &matx);
    _InputArray(const double &val);

    Mat getMat(int idx = -1) const;
    Mat getMat_(int idx = -1) const;
    void getMatVector(std::vector<Mat> &mv) const;

    int getFlags() const;
    void *getObj() const;
    Size getSz() const;

    int kind() const;
    int dims(int i = -1) const;
    int cols(int i = -1) const;
    int rows(int i = -1) const;
    Size size(int i = -1) const;
    int sizend(int *sz, int i = -1) const;
    bool sameSize(const _InputArray &arr) const;
    size_t total(int i = -1) const;
    int type(int i = -1) const;
    int depth(int i = -1) const;
    int channels(int i = -1) const;
    bool isContinuous(int i = -1) const;
    bool isSubmatrix(int i = -1) const;
    bool empty() const;
    void copyTo(const _OutputArray &arr) const;
    void copyTo(const _OutputArray &arr, const _InputArray &mask) const;
    size_t offset(int i = -1) const;
    size_t step(int i = -1) const;
    bool isMat() const;
    bool isMatx() const;
    bool isMatVector() const;
    bool isVector() const;
    ~_InputArray();

   protected:
    int flags;
    void *obj;
    Size sz;

    void init(int _flags, const void *_obj);
    void init(int _flags, const void *_obj, Size _sz);
};

class LYCON_EXPORTS _OutputArray : public _InputArray
{
   public:
    enum
    {
        DEPTH_MASK_8U = 1 << LYCON_8U,
        DEPTH_MASK_8S = 1 << LYCON_8S,
        DEPTH_MASK_16U = 1 << LYCON_16U,
        DEPTH_MASK_16S = 1 << LYCON_16S,
        DEPTH_MASK_32S = 1 << LYCON_32S,
        DEPTH_MASK_32F = 1 << LYCON_32F,
        DEPTH_MASK_64F = 1 << LYCON_64F,
        DEPTH_MASK_ALL = (DEPTH_MASK_64F << 1) - 1,
        DEPTH_MASK_ALL_BUT_8S = DEPTH_MASK_ALL & ~DEPTH_MASK_8S,
        DEPTH_MASK_FLT = DEPTH_MASK_32F + DEPTH_MASK_64F
    };

    _OutputArray();
    _OutputArray(int _flags, void *_obj);
    _OutputArray(Mat &m);
    _OutputArray(std::vector<Mat> &vec);
    template <typename _Tp>
    _OutputArray(std::vector<_Tp> &vec);
    _OutputArray(std::vector<bool> &vec);
    template <typename _Tp>
    _OutputArray(std::vector<std::vector<_Tp>> &vec);
    template <typename _Tp>
    _OutputArray(std::vector<Mat_<_Tp>> &vec);
    template <typename _Tp>
    _OutputArray(Mat_<_Tp> &m);
    template <typename _Tp>
    _OutputArray(_Tp *vec, int n);
    template <typename _Tp, int m, int n>
    _OutputArray(Matx<_Tp, m, n> &matx);

    _OutputArray(const Mat &m);
    _OutputArray(const std::vector<Mat> &vec);
    template <typename _Tp>
    _OutputArray(const std::vector<_Tp> &vec);
    template <typename _Tp>
    _OutputArray(const std::vector<std::vector<_Tp>> &vec);
    template <typename _Tp>
    _OutputArray(const std::vector<Mat_<_Tp>> &vec);
    template <typename _Tp>
    _OutputArray(const Mat_<_Tp> &m);
    template <typename _Tp>
    _OutputArray(const _Tp *vec, int n);
    template <typename _Tp, int m, int n>
    _OutputArray(const Matx<_Tp, m, n> &matx);

    bool fixedSize() const;
    bool fixedType() const;
    bool needed() const;
    Mat &getMatRef(int i = -1) const;
    void create(Size sz, int type, int i = -1, bool allowTransposed = false, int fixedDepthMask = 0) const;
    void create(int rows, int cols, int type, int i = -1, bool allowTransposed = false, int fixedDepthMask = 0) const;
    void create(int dims, const int *size, int type, int i = -1, bool allowTransposed = false,
                int fixedDepthMask = 0) const;
    void createSameSize(const _InputArray &arr, int mtype) const;
    void release() const;
    void clear() const;
    void setTo(const _InputArray &value, const _InputArray &mask = _InputArray()) const;

    void assign(const Mat &m) const;
};

class LYCON_EXPORTS _InputOutputArray : public _OutputArray
{
   public:
    _InputOutputArray();
    _InputOutputArray(int _flags, void *_obj);
    _InputOutputArray(Mat &m);
    _InputOutputArray(std::vector<Mat> &vec);
    template <typename _Tp>
    _InputOutputArray(std::vector<_Tp> &vec);
    _InputOutputArray(std::vector<bool> &vec);
    template <typename _Tp>
    _InputOutputArray(std::vector<std::vector<_Tp>> &vec);
    template <typename _Tp>
    _InputOutputArray(std::vector<Mat_<_Tp>> &vec);
    template <typename _Tp>
    _InputOutputArray(Mat_<_Tp> &m);
    template <typename _Tp>
    _InputOutputArray(_Tp *vec, int n);
    template <typename _Tp, int m, int n>
    _InputOutputArray(Matx<_Tp, m, n> &matx);

    _InputOutputArray(const Mat &m);
    _InputOutputArray(const std::vector<Mat> &vec);
    template <typename _Tp>
    _InputOutputArray(const std::vector<_Tp> &vec);
    template <typename _Tp>
    _InputOutputArray(const std::vector<std::vector<_Tp>> &vec);
    template <typename _Tp>
    _InputOutputArray(const std::vector<Mat_<_Tp>> &vec);
    template <typename _Tp>
    _InputOutputArray(const Mat_<_Tp> &m);
    template <typename _Tp>
    _InputOutputArray(const _Tp *vec, int n);
    template <typename _Tp, int m, int n>
    _InputOutputArray(const Matx<_Tp, m, n> &matx);
};

typedef const _InputArray &InputArray;
typedef InputArray InputArrayOfArrays;
typedef const _OutputArray &OutputArray;
typedef OutputArray OutputArrayOfArrays;
typedef const _InputOutputArray &InputOutputArray;
typedef InputOutputArray InputOutputArrayOfArrays;

LYCON_EXPORTS InputOutputArray noArray();
}
