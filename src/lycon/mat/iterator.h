#pragma once

#include <iterator>

#include "lycon/defs.h"
#include "lycon/types.h"
#include "lycon/util/error.h"

namespace lycon
{
class LYCON_EXPORTS MatConstIterator
{
   public:
    typedef uchar *value_type;
    typedef ptrdiff_t difference_type;
    typedef const uchar **pointer;
    typedef uchar *reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator(const Mat *_m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat *_m, int _row, int _col = 0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat *_m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator(const Mat *_m, const int *_idx);
    //! copy constructor
    MatConstIterator(const MatConstIterator &it);

    //! copy operator
    MatConstIterator &operator=(const MatConstIterator &it);
    //! returns the current matrix element
    const uchar *operator*() const;
    //! returns the i-th matrix element, relative to the current
    const uchar *operator[](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator &operator+=(ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator &operator-=(ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator &operator--();
    //! decrements the iterator
    MatConstIterator operator--(int);
    //! increments the iterator
    MatConstIterator &operator++();
    //! increments the iterator
    MatConstIterator operator++(int);
    //! returns the current iterator position
    Point pos() const;
    //! returns the current iterator position
    void pos(int *_idx) const;

    ptrdiff_t lpos() const;
    void seek(ptrdiff_t ofs, bool relative = false);
    void seek(const int *_idx, bool relative = false);

    const Mat *m;
    size_t elemSize;
    const uchar *ptr;
    const uchar *sliceStart;
    const uchar *sliceEnd;
};

template <typename _Tp>
class MatConstIterator_ : public MatConstIterator
{
   public:
    typedef _Tp value_type;
    typedef ptrdiff_t difference_type;
    typedef const _Tp *pointer;
    typedef const _Tp &reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! default constructor
    MatConstIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatConstIterator_(const Mat_<_Tp> *_m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp> *_m, int _row, int _col = 0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp> *_m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatConstIterator_(const Mat_<_Tp> *_m, const int *_idx);
    //! copy constructor
    MatConstIterator_(const MatConstIterator_ &it);

    //! copy operator
    MatConstIterator_ &operator=(const MatConstIterator_ &it);
    //! returns the current matrix element
    const _Tp &operator*() const;
    //! returns the i-th matrix element, relative to the current
    const _Tp &operator[](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatConstIterator_ &operator+=(ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatConstIterator_ &operator-=(ptrdiff_t ofs);
    //! decrements the iterator
    MatConstIterator_ &operator--();
    //! decrements the iterator
    MatConstIterator_ operator--(int);
    //! increments the iterator
    MatConstIterator_ &operator++();
    //! increments the iterator
    MatConstIterator_ operator++(int);
    //! returns the current iterator position
    Point pos() const;
};

template <typename _Tp>
class MatIterator_ : public MatConstIterator_<_Tp>
{
   public:
    typedef _Tp *pointer;
    typedef _Tp &reference;

    typedef std::random_access_iterator_tag iterator_category;

    //! the default constructor
    MatIterator_();
    //! constructor that sets the iterator to the beginning of the matrix
    MatIterator_(Mat_<_Tp> *_m);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp> *_m, int _row, int _col = 0);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp> *_m, Point _pt);
    //! constructor that sets the iterator to the specified element of the matrix
    MatIterator_(Mat_<_Tp> *_m, const int *_idx);
    //! copy constructor
    MatIterator_(const MatIterator_ &it);
    //! copy operator
    MatIterator_ &operator=(const MatIterator_<_Tp> &it);

    //! returns the current matrix element
    _Tp &operator*() const;
    //! returns the i-th matrix element, relative to the current
    _Tp &operator[](ptrdiff_t i) const;

    //! shifts the iterator forward by the specified number of elements
    MatIterator_ &operator+=(ptrdiff_t ofs);
    //! shifts the iterator backward by the specified number of elements
    MatIterator_ &operator-=(ptrdiff_t ofs);
    //! decrements the iterator
    MatIterator_ &operator--();
    //! decrements the iterator
    MatIterator_ operator--(int);
    //! increments the iterator
    MatIterator_ &operator++();
    //! increments the iterator
    MatIterator_ operator++(int);
};

class LYCON_EXPORTS NAryMatIterator
{
   public:
    //! the default constructor
    NAryMatIterator();
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat **arrays, uchar **ptrs, int narrays = -1);
    //! the full constructor taking arbitrary number of n-dim matrices
    NAryMatIterator(const Mat **arrays, Mat *planes, int narrays = -1);
    //! the separate iterator initialization method
    void init(const Mat **arrays, Mat *planes, uchar **ptrs, int narrays = -1);

    //! proceeds to the next plane of every iterated matrix
    NAryMatIterator &operator++();
    //! proceeds to the next plane of every iterated matrix (postfix increment operator)
    NAryMatIterator operator++(int);

    //! the iterated arrays
    const Mat **arrays;
    //! the current planes
    Mat *planes;
    //! data pointers
    uchar **ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;

   protected:
    int iterdepth;
    size_t idx;
};
}
