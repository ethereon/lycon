#pragma once

#include "lycon/defs.h"
#include "lycon/mat/allocator.h"
#include "lycon/mat/io_array.h"
#include "lycon/mat/iterator.h"
#include "lycon/mat/size.h"
#include "lycon/mat/step.h"
#include "lycon/mat/umat_data.h"
#include "lycon/mat/vec.h"
#include "lycon/types.h"
#include "lycon/util/alloc.h"

namespace lycon
{
class LYCON_EXPORTS Mat
{
   public:
    /**
    These are various constructors that form a matrix. As noted in the AutomaticAllocation, often
    the default constructor is enough, and the proper matrix will be allocated by an OpenCV function.
    The constructed matrix can further be assigned to another matrix or matrix expression or can be
    allocated with Mat::create . In the former case, the old content is de-referenced.
     */
    Mat();

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    */
    Mat(int rows, int cols, int type);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
      */
    Mat(Size size, int type);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int rows, int cols, int type, const Scalar &s);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
      */
    Mat(Size size, int type, const Scalar &s);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    */
    Mat(int ndims, const int *sizes, int type);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    */
    Mat(const std::vector<int> &sizes, int type);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(int ndims, const int *sizes, int type, const Scalar &s);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param s An optional value to initialize each matrix element with. To set all the matrix elements to
    the particular value after the construction, use the assignment operator
    Mat::operator=(const Scalar& value) .
    */
    Mat(const std::vector<int> &sizes, int type, const Scalar &s);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    */
    Mat(const Mat &m);

    /** @overload
    @param rows Number of rows in a 2D array.
    @param cols Number of columns in a 2D array.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(int rows, int cols, int type, void *data, size_t step = AUTO_STEP);

    /** @overload
    @param size 2D array size: Size(cols, rows) . In the Size() constructor, the number of rows and the
    number of columns go in the reverse order.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param step Number of bytes each matrix row occupies. The value should include the padding bytes at
    the end of each row, if any. If the parameter is missing (set to AUTO_STEP ), no padding is assumed
    and the actual step is calculated as cols*elemSize(). See Mat::elemSize.
    */
    Mat(Size size, int type, void *data, size_t step = AUTO_STEP);

    /** @overload
    @param ndims Array dimensionality.
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(int ndims, const int *sizes, int type, void *data, const size_t *steps = 0);

    /** @overload
    @param sizes Array of integers specifying an n-dimensional array shape.
    @param type Array type. Use LYCON_8UC1, ..., LYCON_64FC4 to create 1-4 channel matrices, or
    LYCON_8UC(n), ..., LYCON_64FC(n) to create multi-channel (up to LYCON_CN_MAX channels) matrices.
    @param data Pointer to the user data. Matrix constructors that take data and step parameters do not
    allocate matrix data. Instead, they just initialize the matrix header that points to the specified
    data, which means that no data is copied. This operation is very efficient and can be used to
    process external data using OpenCV functions. The external data is not automatically deallocated, so
    you should take care of it.
    @param steps Array of ndims-1 steps in case of a multi-dimensional array (the last step is always
    set to the element size). If not specified, the matrix is assumed to be continuous.
    */
    Mat(const std::vector<int> &sizes, int type, void *data, const size_t *steps = 0);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param rowRange Range of the m rows to take. As usual, the range start is inclusive and the range
    end is exclusive. Use Range::all() to take all the rows.
    @param colRange Range of the m columns to take. Use Range::all() to take all the columns.
    */
    Mat(const Mat &m, const Range &rowRange, const Range &colRange = Range::all());

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param roi Region of interest.
    */
    Mat(const Mat &m, const Rect &roi);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat &m, const Range *ranges);

    /** @overload
    @param m Array that (as a whole or partly) is assigned to the constructed matrix. No data is copied
    by these constructors. Instead, the header pointing to m data or its sub-array is constructed and
    associated with it. The reference counter, if any, is incremented. So, when you modify the matrix
    formed using such a constructor, you also modify the corresponding elements of m . If you want to
    have an independent copy of the sub-array, use Mat::clone() .
    @param ranges Array of selected ranges of m along each dimensionality.
    */
    Mat(const Mat &m, const std::vector<Range> &ranges);

    /** @overload
    @param vec STL vector whose elements form the matrix. The matrix has a single column and the number
    of rows equal to the number of vector elements. Type of the matrix matches the type of vector
    elements. The constructor can handle arbitrary types, for which there is a properly declared
    DataType . This means that the vector elements must be primitive numbers or uni-type numerical
    tuples of numbers. Mixed-type structures are not supported. The corresponding constructor is
    explicit. Since STL vectors are not automatically converted to Mat instances, you should write
    Mat(vec) explicitly. Unless you copy the data into the matrix ( copyData=true ), no new elements
    will be added to the vector because it can potentially yield vector data reallocation, and, thus,
    the matrix data pointer will be invalid.
    @param copyData Flag to specify whether the underlying data of the STL vector should be copied
    to (true) or shared with (false) the newly constructed matrix. When the data is copied, the
    allocated buffer is managed using Mat reference counting mechanism. While the data is shared,
    the reference counter is NULL, and you should not deallocate the data until the matrix is not
    destructed.
    */
    template <typename _Tp>
    explicit Mat(const std::vector<_Tp> &vec, bool copyData = false);

    /** @overload
    */
    template <typename _Tp, int n>
    explicit Mat(const Vec<_Tp, n> &vec, bool copyData = true);

    /** @overload
    */
    template <typename _Tp, int m, int n>
    explicit Mat(const Matx<_Tp, m, n> &mtx, bool copyData = true);

    /** @overload
    */
    template <typename _Tp>
    explicit Mat(const Point_<_Tp> &pt, bool copyData = true);

    //! destructor - calls release()
    ~Mat();

    /** @brief assignment operators

    These are available assignment operators. Since they all are very different, make sure to read the
    operator parameters description.
    @param m Assigned, right-hand-side matrix. Matrix assignment is an O(1) operation. This means that
    no data is copied but the data is shared and the reference counter, if any, is incremented. Before
    assigning new data, the old data is de-referenced via Mat::release .
     */
    Mat &operator=(const Mat &m);

    /** @brief Creates a matrix header for the specified matrix row.

    The method makes a new header for the specified matrix row and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. Here is the example of one of the classical basic matrix processing operations,
    axpy, used by LU and many other algorithms:
    @code
        inline void matrix_axpy(Mat& A, int i, int j, double alpha)
        {
            A.row(i) += A.row(j)*alpha;
        }
    @endcode
    @note In the current implementation, the following code does not work as expected:
    @code
        Mat A;
        ...
        A.row(i) = A.row(j); // will not work
    @endcode
    This happens because A.row(i) forms a temporary header that is further assigned to another header.
    Remember that each of these operations is O(1), that is, no data is copied. Thus, the above
    assignment is not true if you may have expected the j-th row to be copied to the i-th row. To
    achieve that, you should either turn this simple assignment into an expression or use the
    Mat::copyTo method:
    @code
        Mat A;
        ...
        // works, but looks a bit obscure.
        A.row(i) = A.row(j) + 0;
        // this is a bit longer, but the recommended method.
        A.row(j).copyTo(A.row(i));
    @endcode
    @param y A 0-based row index.
     */
    Mat row(int y) const;

    /** @brief Creates a matrix header for the specified matrix column.

    The method makes a new header for the specified matrix column and returns it. This is an O(1)
    operation, regardless of the matrix size. The underlying data of the new matrix is shared with the
    original matrix. See also the Mat::row description.
    @param x A 0-based column index.
     */
    Mat col(int x) const;

    /** @brief Creates a matrix header for the specified row span.

    The method makes a new header for the specified row span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startrow An inclusive 0-based start index of the row span.
    @param endrow An exclusive 0-based ending index of the row span.
     */
    Mat rowRange(int startrow, int endrow) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat rowRange(const Range &r) const;

    /** @brief Creates a matrix header for the specified column span.

    The method makes a new header for the specified column span of the matrix. Similarly to Mat::row and
    Mat::col , this is an O(1) operation.
    @param startcol An inclusive 0-based start index of the column span.
    @param endcol An exclusive 0-based ending index of the column span.
     */
    Mat colRange(int startcol, int endcol) const;

    /** @overload
    @param r Range structure containing both the start and the end indices.
    */
    Mat colRange(const Range &r) const;

    /** @brief Extracts a diagonal from a matrix

    The method makes a new header for the specified matrix diagonal. The new matrix is represented as a
    single-column matrix. Similarly to Mat::row and Mat::col, this is an O(1) operation.
    @param d index of the diagonal, with the following values:
    - `d=0` is the main diagonal.
    - `d>0` is a diagonal from the lower half. For example, d=1 means the diagonal is set
      immediately below the main one.
    - `d<0` is a diagonal from the upper half. For example, d=-1 means the diagonal is set
      immediately above the main one.
     */
    Mat diag(int d = 0) const;

    /** @brief creates a diagonal matrix

    The method creates a square diagonal matrix from specified main diagonal.
    @param d One-dimensional matrix that represents the main diagonal.
     */
    static Mat diag(const Mat &d);

    /** @brief Creates a full copy of the array and the underlying data.

    The method creates a full copy of the array. The original step[] is not taken into account. So, the
    array copy is a continuous array occupying total()*elemSize() bytes.
     */
    Mat clone() const;

    /** @brief Copies the matrix to another one.

    The method copies the matrix data to another matrix. Before copying the data, the method invokes :
    @code
        m.create(this->size(), this->type());
    @endcode
    so that the destination matrix is reallocated if needed. While m.copyTo(m); works flawlessly, the
    function does not handle the case of a partial overlap between the source and the destination
    matrices.

    When the operation mask is specified, if the Mat::create call shown above reallocates the matrix,
    the newly allocated matrix is initialized with all zeros before copying the data.
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
     */
    void copyTo(OutputArray m) const;

    /** @overload
    @param m Destination matrix. If it does not have a proper size or type before the operation, it is
    reallocated.
    @param mask Operation mask. Its non-zero elements indicate which matrix elements need to be copied.
    The mask has to be of type LYCON_8U and can have 1 or multiple channels.
    */
    void copyTo(OutputArray m, InputArray mask) const;

    /** @brief Converts an array to another data type with optional scaling.

    The method converts source pixel values to the target data type. saturate_cast\<\> is applied at
    the end to avoid possible overflows:

    \f[m(x,y) = saturate \_ cast<rType>( \alpha (*this)(x,y) +  \beta )\f]
    @param m output matrix; if it does not have a proper size or type before the operation, it is
    reallocated.
    @param rtype desired output matrix type or, rather, the depth since the number of channels are the
    same as the input has; if rtype is negative, the output matrix will have the same type as the input.
    @param alpha optional scale factor.
    @param beta optional delta added to the scaled values.
     */
    void convertTo(OutputArray m, int rtype, double alpha = 1, double beta = 0) const;

    /** @brief Provides a functional form of convertTo.

    This is an internally used method called by the @ref MatrixExpressions engine.
    @param m Destination array.
    @param type Desired destination array depth (or -1 if it should be the same as the source type).
     */
    void assignTo(Mat &m, int type = -1) const;

    /** @brief Sets all or some of the array elements to the specified value.
    @param s Assigned scalar converted to the actual array type.
    */
    Mat &operator=(const Scalar &s);

    /** @brief Sets all or some of the array elements to the specified value.

    This is an advanced variant of the Mat::operator=(const Scalar& s) operator.
    @param value Assigned scalar converted to the actual array type.
    @param mask Operation mask of the same size as \*this.
     */
    Mat &setTo(InputArray value, InputArray mask = noArray());

    /** @brief Changes the shape and/or the number of channels of a 2D matrix without copying the data.

    The method makes a new matrix header for \*this elements. The new matrix may have a different size
    and/or different number of channels. Any combination is possible if:
    -   No extra elements are included into the new matrix and no elements are excluded. Consequently,
        the product rows\*cols\*channels() must stay the same after the transformation.
    -   No data is copied. That is, this is an O(1) operation. Consequently, if you change the number of
        rows, or the operation changes the indices of elements row in some other way, the matrix must be
        continuous. See Mat::isContinuous .

    For example, if there is a set of 3D points stored as an STL vector, and you want to represent the
    points as a 3xN matrix, do the following:
    @code
        std::vector<Point3f> vec;
        ...
        Mat pointMat = Mat(vec). // convert vector to Mat, O(1) operation
                          reshape(1). // make Nx3 1-channel matrix out of Nx1 3-channel.
                                      // Also, an O(1) operation
                             t(); // finally, transpose the Nx3 matrix.
                                  // This involves copying all the elements
    @endcode
    @param cn New number of channels. If the parameter is 0, the number of channels remains the same.
    @param rows New number of rows. If the parameter is 0, the number of rows remains the same.
     */
    Mat reshape(int cn, int rows = 0) const;

    /** @overload */
    Mat reshape(int cn, int newndims, const int *newsz) const;

    /** @brief Computes a cross-product of two 3-element vectors.

    The method computes a cross-product of two 3-element vectors. The vectors must be 3-element
    floating-point vectors of the same shape and size. The result is another 3-element vector of the
    same shape and type as operands.
    @param m Another cross-product operand.
     */
    Mat cross(InputArray m) const;

    /** @brief Computes a dot-product of two vectors.

    The method computes a dot-product of two matrices. If the matrices are not single-column or
    single-row vectors, the top-to-bottom left-to-right scan ordering is used to treat them as 1D
    vectors. The vectors must have the same size and type. If the matrices have more than one channel,
    the dot products from all the channels are summed together.
    @param m another dot-product operand.
     */
    double dot(InputArray m) const;

    /** @brief Allocates new array data if needed.

    This is one of the key Mat methods. Most new-style OpenCV functions and methods that produce arrays
    call this method for each output array. The method uses the following algorithm:

    -# If the current array shape and the type match the new ones, return immediately. Otherwise,
       de-reference the previous data by calling Mat::release.
    -# Initialize the new header.
    -# Allocate the new data of total()\*elemSize() bytes.
    -# Allocate the new, associated with the data, reference counter and set it to 1.

    Such a scheme makes the memory management robust and efficient at the same time and helps avoid
    extra typing for you. This means that usually there is no need to explicitly allocate output arrays.
    That is, instead of writing:
    @code
        Mat color;
        ...
        Mat gray(color.rows, color.cols, color.depth());
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    you can simply write:
    @code
        Mat color;
        ...
        Mat gray;
        cvtColor(color, gray, COLOR_BGR2GRAY);
    @endcode
    because cvtColor, as well as the most of OpenCV functions, calls Mat::create() for the output array
    internally.
    @param rows New number of rows.
    @param cols New number of columns.
    @param type New matrix type.
     */
    void create(int rows, int cols, int type);

    /** @overload
    @param size Alternative new matrix size specification: Size(cols, rows)
    @param type New matrix type.
    */
    void create(Size size, int type);

    /** @overload
    @param ndims New array dimensionality.
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(int ndims, const int *sizes, int type);

    /** @overload
    @param sizes Array of integers specifying a new array shape.
    @param type New matrix type.
    */
    void create(const std::vector<int> &sizes, int type);

    /** @brief Increments the reference counter.

    The method increments the reference counter associated with the matrix data. If the matrix header
    points to an external data set (see Mat::Mat ), the reference counter is NULL, and the method has no
    effect in this case. Normally, to avoid memory leaks, the method should not be called explicitly. It
    is called implicitly by the matrix assignment operator. The reference counter increment is an atomic
    operation on the platforms that support it. Thus, it is safe to operate on the same matrices
    asynchronously in different threads.
     */
    void addref();

    /** @brief Decrements the reference counter and deallocates the matrix if needed.

    The method decrements the reference counter associated with the matrix data. When the reference
    counter reaches 0, the matrix data is deallocated and the data and the reference counter pointers
    are set to NULL's. If the matrix header points to an external data set (see Mat::Mat ), the
    reference counter is NULL, and the method has no effect in this case.

    This method can be called manually to force the matrix data deallocation. But since this method is
    automatically called in the destructor, or by any other method that changes the data pointer, it is
    usually not needed. The reference counter decrement and check for 0 is an atomic operation on the
    platforms that support it. Thus, it is safe to operate on the same matrices asynchronously in
    different threads.
     */
    void release();

    //! deallocates the matrix data
    void deallocate();
    //! internal use function; properly re-allocates _size, _step arrays
    void copySize(const Mat &m);

    /** @brief Reserves space for the certain number of rows.

    The method reserves space for sz rows. If the matrix already has enough space to store sz rows,
    nothing happens. If the matrix is reallocated, the first Mat::rows rows are preserved. The method
    emulates the corresponding method of the STL vector class.
    @param sz Number of rows.
     */
    void reserve(size_t sz);

    /** @brief Changes the number of matrix rows.

    The methods change the number of matrix rows. If the matrix is reallocated, the first
    min(Mat::rows, sz) rows are preserved. The methods emulate the corresponding methods of the STL
    vector class.
    @param sz New number of rows.
     */
    void resize(size_t sz);

    /** @overload
    @param sz New number of rows.
    @param s Value assigned to the newly added elements.
     */
    void resize(size_t sz, const Scalar &s);

    //! internal function
    void push_back_(const void *elem);

    /** @brief Adds elements to the bottom of the matrix.

    The methods add one or more elements to the bottom of the matrix. They emulate the corresponding
    method of the STL vector class. When elem is Mat , its type and the number of columns must be the
    same as in the container matrix.
    @param elem Added element(s).
     */
    template <typename _Tp>
    void push_back(const _Tp &elem);

    /** @overload
    @param elem Added element(s).
    */
    template <typename _Tp>
    void push_back(const Mat_<_Tp> &elem);

    /** @overload
    @param m Added line(s).
    */
    void push_back(const Mat &m);

    /** @brief Removes elements from the bottom of the matrix.

    The method removes one or more rows from the bottom of the matrix.
    @param nelems Number of removed rows. If it is greater than the total number of rows, an exception
    is thrown.
     */
    void pop_back(size_t nelems = 1);

    /** @brief Locates the matrix header within a parent matrix.

    After you extracted a submatrix from a matrix using Mat::row, Mat::col, Mat::rowRange,
    Mat::colRange, and others, the resultant submatrix points just to the part of the original big
    matrix. However, each submatrix contains information (represented by datastart and dataend
    fields) that helps reconstruct the original matrix size and the position of the extracted
    submatrix within the original matrix. The method locateROI does exactly that.
    @param wholeSize Output parameter that contains the size of the whole matrix containing *this*
    as a part.
    @param ofs Output parameter that contains an offset of *this* inside the whole matrix.
     */
    void locateROI(Size &wholeSize, Point &ofs) const;

    /** @brief Adjusts a submatrix size and position within the parent matrix.

    The method is complimentary to Mat::locateROI . The typical use of these functions is to determine
    the submatrix position within the parent matrix and then shift the position somehow. Typically, it
    can be required for filtering operations when pixels outside of the ROI should be taken into
    account. When all the method parameters are positive, the ROI needs to grow in all directions by the
    specified amount, for example:
    @code
        A.adjustROI(2, 2, 2, 2);
    @endcode
    In this example, the matrix size is increased by 4 elements in each direction. The matrix is shifted
    by 2 elements to the left and 2 elements up, which brings in all the necessary pixels for the
    filtering with the 5x5 kernel.

    adjustROI forces the adjusted ROI to be inside of the parent matrix that is boundaries of the
    adjusted ROI are constrained by boundaries of the parent matrix. For example, if the submatrix A is
    located in the first row of a parent matrix and you called A.adjustROI(2, 2, 2, 2) then A will not
    be increased in the upward direction.

    The function is used internally by the OpenCV filtering functions, like filter2D , morphological
    operations, and so on.
    @param dtop Shift of the top submatrix boundary upwards.
    @param dbottom Shift of the bottom submatrix boundary downwards.
    @param dleft Shift of the left submatrix boundary to the left.
    @param dright Shift of the right submatrix boundary to the right.
    @sa copyMakeBorder
     */
    Mat &adjustROI(int dtop, int dbottom, int dleft, int dright);

    /** @brief Extracts a rectangular submatrix.

    The operators make a new header for the specified sub-array of \*this . They are the most
    generalized forms of Mat::row, Mat::col, Mat::rowRange, and Mat::colRange . For example,
    `A(Range(0, 10), Range::all())` is equivalent to `A.rowRange(0, 10)`. Similarly to all of the above,
    the operators are O(1) operations, that is, no matrix data is copied.
    @param rowRange Start and end row of the extracted submatrix. The upper boundary is not included. To
    select all the rows, use Range::all().
    @param colRange Start and end column of the extracted submatrix. The upper boundary is not included.
    To select all the columns, use Range::all().
     */
    Mat operator()(Range rowRange, Range colRange) const;

    /** @overload
    @param roi Extracted submatrix specified as a rectangle.
    */
    Mat operator()(const Rect &roi) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()(const Range *ranges) const;

    /** @overload
    @param ranges Array of selected ranges along each array dimension.
    */
    Mat operator()(const std::vector<Range> &ranges) const;

    // //! converts header to CvMat; no data is copied
    // operator CvMat() const;
    // //! converts header to CvMatND; no data is copied
    // operator CvMatND() const;
    // //! converts header to IplImage; no data is copied
    // operator IplImage() const;

    template <typename _Tp>
    operator std::vector<_Tp>() const;
    template <typename _Tp, int n>
    operator Vec<_Tp, n>() const;
    template <typename _Tp, int m, int n>
    operator Matx<_Tp, m, n>() const;

    /** @brief Reports whether the matrix is continuous or not.

    The method returns true if the matrix elements are stored continuously without gaps at the end of
    each row. Otherwise, it returns false. Obviously, 1x1 or 1xN matrices are always continuous.
    Matrices created with Mat::create are always continuous. But if you extract a part of the matrix
    using Mat::col, Mat::diag, and so on, or constructed a matrix header for externally allocated data,
    such matrices may no longer have this property.

    The continuity flag is stored as a bit in the Mat::flags field and is computed automatically when
    you construct a matrix header. Thus, the continuity check is a very fast operation, though
    theoretically it could be done as follows:
    @code
        // alternative implementation of Mat::isContinuous()
        bool myCheckMatContinuity(const Mat& m)
        {
            //return (m.flags & Mat::CONTINUOUS_FLAG) != 0;
            return m.rows == 1 || m.step == m.cols*m.elemSize();
        }
    @endcode
    The method is used in quite a few of OpenCV functions. The point is that element-wise operations
    (such as arithmetic and logical operations, math functions, alpha blending, color space
    transformations, and others) do not depend on the image geometry. Thus, if all the input and output
    arrays are continuous, the functions can process them as very long single-row vectors. The example
    below illustrates how an alpha-blending function can be implemented:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            LYCON_ASSERT( src1.type() == src2.type() &&
                       src1.type() == LYCON_MAKETYPE(DataType<T>::depth, 4) &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            // here is the idiom: check the arrays for continuity and,
            // if this is the case,
            // treat the arrays as 1D vectors
            if( src1.isContinuous() && src2.isContinuous() && dst.isContinuous() )
            {
                size.width *= size.height;
                size.height = 1;
            }
            size.width *= 4;

            for( int i = 0; i < size.height; i++ )
            {
                // when the arrays are continuous,
                // the outer loop is executed only once
                const T* ptr1 = src1.ptr<T>(i);
                const T* ptr2 = src2.ptr<T>(i);
                T* dptr = dst.ptr<T>(i);

                for( int j = 0; j < size.width; j += 4 )
                {
                    float alpha = ptr1[j+3]*inv_scale, beta = ptr2[j+3]*inv_scale;
                    dptr[j] = saturate_cast<T>(ptr1[j]*alpha + ptr2[j]*beta);
                    dptr[j+1] = saturate_cast<T>(ptr1[j+1]*alpha + ptr2[j+1]*beta);
                    dptr[j+2] = saturate_cast<T>(ptr1[j+2]*alpha + ptr2[j+2]*beta);
                    dptr[j+3] = saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale);
                }
            }
        }
    @endcode
    This approach, while being very simple, can boost the performance of a simple element-operation by
    10-20 percents, especially if the image is rather small and the operation is quite simple.

    Another OpenCV idiom in this function, a call of Mat::create for the destination array, that
    allocates the destination array unless it already has the proper size and type. And while the newly
    allocated arrays are always continuous, you still need to check the destination array because
    Mat::create does not always allocate a new matrix.
     */
    bool isContinuous() const;

    //! returns true if the matrix is a submatrix of another matrix
    bool isSubmatrix() const;

    /** @brief Returns the matrix element size in bytes.

    The method returns the matrix element size in bytes. For example, if the matrix type is LYCON_16SC3 ,
    the method returns 3\*sizeof(short) or 6.
     */
    size_t elemSize() const;

    /** @brief Returns the size of each matrix element channel in bytes.

    The method returns the matrix element channel size in bytes, that is, it ignores the number of
    channels. For example, if the matrix type is LYCON_16SC3 , the method returns sizeof(short) or 2.
     */
    size_t elemSize1() const;

    /** @brief Returns the type of a matrix element.

    The method returns a matrix element type. This is an identifier compatible with the CvMat type
    system, like LYCON_16SC3 or 16-bit signed 3-channel array, and so on.
     */
    int type() const;

    /** @brief Returns the depth of a matrix element.

    The method returns the identifier of the matrix element depth (the type of each individual channel).
    For example, for a 16-bit signed element array, the method returns LYCON_16S . A complete list of
    matrix types contains the following values:
    -   LYCON_8U - 8-bit unsigned integers ( 0..255 )
    -   LYCON_8S - 8-bit signed integers ( -128..127 )
    -   LYCON_16U - 16-bit unsigned integers ( 0..65535 )
    -   LYCON_16S - 16-bit signed integers ( -32768..32767 )
    -   LYCON_32S - 32-bit signed integers ( -2147483648..2147483647 )
    -   LYCON_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
    -   LYCON_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
     */
    int depth() const;

    /** @brief Returns the number of matrix channels.

    The method returns the number of matrix channels.
     */
    int channels() const;

    /** @brief Returns a normalized step.

    The method returns a matrix step divided by Mat::elemSize1() . It can be useful to quickly access an
    arbitrary matrix element.
     */
    size_t step1(int i = 0) const;

    /** @brief Returns true if the array has no elements.

    The method returns true if Mat::total() is 0 or if Mat::data is NULL. Because of pop_back() and
    resize() methods `M.total() == 0` does not imply that `M.data == NULL`.
     */
    bool empty() const;

    /** @brief Returns the total number of array elements.

    The method returns the number of array elements (a number of pixels if the array represents an
    image).
     */
    size_t total() const;

    //! returns N if the matrix is 1-channel (N x ptdim) or ptdim-channel (1 x N) or (N x 1); negative number otherwise
    int checkVector(int elemChannels, int depth = -1, bool requireContinuous = true) const;

    /** @brief Returns a pointer to the specified matrix row.

    The methods return `uchar*` or typed pointer to the specified matrix row. See the sample in
    Mat::isContinuous to know how to use these methods.
    @param i0 A 0-based row index.
     */
    uchar *ptr(int i0 = 0);
    /** @overload */
    const uchar *ptr(int i0 = 0) const;

    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    uchar *ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    const uchar *ptr(int row, int col) const;

    /** @overload */
    uchar *ptr(int i0, int i1, int i2);
    /** @overload */
    const uchar *ptr(int i0, int i1, int i2) const;

    /** @overload */
    uchar *ptr(const int *idx);
    /** @overload */
    const uchar *ptr(const int *idx) const;
    /** @overload */
    template <int n>
    uchar *ptr(const Vec<int, n> &idx);
    /** @overload */
    template <int n>
    const uchar *ptr(const Vec<int, n> &idx) const;

    /** @overload */
    template <typename _Tp>
    _Tp *ptr(int i0 = 0);
    /** @overload */
    template <typename _Tp>
    const _Tp *ptr(int i0 = 0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template <typename _Tp>
    _Tp *ptr(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template <typename _Tp>
    const _Tp *ptr(int row, int col) const;
    /** @overload */
    template <typename _Tp>
    _Tp *ptr(int i0, int i1, int i2);
    /** @overload */
    template <typename _Tp>
    const _Tp *ptr(int i0, int i1, int i2) const;
    /** @overload */
    template <typename _Tp>
    _Tp *ptr(const int *idx);
    /** @overload */
    template <typename _Tp>
    const _Tp *ptr(const int *idx) const;
    /** @overload */
    template <typename _Tp, int n>
    _Tp *ptr(const Vec<int, n> &idx);
    /** @overload */
    template <typename _Tp, int n>
    const _Tp *ptr(const Vec<int, n> &idx) const;

    /** @brief Returns a reference to the specified array element.

    The template methods return a reference to the specified array element. For the sake of higher
    performance, the index range checks are only performed in the Debug configuration.

    Note that the variants with a single index (i) can be used to access elements of single-row or
    single-column 2-dimensional arrays. That is, if, for example, A is a 1 x N floating-point matrix and
    B is an M x 1 integer matrix, you can simply write `A.at<float>(k+4)` and `B.at<int>(2*i+1)`
    instead of `A.at<float>(0,k+4)` and `B.at<int>(2*i+1,0)`, respectively.

    The example below initializes a Hilbert matrix:
    @code
        Mat H(100, 100, LYCON_64F);
        for(int i = 0; i < H.rows; i++)
            for(int j = 0; j < H.cols; j++)
                H.at<double>(i,j)=1./(i+j+1);
    @endcode

    Keep in mind that the size identifier used in the at operator cannot be chosen at random. It depends
    on the image from which you are trying to retrieve the data. The table below gives a better insight in this:
     - If matrix is of type `LYCON_8U` then use `Mat.at<uchar>(y,x)`.
     - If matrix is of type `LYCON_8S` then use `Mat.at<schar>(y,x)`.
     - If matrix is of type `LYCON_16U` then use `Mat.at<ushort>(y,x)`.
     - If matrix is of type `LYCON_16S` then use `Mat.at<short>(y,x)`.
     - If matrix is of type `LYCON_32S`  then use `Mat.at<int>(y,x)`.
     - If matrix is of type `LYCON_32F`  then use `Mat.at<float>(y,x)`.
     - If matrix is of type `LYCON_64F` then use `Mat.at<double>(y,x)`.

    @param i0 Index along the dimension 0
     */
    template <typename _Tp>
    _Tp &at(int i0 = 0);
    /** @overload
    @param i0 Index along the dimension 0
    */
    template <typename _Tp>
    const _Tp &at(int i0 = 0) const;
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template <typename _Tp>
    _Tp &at(int row, int col);
    /** @overload
    @param row Index along the dimension 0
    @param col Index along the dimension 1
    */
    template <typename _Tp>
    const _Tp &at(int row, int col) const;

    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template <typename _Tp>
    _Tp &at(int i0, int i1, int i2);
    /** @overload
    @param i0 Index along the dimension 0
    @param i1 Index along the dimension 1
    @param i2 Index along the dimension 2
    */
    template <typename _Tp>
    const _Tp &at(int i0, int i1, int i2) const;

    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template <typename _Tp>
    _Tp &at(const int *idx);
    /** @overload
    @param idx Array of Mat::dims indices.
    */
    template <typename _Tp>
    const _Tp &at(const int *idx) const;

    /** @overload */
    template <typename _Tp, int n>
    _Tp &at(const Vec<int, n> &idx);
    /** @overload */
    template <typename _Tp, int n>
    const _Tp &at(const Vec<int, n> &idx) const;

    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template <typename _Tp>
    _Tp &at(Point pt);
    /** @overload
    special versions for 2D arrays (especially convenient for referencing image pixels)
    @param pt Element position specified as Point(j,i) .
    */
    template <typename _Tp>
    const _Tp &at(Point pt) const;

    /** @brief Returns the matrix iterator and sets it to the first matrix element.

    The methods return the matrix read-only or read-write iterators. The use of matrix iterators is very
    similar to the use of bi-directional STL iterators. In the example below, the alpha blending
    function is rewritten using the matrix iterators:
    @code
        template<typename T>
        void alphaBlendRGBA(const Mat& src1, const Mat& src2, Mat& dst)
        {
            typedef Vec<T, 4> VT;

            const float alpha_scale = (float)std::numeric_limits<T>::max(),
                        inv_scale = 1.f/alpha_scale;

            LYCON_ASSERT( src1.type() == src2.type() &&
                       src1.type() == DataType<VT>::type &&
                       src1.size() == src2.size());
            Size size = src1.size();
            dst.create(size, src1.type());

            MatConstIterator_<VT> it1 = src1.begin<VT>(), it1_end = src1.end<VT>();
            MatConstIterator_<VT> it2 = src2.begin<VT>();
            MatIterator_<VT> dst_it = dst.begin<VT>();

            for( ; it1 != it1_end; ++it1, ++it2, ++dst_it )
            {
                VT pix1 = *it1, pix2 = *it2;
                float alpha = pix1[3]*inv_scale, beta = pix2[3]*inv_scale;
                *dst_it = VT(saturate_cast<T>(pix1[0]*alpha + pix2[0]*beta),
                             saturate_cast<T>(pix1[1]*alpha + pix2[1]*beta),
                             saturate_cast<T>(pix1[2]*alpha + pix2[2]*beta),
                             saturate_cast<T>((1 - (1-alpha)*(1-beta))*alpha_scale));
            }
        }
    @endcode
     */
    template <typename _Tp>
    MatIterator_<_Tp> begin();
    template <typename _Tp>
    MatConstIterator_<_Tp> begin() const;

    /** @brief Returns the matrix iterator and sets it to the after-last matrix element.

    The methods return the matrix read-only or read-write iterators, set to the point following the last
    matrix element.
     */
    template <typename _Tp>
    MatIterator_<_Tp> end();
    template <typename _Tp>
    MatConstIterator_<_Tp> end() const;

    /** @brief Runs the given functor over all matrix elements in parallel.

    The operation passed as argument has to be a function pointer, a function object or a lambda(C++11).

    Example 1. All of the operations below put 0xFF the first channel of all matrix elements:
    @code
        Mat image(1920, 1080, LYCON_8UC3);
        typedef cv::Point3_<uint8_t> Pixel;

        // first. raw pointer access.
        for (int r = 0; r < image.rows; ++r) {
            Pixel* ptr = image.ptr<Pixel>(0, r);
            const Pixel* ptr_end = ptr + image.cols;
            for (; ptr != ptr_end; ++ptr) {
                ptr->x = 255;
            }
        }

        // Using MatIterator. (Simple but there are a Iterator's overhead)
        for (Pixel &p : cv::Mat_<Pixel>(image)) {
            p.x = 255;
        }

        // Parallel execution with function object.
        struct Operator {
            void operator ()(Pixel &pixel, const int * position) {
                pixel.x = 255;
            }
        };
        image.forEach<Pixel>(Operator());

        // Parallel execution using C++11 lambda.
        image.forEach<Pixel>([](Pixel &p, const int * position) -> void {
            p.x = 255;
        });
    @endcode
    Example 2. Using the pixel's position:
    @code
        // Creating 3D matrix (255 x 255 x 255) typed uint8_t
        // and initialize all elements by the value which equals elements position.
        // i.e. pixels (x,y,z) = (1,2,3) is (b,g,r) = (1,2,3).

        int sizes[] = { 255, 255, 255 };
        typedef cv::Point3_<uint8_t> Pixel;

        Mat_<Pixel> image = Mat::zeros(3, sizes, LYCON_8UC3);

        image.forEach<Pixel>([&](Pixel& pixel, const int position[]) -> void {
            pixel.x = position[0];
            pixel.y = position[1];
            pixel.z = position[2];
        });
    @endcode
     */
    template <typename _Tp, typename Functor>
    void forEach(const Functor &operation);
    /** @overload */
    template <typename _Tp, typename Functor>
    void forEach(const Functor &operation) const;

    Mat(Mat &&m);
    Mat &operator=(Mat &&m);

    enum
    {
        MAGIC_VAL = 0x42FF0000,
        AUTO_STEP = 0,
        CONTINUOUS_FLAG = LYCON_MAT_CONT_FLAG,
        SUBMATRIX_FLAG = LYCON_SUBMAT_FLAG
    };
    enum
    {
        MAGIC_MASK = 0xFFFF0000,
        TYPE_MASK = 0x00000FFF,
        DEPTH_MASK = 7
    };

    /*! includes several bit-fields:
         - the magic signature
         - continuity flag
         - depth
         - number of channels
     */
    int flags;
    //! the matrix dimensionality, >= 2
    int dims;
    //! the number of rows and columns or (-1, -1) when the matrix has more than 2 dimensions
    int rows, cols;
    //! pointer to the data
    uchar *data;

    //! helper fields used in locateROI and adjustROI
    const uchar *datastart;
    const uchar *dataend;
    const uchar *datalimit;

    //! custom allocator
    MatAllocator *allocator;
    //! and the standard allocator
    static MatAllocator *getStdAllocator();
    static MatAllocator *getDefaultAllocator();
    static void setDefaultAllocator(MatAllocator *allocator);

    //! interaction with UMat
    UMatData *u;

    MatSize size;
    MatStep step;

   protected:
    template <typename _Tp, typename Functor>
    void forEach_impl(const Functor &operation);
};

template <typename _Tp>
class Mat_ : public Mat
{
   public:
    typedef _Tp value_type;
    typedef typename DataType<_Tp>::channel_type channel_type;
    typedef MatIterator_<_Tp> iterator;
    typedef MatConstIterator_<_Tp> const_iterator;

    //! default constructor
    Mat_();
    //! equivalent to Mat(_rows, _cols, DataType<_Tp>::type)
    Mat_(int _rows, int _cols);
    //! constructor that sets each matrix element to specified value
    Mat_(int _rows, int _cols, const _Tp &value);
    //! equivalent to Mat(_size, DataType<_Tp>::type)
    explicit Mat_(Size _size);
    //! constructor that sets each matrix element to specified value
    Mat_(Size _size, const _Tp &value);
    //! n-dim array constructor
    Mat_(int _ndims, const int *_sizes);
    //! n-dim array constructor that sets each matrix element to specified value
    Mat_(int _ndims, const int *_sizes, const _Tp &value);
    //! copy/conversion contructor. If m is of different type, it's converted
    Mat_(const Mat &m);
    //! copy constructor
    Mat_(const Mat_ &m);
    //! constructs a matrix on top of user-allocated data. step is in bytes(!!!), regardless of the type
    Mat_(int _rows, int _cols, _Tp *_data, size_t _step = AUTO_STEP);
    //! constructs n-dim matrix on top of user-allocated data. steps are in bytes(!!!), regardless of the type
    Mat_(int _ndims, const int *_sizes, _Tp *_data, const size_t *_steps = 0);
    //! selects a submatrix
    Mat_(const Mat_ &m, const Range &rowRange, const Range &colRange = Range::all());
    //! selects a submatrix
    Mat_(const Mat_ &m, const Rect &roi);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_ &m, const Range *ranges);
    //! selects a submatrix, n-dim version
    Mat_(const Mat_ &m, const std::vector<Range> &ranges);
    //! makes a matrix out of Vec, std::vector, Point_ or Point3_. The matrix will have a single column
    explicit Mat_(const std::vector<_Tp> &vec, bool copyData = false);
    template <int n>
    explicit Mat_(const Vec<typename DataType<_Tp>::channel_type, n> &vec, bool copyData = true);
    template <int m, int n>
    explicit Mat_(const Matx<typename DataType<_Tp>::channel_type, m, n> &mtx, bool copyData = true);
    explicit Mat_(const Point_<typename DataType<_Tp>::channel_type> &pt, bool copyData = true);

    Mat_ &operator=(const Mat &m);
    Mat_ &operator=(const Mat_ &m);
    //! set all the elements to s.
    Mat_ &operator=(const _Tp &s);

    //! iterators; they are smart enough to skip gaps in the end of rows
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    //! template methods for for operation over all matrix elements.
    // the operations take care of skipping gaps in the end of rows (if any)
    template <typename Functor>
    void forEach(const Functor &operation);
    template <typename Functor>
    void forEach(const Functor &operation) const;

    //! equivalent to Mat::create(_rows, _cols, DataType<_Tp>::type)
    void create(int _rows, int _cols);
    //! equivalent to Mat::create(_size, DataType<_Tp>::type)
    void create(Size _size);
    //! equivalent to Mat::create(_ndims, _sizes, DatType<_Tp>::type)
    void create(int _ndims, const int *_sizes);
    //! cross-product
    Mat_ cross(const Mat_ &m) const;
    //! data type conversion
    template <typename T2>
    operator Mat_<T2>() const;
    //! overridden forms of Mat::row() etc.
    Mat_ row(int y) const;
    Mat_ col(int x) const;
    Mat_ diag(int d = 0) const;
    Mat_ clone() const;

    //! overridden forms of Mat::elemSize() etc.
    size_t elemSize() const;
    size_t elemSize1() const;
    int type() const;
    int depth() const;
    int channels() const;
    size_t step1(int i = 0) const;
    //! returns step()/sizeof(_Tp)
    size_t stepT(int i = 0) const;

    //! some more overriden methods
    Mat_ &adjustROI(int dtop, int dbottom, int dleft, int dright);
    Mat_ operator()(const Range &rowRange, const Range &colRange) const;
    Mat_ operator()(const Rect &roi) const;
    Mat_ operator()(const Range *ranges) const;
    Mat_ operator()(const std::vector<Range> &ranges) const;

    //! more convenient forms of row and element access operators
    _Tp *operator[](int y);
    const _Tp *operator[](int y) const;

    //! returns reference to the specified element
    _Tp &operator()(const int *idx);
    //! returns read-only reference to the specified element
    const _Tp &operator()(const int *idx) const;

    //! returns reference to the specified element
    template <int n>
    _Tp &operator()(const Vec<int, n> &idx);
    //! returns read-only reference to the specified element
    template <int n>
    const _Tp &operator()(const Vec<int, n> &idx) const;

    //! returns reference to the specified element (1D case)
    _Tp &operator()(int idx0);
    //! returns read-only reference to the specified element (1D case)
    const _Tp &operator()(int idx0) const;
    //! returns reference to the specified element (2D case)
    _Tp &operator()(int row, int col);
    //! returns read-only reference to the specified element (2D case)
    const _Tp &operator()(int row, int col) const;
    //! returns reference to the specified element (3D case)
    _Tp &operator()(int idx0, int idx1, int idx2);
    //! returns read-only reference to the specified element (3D case)
    const _Tp &operator()(int idx0, int idx1, int idx2) const;

    _Tp &operator()(Point pt);
    const _Tp &operator()(Point pt) const;

    //! conversion to vector.
    operator std::vector<_Tp>() const;
    //! conversion to Vec
    template <int n>
    operator Vec<typename DataType<_Tp>::channel_type, n>() const;
    //! conversion to Matx
    template <int m, int n>
    operator Matx<typename DataType<_Tp>::channel_type, m, n>() const;

    Mat_(Mat_ &&m);
    Mat_ &operator=(Mat_ &&m);

    Mat_(Mat &&m);
    Mat_ &operator=(Mat &&m);
};

typedef Mat_<uchar> Mat1b;
typedef Mat_<Vec2b> Mat2b;
typedef Mat_<Vec3b> Mat3b;
typedef Mat_<Vec4b> Mat4b;

typedef Mat_<short> Mat1s;
typedef Mat_<Vec2s> Mat2s;
typedef Mat_<Vec3s> Mat3s;
typedef Mat_<Vec4s> Mat4s;

typedef Mat_<ushort> Mat1w;
typedef Mat_<Vec2w> Mat2w;
typedef Mat_<Vec3w> Mat3w;
typedef Mat_<Vec4w> Mat4w;

typedef Mat_<int> Mat1i;
typedef Mat_<Vec2i> Mat2i;
typedef Mat_<Vec3i> Mat3i;
typedef Mat_<Vec4i> Mat4i;

typedef Mat_<float> Mat1f;
typedef Mat_<Vec2f> Mat2f;
typedef Mat_<Vec3f> Mat3f;
typedef Mat_<Vec4f> Mat4f;

typedef Mat_<double> Mat1d;
typedef Mat_<Vec2d> Mat2d;
typedef Mat_<Vec3d> Mat3d;
typedef Mat_<Vec4d> Mat4d;

#include "lycon/mat/mat_impl.h"

#include "lycon/mat/iterator_impl.h"

#include "lycon/mat/io_array_impl.h"
}
