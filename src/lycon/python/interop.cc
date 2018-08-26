#include "lycon/python/interop.h"
#include "lycon/mat/mat.h"
#include "lycon/python/gil.h"
#include "lycon/python/macros.h"
#include "lycon/python/numpy.h"
#include "lycon/util/auto_buffer.h"

namespace lycon
{
static NumpyAllocator g_numpyAllocator;

static int failmsg(const char* fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

static const int lycon_from_numpy_type(const int numpy_type)
{
    switch (numpy_type)
    {
    case NPY_UBYTE: return LYCON_8U;
    case NPY_BYTE: return LYCON_8S;
    case NPY_USHORT: return LYCON_16U;
    case NPY_SHORT: return LYCON_16S;
    case NPY_INT32: return LYCON_32S;
    case NPY_FLOAT: return LYCON_32F;
    case NPY_DOUBLE: return LYCON_64F;
    }
    return -1;
}

static int numpy_from_lycon_type(const int lycon_type)
{
    switch (lycon_type)
    {
    case LYCON_8U: return NPY_UBYTE;
    case LYCON_8S: return NPY_BYTE;
    case LYCON_16U: return NPY_USHORT;
    case LYCON_16S: return NPY_SHORT;
    case LYCON_32S: return NPY_INT;
    case LYCON_32F: return NPY_FLOAT;
    case LYCON_64F: return NPY_DOUBLE;
    }
    const int f = (int)(sizeof(size_t) / 8);
    return f * NPY_ULONGLONG + (f ^ 1) * NPY_UINT;
}

NumpyAllocator::NumpyAllocator() : stdAllocator(Mat::getStdAllocator())
{
}

NumpyAllocator::~NumpyAllocator()
{
}

UMatData* NumpyAllocator::allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
{
    UMatData* u = new UMatData(this);
    u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*)o);
    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*)o);
    for (int i = 0; i < dims - 1; i++)
        step[i] = (size_t)_strides[i];
    step[dims - 1] = LYCON_ELEM_SIZE(type);
    u->size = sizes[0] * step[0];
    u->userdata = o;
    return u;
}

UMatData* NumpyAllocator::allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags,
                                   UMatUsageFlags usageFlags) const
{
    if (data != 0)
    {
        LYCON_ERROR("The data should be NULL!");
    }
    PyEnsureGIL gil;

    int depth = LYCON_MAT_DEPTH(type);
    int cn = LYCON_MAT_CN(type);
    int typenum = numpy_from_lycon_type(depth);
    int i, dims = dims0;
    AutoBuffer<npy_intp> _sizes(dims + 1);
    for (i = 0; i < dims; i++)
        _sizes[i] = sizes[i];
    if (cn > 1)
        _sizes[dims++] = cn;
    PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
    if (!o)
        LYCON_ERROR("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims);
    return allocate(o, dims0, sizes, type, step);
}

bool NumpyAllocator::allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
{
    return stdAllocator->allocate(u, accessFlags, usageFlags);
}

void NumpyAllocator::deallocate(UMatData* u) const
{
    if (!u)
        return;
    PyEnsureGIL gil;
    LYCON_ASSERT(u->urefcount >= 0);
    LYCON_ASSERT(u->refcount >= 0);
    if (u->refcount == 0)
    {
        PyObject* o = (PyObject*)u->userdata;
        Py_XDECREF(o);
        delete u;
    }
}

NumpyAllocator& NumpyAllocator::getNumpyAllocator()
{
    return g_numpyAllocator;
}

bool mat_from_ndarray(PyObject* py_obj, Mat& mat, bool allow_copy)
{
    if (!py_obj || py_obj == Py_None)
    {
        // Create a new numpy compatible Mat
        if (!mat.data)
        {
            mat.allocator = &g_numpyAllocator;
        }
        return true;
    }

    if (PyInt_Check(py_obj))
    {
        // Integer scalar`
        double v[] = {static_cast<double>(PyInt_AsLong((PyObject*)py_obj)), 0., 0., 0.};
        mat = Mat(4, 1, LYCON_64F, v).clone();
        return true;
    }

    if (PyFloat_Check(py_obj))
    {
        // Float scalar
        double v[] = {PyFloat_AsDouble((PyObject*)py_obj), 0., 0., 0.};
        mat = Mat(4, 1, LYCON_64F, v).clone();
        return true;
    }

    if (PyTuple_Check(py_obj))
    {
        // Tuple
        int i, sz = (int)PyTuple_Size((PyObject*)py_obj);
        mat = Mat(sz, 1, LYCON_64F);
        for (i = 0; i < sz; i++)
        {
            PyObject* oi = PyTuple_GET_ITEM(py_obj, i);
            if (PyInt_Check(oi))
                mat.at<double>(i) = (double)PyInt_AsLong(oi);
            else if (PyFloat_Check(oi))
                mat.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("object is not a numerical tuple");
                mat.release();
                return false;
            }
        }
        return true;
    }

    if (!PyArray_Check(py_obj))
    {
        failmsg("object is not a numpy array, neither a scalar");
        return false;
    }

    // Convert from ndarray
    PyArrayObject* oarr = (PyArrayObject*)py_obj;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = lycon_from_numpy_type(typenum);

    if (type < 0)
    {
        if (typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG)
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = LYCON_32S;
        }
        else
        {
            failmsg("ndarray data type = %d is not supported", typenum);
            return false;
        }
    }

    int ndims = PyArray_NDIM(oarr);
    if (ndims >= LYCON_MAX_DIM)
    {
        failmsg("ndarray dimensionality (=%d) is too high", ndims);
        return false;
    }

    int size[LYCON_MAX_DIM + 1];
    size_t step[LYCON_MAX_DIM + 1];
    size_t elemsize = LYCON_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= LYCON_CN_MAX;

    for (int i = ndims - 1; i >= 0 && !needcopy; i--)
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if ((i == ndims - 1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims - 1 && _sizes[i] > 1 && _strides[i] < _strides[i + 1]))
            needcopy = true;
    }

    if (ismultichannel && _strides[1] != (npy_intp)elemsize * _sizes[2])
        needcopy = true;

    if (needcopy)
    {
        if (!allow_copy)
        {
            failmsg("Layout of the output array is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] "
                    "!= elemsize*nchannels)");
            return false;
        }

        if (needcast)
        {
            py_obj = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*)py_obj;
        }
        else
        {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            py_obj = (PyObject*)oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for (int i = ndims - 1; i >= 0; --i)
    {
        size[i] = (int)_sizes[i];
        if (size[i] > 1)
        {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        }
        else
        {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if (ndims == 0)
    {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if (ismultichannel)
    {
        ndims--;
        type |= LYCON_MAKETYPE(0, size[2]);
    }

    mat = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    mat.u = g_numpyAllocator.allocate(py_obj, ndims, size, type, step);
    mat.addref();

    if (!needcopy)
    {
        Py_INCREF(py_obj);
    }
    mat.allocator = &g_numpyAllocator;

    return true;
}

PyObject* ndarray_from_mat(const Mat& mat)
{
    if (!mat.data)
    {
        Py_RETURN_NONE;
    }
    Mat temp, *p = (Mat*)&mat;
    if (!p->u || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        PYCON_WITHOUT_GIL(mat.copyTo(temp));
        p = &temp;
    }
    PyObject* py_obj = (PyObject*)p->u->userdata;
    Py_INCREF(py_obj);
    return py_obj;
}

std::string string_from_pyobject(PyObject* object)
{
    PYCON_ASSERT_NOT_NONE(object);
    const char* str = PyString_AsString(object);
    LYCON_ASSERT(str)
    return std::string(str);
}

Size size_from_pyobject(PyObject* obj)
{
    std::vector<int> elems = vector_from_pyobject<int>(obj);
    return Size(elems[0], elems[1]);
}
}
