#pragma once

#include <Python.h>

#include <string>
#include <vector>

#include "lycon/mat/allocator.h"
#include "lycon/mat/umat_data.h"
#include "lycon/python/compat.h"
#include "lycon/python/macros.h"
#include "lycon/types.h"

namespace lycon
{
class NumpyAllocator : public MatAllocator
{
  public:
    NumpyAllocator();
    ~NumpyAllocator();

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const;

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags,
                       UMatUsageFlags usageFlags) const;

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const;

    void deallocate(UMatData* u) const;

    const MatAllocator* stdAllocator;

    static NumpyAllocator& getNumpyAllocator();
};

bool mat_from_ndarray(PyObject* py_obj, Mat& mat, bool allow_copy);

PyObject* ndarray_from_mat(const Mat& mat);

std::string string_from_pyobject(PyObject* obj);

Size size_from_pyobject(PyObject* obj);

template <typename _Tp> std::vector<_Tp> vector_from_pyobject(PyObject* obj)
{
    std::vector<_Tp> output_vec;
    PYCON_ASSERT_NOT_NONE(obj);
    LYCON_ASSERT(PySequence_Check(obj))
    PyObject* seq = PySequence_Fast(obj, "seq_extract");
    PYCON_ASSERT_NOT_NONE(seq)
    int num_elems = (int)PySequence_Fast_GET_SIZE(seq);
    output_vec.reserve(num_elems);
    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (int i = 0; i < num_elems; i++)
    {
        PyObject* item = items[i];
        if (PyInt_Check(item))
        {
            int v = (int)PyInt_AsLong(item);
            if (v == -1 && PyErr_Occurred())
                break;
            output_vec.push_back(static_cast<_Tp>(v));
        }
        else if (PyLong_Check(item))
        {
            int v = (int)PyLong_AsLong(item);
            if (v == -1 && PyErr_Occurred())
                break;
            output_vec.push_back(static_cast<_Tp>(v));
        }
        else if (PyFloat_Check(item))
        {
            double v = PyFloat_AsDouble(item);
            if (PyErr_Occurred())
                break;
            output_vec.push_back(static_cast<_Tp>(v));
        }
        else
        {
            break;
        }
    }
    Py_DECREF(seq);
    LYCON_ASSERT(output_vec.size() == num_elems);
    return output_vec;
}
}
