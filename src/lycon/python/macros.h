#pragma once

#include "lycon/python/gil.h"

namespace lycon
{
PyObject *get_pycon_error();

#define ERROR_FENCED(expr)                            \
    try                                               \
    {                                                 \
        expr;                                         \
    }                                                 \
    catch (const lycon::RuntimeError &e)              \
    {                                                 \
        PyErr_SetString(get_pycon_error(), e.what()); \
        return 0;                                     \
    }

#define DEFINE_FUNCTION(name)                                                  \
    static inline PyObject *pycon_impl_##name(PyObject *self, PyObject *args); \
    extern "C" PyObject *pycon_##name(PyObject *self, PyObject *args)          \
    {                                                                          \
        ERROR_FENCED(return pycon_impl_##name(self, args))                     \
    }                                                                          \
    static inline PyObject *pycon_impl_##name(PyObject *self, PyObject *args)

#define DECLARE_METHOD(name)                  \
    {                                         \
        #name, pycon_##name, METH_VARARGS, "" \
    }

#define PYCON_IS_NOT_NONE(obj) (obj != nullptr) && (obj != Py_None)

#define PYCON_ASSERT_NOT_NONE(obj) LYCON_ASSERT(PYCON_IS_NOT_NONE(obj))

#define PYCON_WITHOUT_GIL(expr)   \
    {                             \
        PyReleaseGIL release_gil; \
        expr;                     \
    }
}
