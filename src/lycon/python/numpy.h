#ifndef LYCON_IMPORT_ARRAY
// All sources besides the main module that calls import_array need to define this
// before importing the array header.
#define NO_IMPORT_ARRAY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL LYCON_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
