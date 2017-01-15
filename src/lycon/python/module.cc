#include <Python.h>

#include "lycon/io/io.h"
#include "lycon/python/interop.h"
#include "lycon/python/macros.h"
#include "lycon/transform/resize.h"

#define LYCON_IMPORT_ARRAY
#include "lycon/python/numpy.h"

using namespace lycon;

#include "lycon/python/module.io.h"
#include "lycon/python/module.transform.h"

static const char* module_name = "_lycon";

static PyMethodDef module_methods[] = {
    DECLARE_METHOD(load), DECLARE_METHOD(save), DECLARE_METHOD(resize), {NULL, NULL, 0, NULL}};

static const char* module_docstring = "Lycon image library";

namespace lycon
{
static PyObject* pycon_error = nullptr;

PyObject* get_pycon_error()
{
    return pycon_error;
}
}

extern "C" PyMODINIT_FUNC init_lycon(void)
{
    PyObject* m = Py_InitModule3(module_name, module_methods, module_docstring);
    if (m == NULL)
    {
        return;
    }

    // Initialize numpy
    import_array();

    // Create exception
    PyObject* module_dict = PyModule_GetDict(m);
    PyDict_SetItemString(module_dict, "__version__", PyString_FromString(LYCON_VERSION_STRING));
    pycon_error = PyErr_NewException((char*)("_lycon.PyconError"), NULL, NULL);
}
