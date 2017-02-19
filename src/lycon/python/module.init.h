namespace lycon
{

// The Python exception class used for Lycon errors
static PyObject* pycon_error = nullptr;

PyObject* get_pycon_error()
{
    return pycon_error;
}
} // namespace lycon

static const char* module_name = "_lycon";

static const char* module_docstring = "Lycon image library";

static PyMethodDef module_methods[] = {
    DECLARE_METHOD(load), DECLARE_METHOD(save), DECLARE_METHOD(resize), {NULL, NULL, 0, NULL}};

#if PY_MAJOR_VERSION >= 3

static int pycon_traverse(PyObject* m, visitproc visit, void* arg)
{
    Py_VISIT(pycon_error);
    return 0;
}

static int pycon_clear(PyObject* m)
{
    Py_CLEAR(pycon_error);
    return 0;
}

static struct PyModuleDef module_defs = {PyModuleDef_HEAD_INIT, module_name, NULL, 0, module_methods, NULL,
                                         pycon_traverse,        pycon_clear, NULL};

#define MODULE_INIT_SIGNATURE PyMODINIT_FUNC PyInit__lycon(void)
#define MODULE_INIT_RETURN_ON_ERROR NULL

#else

#define MODULE_INIT_SIGNATURE PyMODINIT_FUNC init_lycon(void)
#define MODULE_INIT_RETURN_ON_ERROR

#endif

MODULE_INIT_SIGNATURE
{
#if PY_MAJOR_VERSION >= 3
    PyObject* module = PyModule_Create(&module_defs);
#else
    PyObject* module = Py_InitModule3(module_name, module_methods, module_docstring);
#endif
    if (module == NULL)
    {
        return MODULE_INIT_RETURN_ON_ERROR;
    }

    // Initialize numpy
    import_array();

    // Create exception
    PyObject* module_dict = PyModule_GetDict(module);
    PyDict_SetItemString(module_dict, "__version__", PyString_FromString(LYCON_VERSION_STRING));
    pycon_error = PyErr_NewException((char*)("_lycon.PyconError"), NULL, NULL);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
