DEFINE_FUNCTION(load)
{
    PyObject* py_path = nullptr;
    int flags = IMREAD_COLOR;
    if (PyArg_ParseTuple(args, "O|i:load", &py_path, &flags))
    {
        std::string path = string_from_pyobject(py_path);
        Mat img;
        PYCON_WITHOUT_GIL(img = imread(path));
        return ndarray_from_mat(img);
    }
    return nullptr;
}

DEFINE_FUNCTION(save)
{
    PyObject* py_path = nullptr;
    PyObject* py_img = nullptr;
    PyObject* py_flags = nullptr;
    if (PyArg_ParseTuple(args, "OO|O:save", &py_path, &py_img, &py_flags))
    {
        std::string path = string_from_pyobject(py_path);

        Mat img;
        mat_from_ndarray(py_img, img, false);

        std::vector<int> options;
        if (PYCON_IS_NOT_NONE(py_flags))
        {
            options = vector_from_pyobject<int>(py_flags);
        }
        PYCON_WITHOUT_GIL(imwrite(path, img, options));
    }
    return Py_None;
}
