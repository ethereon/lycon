DEFINE_FUNCTION(resize)
{
    PyObject* src_ndarray = nullptr;
    PyObject* py_dst_size = nullptr;
    PyObject* dst_ndarray = nullptr;
    int interpolation = INTER_LINEAR;

    if (PyArg_ParseTuple(args, "OO|iO:resize", &src_ndarray, &py_dst_size, &interpolation, &dst_ndarray))
    {
        Mat src_img;
        mat_from_ndarray(src_ndarray, src_img, true);

        Size dst_size = size_from_pyobject(py_dst_size);

        Mat dst_img;
        if (PYCON_IS_NOT_NONE(dst_ndarray))
        {
            mat_from_ndarray(dst_ndarray, dst_img, false);
            LYCON_ASSERT(dst_img.type() == src_img.type());
            LYCON_ASSERT(dst_img.rows == dst_size.height);
            LYCON_ASSERT(dst_img.cols == dst_size.width);
        }
        else
        {
            dst_img.allocator = &(NumpyAllocator::getNumpyAllocator());
        }
        PYCON_WITHOUT_GIL(resize(src_img, dst_img, dst_size, 0, 0, interpolation));

        return ndarray_from_mat(dst_img);
    }
    return Py_None;
}
