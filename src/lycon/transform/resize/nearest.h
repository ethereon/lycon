class resizeNNInvoker : public ParallelLoopBody
{
   public:
    resizeNNInvoker(const Mat& _src, Mat& _dst, int* _x_ofs, int _pix_size4, double _ify)
        : ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4), ify(_ify)
    {
    }

    virtual void operator()(const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x, pix_size = (int)src.elemSize();

        for (y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step * y;
            int sy = std::min(fast_floor(y * ify), ssize.height - 1);
            const uchar* S = src.ptr(sy);

            switch (pix_size)
            {
                case 1:
                    for (x = 0; x <= dsize.width - 2; x += 2)
                    {
                        uchar t0 = S[x_ofs[x]];
                        uchar t1 = S[x_ofs[x + 1]];
                        D[x] = t0;
                        D[x + 1] = t1;
                    }

                    for (; x < dsize.width; x++) D[x] = S[x_ofs[x]];
                    break;
                case 2:
                    for (x = 0; x < dsize.width; x++) *(ushort*)(D + x * 2) = *(ushort*)(S + x_ofs[x]);
                    break;
                case 3:
                    for (x = 0; x < dsize.width; x++, D += 3)
                    {
                        const uchar* _tS = S + x_ofs[x];
                        D[0] = _tS[0];
                        D[1] = _tS[1];
                        D[2] = _tS[2];
                    }
                    break;
                case 4:
                    for (x = 0; x < dsize.width; x++) *(int*)(D + x * 4) = *(int*)(S + x_ofs[x]);
                    break;
                case 6:
                    for (x = 0; x < dsize.width; x++, D += 6)
                    {
                        const ushort* _tS = (const ushort*)(S + x_ofs[x]);
                        ushort* _tD = (ushort*)D;
                        _tD[0] = _tS[0];
                        _tD[1] = _tS[1];
                        _tD[2] = _tS[2];
                    }
                    break;
                case 8:
                    for (x = 0; x < dsize.width; x++, D += 8)
                    {
                        const int* _tS = (const int*)(S + x_ofs[x]);
                        int* _tD = (int*)D;
                        _tD[0] = _tS[0];
                        _tD[1] = _tS[1];
                    }
                    break;
                case 12:
                    for (x = 0; x < dsize.width; x++, D += 12)
                    {
                        const int* _tS = (const int*)(S + x_ofs[x]);
                        int* _tD = (int*)D;
                        _tD[0] = _tS[0];
                        _tD[1] = _tS[1];
                        _tD[2] = _tS[2];
                    }
                    break;
                default:
                    for (x = 0; x < dsize.width; x++, D += pix_size)
                    {
                        const int* _tS = (const int*)(S + x_ofs[x]);
                        int* _tD = (int*)D;
                        for (int k = 0; k < pix_size4; k++) _tD[k] = _tS[k];
                    }
            }
        }
    }

   private:
    const Mat src;
    Mat dst;
    int *x_ofs, pix_size4;
    double ify;

    resizeNNInvoker(const resizeNNInvoker&);
    resizeNNInvoker& operator=(const resizeNNInvoker&);
};

static void resizeNN(const Mat& src, Mat& dst, double fx, double fy)
{
    Size ssize = src.size(), dsize = dst.size();
    AutoBuffer<int> _x_ofs(dsize.width);
    int* x_ofs = _x_ofs;
    int pix_size = (int)src.elemSize();
    int pix_size4 = (int)(pix_size / sizeof(int));
    double ifx = 1. / fx, ify = 1. / fy;
    int x;

    for (x = 0; x < dsize.width; x++)
    {
        int sx = fast_floor(x * ifx);
        x_ofs[x] = std::min(sx, ssize.width - 1) * pix_size;
    }

    Range range(0, dsize.height);
    resizeNNInvoker invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}
