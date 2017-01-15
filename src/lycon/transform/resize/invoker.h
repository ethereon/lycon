template <typename HResize, typename VResize>
class resizeGeneric_Invoker : public ParallelLoopBody
{
   public:
    typedef typename HResize::value_type T;
    typedef typename HResize::buf_type WT;
    typedef typename HResize::alpha_type AT;

    resizeGeneric_Invoker(const Mat& _src, Mat& _dst, const int* _xofs, const int* _yofs, const AT* _alpha,
                          const AT* __beta, const Size& _ssize, const Size& _dsize, int _ksize, int _xmin, int _xmax)
        : ParallelLoopBody(),
          src(_src),
          dst(_dst),
          xofs(_xofs),
          yofs(_yofs),
          alpha(_alpha),
          _beta(__beta),
          ssize(_ssize),
          dsize(_dsize),
          ksize(_ksize),
          xmin(_xmin),
          xmax(_xmax)
    {
        LYCON_ASSERT(ksize <= MAX_ESIZE);
    }

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    virtual void operator()(const Range& range) const
    {
        int dy, cn = src.channels();
        HResize hresize;
        VResize vresize;

        int bufstep = (int)alignSize(dsize.width, 16);
        AutoBuffer<WT> _buffer(bufstep * ksize);
        const T* srows[MAX_ESIZE] = {0};
        WT* rows[MAX_ESIZE] = {0};
        int prev_sy[MAX_ESIZE];

        for (int k = 0; k < ksize; k++)
        {
            prev_sy[k] = -1;
            rows[k] = (WT*)_buffer + bufstep * k;
        }

        const AT* beta = _beta + ksize * range.start;

        for (dy = range.start; dy < range.end; dy++, beta += ksize)
        {
            int sy0 = yofs[dy], k0 = ksize, k1 = 0, ksize2 = ksize / 2;

            for (int k = 0; k < ksize; k++)
            {
                int sy = clip(sy0 - ksize2 + 1 + k, 0, ssize.height);
                for (k1 = std::max(k1, k); k1 < ksize; k1++)
                {
                    if (sy == prev_sy[k1])  // if the sy-th row has been computed already, reuse it.
                    {
                        if (k1 > k) memcpy(rows[k], rows[k1], bufstep * sizeof(rows[0][0]));
                        break;
                    }
                }
                if (k1 == ksize) k0 = std::min(k0, k);  // remember the first row that needs to be computed
                srows[k] = src.template ptr<T>(sy);
                prev_sy[k] = sy;
            }

            if (k0 < ksize)
                hresize((const T**)(srows + k0), (WT**)(rows + k0), ksize - k0, xofs, (const AT*)(alpha), ssize.width,
                        dsize.width, cn, xmin, xmax);
            vresize((const WT**)rows, (T*)(dst.data + dst.step * dy), beta, dsize.width);
        }
    }
#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

   private:
    Mat src;
    Mat dst;
    const int *xofs, *yofs;
    const AT *alpha, *_beta;
    Size ssize, dsize;
    const int ksize, xmin, xmax;

    resizeGeneric_Invoker& operator=(const resizeGeneric_Invoker&);
};

template <class HResize, class VResize>
static void resizeGeneric_(const Mat& src, Mat& dst, const int* xofs, const void* _alpha, const int* yofs,
                           const void* _beta, int xmin, int xmax, int ksize)
{
    typedef typename HResize::alpha_type AT;

    const AT* beta = (const AT*)_beta;
    Size ssize = src.size(), dsize = dst.size();
    int cn = src.channels();
    ssize.width *= cn;
    dsize.width *= cn;
    xmin *= cn;
    xmax *= cn;
    // image resize is a separable operation. In case of not too strong

    Range range(0, dsize.height);
    resizeGeneric_Invoker<HResize, VResize> invoker(src, dst, xofs, yofs, (const AT*)_alpha, beta, ssize, dsize, ksize,
                                                    xmin, xmax);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

typedef void (*ResizeFunc)(const Mat& src, Mat& dst, const int* xofs, const void* alpha, const int* yofs,
                           const void* beta, int xmin, int xmax, int ksize);

void resize(int src_type, const uchar* src_data, size_t src_step, int src_width, int src_height, uchar* dst_data,
            size_t dst_step, int dst_width, int dst_height, double inv_scale_x, double inv_scale_y, int interpolation)
{
    LYCON_ASSERT((dst_width * dst_height > 0) || (inv_scale_x > 0 && inv_scale_y > 0));
    if (inv_scale_x < DBL_EPSILON || inv_scale_y < DBL_EPSILON)
    {
        inv_scale_x = static_cast<double>(dst_width) / src_width;
        inv_scale_y = static_cast<double>(dst_height) / src_height;
    }

    static ResizeFunc linear_tab[] = {
        resizeGeneric_<HResizeLinear<uchar, int, short, INTER_RESIZE_COEF_SCALE, HResizeLinearVec_8u32s>,
                       VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
                                     VResizeLinearVec_32s8u> >,
        0, resizeGeneric_<HResizeLinear<ushort, float, float, 1, HResizeLinearVec_16u32f>,
                          VResizeLinear<ushort, float, float, Cast<float, ushort>, VResizeLinearVec_32f16u> >,
        resizeGeneric_<HResizeLinear<short, float, float, 1, HResizeLinearVec_16s32f>,
                       VResizeLinear<short, float, float, Cast<float, short>, VResizeLinearVec_32f16s> >,
        0, resizeGeneric_<HResizeLinear<float, float, float, 1, HResizeLinearVec_32f>,
                          VResizeLinear<float, float, float, Cast<float, float>, VResizeLinearVec_32f> >,
        resizeGeneric_<HResizeLinear<double, double, float, 1, HResizeNoVec>,
                       VResizeLinear<double, double, float, Cast<double, double>, VResizeNoVec> >,
        0};

    static ResizeFunc cubic_tab[] = {
        resizeGeneric_<HResizeCubic<uchar, int, short>,
                       VResizeCubic<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>,
                                    VResizeCubicVec_32s8u> >,
        0, resizeGeneric_<HResizeCubic<ushort, float, float>,
                          VResizeCubic<ushort, float, float, Cast<float, ushort>, VResizeCubicVec_32f16u> >,
        resizeGeneric_<HResizeCubic<short, float, float>,
                       VResizeCubic<short, float, float, Cast<float, short>, VResizeCubicVec_32f16s> >,
        0, resizeGeneric_<HResizeCubic<float, float, float>,
                          VResizeCubic<float, float, float, Cast<float, float>, VResizeCubicVec_32f> >,
        resizeGeneric_<HResizeCubic<double, double, float>,
                       VResizeCubic<double, double, float, Cast<double, double>, VResizeNoVec> >,
        0};

    static ResizeFunc lanczos4_tab[] = {
        resizeGeneric_<
            HResizeLanczos4<uchar, int, short>,
            VResizeLanczos4<uchar, int, short, FixedPtCast<int, uchar, INTER_RESIZE_COEF_BITS * 2>, VResizeNoVec> >,
        0, resizeGeneric_<HResizeLanczos4<ushort, float, float>,
                          VResizeLanczos4<ushort, float, float, Cast<float, ushort>, VResizeLanczos4Vec_32f16u> >,
        resizeGeneric_<HResizeLanczos4<short, float, float>,
                       VResizeLanczos4<short, float, float, Cast<float, short>, VResizeLanczos4Vec_32f16s> >,
        0, resizeGeneric_<HResizeLanczos4<float, float, float>,
                          VResizeLanczos4<float, float, float, Cast<float, float>, VResizeLanczos4Vec_32f> >,
        resizeGeneric_<HResizeLanczos4<double, double, float>,
                       VResizeLanczos4<double, double, float, Cast<double, double>, VResizeNoVec> >,
        0};

    static ResizeAreaFastFunc areafast_tab[] = {
        resizeAreaFast_<uchar, int, ResizeAreaFastVec<uchar, ResizeAreaFastVec_SIMD_8u> >,
        0,
        resizeAreaFast_<ushort, float, ResizeAreaFastVec<ushort, ResizeAreaFastVec_SIMD_16u> >,
        resizeAreaFast_<short, float, ResizeAreaFastVec<short, ResizeAreaFastVec_SIMD_16s> >,
        0,
        resizeAreaFast_<float, float, ResizeAreaFastVec_SIMD_32f>,
        resizeAreaFast_<double, double, ResizeAreaFastNoVec<double, double> >,
        0};

    static ResizeAreaFunc area_tab[] = {resizeArea_<uchar, float>,   0, resizeArea_<ushort, float>,
                                        resizeArea_<short, float>,   0, resizeArea_<float, float>,
                                        resizeArea_<double, double>, 0};

    int depth = LYCON_MAT_DEPTH(src_type), cn = LYCON_MAT_CN(src_type);
    double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;

    int iscale_x = saturate_cast<int>(scale_x);
    int iscale_y = saturate_cast<int>(scale_y);

    bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON && std::abs(scale_y - iscale_y) < DBL_EPSILON;

    Size dsize = Size(saturate_cast<int>(src_width * inv_scale_x), saturate_cast<int>(src_height * inv_scale_y));
    LYCON_ASSERT(dsize.area() > 0);

    Mat src(Size(src_width, src_height), src_type, const_cast<uchar*>(src_data), src_step);
    Mat dst(dsize, src_type, dst_data, dst_step);

    if (interpolation == INTER_NEAREST)
    {
        resizeNN(src, dst, inv_scale_x, inv_scale_y);
        return;
    }

    int k, sx, sy, dx, dy;

    {
        // in case of scale_x && scale_y is equal to 2
        // INTER_AREA (fast) also is equal to INTER_LINEAR
        if (interpolation == INTER_LINEAR && is_area_fast && iscale_x == 2 && iscale_y == 2) interpolation = INTER_AREA;

        // true "area" interpolation is only implemented for the case (scale_x <= 1 && scale_y <= 1).
        // In other cases it is emulated using some variant of bilinear interpolation
        if (interpolation == INTER_AREA && scale_x >= 1 && scale_y >= 1)
        {
            if (is_area_fast)
            {
                int area = iscale_x * iscale_y;
                size_t srcstep = src_step / src.elemSize1();
                AutoBuffer<int> _ofs(area + dsize.width * cn);
                int* ofs = _ofs;
                int* xofs = ofs + area;
                ResizeAreaFastFunc func = areafast_tab[depth];
                LYCON_ASSERT(func != 0);

                for (sy = 0, k = 0; sy < iscale_y; sy++)
                    for (sx = 0; sx < iscale_x; sx++) ofs[k++] = (int)(sy * srcstep + sx * cn);

                for (dx = 0; dx < dsize.width; dx++)
                {
                    int j = dx * cn;
                    sx = iscale_x * j;
                    for (k = 0; k < cn; k++) xofs[j + k] = sx + k;
                }

                func(src, dst, ofs, xofs, iscale_x, iscale_y);
                return;
            }

            ResizeAreaFunc func = area_tab[depth];
            LYCON_ASSERT(func != 0 && cn <= 4);

            AutoBuffer<DecimateAlpha> _xytab((src_width + src_height) * 2);
            DecimateAlpha *xtab = _xytab, *ytab = xtab + src_width * 2;

            int xtab_size = computeResizeAreaTab(src_width, dsize.width, cn, scale_x, xtab);
            int ytab_size = computeResizeAreaTab(src_height, dsize.height, 1, scale_y, ytab);

            AutoBuffer<int> _tabofs(dsize.height + 1);
            int* tabofs = _tabofs;
            for (k = 0, dy = 0; k < ytab_size; k++)
            {
                if (k == 0 || ytab[k].di != ytab[k - 1].di)
                {
                    assert(ytab[k].di == dy);
                    tabofs[dy++] = k;
                }
            }
            tabofs[dy] = ytab_size;

            func(src, dst, xtab, xtab_size, ytab, ytab_size, tabofs);
            return;
        }
    }

    int xmin = 0, xmax = dsize.width, width = dsize.width * cn;
    bool area_mode = interpolation == INTER_AREA;
    bool fixpt = depth == LYCON_8U;
    float fx, fy;
    ResizeFunc func = 0;
    int ksize = 0, ksize2;
    if (interpolation == INTER_CUBIC)
        ksize = 4, func = cubic_tab[depth];
    else if (interpolation == INTER_LANCZOS4)
        ksize = 8, func = lanczos4_tab[depth];
    else if (interpolation == INTER_LINEAR || interpolation == INTER_AREA)
        ksize = 2, func = linear_tab[depth];
    else
        LYCON_ERROR("Unknown interpolation method");
    ksize2 = ksize / 2;

    LYCON_ASSERT(func != 0);

    AutoBuffer<uchar> _buffer((width + dsize.height) * (sizeof(int) + sizeof(float) * ksize));
    int* xofs = (int*)(uchar*)_buffer;
    int* yofs = xofs + width;
    float* alpha = (float*)(yofs + dsize.height);
    short* ialpha = (short*)alpha;
    float* beta = alpha + width * ksize;
    short* ibeta = ialpha + width * ksize;
    float cbuf[MAX_ESIZE];

    for (dx = 0; dx < dsize.width; dx++)
    {
        if (!area_mode)
        {
            fx = (float)((dx + 0.5) * scale_x - 0.5);
            sx = fast_floor(fx);
            fx -= sx;
        }
        else
        {
            sx = fast_floor(dx * scale_x);
            fx = (float)((dx + 1) - (sx + 1) * inv_scale_x);
            fx = fx <= 0 ? 0.f : fx - fast_floor(fx);
        }

        if (sx < ksize2 - 1)
        {
            xmin = dx + 1;
            if (sx < 0 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4)) fx = 0, sx = 0;
        }

        if (sx + ksize2 >= src_width)
        {
            xmax = std::min(xmax, dx);
            if (sx >= src_width - 1 && (interpolation != INTER_CUBIC && interpolation != INTER_LANCZOS4))
                fx = 0, sx = src_width - 1;
        }

        for (k = 0, sx *= cn; k < cn; k++) xofs[dx * cn + k] = sx + k;

        if (interpolation == INTER_CUBIC)
            interpolateCubic(fx, cbuf);
        else if (interpolation == INTER_LANCZOS4)
            interpolateLanczos4(fx, cbuf);
        else
        {
            cbuf[0] = 1.f - fx;
            cbuf[1] = fx;
        }
        if (fixpt)
        {
            for (k = 0; k < ksize; k++)
                ialpha[dx * cn * ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
            for (; k < cn * ksize; k++) ialpha[dx * cn * ksize + k] = ialpha[dx * cn * ksize + k - ksize];
        }
        else
        {
            for (k = 0; k < ksize; k++) alpha[dx * cn * ksize + k] = cbuf[k];
            for (; k < cn * ksize; k++) alpha[dx * cn * ksize + k] = alpha[dx * cn * ksize + k - ksize];
        }
    }

    for (dy = 0; dy < dsize.height; dy++)
    {
        if (!area_mode)
        {
            fy = (float)((dy + 0.5) * scale_y - 0.5);
            sy = fast_floor(fy);
            fy -= sy;
        }
        else
        {
            sy = fast_floor(dy * scale_y);
            fy = (float)((dy + 1) - (sy + 1) * inv_scale_y);
            fy = fy <= 0 ? 0.f : fy - fast_floor(fy);
        }

        yofs[dy] = sy;
        if (interpolation == INTER_CUBIC)
            interpolateCubic(fy, cbuf);
        else if (interpolation == INTER_LANCZOS4)
            interpolateLanczos4(fy, cbuf);
        else
        {
            cbuf[0] = 1.f - fy;
            cbuf[1] = fy;
        }

        if (fixpt)
        {
            for (k = 0; k < ksize; k++) ibeta[dy * ksize + k] = saturate_cast<short>(cbuf[k] * INTER_RESIZE_COEF_SCALE);
        }
        else
        {
            for (k = 0; k < ksize; k++) beta[dy * ksize + k] = cbuf[k];
        }
    }

    func(src, dst, xofs, fixpt ? (void*)ialpha : (void*)alpha, yofs, fixpt ? (void*)ibeta : (void*)beta, xmin, xmax,
         ksize);
}

void resize(InputArray _src, OutputArray _dst, Size dsize, double inv_scale_x, double inv_scale_y, int interpolation)
{
    Size ssize = _src.size();

    LYCON_ASSERT(ssize.width > 0 && ssize.height > 0);
    LYCON_ASSERT(dsize.area() > 0 || (inv_scale_x > 0 && inv_scale_y > 0));
    if (dsize.area() == 0)
    {
        dsize = Size(saturate_cast<int>(ssize.width * inv_scale_x), saturate_cast<int>(ssize.height * inv_scale_y));
        LYCON_ASSERT(dsize.area() > 0);
    }
    else
    {
        inv_scale_x = (double)dsize.width / ssize.width;
        inv_scale_y = (double)dsize.height / ssize.height;
    }

    Mat src = _src.getMat();
    _dst.create(dsize, src.type());
    Mat dst = _dst.getMat();

    if (dsize == ssize)
    {
        // Source and destination are of same size. Use simple copy.
        src.copyTo(dst);
        return;
    }

    resize(src.type(), src.data, src.step, src.cols, src.rows, dst.data, dst.step, dst.cols, dst.rows, inv_scale_x,
           inv_scale_y, interpolation);
}
