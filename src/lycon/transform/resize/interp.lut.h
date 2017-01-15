const int INTER_RESIZE_COEF_BITS = 11;
const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

const int INTER_REMAP_COEF_BITS = 15;
const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

static uchar NNDeltaTab_i[INTER_TAB_SIZE2][2];

static float BilinearTab_f[INTER_TAB_SIZE2][2][2];
static short BilinearTab_i[INTER_TAB_SIZE2][2][2];

#if LYCON_SSE2 || LYCON_NEON
static short BilinearTab_iC4_buf[INTER_TAB_SIZE2 + 2][2][8];
static short (*BilinearTab_iC4)[2][8] = (short (*)[2][8])alignPtr(BilinearTab_iC4_buf, 16);
#endif

static float BicubicTab_f[INTER_TAB_SIZE2][4][4];
static short BicubicTab_i[INTER_TAB_SIZE2][4][4];

static float Lanczos4Tab_f[INTER_TAB_SIZE2][8][8];
static short Lanczos4Tab_i[INTER_TAB_SIZE2][8][8];

static inline void interpolateLinear(float x, float* coeffs)
{
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

static inline void interpolateCubic(float x, float* coeffs)
{
    const float A = -0.75f;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void interpolateLanczos4(float x, float* coeffs)
{
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2] = {{1, 0},  {-s45, -s45}, {0, 1},  {s45, -s45},
                                   {-1, 0}, {s45, s45},   {0, -1}, {-s45, s45}};

    if (x < FLT_EPSILON)
    {
        for (int i = 0; i < 8; i++) coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }

    float sum = 0;
    double y0 = -(x + 3) * M_PI * 0.25, s0 = sin(y0), c0 = cos(y0);
    for (int i = 0; i < 8; i++)
    {
        double y = -(x + 3 - i) * M_PI * 0.25;
        coeffs[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        sum += coeffs[i];
    }

    sum = 1.f / sum;
    for (int i = 0; i < 8; i++) coeffs[i] *= sum;
}

static void initInterTab1D(int method, float* tab, int tabsz)
{
    float scale = 1.f / tabsz;
    if (method == INTER_LINEAR)
    {
        for (int i = 0; i < tabsz; i++, tab += 2) interpolateLinear(i * scale, tab);
    }
    else if (method == INTER_CUBIC)
    {
        for (int i = 0; i < tabsz; i++, tab += 4) interpolateCubic(i * scale, tab);
    }
    else if (method == INTER_LANCZOS4)
    {
        for (int i = 0; i < tabsz; i++, tab += 8) interpolateLanczos4(i * scale, tab);
    }
    else
        LYCON_ERROR("Unknown interpolation method");
}

static const void* initInterTab2D(int method, bool fixpt)
{
    static bool inittab[INTER_MAX + 1] = {false};
    float* tab = 0;
    short* itab = 0;
    int ksize = 0;
    if (method == INTER_LINEAR)
        tab = BilinearTab_f[0][0], itab = BilinearTab_i[0][0], ksize = 2;
    else if (method == INTER_CUBIC)
        tab = BicubicTab_f[0][0], itab = BicubicTab_i[0][0], ksize = 4;
    else if (method == INTER_LANCZOS4)
        tab = Lanczos4Tab_f[0][0], itab = Lanczos4Tab_i[0][0], ksize = 8;
    else
        LYCON_ERROR("Unknown/unsupported interpolation type");

    if (!inittab[method])
    {
        AutoBuffer<float> _tab(8 * INTER_TAB_SIZE);
        int i, j, k1, k2;
        initInterTab1D(method, _tab, INTER_TAB_SIZE);
        for (i = 0; i < INTER_TAB_SIZE; i++)
            for (j = 0; j < INTER_TAB_SIZE; j++, tab += ksize * ksize, itab += ksize * ksize)
            {
                int isum = 0;
                NNDeltaTab_i[i * INTER_TAB_SIZE + j][0] = j < INTER_TAB_SIZE / 2;
                NNDeltaTab_i[i * INTER_TAB_SIZE + j][1] = i < INTER_TAB_SIZE / 2;

                for (k1 = 0; k1 < ksize; k1++)
                {
                    float vy = _tab[i * ksize + k1];
                    for (k2 = 0; k2 < ksize; k2++)
                    {
                        float v = vy * _tab[j * ksize + k2];
                        tab[k1 * ksize + k2] = v;
                        isum += itab[k1 * ksize + k2] = saturate_cast<short>(v * INTER_REMAP_COEF_SCALE);
                    }
                }

                if (isum != INTER_REMAP_COEF_SCALE)
                {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2, mk1 = ksize2, mk2 = ksize2;
                    for (k1 = ksize2; k1 < ksize2 + 2; k1++)
                        for (k2 = ksize2; k2 < ksize2 + 2; k2++)
                        {
                            if (itab[k1 * ksize + k2] < itab[mk1 * ksize + mk2])
                                mk1 = k1, mk2 = k2;
                            else if (itab[k1 * ksize + k2] > itab[Mk1 * ksize + Mk2])
                                Mk1 = k1, Mk2 = k2;
                        }
                    if (diff < 0)
                        itab[Mk1 * ksize + Mk2] = (short)(itab[Mk1 * ksize + Mk2] - diff);
                    else
                        itab[mk1 * ksize + mk2] = (short)(itab[mk1 * ksize + mk2] - diff);
                }
            }
        tab -= INTER_TAB_SIZE2 * ksize * ksize;
        itab -= INTER_TAB_SIZE2 * ksize * ksize;
#if LYCON_SSE2 || LYCON_NEON
        if (method == INTER_LINEAR)
        {
            for (i = 0; i < INTER_TAB_SIZE2; i++)
                for (j = 0; j < 4; j++)
                {
                    BilinearTab_iC4[i][0][j * 2] = BilinearTab_i[i][0][0];
                    BilinearTab_iC4[i][0][j * 2 + 1] = BilinearTab_i[i][0][1];
                    BilinearTab_iC4[i][1][j * 2] = BilinearTab_i[i][1][0];
                    BilinearTab_iC4[i][1][j * 2 + 1] = BilinearTab_i[i][1][1];
                }
        }
#endif
        inittab[method] = true;
    }
    return fixpt ? (const void*)itab : (const void*)tab;
}

#ifndef __MINGW32__
static bool initAllInterTab2D()
{
    return initInterTab2D(INTER_LINEAR, false) && initInterTab2D(INTER_LINEAR, true) &&
           initInterTab2D(INTER_CUBIC, false) && initInterTab2D(INTER_CUBIC, true) &&
           initInterTab2D(INTER_LANCZOS4, false) && initInterTab2D(INTER_LANCZOS4, true);
}

static volatile bool doInitAllInterTab2D = initAllInterTab2D();
#endif
