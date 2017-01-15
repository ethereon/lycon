struct VResizeNoVec
{
    int operator()(const uchar**, uchar*, const uchar*, int) const { return 0; }
};

struct HResizeNoVec
{
    int operator()(const uchar**, uchar**, int, const int*, const uchar*, int, int, int, int, int) const { return 0; }
};

template <typename ST, typename DT>
struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template <typename ST, typename DT, int bits>
struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum
    {
        SHIFT = bits,
        DELTA = 1 << (bits - 1)
    };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA) >> SHIFT); }
};

static inline int clip(int x, int a, int b) { return x >= a ? (x < b ? x : b - 1) : a; }

static const int MAX_ESIZE = 16;
