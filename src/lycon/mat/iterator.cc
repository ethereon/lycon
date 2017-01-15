#include "lycon/mat/mat.h"

namespace lycon
{
NAryMatIterator::NAryMatIterator()
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, Mat* _planes, int _narrays)
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, _planes, 0, _narrays);
}

NAryMatIterator::NAryMatIterator(const Mat** _arrays, uchar** _ptrs, int _narrays)
    : arrays(0), planes(0), ptrs(0), narrays(0), nplanes(0), size(0), iterdepth(0), idx(0)
{
    init(_arrays, 0, _ptrs, _narrays);
}

void NAryMatIterator::init(const Mat** _arrays, Mat* _planes, uchar** _ptrs, int _narrays)
{
    LYCON_ASSERT(_arrays && (_ptrs || _planes));
    int i, j, d1 = 0, i0 = -1, d = -1;

    arrays = _arrays;
    ptrs = _ptrs;
    planes = _planes;
    narrays = _narrays;
    nplanes = 0;
    size = 0;

    if (narrays < 0)
    {
        for (i = 0; _arrays[i] != 0; i++)
            ;
        narrays = i;
        LYCON_ASSERT(narrays <= 1000);
    }

    iterdepth = 0;

    for (i = 0; i < narrays; i++)
    {
        LYCON_ASSERT(arrays[i] != 0);
        const Mat& A = *arrays[i];
        if (ptrs)
            ptrs[i] = A.data;

        if (!A.data)
            continue;

        if (i0 < 0)
        {
            i0 = i;
            d = A.dims;

            // find the first dimensionality which is different from 1;
            // in any of the arrays the first "d1" step do not affect the continuity
            for (d1 = 0; d1 < d; d1++)
                if (A.size[d1] > 1)
                    break;
        }
        else
            LYCON_ASSERT(A.size == arrays[i0]->size);

        if (!A.isContinuous())
        {
            LYCON_ASSERT(A.step[d - 1] == A.elemSize());
            for (j = d - 1; j > d1; j--)
                if (A.step[j] * A.size[j] < A.step[j - 1])
                    break;
            iterdepth = std::max(iterdepth, j);
        }
    }

    if (i0 >= 0)
    {
        size = arrays[i0]->size[d - 1];
        for (j = d - 1; j > iterdepth; j--)
        {
            int64 total1 = (int64)size * arrays[i0]->size[j - 1];
            if (total1 != (int)total1)
                break;
            size = (int)total1;
        }

        iterdepth = j;
        if (iterdepth == d1)
            iterdepth = 0;

        nplanes = 1;
        for (j = iterdepth - 1; j >= 0; j--)
            nplanes *= arrays[i0]->size[j];
    }
    else
        iterdepth = 0;

    idx = 0;

    if (!planes)
        return;

    for (i = 0; i < narrays; i++)
    {
        LYCON_ASSERT(arrays[i] != 0);
        const Mat& A = *arrays[i];

        if (!A.data)
        {
            planes[i] = Mat();
            continue;
        }

        planes[i] = Mat(1, (int)size, A.type(), A.data);
    }
}

NAryMatIterator& NAryMatIterator::operator++()
{
    if (idx >= nplanes - 1)
        return *this;
    ++idx;

    if (iterdepth == 1)
    {
        if (ptrs)
        {
            for (int i = 0; i < narrays; i++)
            {
                if (!ptrs[i])
                    continue;
                ptrs[i] = arrays[i]->data + arrays[i]->step[0] * idx;
            }
        }
        if (planes)
        {
            for (int i = 0; i < narrays; i++)
            {
                if (!planes[i].data)
                    continue;
                planes[i].data = arrays[i]->data + arrays[i]->step[0] * idx;
            }
        }
    }
    else
    {
        for (int i = 0; i < narrays; i++)
        {
            const Mat& A = *arrays[i];
            if (!A.data)
                continue;
            int _idx = (int)idx;
            uchar* data = A.data;
            for (int j = iterdepth - 1; j >= 0 && _idx > 0; j--)
            {
                int szi = A.size[j], t = _idx / szi;
                data += (_idx - t * szi) * A.step[j];
                _idx = t;
            }
            if (ptrs)
                ptrs[i] = data;
            if (planes)
                planes[i].data = data;
        }
    }

    return *this;
}

NAryMatIterator NAryMatIterator::operator++(int)
{
    NAryMatIterator it = *this;
    ++*this;
    return it;
}
}
