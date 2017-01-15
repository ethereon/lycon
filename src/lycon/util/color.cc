#include "lycon/util/color.h"

namespace lycon
{

#define WRITE_PIX(ptr, clr) (((uchar*)(ptr))[0] = (clr).b, ((uchar*)(ptr))[1] = (clr).g, ((uchar*)(ptr))[2] = (clr).r)

#define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))

#define SCALE 14
#define cR (int)(0.299 * (1 << SCALE) + 0.5)
#define cG (int)(0.587 * (1 << SCALE) + 0.5)
#define cB ((1 << SCALE) - cR - cG)

void convert_BGR2Gray_8u_C3C1R(const uchar* rgb, int rgb_step, uchar* gray, int gray_step, Size size, int _swap_rb)
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for (; size.height--; gray += gray_step)
    {
        for (i = 0; i < size.width; i++, rgb += 3)
        {
            int t = descale(rgb[swap_rb] * cB + rgb[1] * cG + rgb[swap_rb ^ 2] * cR, SCALE);
            gray[i] = (uchar)t;
        }

        rgb += rgb_step - size.width * 3;
    }
}

void convert_BGRA2Gray_16u_CnC1R(const ushort* rgb, int rgb_step, ushort* gray, int gray_step, Size size, int ncn,
                                 int _swap_rb)
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for (; size.height--; gray += gray_step)
    {
        for (i = 0; i < size.width; i++, rgb += ncn)
        {
            int t = descale(rgb[swap_rb] * cB + rgb[1] * cG + rgb[swap_rb ^ 2] * cR, SCALE);
            gray[i] = (ushort)t;
        }

        rgb += rgb_step - size.width * ncn;
    }
}

void convert_BGRA2Gray_8u_C4C1R(const uchar* rgba, int rgba_step, uchar* gray, int gray_step, Size size, int _swap_rb)
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for (; size.height--; gray += gray_step)
    {
        for (i = 0; i < size.width; i++, rgba += 4)
        {
            int t = descale(rgba[swap_rb] * cB + rgba[1] * cG + rgba[swap_rb ^ 2] * cR, SCALE);
            gray[i] = (uchar)t;
        }

        rgba += rgba_step - size.width * 4;
    }
}

void convert_Gray2BGR_8u_C1C3R(const uchar* gray, int gray_step, uchar* bgr, int bgr_step, Size size)
{
    int i;
    for (; size.height--; gray += gray_step)
    {
        for (i = 0; i < size.width; i++, bgr += 3)
        {
            bgr[0] = bgr[1] = bgr[2] = gray[i];
        }
        bgr += bgr_step - size.width * 3;
    }
}

void convert_Gray2BGR_16u_C1C3R(const ushort* gray, int gray_step, ushort* bgr, int bgr_step, Size size)
{
    int i;
    for (; size.height--; gray += gray_step / sizeof(gray[0]))
    {
        for (i = 0; i < size.width; i++, bgr += 3)
        {
            bgr[0] = bgr[1] = bgr[2] = gray[i];
        }
        bgr += bgr_step / sizeof(bgr[0]) - size.width * 3;
    }
}

void convert_RGBA2RGB_8u_C4C3R(const uchar* rgba, int rgba_step, uchar* rgb, int rgb_step, Size size, int _swap_rb)
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, rgb += 3, rgba += 4)
        {
            uchar t0 = rgba[swap_rb], t1 = rgba[1];
            rgb[0] = t0;
            rgb[1] = t1;
            t0 = rgba[swap_rb ^ 2];
            rgb[2] = t0;
        }
        rgb += rgb_step - size.width * 3;
        rgba += rgba_step - size.width * 4;
    }
}

void convert_BGRA2BGR_16u_C4C3R(const ushort* bgra, int bgra_step, ushort* bgr, int bgr_step, Size size, int _swap_rb)
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgr += 3, bgra += 4)
        {
            ushort t0 = bgra[swap_rb], t1 = bgra[1];
            bgr[0] = t0;
            bgr[1] = t1;
            t0 = bgra[swap_rb ^ 2];
            bgr[2] = t0;
        }
        bgr += bgr_step / sizeof(bgr[0]) - size.width * 3;
        bgra += bgra_step / sizeof(bgra[0]) - size.width * 4;
    }
}

void convert_BGRA2RGBA_8u_C4R(const uchar* bgra, int bgra_step, uchar* rgba, int rgba_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgra += 4, rgba += 4)
        {
            uchar t0 = bgra[0], t1 = bgra[1];
            uchar t2 = bgra[2], t3 = bgra[3];
            rgba[0] = t2;
            rgba[1] = t1;
            rgba[2] = t0;
            rgba[3] = t3;
        }
        bgra += bgra_step - size.width * 4;
        rgba += rgba_step - size.width * 4;
    }
}

void convert_BGRA2RGBA_16u_C4R(const ushort* bgra, int bgra_step, ushort* rgba, int rgba_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgra += 4, rgba += 4)
        {
            ushort t0 = bgra[0], t1 = bgra[1];
            ushort t2 = bgra[2], t3 = bgra[3];

            rgba[0] = t2;
            rgba[1] = t1;
            rgba[2] = t0;
            rgba[3] = t3;
        }
        bgra += bgra_step / sizeof(bgra[0]) - size.width * 4;
        rgba += rgba_step / sizeof(rgba[0]) - size.width * 4;
    }
}

void convert_BGR2RGB_8u_C3R(const uchar* bgr, int bgr_step, uchar* rgb, int rgb_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgr += 3, rgb += 3)
        {
            uchar t0 = bgr[0], t1 = bgr[1], t2 = bgr[2];
            rgb[2] = t0;
            rgb[1] = t1;
            rgb[0] = t2;
        }
        bgr += bgr_step - size.width * 3;
        rgb += rgb_step - size.width * 3;
    }
}

void convert_BGR2RGB_16u_C3R(const ushort* bgr, int bgr_step, ushort* rgb, int rgb_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgr += 3, rgb += 3)
        {
            ushort t0 = bgr[0], t1 = bgr[1], t2 = bgr[2];
            rgb[2] = t0;
            rgb[1] = t1;
            rgb[0] = t2;
        }
        bgr += bgr_step - size.width * 3;
        rgb += rgb_step - size.width * 3;
    }
}

typedef unsigned short ushort;

void convert_BGR5552Gray_8u_C2C1R(const uchar* bgr555, int bgr555_step, uchar* gray, int gray_step, Size size)
{
    int i;
    for (; size.height--; gray += gray_step, bgr555 += bgr555_step)
    {
        for (i = 0; i < size.width; i++)
        {
            int t = descale(((((ushort*)bgr555)[i] << 3) & 0xf8) * cB + ((((ushort*)bgr555)[i] >> 2) & 0xf8) * cG +
                                ((((ushort*)bgr555)[i] >> 7) & 0xf8) * cR,
                            SCALE);
            gray[i] = (uchar)t;
        }
    }
}

void convert_BGR5652Gray_8u_C2C1R(const uchar* bgr565, int bgr565_step, uchar* gray, int gray_step, Size size)
{
    int i;
    for (; size.height--; gray += gray_step, bgr565 += bgr565_step)
    {
        for (i = 0; i < size.width; i++)
        {
            int t = descale(((((ushort*)bgr565)[i] << 3) & 0xf8) * cB + ((((ushort*)bgr565)[i] >> 3) & 0xfc) * cG +
                                ((((ushort*)bgr565)[i] >> 8) & 0xf8) * cR,
                            SCALE);
            gray[i] = (uchar)t;
        }
    }
}

void convert_BGR5552BGR_8u_C2C3R(const uchar* bgr555, int bgr555_step, uchar* bgr, int bgr_step, Size size)
{
    int i;
    for (; size.height--; bgr555 += bgr555_step)
    {
        for (i = 0; i < size.width; i++, bgr += 3)
        {
            int t0 = (((ushort*)bgr555)[i] << 3) & 0xf8;
            int t1 = (((ushort*)bgr555)[i] >> 2) & 0xf8;
            int t2 = (((ushort*)bgr555)[i] >> 7) & 0xf8;
            bgr[0] = (uchar)t0;
            bgr[1] = (uchar)t1;
            bgr[2] = (uchar)t2;
        }
        bgr += bgr_step - size.width * 3;
    }
}

void convert_BGR5652BGR_8u_C2C3R(const uchar* bgr565, int bgr565_step, uchar* bgr, int bgr_step, Size size)
{
    int i;
    for (; size.height--; bgr565 += bgr565_step)
    {
        for (i = 0; i < size.width; i++, bgr += 3)
        {
            int t0 = (((ushort*)bgr565)[i] << 3) & 0xf8;
            int t1 = (((ushort*)bgr565)[i] >> 3) & 0xfc;
            int t2 = (((ushort*)bgr565)[i] >> 8) & 0xf8;
            bgr[0] = (uchar)t0;
            bgr[1] = (uchar)t1;
            bgr[2] = (uchar)t2;
        }
        bgr += bgr_step - size.width * 3;
    }
}

void convert_CMYK2BGR_8u_C4C3R(const uchar* cmyk, int cmyk_step, uchar* bgr, int bgr_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, bgr += 3, cmyk += 4)
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c) * k >> 8);
            m = k - ((255 - m) * k >> 8);
            y = k - ((255 - y) * k >> 8);
            bgr[2] = (uchar)c;
            bgr[1] = (uchar)m;
            bgr[0] = (uchar)y;
        }
        bgr += bgr_step - size.width * 3;
        cmyk += cmyk_step - size.width * 4;
    }
}

void convert_CMYK2RGB_8u_C4C3R(const uchar* cmyk, int cmyk_step, uchar* rgb, int rgb_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, rgb += 3, cmyk += 4)
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c) * k >> 8);
            m = k - ((255 - m) * k >> 8);
            y = k - ((255 - y) * k >> 8);
            rgb[0] = (uchar)c;
            rgb[1] = (uchar)m;
            rgb[2] = (uchar)y;
        }
        rgb += rgb_step - size.width * 3;
        cmyk += cmyk_step - size.width * 4;
    }
}

void convert_CMYK2Gray_8u_C4C1R(const uchar* cmyk, int cmyk_step, uchar* gray, int gray_step, Size size)
{
    int i;
    for (; size.height--;)
    {
        for (i = 0; i < size.width; i++, cmyk += 4)
        {
            int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
            c = k - ((255 - c) * k >> 8);
            m = k - ((255 - m) * k >> 8);
            y = k - ((255 - y) * k >> 8);
            int t = descale(y * cB + m * cG + c * cR, SCALE);
            gray[i] = (uchar)t;
        }
        gray += gray_step;
        cmyk += cmyk_step - size.width * 4;
    }
}

void CvtPaletteToGray(const PaletteEntry* palette, uchar* grayPalette, int entries)
{
    int i;
    for (i = 0; i < entries; i++)
    {
        convert_BGR2Gray_8u_C3C1R((uchar*)(palette + i), 0, grayPalette + i, 0, Size(1, 1));
    }
}

void FillGrayPalette(PaletteEntry* palette, int bpp, bool negative)
{
    int i, length = 1 << bpp;
    int xor_mask = negative ? 255 : 0;

    for (i = 0; i < length; i++)
    {
        int val = (i * 255 / (length - 1)) ^ xor_mask;
        palette[i].b = palette[i].g = palette[i].r = (uchar)val;
        palette[i].a = 0;
    }
}

bool IsColorPalette(PaletteEntry* palette, int bpp)
{
    int i, length = 1 << bpp;

    for (i = 0; i < length; i++)
    {
        if (palette[i].b != palette[i].g || palette[i].b != palette[i].r)
            return true;
    }

    return false;
}

uchar* FillUniColor(uchar* data, uchar*& line_end, int step, int width3, int& y, int height, int count3,
                    PaletteEntry clr)
{
    do
    {
        uchar* end = data + count3;

        if (end > line_end)
            end = line_end;

        count3 -= (int)(end - data);

        for (; data < end; data += 3)
        {
            WRITE_PIX(data, clr);
        }

        if (data >= line_end)
        {
            line_end += step;
            data = line_end - width3;
            if (++y >= height)
                break;
        }
    } while (count3 > 0);

    return data;
}

uchar* FillUniGray(uchar* data, uchar*& line_end, int step, int width, int& y, int height, int count, uchar clr)
{
    do
    {
        uchar* end = data + count;

        if (end > line_end)
            end = line_end;

        count -= (int)(end - data);

        for (; data < end; data++)
        {
            *data = clr;
        }

        if (data >= line_end)
        {
            line_end += step;
            data = line_end - width;
            if (++y >= height)
                break;
        }
    } while (count > 0);

    return data;
}

uchar* FillColorRow8(uchar* data, uchar* indices, int len, PaletteEntry* palette)
{
    uchar* end = data + len * 3;
    while ((data += 3) < end)
    {
        *((PaletteEntry*)(data - 3)) = palette[*indices++];
    }
    PaletteEntry clr = palette[indices[0]];
    WRITE_PIX(data - 3, clr);
    return data;
}

uchar* FillGrayRow8(uchar* data, uchar* indices, int len, uchar* palette)
{
    int i;
    for (i = 0; i < len; i++)
    {
        data[i] = palette[indices[i]];
    }
    return data + len;
}

uchar* FillColorRow4(uchar* data, uchar* indices, int len, PaletteEntry* palette)
{
    uchar* end = data + len * 3;

    while ((data += 6) < end)
    {
        int idx = *indices++;
        *((PaletteEntry*)(data - 6)) = palette[idx >> 4];
        *((PaletteEntry*)(data - 3)) = palette[idx & 15];
    }

    int idx = indices[0];
    PaletteEntry clr = palette[idx >> 4];
    WRITE_PIX(data - 6, clr);

    if (data == end)
    {
        clr = palette[idx & 15];
        WRITE_PIX(data - 3, clr);
    }
    return end;
}

uchar* FillGrayRow4(uchar* data, uchar* indices, int len, uchar* palette)
{
    uchar* end = data + len;
    while ((data += 2) < end)
    {
        int idx = *indices++;
        data[-2] = palette[idx >> 4];
        data[-1] = palette[idx & 15];
    }

    int idx = indices[0];
    uchar clr = palette[idx >> 4];
    data[-2] = clr;

    if (data == end)
    {
        clr = palette[idx & 15];
        data[-1] = clr;
    }
    return end;
}

uchar* FillColorRow1(uchar* data, uchar* indices, int len, PaletteEntry* palette)
{
    uchar* end = data + len * 3;

    while ((data += 24) < end)
    {
        int idx = *indices++;
        *((PaletteEntry*)(data - 24)) = palette[(idx & 128) != 0];
        *((PaletteEntry*)(data - 21)) = palette[(idx & 64) != 0];
        *((PaletteEntry*)(data - 18)) = palette[(idx & 32) != 0];
        *((PaletteEntry*)(data - 15)) = palette[(idx & 16) != 0];
        *((PaletteEntry*)(data - 12)) = palette[(idx & 8) != 0];
        *((PaletteEntry*)(data - 9)) = palette[(idx & 4) != 0];
        *((PaletteEntry*)(data - 6)) = palette[(idx & 2) != 0];
        *((PaletteEntry*)(data - 3)) = palette[(idx & 1) != 0];
    }

    int idx = indices[0] << 24;
    for (data -= 24; data < end; data += 3, idx += idx)
    {
        PaletteEntry clr = palette[idx < 0];
        WRITE_PIX(data, clr);
    }

    return data;
}

uchar* FillGrayRow1(uchar* data, uchar* indices, int len, uchar* palette)
{
    uchar* end = data + len;

    while ((data += 8) < end)
    {
        int idx = *indices++;
        *((uchar*)(data - 8)) = palette[(idx & 128) != 0];
        *((uchar*)(data - 7)) = palette[(idx & 64) != 0];
        *((uchar*)(data - 6)) = palette[(idx & 32) != 0];
        *((uchar*)(data - 5)) = palette[(idx & 16) != 0];
        *((uchar*)(data - 4)) = palette[(idx & 8) != 0];
        *((uchar*)(data - 3)) = palette[(idx & 4) != 0];
        *((uchar*)(data - 2)) = palette[(idx & 2) != 0];
        *((uchar*)(data - 1)) = palette[(idx & 1) != 0];
    }

    int idx = indices[0] << 24;
    for (data -= 8; data < end; data++, idx += idx)
    {
        data[0] = palette[idx < 0];
    }

    return data;
}
}
