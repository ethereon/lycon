#pragma once

#include "lycon/defs.h"
#include "lycon/types.h"

namespace lycon
{

struct PaletteEntry
{
    unsigned char b, g, r, a;
};

void convert_BGR2Gray_8u_C3C1R(const uchar *bgr, int bgr_step, uchar *gray, int gray_step, Size size, int swap_rb = 0);
void convert_BGRA2Gray_8u_C4C1R(const uchar *bgra, int bgra_step, uchar *gray, int gray_step, Size size,
                                int swap_rb = 0);
void convert_BGRA2Gray_16u_CnC1R(const ushort *bgra, int bgra_step, ushort *gray, int gray_step, Size size, int ncn,
                                 int swap_rb = 0);

void convert_Gray2BGR_8u_C1C3R(const uchar *gray, int gray_step, uchar *bgr, int bgr_step, Size size);
void convert_Gray2BGR_16u_C1C3R(const ushort *gray, int gray_step, ushort *bgr, int bgr_step, Size size);

void convert_RGBA2RGB_8u_C4C3R(const uchar *rgba, int rgba_step, uchar *rgb, int rgb_step, Size size, int _swap_rb = 0);
void convert_BGRA2BGR_16u_C4C3R(const ushort *bgra, int bgra_step, ushort *bgr, int bgr_step, Size size, int _swap_rb);

void convert_BGR2RGB_8u_C3R(const uchar *bgr, int bgr_step, uchar *rgb, int rgb_step, Size size);
#define convert_RGB2BGR_8u_C3R convert_BGR2RGB_8u_C3R

void convert_BGR2RGB_16u_C3R(const ushort *bgr, int bgr_step, ushort *rgb, int rgb_step, Size size);
#define convert_RGB2BGR_16u_C3R convert_BGR2RGB_16u_C3R

void convert_BGRA2RGBA_8u_C4R(const uchar *bgra, int bgra_step, uchar *rgba, int rgba_step, Size size);
#define convert_RGBA2BGRA_8u_C4R convert_BGRA2RGBA_8u_C4R

void convert_BGRA2RGBA_16u_C4R(const ushort *bgra, int bgra_step, ushort *rgba, int rgba_step, Size size);
#define convert_RGBA2BGRA_16u_C4R convert_BGRA2RGBA_16u_C4R

void convert_BGR5552Gray_8u_C2C1R(const uchar *bgr555, int bgr555_step, uchar *gray, int gray_step, Size size);
void convert_BGR5652Gray_8u_C2C1R(const uchar *bgr565, int bgr565_step, uchar *gray, int gray_step, Size size);
void convert_BGR5552BGR_8u_C2C3R(const uchar *bgr555, int bgr555_step, uchar *bgr, int bgr_step, Size size);
void convert_BGR5652BGR_8u_C2C3R(const uchar *bgr565, int bgr565_step, uchar *bgr, int bgr_step, Size size);
void convert_CMYK2BGR_8u_C4C3R(const uchar *cmyk, int cmyk_step, uchar *bgr, int bgr_step, Size size);
void convert_CMYK2RGB_8u_C4C3R(const uchar *cmyk, int cmyk_step, uchar *rgb, int rgb_step, Size size);
void convert_CMYK2Gray_8u_C4C1R(const uchar *ycck, int ycck_step, uchar *gray, int gray_step, Size size);

void FillGrayPalette(PaletteEntry *palette, int bpp, bool negative = false);
bool IsColorPalette(PaletteEntry *palette, int bpp);
void CvtPaletteToGray(const PaletteEntry *palette, uchar *grayPalette, int entries);
uchar *FillUniColor(uchar *data, uchar *&line_end, int step, int width3, int &y, int height, int count3,
                    PaletteEntry clr);
uchar *FillUniGray(uchar *data, uchar *&line_end, int step, int width3, int &y, int height, int count3, uchar clr);

uchar *FillColorRow8(uchar *data, uchar *indices, int len, PaletteEntry *palette);
uchar *FillGrayRow8(uchar *data, uchar *indices, int len, uchar *palette);
uchar *FillColorRow4(uchar *data, uchar *indices, int len, PaletteEntry *palette);
uchar *FillGrayRow4(uchar *data, uchar *indices, int len, uchar *palette);
uchar *FillColorRow1(uchar *data, uchar *indices, int len, PaletteEntry *palette);
uchar *FillGrayRow1(uchar *data, uchar *indices, int len, uchar *palette);
}
