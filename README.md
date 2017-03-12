# Lycon

A minimal and fast image library for Python and C++.

Lycon is a small subset of optimized image operations derived from [OpenCV](http://opencv.org/).

Current set of features include:

- Reading and writing JPEG and PNG images
- Fast SIMD optimized image resizing
- Zero-copy interop with [NumPy](http://www.numpy.org/) whenever possible

Tested on:

- Linux (Ubuntu 14.04) with Python`2.7.6` and `3.5.2`.
- macOS (Sierra, 10.12) with Python `2.7.11` and `3.5.1`.

## Install

```
pip install lycon
```

Native extension dependencies:

- CMake 2.8 or newer
- C++ toolchain
- LibJPEG
- LibPNG

### Ubuntu

Single-line command for installing all dependencies:

```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
```

### Anaconda

When working within an Anaconda Python distribution, it is recommended to use the latest `cmake` version (`3.6` or newer). Older versions can lead to a mismatch between the `libpng` and `libjpeg` headers used to build Lycon (usually the system headers), and the linked library (which may be preempted by the Anaconda-scoped version). To install the latest `cmake` version:

```
conda install cmake
```

## Example

```python
import lycon

# Load an image as a numpy array
img = lycon.load('mittens.jpg')
# Resize the image using bicubic interpolation
resized = lycon.resize(img, width=256, height=512, interpolation=lycon.Interpolation.CUBIC)
# Crop the image (like any regular numpy array)
cropped = resized[:100, :200]
# Save the image
lycon.save('cropped-mittens.png', cropped)
```

## Limitations

Compared to other image processing libraries ([OpenCV](http://opencv.org/), [pillow](https://python-pillow.org/), [scikit-image](http://scikit-image.org/)), Lycon offers a very limited set of operations. Intended usages include data loaders for deep learning, mass image resizing, etc.

## Advantages over OpenCV

- Drastically smaller (at the cost of drastically fewer features)
- Python module installable via `pip`
- Images use the more common `RGB` ordering (vs OpenCV's `BGR`)

However, if you already have OpenCV installed, Lycon's advantages are minimal.

## Advantages over PIL(low)

- Faster
- First-class NumPy support
- Full support for floating point images

## Advantages over Scikit-Image

- Drastically faster

## Benchmarks

- The table below lists execution time (in seconds), averaged across 10 runs
- The multiplier next to the time is the relative slowdown compared to Lycon

| Operation            |  Lycon |        OpenCV |             PIL |      Scikit-Image |
|----------------------|-------:|--------------:|----------------:|------------------:|
| Upsample: Nearest    | 0.1944 |   0.1948 (1x) |    2.1342 (11x) |  30.8982 (158.9x) |
| Upsample: Bilinear   | 0.4852 |   0.4940 (1x) |    7.2940 (15x) |   45.9095 (94.6x) |
| Upsample: Bicubic    | 1.8162 |   1.8182 (1x) |   8.9589 (4.9x) |  120.1645 (66.1x) |
| Upsample: Lanczos    | 4.5641 |   4.5714 (1x) |  10.7517 (2.3x) |                   |
| Upsample: Area       | 0.4801 |   0.4931 (1x) |                 |                   |
| Downsample: Nearest  | 0.0183 |   0.0181 (1x) |  0.4379 (24.2x) |   3.6101 (199.9x) |
| Downsample: Bilinear | 0.0258 |   0.0257 (1x) |    1.3122 (51x) |   4.8487 (188.4x) |
| Downsample: Bicubic  | 0.1324 |   0.1329 (1x) |  1.8153 (13.7x) |    9.4905 (71.6x) |
| Downsample: Lanczos  | 0.3317 |   0.3328 (1x) |   2.4058 (7.2x) |                   |
| Downsample: Area     | 0.0258 |   0.0259 (1x) |                 |                   |
| Read: JPG            | 0.3409 | 0.5085 (1.5x) |   1.4081 (4.1x) |     1.4628 (4.3x) |
| Read: PNG            | 1.2114 | 1.3245 (1.1x) |   1.8274 (1.5x) |     1.8674 (1.5x) |
| Write: JPG           | 0.4760 | 0.6046 (1.3x) |     2.3823 (5x) |    5.0159 (10.5x) |
| Write: PNG           | 2.1421 |   2.2370 (1x) |   9.0580 (4.2x) |    11.6060 (5.4x) |

- Blank cells indicate that the operation is not supported by the library
- All operations performed on a 16k (15360 x 8640) RGB image
- Tests performed on Ubuntu 14.04 running on an Intel Core i7 (Skylake)
- OpenCV `3.2+ (master: a85b4b5)`, Pillow `4.0.0`, skimage `0.12.3`, Python `2.7.3`
- OpenCV can potentially achieve better performance with GPU implementations and proprietary libraries like Intel IPP

## License

- All code derived from the OpenCV project is licensed under the 3-clause BSD License.
- All Lycon-specific modifications are licensed under the MIT license.

See `LICENSE` for further details.
