try:
    from enum import IntEnum
except ImportError:
    class IntEnumMeta(type):
        def __iter__(cls):
            return cls.values()

        def __contains__(cls, value):
            return value in cls.values()

    class IntEnum:
        """
        A rudimentary Python 3 style IntEnum Implementation that supports
        iteration and membership testing.
        """
        __metaclass__ = IntEnumMeta

        @classmethod
        def values(cls):
            return (getattr(cls, elem) for elem in cls.__dict__ if elem.upper() == elem)

class Decode(IntEnum):
    """
    Modes for the load function.
    """

    # Load either a grayscale or color image (including alpha channel), 8-bit format
    UNCHANGED = -1

    # Load as a grayscale 8-bit image.
    # Color images will be converted to grayscale.
    GRAYSCALE = 0

    # Load as a three-channeled 8-bit image.
    # Grayscale images will be converted.
    # Any alpha channels will be discarded.
    COLOR = 1

    # If set, 16-bit and 32-bit images are returned as such.
    # Otherwise, an 8-bit image is returned.
    ANY_DEPTH = 2

class Encode(IntEnum):
    """
    Options for the save function.
    """

    # An integer from 0 to 100 (the higher is the better).
    # The default value is 95.
    JPEG_QUALITY = 1

    JPEG_PROGRESSIVE = 2

    JPEG_OPTIMIZE = 3

    JPEG_RST_INTERVAL = 4

    JPEG_LUMA_QUALITY = 5

    JPEG_CHROMA_QUALITY = 6

    # An integer from 0 to 9.
    # A higher value means a smaller size and longer compression time.
    # Default value is 3.
    PNG_COMPRESSION = 16

    PNG_STRATEGY = 17

    PNG_BILEVEL = 18

    PNG_STRATEGY_DEFAULT = 0

    PNG_STRATEGY_FILTERED = 1

    PNG_STRATEGY_HUFFMAN_ONLY = 2

    PNG_STRATEGY_RLE = 3

    PNG_STRATEGY_FIXED = 4


class Interpolation(IntEnum):
    """
    Interpolation methods for the resize function.
    """
    # Nearest Neighbor interpolation
    NEAREST = 0

    # Bilinear interpolation
    LINEAR = 1

    # Bicubic interpolation
    CUBIC = 2

    # Resampling using pixel area relation.
    # It may be a preferred method for image decimation, as it gives moire free results.
    # When the image is zoomed, it is similar to nearest neighborhood interpolation.
    AREA = 3

    # Lanczos interpolation over 8x8 neighborhood
    LANCZOS = 4
