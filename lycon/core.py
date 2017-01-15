import _lycon

import itertools

from .enum import (Decode, Encode, Interpolation)

def load(path, mode=Decode.UNCHANGED):
    """
    Loads and returns the image at the given path as a numpy ndarray.
    """
    return _lycon.load(path, mode)

def save(path, image, options=None):
    """
    Saves the given image (a numpy ndarray) at the given path.
    The image format is inferred from the extension.

    The options argument, if provided, should be a dictionary where the keys are constants
    from the Encode enum and the values are integers.
    """
    if options is not None:
        # Convert to a flat (key_1, value_1, key_2, value_2, ...) list
        options = list(itertools.chain(options.items()))
    _lycon.save(path, image, options)

def resize(image, width, height, interpolation=Interpolation.LINEAR, output=None):
    """
    Resize the image to the given dimensions, resampled using the given interpolation method.

    If an output ndarray is provided, it must be the same type as the input and have the
    dimensions of the resized image.
    """
    assert 2 <= len(image.shape) <= 4
    if output is not None:
        assert output.dtype == image.dtype
        assert len(output.shape) == len(input.shape)
        assert output.shape[:2] == (height, width)
    return _lycon.resize(image, (width, height), interpolation, output)

def get_supported_extensions():
    """
    Returns a list of supported image extensions.
    """
    return ('jpeg', 'jpg', 'png')
