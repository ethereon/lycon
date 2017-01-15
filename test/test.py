import cv2
import lycon

import hashlib
import numpy as np
import os
import shutil
import tempfile
import unittest

def random_rgb_image():
    return (255*np.random.rand(128, 42, 3)).astype(np.uint8)

def rgb_bgr(img):
    return img[:, :, (2, 1, 0)]

def filehash(path):
    buffer_size = 65536
    md5 = hashlib.md5()
    with open(path, 'rb') as infile:
        while True:
            data = infile.read(buffer_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

class TestAgainstOpenCV(unittest.TestCase):

    def setUp(self):
        self.temp_path = tempfile.mkdtemp(prefix='lycon_test_')

    def tearDown(self):
        shutil.rmtree(self.temp_path)

    def get_path(self, filename):
        return os.path.join(self.temp_path, filename)

    def test_save(self):
        img = random_rgb_image()
        for extension in lycon.get_supported_extensions():
            mkpath = lambda name : self.get_path('{}.{}'.format(name, extension))
            # Write using Lycon
            lycon.save(mkpath('opencv'), img)
            # Write using OpenCV
            cv2.imwrite(mkpath('lycon'), rgb_bgr(img))
            self.assertEqual(filehash(mkpath('opencv')), filehash(mkpath('lycon')))

    def test_load(self):
        img = random_rgb_image()
        for extension in lycon.get_supported_extensions():
            mkpath = lambda name : self.get_path('{}.{}'.format(name, extension))
            # Write using OpenCV
            cv2.imwrite(mkpath('opencv'), img)
            # Read using OpenCV
            cv_img = cv2.imread(mkpath('opencv'))
            # Read using Lycon
            lycon_img = rgb_bgr(lycon.load(mkpath('opencv')))
            np.testing.assert_array_equal(cv_img, lycon_img)

    def test_resize(self):
        src_img = random_rgb_image()
        images = [src_img,
                  src_img.astype(np.float32),
                  src_img.astype(np.float64),
                  src_img.astype(np.int16)]                  
        new_shapes = [
            # No change
            src_img.shape[:2],
            # Upsample
            tuple(map(int, np.array(src_img.shape[:2]) * 3)),
            # Downsample
            tuple(map(int, np.array(src_img.shape[:2]) / 2))
        ]
        for img in images:
            for (h, w) in new_shapes:
                for interp in lycon.Interpolation:
                    cv_resized = cv2.resize(img, (w, h), interpolation=interp)
                    lycon_resized = lycon.resize(img, width=w, height=h, interpolation=interp)
                    np.testing.assert_array_equal(
                        cv_resized,
                        lycon_resized,
                        err_msg='Mismatch for dtype={}, interp={}, size=({}, {})'.format(
                            img.dtype, interp, w, h
                        )
                    )


if __name__ == '__main__':
    unittest.main()
