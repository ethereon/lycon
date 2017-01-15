import cv2
import lycon
import PIL
import skimage.io
import skimage.transform

import numpy as np
import os
from timeit import default_timer as timer

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def get_path(name):
    return os.path.join(DATA_PATH, name)

def time(func, count=10):
    times = []
    # Perform an extra iteration and discard the first run to account for one-time initializations.
    count += 1
    for idx in xrange(count):
        start = timer()
        func()
        times.append(timer() - start)
    return np.mean(times[1:]) # [seconds]

def benchmark(*ops):
    results = [(tag, time(func)) for tag, func in ops]
    results.sort(key=lambda pair: pair[1])
    min_time = results[0][1]
    for tag, mean_time in results:
        print('{:50}: {:6.4f} | {:6.3f}x'.format(tag, mean_time, mean_time/min_time))
    print('-'*80)

def benchmark_read(path):
    # Read once to account for disk-caching effects
    with open(path) as infile:
        infile.read()
    msg = lambda tag: '[READ ({})] {}'.format(path.split('.')[-1], tag)
    benchmark(
        (msg('Lycon'), lambda: lycon.load(path)),
        (msg('OpenCV'), lambda: cv2.imread(path)),
        (msg('PIL'), lambda: np.asarray(PIL.Image.open(path))),
        (msg('SKImage'), lambda: skimage.io.imread(path)),
    )

def benchmark_write(img):
    for ext in ('png', 'jpg'):
        output = '/tmp/lycon_test.' + ext
        msg = lambda tag : '[WRITE ({})] {}'.format(ext, tag)
        benchmark(
            (msg('Lycon'), lambda: lycon.save(output, img)),
            (msg('OpenCV'), lambda: cv2.imwrite(output, img)),
            (msg('PIL'), lambda: PIL.Image.fromarray(img).save(output)),
            (msg('SKImage'), lambda: skimage.io.imsave(output, img)),
        )

def benchmark_resize(img):
    h, w = img.shape[:2]
    new_sizes = [(2*w, 2*h), (int(w/2), int(h/2))]
    interpolations = {
        'nearest':{
            'Lycon': lycon.Interpolation.NEAREST,
            'OpenCV': cv2.INTER_NEAREST,
            'PIL': PIL.Image.NEAREST,
            'SKImage': 0
        },
        'bilinear':{
            'Lycon': lycon.Interpolation.LINEAR,
            'OpenCV': cv2.INTER_LINEAR,
            'PIL': PIL.Image.BILINEAR,
            'SKImage': 1
        },
        'bicubic':{
            'Lycon': lycon.Interpolation.CUBIC,
            'OpenCV': cv2.INTER_CUBIC,
            'PIL': PIL.Image.BICUBIC,
            'SKImage': 3
        },
        'lanczos':{
            'Lycon': lycon.Interpolation.LANCZOS,
            'OpenCV': cv2.INTER_LANCZOS4,
            'PIL': PIL.Image.LANCZOS,
        },
        'area':{
            'Lycon': lycon.Interpolation.AREA,
            'OpenCV': cv2.INTER_AREA,
        }
    }
    for w, h in new_sizes:
        for interp in interpolations:
            msg = lambda tag : '[RESIZE ({} - {} x {})] {}'.format(interp, w, h, tag)
            modes = interpolations[interp]
            op = lambda tag, func: (msg(tag), lambda: func(modes[tag])) if tag in modes else None
            benchmark(*filter(None,[
                op('Lycon', lambda i: lycon.resize(img, width=w, height=h, interpolation=i)),
                op('OpenCV', lambda i: cv2.resize(img, (w, h), interpolation=i)),
                op('PIL', lambda i: np.asarray(PIL.Image.fromarray(img).resize((w, h), i))),
                op('SKImage', lambda i: skimage.transform.resize(img, (h, w), order=i))
            ]))


benchmark_read(get_path('16k.png'))
benchmark_read(get_path('16k.jpg'))
benchmark_write(lycon.load(get_path('16k.jpg')))
benchmark_resize(lycon.load(get_path('16k.jpg')))
