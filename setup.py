import os
import sys
import shutil
import multiprocessing
import subprocess as sp

from distutils.spawn import find_executable
from distutils.command.build_ext import build_ext

from setuptools import Extension, setup, find_packages

class BuilderError(Exception): pass

class LyconBuilder(build_ext):
    """
    Builds the C++ Lycon extension using CMake.
    """

    LYCON_NATIVE_EXETENSION_NAME = '_lycon.so'

    def locate(self, name):
        """
        Locate the executable with the given name.
        """
        exec_path = find_executable(name)
        if exec_path is None:
            raise BuilderError('{} not found. Please install it first.'.format(name))
        return exec_path

    def execute(self, args):
        """
        Execute the given command in the temporary build directory.
        """
        proc = sp.Popen(args, cwd=self.build_temp)
        if proc.wait() != 0:
            raise BuilderError('Failed to exceute: {}'.format(' '.join(args)))

    def prepare(self):
        """
        Prepare to build (check everything required exists).
        """
        self.make_path = self.locate('make')
        self.cmake_path = self.locate('cmake')
        # Make sure CMakeLists.txt exists
        if not os.path.exists(os.path.join(self.source_path, 'CMakeLists.txt')):
            raise BuilderError('Could not locate CMakeLists.txt')

    def cmake(self):
        """
        Run cmake.
        """
        print('Source path is {}'.format(self.source_path))
        python_bin = '-DPYTHON_BIN={}'.format(sys.executable)
        self.execute([self.cmake_path, self.source_path, python_bin])

    def make(self, parallel=True):
        """
        Run make.
        """
        num_jobs = multiprocessing.cpu_count() if parallel else 1
        print('Starting build with {} jobs'.format(num_jobs))
        self.execute([self.make_path, '-j', str(num_jobs)])

    def move(self):
        """
        Move the built library to the libs directory.
        """
        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)
        src_path = os.path.join(self.build_temp, self.LYCON_NATIVE_EXETENSION_NAME)
        dst_path = os.path.join(self.build_lib, self.LYCON_NATIVE_EXETENSION_NAME)
        shutil.move(src_path, dst_path)

    def build_extensions(self):
        """
        Build the native extension. [overriden]
        """
        # Setup paths
        self.source_path = os.path.realpath(os.path.dirname(__file__))
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build lycon
        try:
            self.prepare()
            self.cmake()
            self.make()
            self.move()
        except BuilderError as err:
            print('\t* Failed to build the Lycon native extension.')
            print('\t* [Error] {}'.format(err))
            exit(-1)

setup(name='lycon',
      version='0.1.8',
      description='A minimal and fast image library',
      author='Saumitro Dasgupta',
      author_email='sd@cs.stanford.edu',
      url='https://github.com/ethereon/lycon',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Topic :: Multimedia :: Graphics',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: Implementation :: CPython',
      ],
      cmdclass={'build_ext': LyconBuilder},
      ext_modules=[Extension('_lycon', ['lycon.placeholder.c'])],
      packages=find_packages(),
      install_requires=['numpy'],
      include_package_data=True,
      keywords=['Imaging',],
      zip_safe=True,
      license='MIT + 3-clause BSD')
