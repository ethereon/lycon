cmake_minimum_required(VERSION 2.8.11)

project(lycon)

include(cmake/LyconUtils.cmake)

set(LYCON_SOURCES
  src/lycon/io/base.cc
  src/lycon/io/bitstream.cc
  src/lycon/io/exif.cc
  src/lycon/io/io.cc
  src/lycon/io/jpeg.cc
  src/lycon/io/png.cc

  src/lycon/mat/allocator.cc
  src/lycon/mat/convert.cc
  src/lycon/mat/copy.cc
  src/lycon/mat/io_array.cc
  src/lycon/mat/iterator.cc
  src/lycon/mat/mat.cc
  src/lycon/mat/umat_data.cc

  src/lycon/transform/resize.cc
  src/lycon/transform/rotate.cc

  src/lycon/util/alloc.cc
  src/lycon/util/color.cc
  src/lycon/util/file.cc
  src/lycon/util/hardware.cc
  src/lycon/util/parallel_pthreads.cc
  src/lycon/util/parallel.cc
  src/lycon/util/singleton.cc
  src/lycon/util/string.cc
  src/lycon/util/tls.cc
)

set(LYCON_PYTHON_SOURCES
  src/lycon/python/interop.cc
  src/lycon/python/module.cc
)

set(CMAKE_MACOSX_RPATH TRUE)

# Build options
lycon_option(LYCON_BUILD_STATIC "Build Lycon as a static library" ON)
lycon_option(LYCON_BUILD_PYTHON "Build the Python native extension" ON)
lycon_option(LYCON_NUMPY_ALLOCATOR_BY_DEFAULT "Use the NumPy allocator by default" ${LYCON_BUILD_PYTHON})
# Enabling this can avoid libstdc++ compatibility issues under environments like Conda
lycon_option(LYCON_STATIC_LIBSTDCPP "Statically link against libstdc++" ON IF NOT APPLE)

include_directories(src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -fPIC -DNDEBUG -pthread")
if(LYCON_STATIC_LIBSTDCPP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# If enabled, use the NumPy allocator as the default Mat allocator
if(LYCON_NUMPY_ALLOCATOR_BY_DEFAULT)
  add_definitions(-DLYCON_USE_NUMPY_ALLOCATOR_BY_DEFAULT)
endif()

# The main library
if(LYCON_BUILD_STATIC)
  add_library(lycon STATIC ${LYCON_SOURCES})
else()
  add_library(lycon SHARED ${LYCON_SOURCES})
endif()

# LibPNG
find_package(PNG REQUIRED)
include_directories(${PNG_INCLUDE_DIR})
target_link_libraries(lycon ${PNG_LIBRARY})

# LibJPEG
find_package(JPEG REQUIRED)
include_directories(${JPEG_INCLUDE_DIR})
target_link_libraries(lycon ${JPEG_LIBRARY})

# The Python extension
if (LYCON_BUILD_PYTHON)
  include(cmake/LocatePythonLibs.cmake)

  include_directories(${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})

  add_library(pycon SHARED ${LYCON_PYTHON_SOURCES})
  target_link_libraries(pycon lycon ${PYTHON_LIB_PATH})
  # NOTE(saumitro): Even on macOS, Python expects the ".so" suffix rather than ".dylib"
  set_target_properties(pycon PROPERTIES PREFIX "_" SUFFIX ".so" OUTPUT_NAME "lycon")
endif()
