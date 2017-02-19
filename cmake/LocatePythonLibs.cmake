# Find Python binary
find_program(PYTHON_BIN python)

# Python include path
if (NOT DEFINED ${PYTHON_INCLUDE_DIR})
  execute_process(COMMAND ${PYTHON_BIN}
                  -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
                  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Python include path: ${PYTHON_INCLUDE_DIR}")
endif()

# Python library path
if (NOT DEFINED ${PYTHON_LIB_PATH})
  # Find the Python lib dir
  execute_process(COMMAND ${PYTHON_BIN}
                  -c "from distutils.sysconfig import get_config_var; print(get_config_var('LIBDIR'))"
                  OUTPUT_VARIABLE PYTHON_LIB_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Python library path: ${PYTHON_LIB_DIR}")

  # Get the Python version
  execute_process(COMMAND ${PYTHON_BIN} -c "from distutils.sysconfig import get_python_version; print(get_python_version())"
                  OUTPUT_VARIABLE PYTHON_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Python version: ${PYTHON_VERSION}")

  # Get the full Python lib path
  find_library(PYTHON_LIB_PATH
               NAMES python${PYTHON_VERSION}
               PATHS ${PYTHON_LIB_DIR}
               NO_SYSTEM_ENVIRONMENT_PATH)
  message(STATUS "Python library path: ${PYTHON_LIB_PATH}")
endif()

# NumPy include path
if (NOT DEFINED ${NUMPY_INCLUDE_DIR})
  execute_process(COMMAND ${PYTHON_BIN}
                  -c "import numpy; print(numpy.get_include())"
                  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "NumPy include path: ${NUMPY_INCLUDE_DIR}")
endif()
