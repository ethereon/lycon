# Python executable
if (NOT DEFINED ${PYTHON_BIN})
  find_program(PYTHON_BIN python)
endif()
message(STATUS "Python binary: ${PYTHON_BIN}")

# Python include path
if (NOT DEFINED ${PYTHON_INCLUDE_DIR})
  execute_process(COMMAND ${PYTHON_BIN}
                  -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
                  OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "Python include path: ${PYTHON_INCLUDE_DIR}")

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
  # First search in the precise location indicated by Python
  # The following Python lib suffixes are known and supported:
  #   * m   (configured --with-pymalloc)
  #   * dm  (configured --with-pydebug and --with-pymalloc)
  set (PYTHON_LIB_PREFIX python${PYTHON_VERSION})
  find_library(PYTHON_LIB_PATH
               NAMES ${PYTHON_LIB_PREFIX} ${PYTHON_LIB_PREFIX}m ${PYTHON_LIB_PREFIX}dm
               PATHS ${PYTHON_LIB_DIR}
               NO_DEFAULT_PATH)

  # If the targeted search fails, look in cmake default locations
  if (NOT PYTHON_LIB_PATH)
    message(STATUS "Expanding search for libpython")
    find_library(PYTHON_LIB_PATH
                 NAMES ${PYTHON_LIB_PREFIX} ${PYTHON_LIB_PREFIX}m ${PYTHON_LIB_PREFIX}dm
                 PATHS ${PYTHON_LIB_DIR})
  endif()
endif()
message(STATUS "Python library path: ${PYTHON_LIB_PATH}")

# NumPy include path
if (NOT DEFINED ${NUMPY_INCLUDE_DIR})
  execute_process(COMMAND ${PYTHON_BIN}
                  -c "import numpy; print(numpy.get_include())"
                  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "NumPy include path: ${NUMPY_INCLUDE_DIR}")
