#include <Python.h>

#include "lycon/io/io.h"
#include "lycon/python/interop.h"
#include "lycon/python/macros.h"
#include "lycon/transform/resize.h"

#define LYCON_IMPORT_ARRAY
#include "lycon/python/numpy.h"

using namespace lycon;

#include "lycon/python/module.io.h"

#include "lycon/python/module.transform.h"

#include "lycon/python/module.init.h"
