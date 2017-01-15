#pragma once

#include <Python.h>

namespace lycon
{
// RAII style GIL acquisition
// On construction, saves the thread's state and acquires the GIL.
// On destruction, resets the thread state pointer and releases the GIL.
class PyEnsureGIL
{
   public:
    PyEnsureGIL() : state_(PyGILState_Ensure()) {}
    ~PyEnsureGIL() { PyGILState_Release(state_); }

   private:
    PyGILState_STATE state_;
};

// RAII style GIL release
// On construction, saves the current thread state and releases the GIL.
// On destruction, re-acquires the GIL and restores the thread state.
class PyReleaseGIL
{
   public:
    PyReleaseGIL() : state_(PyEval_SaveThread()) {}
    ~PyReleaseGIL() { PyEval_RestoreThread(state_); }

   private:
    PyThreadState *state_;
};
}
