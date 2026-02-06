/*
 * Shared native extension module for catley.
 *
 * This module is the single entry point for native routines.  Feature-specific
 * implementations (pathfinding, FOV, etc.) live in separate C source files and
 * expose Python-callable functions referenced in the methods table here.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Pathfinding entry points provided by _native_pathfinding.c. */
PyObject *catley_native_astar(PyObject *self, PyObject *args);

static PyMethodDef methods[] = {
    {"astar", catley_native_astar, METH_VARARGS,
     "astar(cost, start_x, start_y, goal_x, goal_y) -> list[(x,y)]\n\n"
     "A* pathfinding on a 2D int16 cost grid with octile diagonal costs.\n"
     "cost: numpy int16 array shape (width, height), 0=blocked.\n"
     "Returns path excluding start, or empty list if no path."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Shared native algorithms for catley.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&module);
}
