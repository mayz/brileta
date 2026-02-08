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
/* FOV entry point provided by _native_fov.c. */
PyObject *catley_native_fov(PyObject *self, PyObject *args);
/* WFC entry point provided by _native_wfc.c. */
PyObject *catley_native_wfc_solve(PyObject *self, PyObject *args);
/* Popcount table initializer provided by _native_wfc.c. */
void catley_native_init_popcount_table(void);
/* Shared native WFC contradiction exception type. */
PyObject *catley_native_wfc_contradiction_error = NULL;

static PyMethodDef methods[] = {
    {"astar", catley_native_astar, METH_VARARGS,
     "astar(cost, start_x, start_y, goal_x, goal_y) -> list[(x,y)]\n\n"
     "A* pathfinding on a 2D int16 cost grid with octile diagonal costs.\n"
     "cost: numpy int16 array shape (width, height), 0=blocked.\n"
     "Returns path excluding start, or empty list if no path."},
    {"fov", catley_native_fov, METH_VARARGS,
     "fov(transparent, visible, origin_x, origin_y, radius) -> None\n\n"
     "Compute symmetric shadowcasting FOV in-place into visible."},
    {"wfc_solve", catley_native_wfc_solve, METH_VARARGS,
     "wfc_solve(width, height, num_patterns, propagation_masks, pattern_weights, "
     "initial_wave, seed) -> list[list[int]]\n\n"
     "Run native Wave Function Collapse and return bit-index grid."},
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
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;

    /* Initialize lookup tables once at module load time (thread-safe). */
    catley_native_init_popcount_table();

    catley_native_wfc_contradiction_error = PyErr_NewException(
        "catley.util._native.WFCContradictionError",
        PyExc_Exception,
        NULL
    );
    if (!catley_native_wfc_contradiction_error) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObjectRef(
            m,
            "WFCContradictionError",
            catley_native_wfc_contradiction_error
        ) < 0) {
        Py_DECREF(catley_native_wfc_contradiction_error);
        catley_native_wfc_contradiction_error = NULL;
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
