/*
 * Shared native extension module for brileta.
 *
 * This module is the single entry point for native routines.  Feature-specific
 * implementations (pathfinding, FOV, etc.) live in separate C source files and
 * expose Python-callable functions referenced in the methods table here.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Pathfinding entry points provided by _native_pathfinding.c. */
PyObject *brileta_native_astar(PyObject *self, PyObject *args);
/* FOV entry point provided by _native_fov.c. */
PyObject *brileta_native_fov(PyObject *self, PyObject *args);
/* WFC entry point provided by _native_wfc.c. */
PyObject *brileta_native_wfc_solve(PyObject *self, PyObject *args);
/* Popcount table initializer provided by _native_wfc.c. */
void brileta_native_init_popcount_table(void);
/* Shared native WFC contradiction exception type. */
PyObject *brileta_native_wfc_contradiction_error = NULL;

static PyMethodDef methods[] = {
    {"astar", brileta_native_astar, METH_VARARGS,
     "astar(cost, start_x, start_y, goal_x, goal_y) -> list[(x,y)]\n\n"
     "A* pathfinding on a 2D int16 cost grid with octile diagonal costs.\n"
     "cost: numpy int16 array shape (width, height), 0=blocked.\n"
     "Returns path excluding start, or empty list if no path."},
    {"fov", brileta_native_fov, METH_VARARGS,
     "fov(transparent, visible, origin_x, origin_y, radius) -> None\n\n"
     "Compute symmetric shadowcasting FOV in-place into visible."},
    {"wfc_solve", brileta_native_wfc_solve, METH_VARARGS,
     "wfc_solve(width, height, num_patterns, propagation_masks, pattern_weights, "
     "initial_wave, seed) -> list[list[int]]\n\n"
     "Run native Wave Function Collapse and return bit-index grid."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Shared native algorithms for brileta.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__native(void) {
    PyObject *m = PyModule_Create(&module);
    if (!m) return NULL;

    /* Initialize lookup tables once at module load time (thread-safe). */
    brileta_native_init_popcount_table();

    brileta_native_wfc_contradiction_error = PyErr_NewException(
        "brileta.util._native.WFCContradictionError",
        PyExc_Exception,
        NULL
    );
    if (!brileta_native_wfc_contradiction_error) {
        Py_DECREF(m);
        return NULL;
    }

    if (PyModule_AddObjectRef(
            m,
            "WFCContradictionError",
            brileta_native_wfc_contradiction_error
        ) < 0) {
        Py_DECREF(brileta_native_wfc_contradiction_error);
        brileta_native_wfc_contradiction_error = NULL;
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
