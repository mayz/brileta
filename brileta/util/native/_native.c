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
/* Noise type registration provided by _native_noise.c. */
int brileta_native_init_noise_type(PyObject *module);

/* Sprite drawing primitives provided by _native_sprites.c. */
PyObject *brileta_native_sprite_alpha_blend(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_composite_over(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_draw_line(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_draw_thick_line(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_draw_tapered_trunk(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_stamp_fuzzy_circle(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_stamp_ellipse(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_batch_stamp_ellipses(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_batch_stamp_circles(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_generate_deciduous_canopy(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_fill_triangle(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_paste_sprite(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_darken_rim(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_nibble_canopy(PyObject *self, PyObject *args);
PyObject *brileta_native_sprite_nibble_boulder(PyObject *self, PyObject *args);
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
    {"sprite_alpha_blend", brileta_native_sprite_alpha_blend, METH_VARARGS,
     "sprite_alpha_blend(canvas, x, y, r, g, b, a) -> None\n\n"
     "Alpha-composite a single RGBA pixel onto the canvas."},
    {"sprite_composite_over", brileta_native_sprite_composite_over, METH_VARARGS,
     "sprite_composite_over(canvas, y_min, y_max, x_min, x_max, src_alpha, r, g, b) -> None\n\n"
     "Composite uniform RGB over a canvas region with per-pixel alpha."},
    {"sprite_draw_line", brileta_native_sprite_draw_line, METH_VARARGS,
     "sprite_draw_line(canvas, x0, y0, x1, y1, r, g, b, a) -> None\n\n"
     "Draw a 1px Bresenham line."},
    {"sprite_draw_thick_line", brileta_native_sprite_draw_thick_line, METH_VARARGS,
     "sprite_draw_thick_line(canvas, x0, y0, x1, y1, r, g, b, a, thickness) -> None\n\n"
     "Draw a line with integer thickness using parallel Bresenham lines."},
    {"sprite_draw_tapered_trunk", brileta_native_sprite_draw_tapered_trunk, METH_VARARGS,
     "sprite_draw_tapered_trunk(canvas, cx, y_bottom, y_top, w_bottom, w_top, r, g, b, a, root_flare) -> None\n\n"
     "Draw a trunk as a vertically tapered filled column."},
    {"sprite_stamp_fuzzy_circle", brileta_native_sprite_stamp_fuzzy_circle, METH_VARARGS,
     "sprite_stamp_fuzzy_circle(canvas, cx, cy, radius, r, g, b, a, falloff, hardness) -> None\n\n"
     "Stamp a soft circle with alpha falloff."},
    {"sprite_stamp_ellipse", brileta_native_sprite_stamp_ellipse, METH_VARARGS,
     "sprite_stamp_ellipse(canvas, cx, cy, rx, ry, r, g, b, a, falloff, hardness) -> None\n\n"
     "Stamp a soft ellipse with alpha falloff."},
    {"sprite_batch_stamp_ellipses", brileta_native_sprite_batch_stamp_ellipses, METH_VARARGS,
     "sprite_batch_stamp_ellipses(canvas, ellipses, r, g, b, a, falloff, hardness) -> None\n\n"
     "Batch-stamp ellipses with screen-blend alpha accumulation."},
    {"sprite_batch_stamp_circles", brileta_native_sprite_batch_stamp_circles, METH_VARARGS,
     "sprite_batch_stamp_circles(canvas, circles, r, g, b, a, falloff, hardness) -> None\n\n"
     "Batch-stamp circles (delegates to batch_stamp_ellipses)."},
    {"sprite_generate_deciduous_canopy", brileta_native_sprite_generate_deciduous_canopy, METH_VARARGS,
     "sprite_generate_deciduous_canopy(canvas, seed, size, canopy_cx, canopy_cy, base_radius, crown_rx_scale, crown_ry_scale, canopy_center_x_offset, tips, shadow..., mid..., highlight...) -> list[(x,y)]\n\n"
     "Generate deciduous canopy lobes/shading in C and return lobe centers."},
    {"sprite_fill_triangle", brileta_native_sprite_fill_triangle, METH_VARARGS,
     "sprite_fill_triangle(canvas, cx, top_y, base_width, height, r, g, b, a) -> None\n\n"
     "Draw a filled isoceles triangle pointing upward."},
    {"sprite_paste_sprite", brileta_native_sprite_paste_sprite, METH_VARARGS,
     "sprite_paste_sprite(sheet, sprite, x0, y0) -> None\n\n"
     "Alpha-composite sprite onto sheet at pixel offset."},
    {"sprite_darken_rim", brileta_native_sprite_darken_rim, METH_VARARGS,
     "sprite_darken_rim(canvas, darken_r, darken_g, darken_b) -> None\n\n"
     "Darken 1px silhouette rim pixels bordering transparency."},
    {"sprite_nibble_canopy", brileta_native_sprite_nibble_canopy, METH_VARARGS,
     "sprite_nibble_canopy(canvas, seed, center_x, center_y, canopy_radius, nibble_prob, interior_prob) -> None\n\n"
     "Carve random notches in canopy edge for silhouette variety."},
    {"sprite_nibble_boulder", brileta_native_sprite_nibble_boulder, METH_VARARGS,
     "sprite_nibble_boulder(canvas, seed, nibble_prob) -> None\n\n"
     "Remove random edge pixels from upper half of boulder."},
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

    /* Register the _NoiseState type (FastNoiseLite wrapper). */
    if (brileta_native_init_noise_type(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
