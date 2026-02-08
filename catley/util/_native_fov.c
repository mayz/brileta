/*
 * Native symmetric shadowcasting FOV implementation for catley.
 *
 * This file implements the `fov` callable exported by the shared
 * `catley.util._native` extension module.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int depth;
    int s_num;
    int s_den;
    int e_num;
    int e_den;
} Sector;

/* Grow the sector stack if full. Returns 0 on success, -1 on OOM. */
static int stack_push(
    Sector **stack, int *capacity, int *top, Sector sector
) {
    if (*top >= *capacity) {
        int new_cap = *capacity * 2;
        Sector *new_stack = (Sector *)realloc(*stack, sizeof(Sector) * new_cap);
        if (!new_stack) {
            free(*stack);
            *stack = NULL;
            return -1;
        }
        *stack = new_stack;
        *capacity = new_cap;
    }
    (*stack)[(*top)++] = sector;
    return 0;
}

static const int QUADRANT_TRANSFORMS[4][4] = {
    {1, 0, 0, -1},   /* North */
    {0, 1, 1, 0},    /* East */
    {1, 0, 0, 1},    /* South */
    {0, -1, 1, 0},   /* West */
};

static inline int in_bounds(int x, int y, int width, int height) {
    return x >= 0 && x < width && y >= 0 && y < height;
}


static inline int floor_div(int num, int den) {
    int q = num / den;
    int r = num % den;
    if (r != 0 && ((r > 0) != (den > 0))) q -= 1;
    return q;
}

static inline int ceil_div(int num, int den) {
    int q = num / den;
    int r = num % den;
    if (r != 0 && ((r > 0) == (den > 0))) q += 1;
    return q;
}

static int scan_quadrant(
    int cx,
    int dx,
    int cy,
    int dy,
    int ox,
    int oy,
    int radius,
    int width,
    int height,
    const unsigned char *transparent,
    unsigned char *visible,
    Py_ssize_t t_stride_x,
    Py_ssize_t t_stride_y,
    Py_ssize_t v_stride_x,
    Py_ssize_t v_stride_y
) {
    int capacity = radius > 8 ? radius * 4 : 32;
    Sector *stack = (Sector *)malloc(sizeof(Sector) * capacity);
    if (!stack) return -1;

    int top = 0;
    stack[top++] = (Sector){1, -1, 1, 1, 1};

    while (top > 0) {
        Sector sector = stack[--top];
        int depth = sector.depth;
        int s_num = sector.s_num;
        int s_den = sector.s_den;
        int e_num = sector.e_num;
        int e_den = sector.e_den;

        if (depth > radius) {
            continue;
        }

        int two_s_den = 2 * s_den;
        int min_col = floor_div(2 * depth * s_num + s_den, two_s_den);

        int two_e_den = 2 * e_den;
        int numerator = 2 * depth * e_num - e_den;
        int max_col = ceil_div(numerator, two_e_den);

        int prev_was_wall = -1;

        for (int col = min_col; col <= max_col; col++) {
            int wx = ox + col * cx + depth * dx;
            int wy = oy + col * cy + depth * dy;

            int tile_in_bounds = in_bounds(wx, wy, width, height);
            int is_wall = 1;
            if (tile_in_bounds) {
                is_wall = *(const unsigned char *)(transparent + wx * t_stride_x + wy * t_stride_y) == 0;
            }
            int is_floor = !is_wall;

            if (
                tile_in_bounds
                && (
                    is_wall
                    || (col * s_den >= depth * s_num && col * e_den <= depth * e_num)
                )
            ) {
                *(unsigned char *)(visible + wx * v_stride_x + wy * v_stride_y) = 1;
            }

            if (prev_was_wall != -1) {
                if (prev_was_wall && is_floor) {
                    s_num = 2 * col - 1;
                    s_den = 2 * depth;
                } else if (!prev_was_wall && is_wall) {
                    Sector s = {depth + 1, s_num, s_den, 2*col - 1, 2*depth};
                    if (stack_push(&stack, &capacity, &top, s) < 0)
                        return -1;
                }
            }

            prev_was_wall = is_wall;
        }

        if (prev_was_wall == 0) {
            Sector s = {depth + 1, s_num, s_den, e_num, e_den};
            if (stack_push(&stack, &capacity, &top, s) < 0)
                return -1;
        }
    }

    free(stack);
    return 0;
}

/*
 * fov(transparent, visible, origin_x, origin_y, radius) -> None
 *
 * transparent: 2D bool C-contiguous array shape (width, height)
 * visible:     2D bool C-contiguous array shape (width, height)
 */
PyObject *catley_native_fov(PyObject *self, PyObject *args) {
    PyObject *transparent_obj;
    PyObject *visible_obj;
    int ox, oy, radius;

    if (!PyArg_ParseTuple(
            args,
            "OOiii",
            &transparent_obj,
            &visible_obj,
            &ox,
            &oy,
            &radius
        )) {
        return NULL;
    }

    Py_buffer transparent_buf;
    Py_buffer visible_buf;

    if (PyObject_GetBuffer(
            transparent_obj,
            &transparent_buf,
            PyBUF_STRIDES | PyBUF_FORMAT
        ) < 0) {
        return NULL;
    }

    if (PyObject_GetBuffer(
            visible_obj,
            &visible_buf,
            PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT
        ) < 0) {
        PyBuffer_Release(&transparent_buf);
        return NULL;
    }

    int transparent_ok = transparent_buf.ndim == 2
        && transparent_buf.itemsize == 1
        && (strcmp(transparent_buf.format, "?") == 0
            || strcmp(transparent_buf.format, "b") == 0
            || strcmp(transparent_buf.format, "B") == 0);

    int visible_ok = visible_buf.ndim == 2
        && visible_buf.itemsize == 1
        && (strcmp(visible_buf.format, "?") == 0
            || strcmp(visible_buf.format, "b") == 0
            || strcmp(visible_buf.format, "B") == 0);

    if (!transparent_ok || !visible_ok) {
        PyBuffer_Release(&transparent_buf);
        PyBuffer_Release(&visible_buf);
        PyErr_SetString(
            PyExc_TypeError,
            "transparent and visible must be 2D bool arrays"
        );
        return NULL;
    }

    int width = (int)transparent_buf.shape[0];
    int height = (int)transparent_buf.shape[1];

    if ((int)visible_buf.shape[0] != width || (int)visible_buf.shape[1] != height) {
        PyBuffer_Release(&transparent_buf);
        PyBuffer_Release(&visible_buf);
        PyErr_SetString(
            PyExc_ValueError,
            "transparent and visible must have the same shape"
        );
        return NULL;
    }

    unsigned char *visible = (unsigned char *)visible_buf.buf;
    const unsigned char *transparent = (const unsigned char *)transparent_buf.buf;
    Py_ssize_t t_stride_x = transparent_buf.strides[0];
    Py_ssize_t t_stride_y = transparent_buf.strides[1];
    Py_ssize_t v_stride_x = visible_buf.strides[0];
    Py_ssize_t v_stride_y = visible_buf.strides[1];

    int fov_rc = 0;
    Py_BEGIN_ALLOW_THREADS

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            *(unsigned char *)(visible + x * v_stride_x + y * v_stride_y) = 0;
        }
    }

    if (in_bounds(ox, oy, width, height)) {
        *(unsigned char *)(visible + ox * v_stride_x + oy * v_stride_y) = 1;
    }

    for (int i = 0; i < 4; i++) {
        int rc = scan_quadrant(
            QUADRANT_TRANSFORMS[i][0],
            QUADRANT_TRANSFORMS[i][1],
            QUADRANT_TRANSFORMS[i][2],
            QUADRANT_TRANSFORMS[i][3],
            ox,
            oy,
            radius,
            width,
            height,
            transparent,
            visible,
            t_stride_x,
            t_stride_y,
            v_stride_x,
            v_stride_y
        );
        if (rc < 0) {
            fov_rc = -1;
            break;
        }
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&transparent_buf);
    PyBuffer_Release(&visible_buf);

    if (fov_rc < 0) {
        return PyErr_NoMemory();
    }
    Py_RETURN_NONE;
}
