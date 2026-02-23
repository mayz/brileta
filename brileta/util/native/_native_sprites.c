/*
 * Native sprite drawing primitives for brileta.
 *
 * This file implements fast C versions of the Python drawing functions in
 * brileta/sprites/primitives.py, plus the rim-darkening and silhouette
 * nibbling post-processing used by tree and boulder sprite generators.
 *
 * All canvas parameters are (height, width, 4) uint8 RGBA buffers accessed
 * via the Python buffer protocol (same pattern as _native_fov.c).
 *
 * Exported Python-callable functions (referenced in _native.c method table):
 *   sprite_alpha_blend
 *   sprite_composite_over
 *   sprite_draw_line
 *   sprite_draw_thick_line
 *   sprite_draw_tapered_trunk
 *   sprite_stamp_fuzzy_circle
 *   sprite_stamp_ellipse
 *   sprite_batch_stamp_ellipses
 *   sprite_batch_stamp_circles
 *   sprite_generate_deciduous_canopy
 *   sprite_fill_triangle
 *   sprite_paste_sprite
 *   sprite_darken_rim
 *   sprite_nibble_canopy
 *   sprite_nibble_boulder
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* xoshiro128++ PRNG (same algorithm as _native_wfc.c)                 */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t s[4];
} SpriteRng;

static inline uint32_t sprite_rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint64_t sprite_splitmix64_next(uint64_t *state) {
    uint64_t z;

    *state += 0x9E3779B97F4A7C15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void sprite_rng_init(SpriteRng *rng, uint64_t seed) {
    uint64_t sm = seed;
    uint64_t a = sprite_splitmix64_next(&sm);
    uint64_t b = sprite_splitmix64_next(&sm);

    rng->s[0] = (uint32_t)a;
    rng->s[1] = (uint32_t)(a >> 32);
    rng->s[2] = (uint32_t)b;
    rng->s[3] = (uint32_t)(b >> 32);

    /* xoshiro cannot run with an all-zero state. */
    if ((rng->s[0] | rng->s[1] | rng->s[2] | rng->s[3]) == 0) {
        rng->s[0] = 0x9E3779B9U;
        rng->s[1] = 0x243F6A88U;
        rng->s[2] = 0xB7E15162U;
        rng->s[3] = 0x8AED2A6BU;
    }
}

static inline uint32_t sprite_rng_next_u32(SpriteRng *rng) {
    uint32_t result = sprite_rotl32(rng->s[0] + rng->s[3], 7) + rng->s[0];
    uint32_t t = rng->s[1] << 9;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = sprite_rotl32(rng->s[3], 11);

    return result;
}

/* Return a float in [0, 1). */
static inline double sprite_rng_next_double(SpriteRng *rng) {
    uint64_t hi = (uint64_t)(sprite_rng_next_u32(rng) >> 5);
    uint64_t lo = (uint64_t)(sprite_rng_next_u32(rng) >> 6);
    uint64_t mantissa = (hi << 26) | lo;
    return (double)mantissa * (1.0 / 9007199254740992.0);
}

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float maxf(float a, float b) { return a > b ? a : b; }
static inline float minf(float a, float b) { return a < b ? a : b; }

/* Pixel access macros for (height, width, 4) uint8 RGBA buffer.
 * px is a uint8_t* pointing to canvas data with known contiguous strides. */
#define PX(px, y, x, w) ((px) + ((y) * (w) + (x)) * 4)

/*
 * Acquire a (height, width, 4) uint8 canvas buffer.
 * When writable is true the buffer is requested as writable; when false
 * a read-only buffer is accepted.
 * Sets h, w, data and returns 0 on success, -1 on error (exception set).
 */
static int get_canvas_buffer_ex(
    PyObject *obj,
    Py_buffer *buf,
    int *h,
    int *w,
    uint8_t **data,
    int writable
) {
    int flags = PyBUF_C_CONTIGUOUS;
    if (writable) flags |= PyBUF_WRITABLE;
    if (PyObject_GetBuffer(obj, buf, flags) < 0)
        return -1;

    if (buf->ndim != 3 || buf->shape[2] != 4 || buf->itemsize != 1) {
        PyBuffer_Release(buf);
        PyErr_SetString(
            PyExc_TypeError,
            "canvas must be a contiguous (H, W, 4) uint8 array"
        );
        return -1;
    }

    *h = (int)buf->shape[0];
    *w = (int)buf->shape[1];
    *data = (uint8_t *)buf->buf;
    return 0;
}

/* Convenience wrappers. */
static int get_canvas_buffer(
    PyObject *obj, Py_buffer *buf, int *h, int *w, uint8_t **data
) {
    return get_canvas_buffer_ex(obj, buf, h, w, data, /*writable=*/1);
}

static int get_readonly_canvas_buffer(
    PyObject *obj, Py_buffer *buf, int *h, int *w, uint8_t **data
) {
    return get_canvas_buffer_ex(obj, buf, h, w, data, /*writable=*/0);
}

/* ------------------------------------------------------------------ */
/* Core drawing: alpha_blend (single pixel)                             */
/* ------------------------------------------------------------------ */

/* In-place Porter-Duff "over" for one pixel.  Bounds-checked. */
static inline void c_alpha_blend(
    uint8_t *data, int h, int w,
    int x, int y,
    int r, int g, int b, int a
) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    if (a <= 0) return;

    uint8_t *px = PX(data, y, x, w);
    float src_a = a / 255.0f;
    float dst_a = px[3] / 255.0f;
    float out_a = src_a + dst_a * (1.0f - src_a);
    if (out_a <= 0.0f) return;

    float inv_src = 1.0f - src_a;
    px[0] = (uint8_t)((r * src_a + px[0] * dst_a * inv_src) / out_a);
    px[1] = (uint8_t)((g * src_a + px[1] * dst_a * inv_src) / out_a);
    px[2] = (uint8_t)((b * src_a + px[2] * dst_a * inv_src) / out_a);
    px[3] = (uint8_t)(out_a * 255.0f);
}

/*
 * sprite_alpha_blend(canvas, x, y, r, g, b, a)
 */
PyObject *brileta_native_sprite_alpha_blend(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    int x, y, r, g, b, a;

    if (!PyArg_ParseTuple(args, "Oiiiiii", &canvas_obj, &x, &y, &r, &g, &b, &a))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    c_alpha_blend(data, h, w, x, y, r, g, b, a);

    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Core drawing: composite_over (region with uniform RGB + alpha map)   */
/* ------------------------------------------------------------------ */

/*
 * In-place Porter-Duff "over" for a rectangular region with per-pixel
 * source alpha and uniform source RGB.
 */
static void c_composite_over(
    uint8_t *data, int canvas_w,
    int y_min, int y_max, int x_min, int x_max,
    const float *src_a, /* row-major (rows, cols) */
    int sr, int sg, int sb
) {
    int cols = x_max - x_min + 1;
    for (int row = y_min; row <= y_max; row++) {
        for (int col = x_min; col <= x_max; col++) {
            int ai = (row - y_min) * cols + (col - x_min);
            float sa = src_a[ai];
            if (sa <= 0.0f) continue;

            uint8_t *px = PX(data, row, col, canvas_w);
            float da = px[3] / 255.0f;
            float out_a = sa + da * (1.0f - sa);
            if (out_a <= 0.0f) continue;

            float inv_src = 1.0f - sa;
            float inv_out = 1.0f / out_a;
            px[0] = (uint8_t)clampi(
                (int)((sr * sa + px[0] * da * inv_src) * inv_out), 0, 255
            );
            px[1] = (uint8_t)clampi(
                (int)((sg * sa + px[1] * da * inv_src) * inv_out), 0, 255
            );
            px[2] = (uint8_t)clampi(
                (int)((sb * sa + px[2] * da * inv_src) * inv_out), 0, 255
            );
            px[3] = (uint8_t)clampi((int)(out_a * 255.0f), 0, 255);
        }
    }
}

/*
 * sprite_composite_over(canvas, y_min, y_max, x_min, x_max, src_alpha, r, g, b)
 *
 * src_alpha: 2D float32 array matching the region shape.
 */
PyObject *brileta_native_sprite_composite_over(PyObject *self, PyObject *args) {
    PyObject *canvas_obj, *alpha_obj;
    int y_min, y_max, x_min, x_max, r, g, b;

    if (!PyArg_ParseTuple(
            args, "OiiiiOiii",
            &canvas_obj, &y_min, &y_max, &x_min, &x_max,
            &alpha_obj, &r, &g, &b
        ))
        return NULL;

    Py_buffer canvas_buf, alpha_buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &canvas_buf, &h, &w, &data) < 0)
        return NULL;

    if (PyObject_GetBuffer(alpha_obj, &alpha_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&canvas_buf);
        return NULL;
    }

    const float *src_alpha = (const float *)alpha_buf.buf;

    Py_BEGIN_ALLOW_THREADS
    c_composite_over(data, w, y_min, y_max, x_min, x_max, src_alpha, r, g, b);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&alpha_buf);
    PyBuffer_Release(&canvas_buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* draw_line: Bresenham 1px line                                        */
/* ------------------------------------------------------------------ */

static void c_draw_line(
    uint8_t *data, int h, int w,
    int x0, int y0, int x1, int y1,
    int r, int g, int b, int a
) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    for (;;) {
        c_alpha_blend(data, h, w, x0, y0, r, g, b, a);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx)  { err += dx; y0 += sy; }
    }
}

/*
 * sprite_draw_line(canvas, x0, y0, x1, y1, r, g, b, a)
 *
 * Coordinates are floats, rounded to nearest int for Bresenham.
 */
PyObject *brileta_native_sprite_draw_line(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    double fx0, fy0, fx1, fy1;
    int r, g, b, a;

    if (!PyArg_ParseTuple(
            args, "Oddddiiii",
            &canvas_obj, &fx0, &fy0, &fx1, &fy1, &r, &g, &b, &a
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int ix0 = (int)round(fx0), iy0 = (int)round(fy0);
    int ix1 = (int)round(fx1), iy1 = (int)round(fy1);
    c_draw_line(data, h, w, ix0, iy0, ix1, iy1, r, g, b, a);

    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* draw_thick_line: parallel Bresenham lines                            */
/* ------------------------------------------------------------------ */

/*
 * sprite_draw_thick_line(canvas, x0, y0, x1, y1, r, g, b, a, thickness)
 */
PyObject *brileta_native_sprite_draw_thick_line(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    double fx0, fy0, fx1, fy1;
    int r, g, b, a, thickness;

    if (!PyArg_ParseTuple(
            args, "Oddddiiiii",
            &canvas_obj, &fx0, &fy0, &fx1, &fy1, &r, &g, &b, &a, &thickness
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    if (thickness <= 1) {
        int ix0 = (int)round(fx0), iy0 = (int)round(fy0);
        int ix1 = (int)round(fx1), iy1 = (int)round(fy1);
        c_draw_line(data, h, w, ix0, iy0, ix1, iy1, r, g, b, a);
    } else {
        double dx = fx1 - fx0;
        double dy = fy1 - fy0;
        double length = sqrt(dx * dx + dy * dy);
        if (length < 0.01) {
            c_alpha_blend(data, h, w, (int)round(fx0), (int)round(fy0), r, g, b, a);
        } else {
            /* Unit perpendicular vector. */
            double px = -dy / length;
            double py = dx / length;
            double half = thickness / 2.0;

            for (int i = 0; i < thickness; i++) {
                double offset = -half + 0.5 + i;
                double ox = px * offset;
                double oy = py * offset;
                int ix0 = (int)round(fx0 + ox);
                int iy0 = (int)round(fy0 + oy);
                int ix1 = (int)round(fx1 + ox);
                int iy1 = (int)round(fy1 + oy);
                c_draw_line(data, h, w, ix0, iy0, ix1, iy1, r, g, b, a);
            }
        }
    }

    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* draw_tapered_trunk                                                   */
/* ------------------------------------------------------------------ */

/*
 * sprite_draw_tapered_trunk(canvas, cx, y_bottom, y_top,
 *                           w_bottom, w_top, r, g, b, a, root_flare)
 */
PyObject *brileta_native_sprite_draw_tapered_trunk(
    PyObject *self, PyObject *args
) {
    PyObject *canvas_obj;
    double cx, w_bottom, w_top;
    int y_bottom, y_top, r, g, b, a, root_flare;

    if (!PyArg_ParseTuple(
            args, "Odiiddiiiii",
            &canvas_obj, &cx, &y_bottom, &y_top,
            &w_bottom, &w_top, &r, &g, &b, &a, &root_flare
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int height = y_bottom - y_top + 1;
    if (height <= 0) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    int draw_y_min = clampi(y_top, 0, h - 1);
    int draw_y_max = clampi(y_bottom, 0, h - 1);
    if (draw_y_min > draw_y_max) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    /* Release the GIL for the computation-heavy section.  All data has
     * been extracted from Python objects; only raw C buffers are touched. */
    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Pre-compute half-widths for all rows. */
    float inv_height = 1.0f / (float)(height - 1 > 0 ? height - 1 : 1);
    float *half_w = (float *)malloc(sizeof(float) * height);
    if (!half_w) { oom = 1; goto trunk_done; }

    for (int i = 0; i < height; i++) {
        float t = (float)i * inv_height;  /* 0 at top, 1 at bottom */
        float row_w = (float)w_top + (float)(w_bottom - w_top) * t;

        /* Root flare: widen bottom rows. */
        if (root_flare > 0) {
            float dist_to_bottom = (float)(y_bottom - (y_top + i));
            if (dist_to_bottom < (float)root_flare) {
                float flare_t = 1.0f - dist_to_bottom / (float)root_flare;
                row_w += flare_t * 1.5f;
            }
        }
        half_w[i] = row_w * 0.5f;
    }

    /* Find max half-width for x bounds. */
    float max_hw = 0.0f;
    int local_start = draw_y_min - y_top;
    int local_end = draw_y_max - y_top + 1;
    for (int i = local_start; i < local_end; i++) {
        if (half_w[i] > max_hw) max_hw = half_w[i];
    }

    int x_min = clampi((int)floor(cx - max_hw - 0.5), 0, w - 1);
    int x_max = clampi((int)ceil(cx + max_hw + 0.5), 0, w - 1);
    if (x_min > x_max) { free(half_w); goto trunk_done; }

    /* Composite row by row. Build per-pixel source alpha and call
     * c_composite_over for each row to match Python's vectorized path. */
    {
        int n_cols = x_max - x_min + 1;
        int n_rows = draw_y_max - draw_y_min + 1;
        float *src_a = (float *)calloc((size_t)(n_rows * n_cols), sizeof(float));
        if (!src_a) { free(half_w); oom = 1; goto trunk_done; }

        for (int row = draw_y_min; row <= draw_y_max; row++) {
            int li = row - y_top;
            float hw = half_w[li];
            int ri = row - draw_y_min;

            for (int col = x_min; col <= x_max; col++) {
                float dist = fabsf((float)col - (float)cx);
                int ci = col - x_min;

                if (dist > hw + 0.5f) continue;  /* outside */

                float alpha_f;
                if (dist > hw - 0.5f) {
                    /* Edge pixel: anti-alias. */
                    float edge = maxf(0.0f, 1.0f - (dist - hw + 0.5f));
                    alpha_f = floorf((float)a * edge) / 255.0f;
                } else {
                    alpha_f = (float)a / 255.0f;
                }
                src_a[ri * n_cols + ci] = alpha_f;
            }
        }

        c_composite_over(data, w, draw_y_min, draw_y_max, x_min, x_max, src_a, r, g, b);
        free(src_a);
        free(half_w);
    }

trunk_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* RNG helpers for canopy generators                                     */
/* ------------------------------------------------------------------ */

/* Return a double uniformly in [lo, hi). */
static inline double rng_uniform(SpriteRng *rng, double lo, double hi) {
    return lo + sprite_rng_next_double(rng) * (hi - lo);
}

/* Return an int uniformly in [lo, hi) (exclusive end). */
static inline int rng_int(SpriteRng *rng, int lo, int hi) {
    if (hi <= lo) return lo;
    return lo + (int)(sprite_rng_next_u32(rng) % (unsigned)(hi - lo));
}

/* ------------------------------------------------------------------ */
/* Ellipse spec for internal batch stamping                             */
/* ------------------------------------------------------------------ */

typedef struct {
    float cx, cy, rx, ry;
} EllipseSpec;

#define MAX_CANOPY_ELLIPSES 64

/* ------------------------------------------------------------------ */
/* Ellipse/circle alpha profile (shared by stamp + batch)               */
/* ------------------------------------------------------------------ */

/*
 * Compute the source alpha for an ellipse stamp at distance `dist`
 * (normalized ellipse-space, 1.0 = boundary) with the given falloff/hardness
 * profile, pre-multiplied by opacity.
 */
static inline float ellipse_alpha(
    float dist,
    float inner_fraction,
    float outer_fraction,
    float effective_falloff,
    float edge_limit,
    float opacity
) {
    float alpha;
    if (dist <= inner_fraction) {
        alpha = 1.0f;
    } else if (dist > edge_limit) {
        return 0.0f;
    } else {
        float fr = (dist - inner_fraction) / outer_fraction;
        if (fr < 0.0f) fr = 0.0f;
        alpha = 1.0f - powf(fr, effective_falloff);
        if (alpha < 0.0f) alpha = 0.0f;
        if (alpha > 1.0f) alpha = 1.0f;
    }
    return alpha * opacity;
}

/* ------------------------------------------------------------------ */
/* Internal batch stamp from EllipseSpec array (no Python objects)       */
/* ------------------------------------------------------------------ */

/*
 * Batch-stamp ellipses using screen-blend alpha accumulation, reading
 * from a C array of EllipseSpec instead of a Python list.
 * Used by canopy generators to avoid Python-to-C round-trips.
 */
static void batch_stamp_from_specs(
    uint8_t *data, int canvas_h, int canvas_w,
    const EllipseSpec *specs, int n_specs,
    int r, int g, int b, int a,
    float falloff, float hardness
) {
    if (n_specs <= 0) return;

    float hc = clampf(hardness, 0.0f, 1.0f);
    float inner_fraction = 0.3f + 0.55f * hc;
    float ef = falloff + 2.5f * hc;
    float outer_fraction = maxf(1e-6f, 1.0f - inner_fraction);
    float opacity = (float)a / 255.0f;

    /* Compute union bounding box of all ellipses. */
    int u_y0 = canvas_h, u_y1 = -1;
    int u_x0 = canvas_w, u_x1 = -1;
    for (int i = 0; i < n_specs; i++) {
        if (specs[i].rx <= 0.0f || specs[i].ry <= 0.0f) continue;
        int rxc = (int)ceilf(specs[i].rx) + 1;
        int ryc = (int)ceilf(specs[i].ry) + 1;
        int ey0 = clampi((int)specs[i].cy - ryc, 0, canvas_h - 1);
        int ey1 = clampi((int)specs[i].cy + ryc, 0, canvas_h - 1);
        int ex0 = clampi((int)specs[i].cx - rxc, 0, canvas_w - 1);
        int ex1 = clampi((int)specs[i].cx + rxc, 0, canvas_w - 1);
        if (ey0 < u_y0) u_y0 = ey0;
        if (ey1 > u_y1) u_y1 = ey1;
        if (ex0 < u_x0) u_x0 = ex0;
        if (ex1 > u_x1) u_x1 = ex1;
    }
    if (u_y0 > u_y1 || u_x0 > u_x1) return;

    int n_rows = u_y1 - u_y0 + 1;
    int n_cols = u_x1 - u_x0 + 1;
    size_t n_px = (size_t)n_rows * (size_t)n_cols;

    float *remaining = (float *)malloc(sizeof(float) * n_px);
    if (!remaining) return;
    for (size_t i = 0; i < n_px; i++) remaining[i] = 1.0f;

    /* Accumulate: remaining *= (1 - alpha_i * opacity) per ellipse. */
    for (int ei = 0; ei < n_specs; ei++) {
        float ecx = specs[ei].cx, ecy = specs[ei].cy;
        float erx = specs[ei].rx, ery = specs[ei].ry;
        if (erx <= 0.0f || ery <= 0.0f) continue;
        float max_r = maxf(erx, ery);
        float edge_limit = 1.0f + 0.5f / max_r;

        int rxc = (int)ceilf(erx) + 1;
        int ryc = (int)ceilf(ery) + 1;
        int ey0 = clampi((int)ecy - ryc, 0, canvas_h - 1);
        int ey1 = clampi((int)ecy + ryc, 0, canvas_h - 1);
        int ex0 = clampi((int)ecx - rxc, 0, canvas_w - 1);
        int ex1 = clampi((int)ecx + rxc, 0, canvas_w - 1);

        for (int row = ey0; row <= ey1; row++) {
            for (int col = ex0; col <= ex1; col++) {
                float ddx = ((float)col - ecx) / erx;
                float ddy = ((float)row - ecy) / ery;
                float dist = sqrtf(ddx * ddx + ddy * ddy);

                float alpha = ellipse_alpha(
                    dist, inner_fraction, outer_fraction, ef, edge_limit, 1.0f
                );
                if (alpha <= 0.0f) continue;

                int idx = (row - u_y0) * n_cols + (col - u_x0);
                remaining[idx] *= (1.0f - alpha * opacity);
            }
        }
    }

    for (size_t i = 0; i < n_px; i++) remaining[i] = 1.0f - remaining[i];
    c_composite_over(data, canvas_w, u_y0, u_y1, u_x0, u_x1, remaining, r, g, b);
    free(remaining);
}

/* ------------------------------------------------------------------ */
/* stamp_fuzzy_circle                                                   */
/* ------------------------------------------------------------------ */

/*
 * sprite_stamp_fuzzy_circle(canvas, cx, cy, radius, r, g, b, a,
 *                           falloff, hardness)
 */
PyObject *brileta_native_sprite_stamp_fuzzy_circle(
    PyObject *self, PyObject *args
) {
    PyObject *canvas_obj;
    double cx, cy, radius, falloff, hardness;
    int r, g, b, a;

    if (!PyArg_ParseTuple(
            args, "Odddiiiidd",
            &canvas_obj, &cx, &cy, &radius, &r, &g, &b, &a,
            &falloff, &hardness
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int r_ceil = (int)ceil(radius) + 1;
    int y_min = clampi((int)cy - r_ceil, 0, h - 1);
    int y_max = clampi((int)cy + r_ceil, 0, h - 1);
    int x_min = clampi((int)cx - r_ceil, 0, w - 1);
    int x_max = clampi((int)cx + r_ceil, 0, w - 1);
    if (y_min > y_max || x_min > x_max) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Alpha profile parameters. */
    float hc = clampf((float)hardness, 0.0f, 1.0f);
    float inner_fraction = 0.3f + (0.85f - 0.3f) * hc;
    float ef = (float)falloff + 2.5f * hc;
    float inner_r = (float)radius * inner_fraction;
    float outer_r = (float)radius * (1.0f - inner_fraction);
    float safe_outer_r = maxf(outer_r, 1e-6f);
    float opacity = (float)a / 255.0f;
    float edge_limit = (float)radius + 0.5f;

    /* Build source alpha and composite. */
    int n_rows = y_max - y_min + 1;
    int n_cols = x_max - x_min + 1;
    float *src_a = (float *)malloc(sizeof(float) * n_rows * n_cols);
    if (!src_a) { oom = 1; goto fuzzy_done; }

    for (int row = y_min; row <= y_max; row++) {
        for (int col = x_min; col <= x_max; col++) {
            float ddx = (float)col - (float)cx;
            float ddy = (float)row - (float)cy;
            float dist = sqrtf(ddx * ddx + ddy * ddy);

            float alpha;
            if (dist <= inner_r) {
                alpha = 1.0f;
            } else if (dist > edge_limit) {
                alpha = 0.0f;
            } else {
                float fr = maxf((dist - inner_r) / safe_outer_r, 0.0f);
                alpha = clampf(1.0f - powf(fr, ef), 0.0f, 1.0f);
            }

            int idx = (row - y_min) * n_cols + (col - x_min);
            src_a[idx] = alpha * opacity;
        }
    }

    /* Inline composite over (more efficient than calling c_composite_over
     * because we already have the alpha buffer). */
    for (int row = y_min; row <= y_max; row++) {
        for (int col = x_min; col <= x_max; col++) {
            int idx = (row - y_min) * n_cols + (col - x_min);
            float sa = src_a[idx];
            if (sa <= 0.0f) continue;

            uint8_t *px = PX(data, row, col, w);
            float da = px[3] / 255.0f;
            float out_a = sa + da * (1.0f - sa);
            if (out_a <= 0.0f) continue;

            float inv_src = 1.0f - sa;
            float inv_out = 1.0f / out_a;
            px[0] = (uint8_t)clampi(
                (int)((r * sa + px[0] * da * inv_src) * inv_out), 0, 255
            );
            px[1] = (uint8_t)clampi(
                (int)((g * sa + px[1] * da * inv_src) * inv_out), 0, 255
            );
            px[2] = (uint8_t)clampi(
                (int)((b * sa + px[2] * da * inv_src) * inv_out), 0, 255
            );
            px[3] = (uint8_t)clampi((int)(out_a * 255.0f), 0, 255);
        }
    }

    free(src_a);

fuzzy_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* stamp_ellipse                                                        */
/* ------------------------------------------------------------------ */

/*
 * sprite_stamp_ellipse(canvas, cx, cy, rx, ry, r, g, b, a,
 *                      falloff, hardness)
 */
PyObject *brileta_native_sprite_stamp_ellipse(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    double cx, cy, rx, ry, falloff, hardness;
    int r, g, b, a;

    if (!PyArg_ParseTuple(
            args, "Oddddiiiidd",
            &canvas_obj, &cx, &cy, &rx, &ry,
            &r, &g, &b, &a, &falloff, &hardness
        ))
        return NULL;

    if (rx <= 0.0 || ry <= 0.0) Py_RETURN_NONE;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int rx_ceil = (int)ceil(rx) + 1;
    int ry_ceil = (int)ceil(ry) + 1;
    int y_min = clampi((int)cy - ry_ceil, 0, h - 1);
    int y_max = clampi((int)cy + ry_ceil, 0, h - 1);
    int x_min = clampi((int)cx - rx_ceil, 0, w - 1);
    int x_max = clampi((int)cx + rx_ceil, 0, w - 1);
    if (y_min > y_max || x_min > x_max) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    Py_BEGIN_ALLOW_THREADS

    float hc = clampf((float)hardness, 0.0f, 1.0f);
    float inner_fraction = 0.3f + 0.55f * hc;
    float ef = (float)falloff + 2.5f * hc;
    float outer_fraction = maxf(1e-6f, 1.0f - inner_fraction);
    float opacity = (float)a / 255.0f;
    float max_r = maxf((float)rx, (float)ry);
    float edge_limit = 1.0f + 0.5f / max_r;

    for (int row = y_min; row <= y_max; row++) {
        for (int col = x_min; col <= x_max; col++) {
            float ddx = ((float)col - (float)cx) / (float)rx;
            float ddy = ((float)row - (float)cy) / (float)ry;
            float dist = sqrtf(ddx * ddx + ddy * ddy);

            float sa = ellipse_alpha(
                dist, inner_fraction, outer_fraction, ef, edge_limit, opacity
            );
            if (sa <= 0.0f) continue;

            uint8_t *px = PX(data, row, col, w);
            float da = px[3] / 255.0f;
            float out_a = sa + da * (1.0f - sa);
            if (out_a <= 0.0f) continue;

            float inv_src = 1.0f - sa;
            float inv_out = 1.0f / out_a;
            px[0] = (uint8_t)clampi(
                (int)((r * sa + px[0] * da * inv_src) * inv_out), 0, 255
            );
            px[1] = (uint8_t)clampi(
                (int)((g * sa + px[1] * da * inv_src) * inv_out), 0, 255
            );
            px[2] = (uint8_t)clampi(
                (int)((b * sa + px[2] * da * inv_src) * inv_out), 0, 255
            );
            px[3] = (uint8_t)clampi((int)(out_a * 255.0f), 0, 255);
        }
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* batch_stamp_ellipses / batch_stamp_circles                           */
/* ------------------------------------------------------------------ */

/*
 * Parse a list of (cx, cy, rx, ry) tuples.
 * Returns 0 on success, -1 on error (exception set).
 * Caller must free *out on success.
 */
static int parse_ellipse_list(
    PyObject *list,
    int *n_out,
    float **cx_out, float **cy_out, float **rx_out, float **ry_out
) {
    Py_ssize_t n = PyList_Size(list);
    if (n <= 0) {
        *n_out = 0;
        *cx_out = *cy_out = *rx_out = *ry_out = NULL;
        return 0;
    }

    float *cxs = (float *)malloc(sizeof(float) * n);
    float *cys = (float *)malloc(sizeof(float) * n);
    float *rxs = (float *)malloc(sizeof(float) * n);
    float *rys = (float *)malloc(sizeof(float) * n);
    if (!cxs || !cys || !rxs || !rys) {
        free(cxs); free(cys); free(rxs); free(rys);
        PyErr_NoMemory();
        return -1;
    }

    int count = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        double ecx, ecy, erx, ery;
        if (!PyArg_ParseTuple(item, "dddd", &ecx, &ecy, &erx, &ery)) {
            free(cxs); free(cys); free(rxs); free(rys);
            return -1;
        }
        if (erx <= 0.0 || ery <= 0.0) continue;
        cxs[count] = (float)ecx;
        cys[count] = (float)ecy;
        rxs[count] = (float)erx;
        rys[count] = (float)ery;
        count++;
    }

    *n_out = count;
    *cx_out = cxs;
    *cy_out = cys;
    *rx_out = rxs;
    *ry_out = rys;
    return 0;
}

/*
 * sprite_batch_stamp_ellipses(canvas, ellipses, r, g, b, a,
 *                             falloff, hardness)
 *
 * ellipses: list of (cx, cy, rx, ry) tuples.
 * Uses screen-blend accumulation of per-ellipse alpha, matching the Python
 * batch path's `remaining *= (1 - alpha_i)` pattern.
 */
PyObject *brileta_native_sprite_batch_stamp_ellipses(
    PyObject *self, PyObject *args
) {
    PyObject *canvas_obj, *ellipse_list;
    int r, g, b, a;
    double falloff, hardness;

    if (!PyArg_ParseTuple(
            args, "OOiiiidd",
            &canvas_obj, &ellipse_list, &r, &g, &b, &a,
            &falloff, &hardness
        ))
        return NULL;

    int n_ell;
    float *cxs, *cys, *rxs, *rys;
    if (parse_ellipse_list(ellipse_list, &n_ell, &cxs, &cys, &rxs, &rys) < 0)
        return NULL;

    if (n_ell == 0) Py_RETURN_NONE;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0) {
        free(cxs); free(cys); free(rxs); free(rys);
        return NULL;
    }

    /* All Python data extracted into C arrays. Release GIL for computation. */
    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Compute profile parameters (shared across all ellipses). */
    float hc = clampf((float)hardness, 0.0f, 1.0f);
    float inner_fraction = 0.3f + 0.55f * hc;
    float ef = (float)falloff + 2.5f * hc;
    float outer_fraction = maxf(1e-6f, 1.0f - inner_fraction);
    float opacity = (float)a / 255.0f;

    /* Compute union bounding box of all ellipses. */
    int union_y_min = h, union_y_max = -1;
    int union_x_min = w, union_x_max = -1;
    for (int i = 0; i < n_ell; i++) {
        int rx_ceil = (int)ceilf(rxs[i]) + 1;
        int ry_ceil = (int)ceilf(rys[i]) + 1;
        int ey_min = clampi((int)cys[i] - ry_ceil, 0, h - 1);
        int ey_max = clampi((int)cys[i] + ry_ceil, 0, h - 1);
        int ex_min = clampi((int)cxs[i] - rx_ceil, 0, w - 1);
        int ex_max = clampi((int)cxs[i] + rx_ceil, 0, w - 1);
        if (ey_min < union_y_min) union_y_min = ey_min;
        if (ey_max > union_y_max) union_y_max = ey_max;
        if (ex_min < union_x_min) union_x_min = ex_min;
        if (ex_max > union_x_max) union_x_max = ex_max;
    }
    if (union_y_min > union_y_max || union_x_min > union_x_max) {
        goto batch_ell_done;
    }

    {
        int n_rows = union_y_max - union_y_min + 1;
        int n_cols = union_x_max - union_x_min + 1;
        size_t n_pixels = (size_t)n_rows * (size_t)n_cols;

        /* Allocate `remaining` buffer: starts at 1.0 everywhere. */
        float *remaining = (float *)malloc(sizeof(float) * n_pixels);
        if (!remaining) { oom = 1; goto batch_ell_done; }
        for (size_t i = 0; i < n_pixels; i++) remaining[i] = 1.0f;

        /* Accumulate: remaining *= (1 - alpha_i * opacity) for each ellipse. */
        for (int ei = 0; ei < n_ell; ei++) {
            float ecx = cxs[ei], ecy = cys[ei];
            float erx = rxs[ei], ery = rys[ei];
            float max_r = maxf(erx, ery);
            float edge_limit = 1.0f + 0.5f / max_r;

            int rx_ceil = (int)ceilf(erx) + 1;
            int ry_ceil = (int)ceilf(ery) + 1;
            int ey_min = clampi((int)ecy - ry_ceil, 0, h - 1);
            int ey_max = clampi((int)ecy + ry_ceil, 0, h - 1);
            int ex_min = clampi((int)ecx - rx_ceil, 0, w - 1);
            int ex_max = clampi((int)ecx + rx_ceil, 0, w - 1);

            for (int row = ey_min; row <= ey_max; row++) {
                for (int col = ex_min; col <= ex_max; col++) {
                    float ddx = ((float)col - ecx) / erx;
                    float ddy = ((float)row - ecy) / ery;
                    float dist = sqrtf(ddx * ddx + ddy * ddy);

                    float alpha = ellipse_alpha(
                        dist, inner_fraction, outer_fraction, ef, edge_limit, 1.0f
                    );
                    if (alpha <= 0.0f) continue;

                    int idx = (row - union_y_min) * n_cols + (col - union_x_min);
                    remaining[idx] *= (1.0f - alpha * opacity);
                }
            }
        }

        /* Convert remaining to source alpha and composite. */
        for (size_t i = 0; i < n_pixels; i++)
            remaining[i] = 1.0f - remaining[i];

        c_composite_over(
            data, w, union_y_min, union_y_max, union_x_min, union_x_max,
            remaining, r, g, b
        );

        free(remaining);
    }

batch_ell_done:
    Py_END_ALLOW_THREADS

    free(cxs); free(cys); free(rxs); free(rys);
    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/*
 * sprite_batch_stamp_circles(canvas, circles, r, g, b, a,
 *                            falloff, hardness)
 *
 * circles: list of (cx, cy, radius) tuples.
 * Internally converts to ellipses with rx = ry = radius and calls
 * the batch_stamp_ellipses logic.
 */
PyObject *brileta_native_sprite_batch_stamp_circles(
    PyObject *self, PyObject *args
) {
    PyObject *canvas_obj, *circle_list;
    int r, g, b, a;
    double falloff, hardness;

    if (!PyArg_ParseTuple(
            args, "OOiiiidd",
            &canvas_obj, &circle_list, &r, &g, &b, &a,
            &falloff, &hardness
        ))
        return NULL;

    /* Convert circles to ellipses list. */
    Py_ssize_t n = PyList_Size(circle_list);
    if (n <= 0) Py_RETURN_NONE;

    PyObject *ellipse_list = PyList_New(n);
    if (!ellipse_list) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(circle_list, i);
        double ccx, ccy, cr;
        if (!PyArg_ParseTuple(item, "ddd", &ccx, &ccy, &cr)) {
            Py_DECREF(ellipse_list);
            return NULL;
        }
        PyObject *ell = Py_BuildValue("(dddd)", ccx, ccy, cr, cr);
        if (!ell) {
            Py_DECREF(ellipse_list);
            return NULL;
        }
        PyList_SET_ITEM(ellipse_list, i, ell);
    }

    /* Build new args tuple and delegate to batch_stamp_ellipses. */
    PyObject *new_args = Py_BuildValue(
        "(OOiiiidd)",
        canvas_obj, ellipse_list, r, g, b, a, falloff, hardness
    );
    Py_DECREF(ellipse_list);
    if (!new_args) return NULL;

    PyObject *result = brileta_native_sprite_batch_stamp_ellipses(self, new_args);
    Py_DECREF(new_args);
    return result;
}

/* ------------------------------------------------------------------ */
/* generate_deciduous_canopy: native lobe generation + batch stamping    */
/* ------------------------------------------------------------------ */

/*
 * Parse a Python list of (x, y) tip positions into contiguous float arrays.
 * Returns 0 on success, -1 on error (exception set).
 * Caller must free *x_out and *y_out on success.
 */
static int parse_point_list(
    PyObject *list,
    int *n_out,
    float **x_out,
    float **y_out
) {
    Py_ssize_t n = PyList_Size(list);
    if (n <= 0) {
        *n_out = 0;
        *x_out = NULL;
        *y_out = NULL;
        return 0;
    }

    float *xs = (float *)malloc(sizeof(float) * n);
    float *ys = (float *)malloc(sizeof(float) * n);
    if (!xs || !ys) {
        free(xs);
        free(ys);
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyList_GET_ITEM(list, i);
        double px, py;
        if (!PyArg_ParseTuple(item, "dd", &px, &py)) {
            free(xs);
            free(ys);
            return -1;
        }
        xs[i] = (float)px;
        ys[i] = (float)py;
    }

    *n_out = (int)n;
    *x_out = xs;
    *y_out = ys;
    return 0;
}

static inline void append_ellipse_spec(
    EllipseSpec *specs, int *count, float cx, float cy, float rx, float ry
) {
    assert(*count < MAX_CANOPY_ELLIPSES &&
           "canopy ellipse array overflow - increase MAX_CANOPY_ELLIPSES");
    if (*count >= MAX_CANOPY_ELLIPSES) return;
    if (rx <= 0.0f || ry <= 0.0f) return;
    specs[*count].cx = cx;
    specs[*count].cy = cy;
    specs[*count].rx = rx;
    specs[*count].ry = ry;
    (*count)++;
}

/*
 * Select up to 2 unique indices in [0, n) using the local PRNG.
 * Returns the selected count. A retry cap prevents infinite loops
 * in the (statistically negligible) case of repeated collisions.
 */
static int select_unique_indices(
    SpriteRng *rng, int n, int want, int out_idx[2]
) {
    if (n <= 0 || want <= 0) return 0;
    int target = want < n ? want : n;
    int count = 0;
    int max_retries = n * 10;
    int retries = 0;
    while (count < target && retries < max_retries) {
        int idx = rng_int(rng, 0, n);
        int duplicate = 0;
        for (int i = 0; i < count; i++) {
            if (out_idx[i] == idx) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) { retries++; continue; }
        out_idx[count++] = idx;
    }
    return count;
}

/*
 * sprite_generate_deciduous_canopy(
 *   canvas, seed, size, canopy_cx, canopy_cy, base_radius,
 *   crown_rx_scale, crown_ry_scale, canopy_center_x_offset, tips,
 *   shadow_r, shadow_g, shadow_b, shadow_a,
 *   mid_r, mid_g, mid_b, mid_a,
 *   hi_r, hi_g, hi_b, hi_a
 * ) -> list[(x, y)]
 *
 * Generate the hot lobe-placement + ellipse-batching portion of the
 * deciduous canopy in C and return lobe centers for follow-on Python
 * embellishments (small branch extensions).
 */
PyObject *brileta_native_sprite_generate_deciduous_canopy(
    PyObject *self, PyObject *args
) {
    PyObject *canvas_obj, *tips_obj;
    unsigned long long seed;
    int size;
    double canopy_cx, canopy_cy, base_radius;
    double crown_rx_scale, crown_ry_scale, canopy_center_x_offset;
    int sh_r, sh_g, sh_b, sh_a;
    int mid_r, mid_g, mid_b, mid_a;
    int hi_r, hi_g, hi_b, hi_a;

    if (!PyArg_ParseTuple(
            args,
            "OKiddddddOiiiiiiiiiiii",
            &canvas_obj,
            &seed,
            &size,
            &canopy_cx,
            &canopy_cy,
            &base_radius,
            &crown_rx_scale,
            &crown_ry_scale,
            &canopy_center_x_offset,
            &tips_obj,
            &sh_r, &sh_g, &sh_b, &sh_a,
            &mid_r, &mid_g, &mid_b, &mid_a,
            &hi_r, &hi_g, &hi_b, &hi_a
        ))
        return NULL;

    float *tip_xs = NULL;
    float *tip_ys = NULL;
    int n_tips = 0;
    if (parse_point_list(tips_obj, &n_tips, &tip_xs, &tip_ys) < 0)
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0) {
        free(tip_xs);
        free(tip_ys);
        return NULL;
    }

    /* All Python data extracted. Release GIL for the heavy canopy
     * generation (lobe placement + 4x batch stamp passes). */
    int n_lobes;
    float lobe_xs[8];
    float lobe_ys[8];

    Py_BEGIN_ALLOW_THREADS

    SpriteRng rng;
    sprite_rng_init(&rng, (uint64_t)seed);

    const double kPi = 3.14159265358979323846;

    n_lobes = rng_int(&rng, 3, 6);
    double base_angle_step = 2.0 * kPi / (double)n_lobes;
    double lobe_angle_offset = rng_uniform(&rng, -kPi, kPi);

    for (int i = 0; i < n_lobes; i++) {
        double angle = lobe_angle_offset + base_angle_step * (double)i
            + rng_uniform(&rng, -0.4, 0.4);
        double dist = base_radius * rng_uniform(&rng, 0.45, 0.65);
        double sin_angle = sin(angle);
        double vertical_scale = sin_angle > 0.0 ? 0.9 : 0.7;
        lobe_xs[i] = (float)(canopy_cx + cos(angle) * dist);
        lobe_ys[i] = (float)(canopy_cy + sin_angle * dist * vertical_scale);
    }

    EllipseSpec central_fills[MAX_CANOPY_ELLIPSES];
    EllipseSpec shadows[MAX_CANOPY_ELLIPSES];
    EllipseSpec mids[MAX_CANOPY_ELLIPSES];
    EllipseSpec highlights[MAX_CANOPY_ELLIPSES];
    int n_central = 0;
    int n_shadows = 0;
    int n_mids = 0;
    int n_highlights = 0;

    /* Central fill: connect lobe masses into one canopy silhouette. */
    int n_central_fills = rng_int(&rng, 1, 3);
    for (int i = 0; i < n_central_fills; i++) {
        float cr = (float)(base_radius * rng_uniform(&rng, 0.55, 0.70));
        append_ellipse_spec(
            central_fills,
            &n_central,
            (float)(canopy_cx + rng_uniform(&rng, -0.5, 0.5)),
            (float)(canopy_cy + rng_uniform(&rng, -0.5, 0.3)),
            cr * (float)crown_rx_scale,
            cr * (float)crown_ry_scale
        );
    }

    /* Pass 1: shadow lobes. */
    for (int i = 0; i < n_lobes; i++) {
        int n_for_lobe = rng_int(&rng, 1, 3);
        for (int j = 0; j < n_for_lobe; j++) {
            float radius = (float)(base_radius * rng_uniform(&rng, 0.62, 0.76));
            append_ellipse_spec(
                shadows,
                &n_shadows,
                lobe_xs[i] + (float)rng_uniform(&rng, -size * 0.05, size * 0.05),
                lobe_ys[i] + (float)rng_uniform(&rng, -size * 0.07, size * 0.04),
                radius * (float)crown_rx_scale,
                radius * (float)crown_ry_scale
            );
        }
    }

    int selected_tip_idx[2];
    int n_tip_shadow = select_unique_indices(
        &rng, n_tips, rng_int(&rng, 1, 3), selected_tip_idx
    );
    for (int i = 0; i < n_tip_shadow; i++) {
        int tip_idx = selected_tip_idx[i];
        float radius = (float)(base_radius * rng_uniform(&rng, 0.50, 0.68));
        append_ellipse_spec(
            shadows,
            &n_shadows,
            tip_xs[tip_idx]
                + (float)canopy_center_x_offset
                + (float)rng_uniform(&rng, -0.5, 0.5),
            tip_ys[tip_idx] - 1.0f + (float)rng_uniform(&rng, -0.5, 0.3),
            radius * (float)crown_rx_scale,
            radius * (float)crown_ry_scale
        );
    }

    /* Pass 2: mid-tone lobes. */
    for (int i = 0; i < n_lobes; i++) {
        int n_for_lobe = rng_int(&rng, 1, 3);
        for (int j = 0; j < n_for_lobe; j++) {
            float radius = (float)(base_radius * rng_uniform(&rng, 0.58, 0.72));
            append_ellipse_spec(
                mids,
                &n_mids,
                lobe_xs[i] + (float)rng_uniform(&rng, -size * 0.04, size * 0.04),
                lobe_ys[i] + (float)rng_uniform(&rng, -size * 0.05, size * 0.03),
                radius * (float)crown_rx_scale,
                radius * (float)crown_ry_scale
            );
        }
    }

    int n_tip_mid = select_unique_indices(
        &rng, n_tips, rng_int(&rng, 1, 3), selected_tip_idx
    );
    for (int i = 0; i < n_tip_mid; i++) {
        int tip_idx = selected_tip_idx[i];
        float radius = (float)(base_radius * rng_uniform(&rng, 0.46, 0.62));
        append_ellipse_spec(
            mids,
            &n_mids,
            tip_xs[tip_idx]
                + (float)canopy_center_x_offset
                + (float)rng_uniform(&rng, -0.6, 0.6),
            tip_ys[tip_idx] - 1.1f + (float)rng_uniform(&rng, -0.6, 0.2),
            radius * (float)crown_rx_scale,
            radius * (float)crown_ry_scale
        );
    }

    /* Pass 3: highlights biased upward. */
    for (int i = 0; i < n_lobes; i++) {
        int n_for_lobe = rng_int(&rng, 1, 3);
        for (int j = 0; j < n_for_lobe; j++) {
            float radius = (float)(base_radius * rng_uniform(&rng, 0.45, 0.60));
            append_ellipse_spec(
                highlights,
                &n_highlights,
                lobe_xs[i] + (float)rng_uniform(&rng, -size * 0.03, size * 0.03),
                lobe_ys[i]
                    - (float)(size * 0.05)
                    + (float)rng_uniform(&rng, -size * 0.03, size * 0.02),
                radius * (float)crown_rx_scale,
                radius * (float)crown_ry_scale
            );
        }
    }

    int n_tip_highlight = select_unique_indices(
        &rng, n_tips, rng_int(&rng, 1, 3), selected_tip_idx
    );
    for (int i = 0; i < n_tip_highlight; i++) {
        int tip_idx = selected_tip_idx[i];
        float radius = (float)(base_radius * rng_uniform(&rng, 0.38, 0.50));
        append_ellipse_spec(
            highlights,
            &n_highlights,
            tip_xs[tip_idx]
                + (float)canopy_center_x_offset
                + (float)rng_uniform(&rng, -0.5, 0.5),
            tip_ys[tip_idx] - 1.3f + (float)rng_uniform(&rng, -0.4, 0.2),
            radius * (float)crown_rx_scale,
            radius * (float)crown_ry_scale
        );
    }

    batch_stamp_from_specs(
        data, h, w, central_fills, n_central,
        sh_r, sh_g, sh_b, sh_a, 1.6f, 0.7f
    );
    batch_stamp_from_specs(
        data, h, w, shadows, n_shadows,
        sh_r, sh_g, sh_b, sh_a, 1.8f, 0.8f
    );
    batch_stamp_from_specs(
        data, h, w, mids, n_mids,
        mid_r, mid_g, mid_b, mid_a, 1.5f, 0.7f
    );
    batch_stamp_from_specs(
        data, h, w, highlights, n_highlights,
        hi_r, hi_g, hi_b, hi_a, 1.3f, 0.6f
    );

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    free(tip_xs);
    free(tip_ys);

    /* Build Python return list (needs GIL). */
    PyObject *lobe_centers = PyList_New(n_lobes);
    if (!lobe_centers) return NULL;
    for (int i = 0; i < n_lobes; i++) {
        PyObject *pt = Py_BuildValue("(ff)", lobe_xs[i], lobe_ys[i]);
        if (!pt) {
            Py_DECREF(lobe_centers);
            return NULL;
        }
        PyList_SET_ITEM(lobe_centers, i, pt);
    }
    return lobe_centers;
}

/* ------------------------------------------------------------------ */
/* fill_triangle                                                        */
/* ------------------------------------------------------------------ */

/*
 * sprite_fill_triangle(canvas, cx, top_y, base_width, height, r, g, b, a)
 */
PyObject *brileta_native_sprite_fill_triangle(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    double cx, base_width;
    int top_y, tri_height, r, g, b, a;

    if (!PyArg_ParseTuple(
            args, "Odidiiiii",
            &canvas_obj, &cx, &top_y, &base_width, &tri_height,
            &r, &g, &b, &a
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    if (tri_height <= 0) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    int bottom_y = top_y + tri_height - 1;
    int draw_y_min = clampi(top_y, 0, h - 1);
    int draw_y_max = clampi(bottom_y, 0, h - 1);
    if (draw_y_min > draw_y_max) {
        PyBuffer_Release(&buf);
        Py_RETURN_NONE;
    }

    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Pre-compute half-widths per row (power-shaped taper). */
    float inv_h = 1.0f / (float)(tri_height - 1 > 0 ? tri_height - 1 : 1);
    int local_start = draw_y_min - top_y;
    int local_end = draw_y_max - top_y + 1;

    /* Find max half-width for x bounds. */
    float max_hw = 0.0f;
    for (int i = local_start; i < local_end; i++) {
        float t = (float)i * inv_h;
        float row_w = 1.0f + ((float)base_width - 1.0f) * powf(t, 0.8f);
        float hw = row_w * 0.5f;
        if (hw > max_hw) max_hw = hw;
    }

    int x_min = clampi((int)floor(cx - max_hw - 0.5), 0, w - 1);
    int x_max = clampi((int)ceil(cx + max_hw + 0.5), 0, w - 1);
    if (x_min > x_max) { goto tri_done; }

    {
        int n_rows = draw_y_max - draw_y_min + 1;
        int n_cols = x_max - x_min + 1;
        float *src_a = (float *)calloc((size_t)(n_rows * n_cols), sizeof(float));
        if (!src_a) { oom = 1; goto tri_done; }

        for (int row = draw_y_min; row <= draw_y_max; row++) {
            int li = row - top_y;
            float t = (float)li * inv_h;
            float row_w = 1.0f + ((float)base_width - 1.0f) * powf(t, 0.8f);
            float hw = row_w * 0.5f;
            int ri = row - draw_y_min;

            for (int col = x_min; col <= x_max; col++) {
                float dist = fabsf((float)col - (float)cx);
                int ci = col - x_min;

                if (dist > hw + 0.5f) continue;  /* outside */

                float alpha_f;
                if (dist > hw - 0.5f) {
                    float edge = maxf(0.0f, 1.0f - (dist - hw + 0.5f));
                    alpha_f = floorf((float)a * edge) / 255.0f;
                } else {
                    alpha_f = (float)a / 255.0f;
                }
                src_a[ri * n_cols + ci] = alpha_f;
            }
        }

        c_composite_over(data, w, draw_y_min, draw_y_max, x_min, x_max, src_a, r, g, b);
        free(src_a);
    }

tri_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* paste_sprite                                                         */
/* ------------------------------------------------------------------ */

/*
 * sprite_paste_sprite(sheet, sprite, x0, y0)
 */
PyObject *brileta_native_sprite_paste_sprite(PyObject *self, PyObject *args) {
    PyObject *sheet_obj, *sprite_obj;
    int x0, y0;

    if (!PyArg_ParseTuple(args, "OOii", &sheet_obj, &sprite_obj, &x0, &y0))
        return NULL;

    Py_buffer sheet_buf, sprite_buf;
    int sh, sw, sph, spw;
    uint8_t *sheet_data, *sprite_data;

    if (get_canvas_buffer(sheet_obj, &sheet_buf, &sh, &sw, &sheet_data) < 0)
        return NULL;

    if (get_readonly_canvas_buffer(sprite_obj, &sprite_buf, &sph, &spw, &sprite_data) < 0) {
        PyBuffer_Release(&sheet_buf);
        return NULL;
    }

    /* Clip paste region to sheet bounds. */
    int src_y0 = 0, src_x0 = 0;
    int dst_y0 = y0, dst_x0 = x0;
    if (y0 < 0) { src_y0 = -y0; dst_y0 = 0; }
    if (x0 < 0) { src_x0 = -x0; dst_x0 = 0; }

    int paste_h = sph - src_y0;
    int paste_w = spw - src_x0;
    if (dst_y0 + paste_h > sh) paste_h = sh - dst_y0;
    if (dst_x0 + paste_w > sw) paste_w = sw - dst_x0;

    if (paste_h <= 0 || paste_w <= 0) {
        PyBuffer_Release(&sprite_buf);
        PyBuffer_Release(&sheet_buf);
        Py_RETURN_NONE;
    }

    Py_BEGIN_ALLOW_THREADS

    /* Per-pixel Porter-Duff "over" composite. */
    for (int row = 0; row < paste_h; row++) {
        for (int col = 0; col < paste_w; col++) {
            uint8_t *src = PX(sprite_data, src_y0 + row, src_x0 + col, spw);
            uint8_t *dst = PX(sheet_data, dst_y0 + row, dst_x0 + col, sw);

            float sa = src[3] / 255.0f;
            if (sa <= 0.0f) continue;

            float da = dst[3] / 255.0f;
            float out_a = sa + da * (1.0f - sa);
            if (out_a <= 0.0f) continue;

            float inv_src = 1.0f - sa;
            float inv_out = 1.0f / out_a;
            dst[0] = (uint8_t)clampi(
                (int)((src[0] * sa + dst[0] * da * inv_src) * inv_out), 0, 255
            );
            dst[1] = (uint8_t)clampi(
                (int)((src[1] * sa + dst[1] * da * inv_src) * inv_out), 0, 255
            );
            dst[2] = (uint8_t)clampi(
                (int)((src[2] * sa + dst[2] * da * inv_src) * inv_out), 0, 255
            );
            dst[3] = (uint8_t)clampi((int)(out_a * 255.0f), 0, 255);
        }
    }

    Py_END_ALLOW_THREADS

    PyBuffer_Release(&sprite_buf);
    PyBuffer_Release(&sheet_buf);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* darken_rim: edge-detection + darkening in one pass                    */
/* ------------------------------------------------------------------ */

/*
 * sprite_darken_rim(canvas, darken_r, darken_g, darken_b)
 *
 * For every opaque pixel (alpha > 128) adjacent to a transparent pixel
 * (alpha == 0) in any cardinal direction, subtract darken values from RGB.
 */
PyObject *brileta_native_sprite_darken_rim(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    int dr, dg, db;

    if (!PyArg_ParseTuple(args, "Oiii", &canvas_obj, &dr, &dg, &db))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /*
     * Two-pass approach: first identify rim pixels, then darken.
     * We need two passes because darkening while scanning would affect
     * neighbor alpha checks.
     */
    size_t n_pixels = (size_t)h * (size_t)w;
    uint8_t *rim = (uint8_t *)calloc(n_pixels, 1);
    if (!rim) { oom = 1; goto rim_done; }

    /* Pass 1: identify rim pixels. */
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            uint8_t *px = PX(data, row, col, w);
            if (px[3] <= 128) continue;  /* Not opaque enough. */

            /* Check cardinal neighbors for transparency. Border counts
             * as transparent (same as np.pad with constant_values=True). */
            int has_transparent_neighbor = 0;
            if (row == 0 || PX(data, row - 1, col, w)[3] == 0)
                has_transparent_neighbor = 1;
            else if (row == h - 1 || PX(data, row + 1, col, w)[3] == 0)
                has_transparent_neighbor = 1;
            else if (col == 0 || PX(data, row, col - 1, w)[3] == 0)
                has_transparent_neighbor = 1;
            else if (col == w - 1 || PX(data, row, col + 1, w)[3] == 0)
                has_transparent_neighbor = 1;

            if (has_transparent_neighbor)
                rim[row * w + col] = 1;
        }
    }

    /* Pass 2: darken rim pixels. */
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            if (!rim[row * w + col]) continue;
            uint8_t *px = PX(data, row, col, w);
            px[0] = (uint8_t)clampi((int)px[0] - dr, 0, 255);
            px[1] = (uint8_t)clampi((int)px[1] - dg, 0, 255);
            px[2] = (uint8_t)clampi((int)px[2] - db, 0, 255);
        }
    }

    free(rim);

rim_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* nibble_canopy: random edge erosion for tree canopy silhouettes        */
/* ------------------------------------------------------------------ */

/*
 * Helper: compute rim mask into caller-allocated buffer.
 * rim must be pre-zeroed, sized h*w.
 */
static void compute_rim_mask(
    const uint8_t *data, int h, int w, uint8_t *rim
) {
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            const uint8_t *px = PX(data, row, col, w);
            if (px[3] <= 128) continue;

            int has_transparent = 0;
            if (row == 0 || PX(data, row - 1, col, w)[3] == 0)
                has_transparent = 1;
            else if (row == h - 1 || PX(data, row + 1, col, w)[3] == 0)
                has_transparent = 1;
            else if (col == 0 || PX(data, row, col - 1, w)[3] == 0)
                has_transparent = 1;
            else if (col == w - 1 || PX(data, row, col + 1, w)[3] == 0)
                has_transparent = 1;

            if (has_transparent)
                rim[row * w + col] = 1;
        }
    }
}

/*
 * sprite_nibble_canopy(canvas, seed, center_x, center_y, canopy_radius,
 *                      nibble_prob, interior_prob)
 *
 * Carve random notches in the canopy edge for silhouette variety.
 * Uses an internal PRNG seeded from `seed` instead of numpy rng.
 */
PyObject *brileta_native_sprite_nibble_canopy(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    unsigned long long seed;
    double center_x, center_y, canopy_radius, nibble_prob, interior_prob;

    if (!PyArg_ParseTuple(
            args, "OKddddd",
            &canvas_obj, &seed,
            &center_x, &center_y, &canopy_radius,
            &nibble_prob, &interior_prob
        ))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Compute rim mask. */
    size_t n_pixels = (size_t)h * (size_t)w;
    uint8_t *rim = (uint8_t *)calloc(n_pixels, 1);
    if (!rim) { oom = 1; goto nibble_canopy_done; }
    compute_rim_mask(data, h, w, rim);

    /* Restrict to canopy region (elliptical envelope, same as Python). */
    float safe_radius = maxf((float)canopy_radius, 1e-6f);
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            if (!rim[row * w + col]) continue;
            float dx = ((float)col - (float)center_x) / safe_radius;
            float dy = ((float)row - (float)center_y) / (safe_radius * 0.9f);
            if (dx * dx + dy * dy > 1.6f)
                rim[row * w + col] = 0;
        }
    }

    /* Check if any rim pixels remain. */
    int has_rim = 0;
    for (size_t i = 0; i < n_pixels; i++) {
        if (rim[i]) { has_rim = 1; break; }
    }
    if (!has_rim) { free(rim); goto nibble_canopy_done; }

    {
        SpriteRng rng;
        sprite_rng_init(&rng, (uint64_t)seed);

        float np = clampf((float)nibble_prob, 0.0f, 1.0f);
        float ip = clampf((float)interior_prob, 0.0f, 1.0f);

        /* Nibble pass: erase random rim pixels. */
        int any_nibbled = 0;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (!rim[row * w + col]) continue;
                if (sprite_rng_next_double(&rng) < np) {
                    PX(data, row, col, w)[3] = 0;
                    rim[row * w + col] = 2;  /* Mark as nibbled for interior pass. */
                    any_nibbled = 1;
                }
            }
        }

        if (any_nibbled && ip > 0.0f) {
            /* Interior nibble: for nibbled pixels, also clear one pixel toward center. */
            for (int row = 0; row < h; row++) {
                for (int col = 0; col < w; col++) {
                    if (rim[row * w + col] != 2) continue;
                    if (sprite_rng_next_double(&rng) >= ip) continue;

                    /* Step toward center. */
                    int step_x = 0, step_y = 0;
                    float fdx = (float)center_x - (float)col;
                    float fdy = (float)center_y - (float)row;
                    if (fdx > 0.0f) step_x = 1;
                    else if (fdx < 0.0f) step_x = -1;
                    if (fdy > 0.0f) step_y = 1;
                    else if (fdy < 0.0f) step_y = -1;

                    int inner_x = clampi(col + step_x, 0, w - 1);
                    int inner_y = clampi(row + step_y, 0, h - 1);

                    if (PX(data, inner_y, inner_x, w)[3] > 128)
                        PX(data, inner_y, inner_x, w)[3] = 0;
                }
            }
        }

        free(rim);
    }

nibble_canopy_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* nibble_boulder: upper-half edge erosion for boulder silhouettes       */
/* ------------------------------------------------------------------ */

/*
 * sprite_nibble_boulder(canvas, seed, nibble_prob)
 *
 * Remove random edge pixels from the upper half of the boulder.
 * The bottom edge stays solid (ground contact).
 */
PyObject *brileta_native_sprite_nibble_boulder(PyObject *self, PyObject *args) {
    PyObject *canvas_obj;
    unsigned long long seed;
    double nibble_prob;

    if (!PyArg_ParseTuple(args, "OKd", &canvas_obj, &seed, &nibble_prob))
        return NULL;

    Py_buffer buf;
    int h, w;
    uint8_t *data;
    if (get_canvas_buffer(canvas_obj, &buf, &h, &w, &data) < 0)
        return NULL;

    int oom = 0;

    Py_BEGIN_ALLOW_THREADS

    /* Compute rim mask. */
    size_t n_pixels = (size_t)h * (size_t)w;
    uint8_t *rim = (uint8_t *)calloc(n_pixels, 1);
    if (!rim) { oom = 1; goto nibble_boulder_done; }
    compute_rim_mask(data, h, w, rim);

    /* Check if any rim pixels exist. */
    int has_rim = 0;
    for (size_t i = 0; i < n_pixels; i++) {
        if (rim[i]) { has_rim = 1; break; }
    }
    if (!has_rim) { free(rim); goto nibble_boulder_done; }

    {
        /* Find opaque region vertical extent to compute midpoint. */
        int first_opaque_row = h, last_opaque_row = -1;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (PX(data, row, col, w)[3] > 0) {
                    if (row < first_opaque_row) first_opaque_row = row;
                    if (row > last_opaque_row) last_opaque_row = row;
                }
            }
        }
        if (first_opaque_row > last_opaque_row) { free(rim); goto nibble_boulder_done; }
        int midpoint = (first_opaque_row + last_opaque_row) / 2;

        /* Only nibble in the upper half. */
        for (int row = midpoint; row < h; row++) {
            for (int col = 0; col < w; col++) {
                rim[row * w + col] = 0;
            }
        }

        SpriteRng rng;
        sprite_rng_init(&rng, (uint64_t)seed);

        float np = clampf((float)nibble_prob, 0.0f, 1.0f);

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                if (!rim[row * w + col]) continue;
                if (sprite_rng_next_double(&rng) < np) {
                    PX(data, row, col, w)[3] = 0;
                }
            }
        }

        free(rim);
    }

nibble_boulder_done:
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);
    if (oom) return PyErr_NoMemory();
    Py_RETURN_NONE;
}
