/*
 * Fast glyph-buffer → vertex-buffer encoder for the WGPU glyph renderer.
 *
 * Replaces the Python/numpy _build_glyph_vertices() method.  A single tight
 * loop reads each GlyphCell, converts characters to CP437 UV coordinates,
 * expands per-cell colours/noise/edge data to 6 triangle vertices, and writes
 * them sequentially into the pre-allocated output buffer.
 *
 * The GIL is released during the encoding pass so other Python threads can
 * run concurrently.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

/* ── Glyph buffer cell layout (must match brileta.util.glyph_buffer.GLYPH_DTYPE) ── */

#pragma pack(push, 1)
typedef struct {
    int32_t ch;                     /* Character code (Unicode codepoint)       */
    uint8_t fg[4];                  /* Foreground RGBA                          */
    uint8_t bg[4];                  /* Background RGBA                          */
    float noise;                    /* Sub-tile noise amplitude                 */
    uint8_t noise_pattern;          /* Sub-tile noise pattern ID                */
    uint8_t edge_neighbor_mask;     /* Cardinal diff mask (W/N/S/E bits)        */
    float edge_blend;               /* Edge feathering amplitude (0.0-1.0)      */
    uint8_t edge_neighbor_bg[4][3]; /* Neighbor background RGB [W,N,S,E]×[R,G,B] */
    /* Sub-tile split fields for perspective offset boundary tiles. */
    float split_y;               /* Y threshold [0,1]: 0=no split            */
    uint8_t split_bg[4];         /* Background RGBA for below-split portion  */
    uint8_t split_fg[4];         /* Foreground RGBA for below-split portion  */
    float split_noise;           /* Noise amplitude for below-split portion  */
    uint8_t split_noise_pattern; /* Noise pattern for below-split portion    */
} GlyphCell;
#pragma pack(pop)

/* ── Vertex layout (must match glyph_renderer.TEXTURE_VERTEX_DTYPE) ── */

typedef struct {
    float position[2];            /* Screen (x, y)                            */
    float uv[2];                  /* Tileset (u, v)                           */
    float fg_color[4];            /* Foreground RGBA as 0.0-1.0               */
    float bg_color[4];            /* Background RGBA as 0.0-1.0               */
    float noise_amplitude;        /* Sub-tile noise amplitude                 */
    uint32_t noise_pattern;       /* Noise pattern ID (u32 for WGSL)          */
    uint32_t edge_neighbor_mask;  /* Cardinal boundary mask (u32 for WGSL)    */
    float edge_blend;             /* Edge feathering amplitude                */
    float edge_neighbor_bg[4][3]; /* Neighbor bg as 0.0-1.0 [W,N,S,E]×[R,G,B] */
    /* Sub-tile split fields for perspective offset boundary tiles. */
    float split_y;                /* Y threshold [0,1]: 0=no split            */
    float split_bg_color[4];      /* Background RGBA as 0.0-1.0               */
    float split_fg_color[4];      /* Foreground RGBA as 0.0-1.0               */
    float split_noise_amplitude;  /* Noise amplitude for below-split portion  */
    uint32_t split_noise_pattern; /* Noise pattern for below-split portion    */
} GlyphVertex;

/* Compile-time size checks against the numpy dtype sizes. */
_Static_assert(sizeof(GlyphCell) == 51, "GlyphCell size must match GLYPH_DTYPE (51 bytes)");
_Static_assert(sizeof(GlyphVertex) == 156,
               "GlyphVertex size must match TEXTURE_VERTEX_DTYPE (156 bytes)");

/* ── Core encoding loop (GIL-free) ── */

static int encode_glyph_vertices(const GlyphCell *glyph_data, /* (w, h) C-contiguous glyph cells */
                                 GlyphVertex *output, /* output vertex buffer, flat         */
                                 const float *uv_map, /* (256, 4) float32: u1, v1, u2, v2 per idx */
                                 const uint8_t *cp437_map, /* unicode-to-CP437 lookup table */
                                 int cp437_size, /* length of cp437_map                       */
                                 int w,          /* glyph buffer width (axis 0)               */
                                 int h,          /* glyph buffer height (axis 1)              */
                                 float tile_w,   /* tile width in pixels                      */
                                 float tile_h)   /* tile height in pixels                     */
{
    const float inv255 = 1.0f / 255.0f;
    int total_vertices = 0;

    /*
     * Iteration order: y-outer, x-inner.  This writes the vertex buffer
     * sequentially (it is shaped (h, w, 6) in Python) which is better for
     * cache since each vertex is 112 bytes.  Glyph reads stride by h
     * (glyph_data is (w, h) C-contiguous) but at 34 bytes per cell this
     * is a much smaller working set.
     */
    for (int y = 0; y < h; y++) {
        float y1_px = (float)y * tile_h;
        float y2_px = y1_px + tile_h;

        for (int x = 0; x < w; x++) {
            const GlyphCell *cell = &glyph_data[x * h + y];
            GlyphVertex *v = &output[(y * w + x) * 6];

            /* ── Character → CP437 → UV lookup ── */
            int32_t ch = cell->ch;
            uint8_t cp437_idx;
            if (ch >= 0 && ch < cp437_size)
                cp437_idx = cp437_map[ch];
            else
                cp437_idx = cp437_map[(int)'?'];

            const float *uv = &uv_map[cp437_idx * 4];
            float u1 = uv[0], v1_uv = uv[1], u2 = uv[2], v2_uv = uv[3];

            /* ── Screen positions ── */
            float x1_px = (float)x * tile_w;
            float x2_px = x1_px + tile_w;

            /* ── Colours: uint8 RGBA → float 0.0-1.0 ── */
            float fg_r = cell->fg[0] * inv255;
            float fg_g = cell->fg[1] * inv255;
            float fg_b = cell->fg[2] * inv255;
            float fg_a = cell->fg[3] * inv255;
            float bg_r = cell->bg[0] * inv255;
            float bg_g = cell->bg[1] * inv255;
            float bg_b = cell->bg[2] * inv255;
            float bg_a = cell->bg[3] * inv255;

            /* ── Per-cell attributes (shared by all 6 vertices) ── */
            float noise_amp = cell->noise;
            uint32_t noise_pat = (uint32_t)cell->noise_pattern;
            uint32_t edge_mask = (uint32_t)cell->edge_neighbor_mask;
            float edge_bld = cell->edge_blend;

            /* Neighbor bg: uint8 [4][3] → float [4][3] */
            float nbg[4][3];
            for (int n = 0; n < 4; n++) {
                nbg[n][0] = cell->edge_neighbor_bg[n][0] * inv255;
                nbg[n][1] = cell->edge_neighbor_bg[n][1] * inv255;
                nbg[n][2] = cell->edge_neighbor_bg[n][2] * inv255;
            }

            /* ── Split fields: uint8 RGBA → float, pattern → u32 ── */
            float s_y = cell->split_y;
            float s_bg[4] = {cell->split_bg[0] * inv255,
                             cell->split_bg[1] * inv255,
                             cell->split_bg[2] * inv255,
                             cell->split_bg[3] * inv255};
            float s_fg[4] = {cell->split_fg[0] * inv255,
                             cell->split_fg[1] * inv255,
                             cell->split_fg[2] * inv255,
                             cell->split_fg[3] * inv255};
            float s_noise = cell->split_noise;
            uint32_t s_npat = (uint32_t)cell->split_noise_pattern;

            /* ── Write shared data to all 6 vertices ── */
            for (int i = 0; i < 6; i++) {
                v[i].fg_color[0] = fg_r;
                v[i].fg_color[1] = fg_g;
                v[i].fg_color[2] = fg_b;
                v[i].fg_color[3] = fg_a;
                v[i].bg_color[0] = bg_r;
                v[i].bg_color[1] = bg_g;
                v[i].bg_color[2] = bg_b;
                v[i].bg_color[3] = bg_a;
                v[i].noise_amplitude = noise_amp;
                v[i].noise_pattern = noise_pat;
                v[i].edge_neighbor_mask = edge_mask;
                v[i].edge_blend = edge_bld;
                memcpy(v[i].edge_neighbor_bg, nbg, sizeof(nbg));
                v[i].split_y = s_y;
                memcpy(v[i].split_bg_color, s_bg, sizeof(s_bg));
                memcpy(v[i].split_fg_color, s_fg, sizeof(s_fg));
                v[i].split_noise_amplitude = s_noise;
                v[i].split_noise_pattern = s_npat;
            }

            /* ── Per-vertex position and UV (two triangles forming a quad) ── */

            /* Vertex 0: bottom-left */
            v[0].position[0] = x1_px;
            v[0].position[1] = y1_px;
            v[0].uv[0] = u1;
            v[0].uv[1] = v1_uv;

            /* Vertex 1: bottom-right */
            v[1].position[0] = x2_px;
            v[1].position[1] = y1_px;
            v[1].uv[0] = u2;
            v[1].uv[1] = v1_uv;

            /* Vertex 2: top-left */
            v[2].position[0] = x1_px;
            v[2].position[1] = y2_px;
            v[2].uv[0] = u1;
            v[2].uv[1] = v2_uv;

            /* Vertex 3: bottom-right (same as 1) */
            v[3].position[0] = x2_px;
            v[3].position[1] = y1_px;
            v[3].uv[0] = u2;
            v[3].uv[1] = v1_uv;

            /* Vertex 4: top-left (same as 2) */
            v[4].position[0] = x1_px;
            v[4].position[1] = y2_px;
            v[4].uv[0] = u1;
            v[4].uv[1] = v2_uv;

            /* Vertex 5: top-right */
            v[5].position[0] = x2_px;
            v[5].position[1] = y2_px;
            v[5].uv[0] = u2;
            v[5].uv[1] = v2_uv;

            total_vertices += 6;
        }
    }

    return total_vertices;
}

/* ── Python wrapper ── */

PyObject *brileta_native_build_glyph_vertices(PyObject *self, PyObject *args) {
    PyObject *glyph_obj, *output_obj, *uv_obj, *cp437_obj;
    float tile_w, tile_h;

    if (!PyArg_ParseTuple(args,
                          "OOOOff",
                          &glyph_obj,  /* glyph buffer data array     */
                          &output_obj, /* output vertex buffer         */
                          &uv_obj,     /* UV map (256, 4) float32     */
                          &cp437_obj,  /* unicode→CP437 lookup table  */
                          &tile_w,
                          &tile_h))
        return NULL;

    /* ── Acquire buffer views ── */

    Py_buffer glyph_buf;
    if (PyObject_GetBuffer(glyph_obj, &glyph_buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    Py_buffer output_buf;
    if (PyObject_GetBuffer(output_obj, &output_buf, PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&glyph_buf);
        return NULL;
    }

    Py_buffer uv_buf;
    if (PyObject_GetBuffer(uv_obj, &uv_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&output_buf);
        PyBuffer_Release(&glyph_buf);
        return NULL;
    }

    Py_buffer cp437_buf;
    if (PyObject_GetBuffer(cp437_obj, &cp437_buf, PyBUF_C_CONTIGUOUS) < 0) {
        PyBuffer_Release(&uv_buf);
        PyBuffer_Release(&output_buf);
        PyBuffer_Release(&glyph_buf);
        return NULL;
    }

    /* ── Validate glyph buffer ── */
    if (glyph_buf.ndim != 2) {
        PyErr_SetString(PyExc_TypeError, "glyph_data must be a 2D array (width, height)");
        goto error;
    }
    if ((Py_ssize_t)glyph_buf.itemsize != (Py_ssize_t)sizeof(GlyphCell)) {
        PyErr_Format(PyExc_TypeError,
                     "glyph_data itemsize %zd does not match GlyphCell (%zu)",
                     glyph_buf.itemsize,
                     sizeof(GlyphCell));
        goto error;
    }

    int w = (int)glyph_buf.shape[0];
    int h = (int)glyph_buf.shape[1];
    int num_cells = w * h;
    int num_vertices = num_cells * 6;

    /* ── Validate output buffer ── */
    Py_ssize_t output_capacity = output_buf.len / (Py_ssize_t)sizeof(GlyphVertex);
    if (output_capacity < num_vertices) {
        PyErr_Format(PyExc_ValueError,
                     "output buffer too small: needs %d vertices, has %zd",
                     num_vertices,
                     output_capacity);
        goto error;
    }

    /* ── Validate UV map (256 entries × 4 floats) ── */
    if (uv_buf.len < (Py_ssize_t)(256 * 4 * sizeof(float))) {
        PyErr_SetString(PyExc_TypeError, "uv_map must have at least 256 entries of 4 floats");
        goto error;
    }

    /* ── Extract pointers ── */
    const GlyphCell *glyph_data = (const GlyphCell *)glyph_buf.buf;
    GlyphVertex *output = (GlyphVertex *)output_buf.buf;
    const float *uv_map = (const float *)uv_buf.buf;
    const uint8_t *cp437_map = (const uint8_t *)cp437_buf.buf;
    int cp437_size = (int)(cp437_buf.len / cp437_buf.itemsize);

    /* ── Encode (GIL released) ── */
    int result;
    /* clang-format off */
    Py_BEGIN_ALLOW_THREADS
    result = encode_glyph_vertices(
        glyph_data, output, uv_map, cp437_map, cp437_size, w, h, tile_w, tile_h);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&cp437_buf);
    PyBuffer_Release(&uv_buf);
    PyBuffer_Release(&output_buf);
    PyBuffer_Release(&glyph_buf);
    /* clang-format on */
    return PyLong_FromLong(result);

error:
    PyBuffer_Release(&cp437_buf);
    PyBuffer_Release(&uv_buf);
    PyBuffer_Release(&output_buf);
    PyBuffer_Release(&glyph_buf);
    return NULL;
}
