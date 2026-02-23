/*
 * Native FastNoiseLite wrapper for brileta.
 *
 * Exposes a _NoiseState Python type that wraps the FastNoiseLite fnl_state
 * struct.  The type supports configurable noise generation via sample_2d()
 * and sample_3d() methods, plus domain warping via domain_warp_2d() and
 * domain_warp_3d().
 *
 * The implementation is included here (header-only library) via FNL_IMPL.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define FNL_IMPL
#include "FastNoiseLite.h"

/* ------------------------------------------------------------------ */
/* Python type wrapping fnl_state                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD
    fnl_state state;
} NoiseStateObject;

static int NoiseState_init(
    NoiseStateObject *self, PyObject *args, PyObject *kwds
) {
    static char *kwlist[] = {
        "seed", "noise_type", "frequency",
        "fractal_type", "octaves", "lacunarity", "gain",
        "weighted_strength", "ping_pong_strength",
        "cellular_distance_func", "cellular_return_type",
        "cellular_jitter_mod",
        "domain_warp_type", "domain_warp_amp",
        NULL
    };

    int seed = 1337;
    int noise_type = FNL_NOISE_OPENSIMPLEX2;
    float frequency = 0.01f;
    int fractal_type = FNL_FRACTAL_NONE;
    int octaves = 3;
    float lacunarity = 2.0f;
    float gain = 0.5f;
    float weighted_strength = 0.0f;
    float ping_pong_strength = 2.0f;
    int cellular_distance_func = FNL_CELLULAR_DISTANCE_EUCLIDEANSQ;
    int cellular_return_type = FNL_CELLULAR_RETURN_TYPE_DISTANCE;
    float cellular_jitter_mod = 1.0f;
    int domain_warp_type = FNL_DOMAIN_WARP_OPENSIMPLEX2;
    float domain_warp_amp = 1.0f;

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|iifiiffffiifif", kwlist,
            &seed, &noise_type, &frequency,
            &fractal_type, &octaves, &lacunarity, &gain,
            &weighted_strength, &ping_pong_strength,
            &cellular_distance_func, &cellular_return_type,
            &cellular_jitter_mod,
            &domain_warp_type, &domain_warp_amp
        )) {
        return -1;
    }

    self->state = fnlCreateState();
    self->state.seed = seed;
    self->state.noise_type = (fnl_noise_type)noise_type;
    self->state.frequency = frequency;
    self->state.fractal_type = (fnl_fractal_type)fractal_type;
    self->state.octaves = octaves;
    self->state.lacunarity = lacunarity;
    self->state.gain = gain;
    self->state.weighted_strength = weighted_strength;
    self->state.ping_pong_strength = ping_pong_strength;
    self->state.cellular_distance_func =
        (fnl_cellular_distance_func)cellular_distance_func;
    self->state.cellular_return_type =
        (fnl_cellular_return_type)cellular_return_type;
    self->state.cellular_jitter_mod = cellular_jitter_mod;
    self->state.domain_warp_type = (fnl_domain_warp_type)domain_warp_type;
    self->state.domain_warp_amp = domain_warp_amp;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Methods                                                            */
/* ------------------------------------------------------------------ */

static PyObject *NoiseState_sample_2d(
    NoiseStateObject *self, PyObject *args
) {
    float x, y;
    if (!PyArg_ParseTuple(args, "ff", &x, &y))
        return NULL;

    float val = fnlGetNoise2D(&self->state, x, y);
    return PyFloat_FromDouble((double)val);
}

static PyObject *NoiseState_sample_3d(
    NoiseStateObject *self, PyObject *args
) {
    float x, y, z;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z))
        return NULL;

    float val = fnlGetNoise3D(&self->state, x, y, z);
    return PyFloat_FromDouble((double)val);
}

static PyObject *NoiseState_domain_warp_2d(
    NoiseStateObject *self, PyObject *args
) {
    FNLfloat x, y;
    if (!PyArg_ParseTuple(args, "ff", &x, &y))
        return NULL;

    fnlDomainWarp2D(&self->state, &x, &y);
    return Py_BuildValue("ff", x, y);
}

static PyObject *NoiseState_domain_warp_3d(
    NoiseStateObject *self, PyObject *args
) {
    FNLfloat x, y, z;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z))
        return NULL;

    fnlDomainWarp3D(&self->state, &x, &y, &z);
    return Py_BuildValue("fff", x, y, z);
}

/* ------------------------------------------------------------------ */
/* Batch methods - sample arrays of coordinates in a tight C loop.    */
/* Uses the Python buffer protocol (no numpy C API dependency).       */
/* All coordinate buffers must be contiguous float32 ('f') with the   */
/* same length.  The output buffer must be writable float32.          */
/* ------------------------------------------------------------------ */

/* Helper: acquire a contiguous float32 buffer and return element count.
 * Sets a Python exception and returns -1 on failure. */
static Py_ssize_t acquire_f32_buffer(
    PyObject *obj, Py_buffer *buf, int writable
) {
    int flags = PyBUF_C_CONTIGUOUS | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;

    if (PyObject_GetBuffer(obj, buf, flags) < 0)
        return -1;

    if (buf->format == NULL || buf->format[0] != 'f' || buf->format[1] != '\0') {
        PyBuffer_Release(buf);
        PyErr_SetString(PyExc_TypeError, "buffer must be float32 ('f') format");
        return -1;
    }
    if (buf->ndim != 1) {
        PyBuffer_Release(buf);
        PyErr_SetString(PyExc_TypeError, "buffer must be 1-dimensional");
        return -1;
    }
    return buf->shape[0];
}

static PyObject *NoiseState_sample_2d_array(
    NoiseStateObject *self, PyObject *args
) {
    PyObject *xobj, *yobj, *outobj;
    if (!PyArg_ParseTuple(args, "OOO", &xobj, &yobj, &outobj))
        return NULL;

    Py_buffer xbuf, ybuf, obuf;
    Py_ssize_t nx = acquire_f32_buffer(xobj, &xbuf, 0);
    if (nx < 0) return NULL;

    Py_ssize_t ny = acquire_f32_buffer(yobj, &ybuf, 0);
    if (ny < 0) { PyBuffer_Release(&xbuf); return NULL; }

    Py_ssize_t no = acquire_f32_buffer(outobj, &obuf, 1);
    if (no < 0) { PyBuffer_Release(&xbuf); PyBuffer_Release(&ybuf); return NULL; }

    if (nx != ny || nx != no) {
        PyBuffer_Release(&xbuf);
        PyBuffer_Release(&ybuf);
        PyBuffer_Release(&obuf);
        PyErr_SetString(PyExc_ValueError, "all buffers must have the same length");
        return NULL;
    }

    const float *xs = (const float *)xbuf.buf;
    const float *ys = (const float *)ybuf.buf;
    float *out = (float *)obuf.buf;
    fnl_state local_state = self->state;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nx; i++) {
        out[i] = fnlGetNoise2D(&local_state, xs[i], ys[i]);
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&xbuf);
    PyBuffer_Release(&ybuf);
    PyBuffer_Release(&obuf);
    Py_RETURN_NONE;
}

static PyObject *NoiseState_sample_3d_array(
    NoiseStateObject *self, PyObject *args
) {
    PyObject *xobj, *yobj, *zobj, *outobj;
    if (!PyArg_ParseTuple(args, "OOOO", &xobj, &yobj, &zobj, &outobj))
        return NULL;

    Py_buffer xbuf, ybuf, zbuf, obuf;
    Py_ssize_t nx = acquire_f32_buffer(xobj, &xbuf, 0);
    if (nx < 0) return NULL;

    Py_ssize_t ny = acquire_f32_buffer(yobj, &ybuf, 0);
    if (ny < 0) { PyBuffer_Release(&xbuf); return NULL; }

    Py_ssize_t nz = acquire_f32_buffer(zobj, &zbuf, 0);
    if (nz < 0) { PyBuffer_Release(&xbuf); PyBuffer_Release(&ybuf); return NULL; }

    Py_ssize_t no = acquire_f32_buffer(outobj, &obuf, 1);
    if (no < 0) {
        PyBuffer_Release(&xbuf); PyBuffer_Release(&ybuf);
        PyBuffer_Release(&zbuf); return NULL;
    }

    if (nx != ny || nx != nz || nx != no) {
        PyBuffer_Release(&xbuf);
        PyBuffer_Release(&ybuf);
        PyBuffer_Release(&zbuf);
        PyBuffer_Release(&obuf);
        PyErr_SetString(PyExc_ValueError, "all buffers must have the same length");
        return NULL;
    }

    const float *xs = (const float *)xbuf.buf;
    const float *ys = (const float *)ybuf.buf;
    const float *zs = (const float *)zbuf.buf;
    float *out = (float *)obuf.buf;
    fnl_state local_state = self->state;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < nx; i++) {
        out[i] = fnlGetNoise3D(&local_state, xs[i], ys[i], zs[i]);
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&xbuf);
    PyBuffer_Release(&ybuf);
    PyBuffer_Release(&zbuf);
    PyBuffer_Release(&obuf);
    Py_RETURN_NONE;
}

static PyMethodDef NoiseState_methods[] = {
    {"sample_2d", (PyCFunction)NoiseState_sample_2d, METH_VARARGS,
     "sample_2d(x, y) -> float\n\n"
     "Sample 2D noise at the given position. Returns value in [-1, 1]."},
    {"sample_3d", (PyCFunction)NoiseState_sample_3d, METH_VARARGS,
     "sample_3d(x, y, z) -> float\n\n"
     "Sample 3D noise at the given position. Returns value in [-1, 1]."},
    {"sample_2d_array",
     (PyCFunction)NoiseState_sample_2d_array, METH_VARARGS,
     "sample_2d_array(xs, ys, out) -> None\n\n"
     "Sample 2D noise for arrays of coordinates.\n"
     "All arguments must be contiguous float32 buffers of the same length.\n"
     "Results are written into *out* in-place (values in [-1, 1])."},
    {"sample_3d_array",
     (PyCFunction)NoiseState_sample_3d_array, METH_VARARGS,
     "sample_3d_array(xs, ys, zs, out) -> None\n\n"
     "Sample 3D noise for arrays of coordinates.\n"
     "All arguments must be contiguous float32 buffers of the same length.\n"
     "Results are written into *out* in-place (values in [-1, 1])."},
    {"domain_warp_2d", (PyCFunction)NoiseState_domain_warp_2d, METH_VARARGS,
     "domain_warp_2d(x, y) -> (float, float)\n\n"
     "Apply domain warping to a 2D position. Returns warped (x, y)."},
    {"domain_warp_3d", (PyCFunction)NoiseState_domain_warp_3d, METH_VARARGS,
     "domain_warp_3d(x, y, z) -> (float, float, float)\n\n"
     "Apply domain warping to a 3D position. Returns warped (x, y, z)."},
    {NULL, NULL, 0, NULL}
};

/* ------------------------------------------------------------------ */
/* Properties (read-write access to the most commonly tweaked fields)  */
/* ------------------------------------------------------------------ */

static PyObject *NoiseState_get_seed(
    NoiseStateObject *self, void *closure
) {
    (void)closure;
    return PyLong_FromLong(self->state.seed);
}

static int NoiseState_set_seed(
    NoiseStateObject *self, PyObject *value, void *closure
) {
    (void)closure;
    if (!value) {
        PyErr_SetString(PyExc_AttributeError, "cannot delete seed");
        return -1;
    }
    long seed = PyLong_AsLong(value);
    if (seed == -1 && PyErr_Occurred())
        return -1;
    self->state.seed = (int)seed;
    return 0;
}

static PyObject *NoiseState_get_frequency(
    NoiseStateObject *self, void *closure
) {
    (void)closure;
    return PyFloat_FromDouble((double)self->state.frequency);
}

static int NoiseState_set_frequency(
    NoiseStateObject *self, PyObject *value, void *closure
) {
    (void)closure;
    if (!value) {
        PyErr_SetString(PyExc_AttributeError, "cannot delete frequency");
        return -1;
    }
    double freq = PyFloat_AsDouble(value);
    if (freq == -1.0 && PyErr_Occurred())
        return -1;
    self->state.frequency = (float)freq;
    return 0;
}

static PyGetSetDef NoiseState_getset[] = {
    {"seed", (getter)NoiseState_get_seed, (setter)NoiseState_set_seed,
     "Seed used for noise generation.", NULL},
    {"frequency", (getter)NoiseState_get_frequency,
     (setter)NoiseState_set_frequency,
     "Frequency for noise generation.", NULL},
    {NULL, NULL, NULL, NULL, NULL}
};

/* ------------------------------------------------------------------ */
/* Type definition                                                    */
/* ------------------------------------------------------------------ */

static PyTypeObject NoiseStateType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "brileta.util._native._NoiseState",
    .tp_doc =
        "_NoiseState(seed=1337, noise_type=0, frequency=0.01, ...)\n\n"
        "Low-level wrapper around FastNoiseLite fnl_state.\n"
        "Use brileta.util.noise.NoiseGenerator for the high-level API.",
    .tp_basicsize = sizeof(NoiseStateObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)NoiseState_init,
    .tp_methods = NoiseState_methods,
    .tp_getset = NoiseState_getset,
};

/* ------------------------------------------------------------------ */
/* Module-level registration (called from _native.c PyInit__native)   */
/* ------------------------------------------------------------------ */

int brileta_native_init_noise_type(PyObject *module) {
    if (PyType_Ready(&NoiseStateType) < 0)
        return -1;

    if (PyModule_AddObjectRef(
            module, "_NoiseState", (PyObject *)&NoiseStateType
        ) < 0) {
        return -1;
    }

    return 0;
}
