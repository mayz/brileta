/*
 * Native A* pathfinding implementation for brileta.
 *
 * This file implements the `astar` callable that is exported by the shared
 * `brileta.util._native` extension module.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Binary min-heap for the open set                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    double f;    /* f-score (g + heuristic) */
    int    idx;  /* flat grid index */
} HeapEntry;

typedef struct {
    HeapEntry *data;
    int        size;
    int        capacity;
} MinHeap;

static int heap_init(MinHeap *h, int capacity) {
    h->data = (HeapEntry *)malloc(sizeof(HeapEntry) * capacity);
    if (!h->data) return -1;
    h->size = 0;
    h->capacity = capacity;
    return 0;
}

static void heap_free(MinHeap *h) {
    free(h->data);
    h->data = NULL;
}

static inline void heap_swap(MinHeap *h, int *pos, int a, int b) {
    HeapEntry tmp = h->data[a];
    h->data[a] = h->data[b];
    h->data[b] = tmp;
    pos[h->data[a].idx] = a;
    pos[h->data[b].idx] = b;
}

static void heap_sift_up(MinHeap *h, int *pos, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (h->data[parent].f <= h->data[i].f) break;
        heap_swap(h, pos, parent, i);
        i = parent;
    }
}

static void heap_sift_down(MinHeap *h, int *pos, int i) {
    for (;;) {
        int left = 2 * i + 1;
        int right = left + 1;
        int smallest = i;
        if (left < h->size && h->data[left].f < h->data[smallest].f)
            smallest = left;
        if (right < h->size && h->data[right].f < h->data[smallest].f)
            smallest = right;
        if (smallest == i) break;
        heap_swap(h, pos, smallest, i);
        i = smallest;
    }
}

/*
 * Push a new node or decrease the key of an existing open node.
 *
 * pos[idx] stores the current heap slot for idx, or -1 if idx is not in heap.
 */
static int heap_push_or_decrease(MinHeap *h, int *pos, double f, int idx) {
    int p = pos[idx];
    if (p >= 0) {
        if (f >= h->data[p].f) return 0;
        h->data[p].f = f;
        heap_sift_up(h, pos, p);
        return 0;
    }

    if (h->size >= h->capacity) {
        int new_cap = h->capacity * 2;
        if (new_cap < 256) new_cap = 256;
        HeapEntry *new_data = (HeapEntry *)realloc(h->data, sizeof(HeapEntry) * new_cap);
        if (!new_data) return -1;
        h->data = new_data;
        h->capacity = new_cap;
    }
    int i = h->size++;
    h->data[i].f = f;
    h->data[i].idx = idx;
    pos[idx] = i;
    heap_sift_up(h, pos, i);
    return 0;
}

static HeapEntry heap_pop(MinHeap *h, int *pos) {
    HeapEntry top = h->data[0];
    pos[top.idx] = -1;

    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size];
        pos[h->data[0].idx] = 0;
        heap_sift_down(h, pos, 0);
    }
    return top;
}

/* ------------------------------------------------------------------ */
/* A* implementation                                                  */
/* ------------------------------------------------------------------ */

static const double SQRT2 = 1.4142135623730951;
static const double SQRT2_MINUS_2 = -0.5857864376269049;  /* sqrt(2) - 2 */
/*
 * Slight heuristic inflation reduces search fan-out in dense obstacle maps.
 * Costs remain octile; this only changes node expansion order.
 */
static const double HEURISTIC_WEIGHT = 1.01;

/* 8 neighbors: dx, dy, cost_multiplier */
static const int    DX[8] = {-1,  1,  0,  0, -1, -1,  1,  1};
static const int    DY[8] = { 0,  0, -1,  1, -1,  1, -1,  1};
static inline double octile_h_from_deltas(int dx, int dy) {
    int mn = dx < dy ? dx : dy;
    return (double)(dx + dy) + SQRT2_MINUS_2 * (double)mn;
}

/*
 * Run A* on a flat cost grid.
 *
 * cost:   flat int16 array of size w*h, indexed as cost[x * h + y].
 *         0 = impassable, positive = traversal weight.
 * w, h:   grid dimensions (width, height).
 * sx, sy: start position.
 * gx, gy: goal position.
 * out_path: output buffer for flat indices (caller allocates, size >= w*h).
 * out_len:  output path length (excluding start).
 *
 * Returns 0 on success, -1 on allocation failure.
 * If no path exists, *out_len = 0.
 */
static int astar_search(
    const short *cost, int w, int h,
    int sx, int sy, int gx, int gy,
    int *out_path, int *out_len
) {
    int size = w * h;
    int start_idx = sx * h + sy;
    int goal_idx = gx * h + gy;

    *out_len = 0;

    /* Quick exit: start or goal is blocked. */
    if (cost[start_idx] == 0 || cost[goal_idx] == 0) return 0;

    /* Allocate working arrays. */
    double *g_score = (double *)malloc(sizeof(double) * size);
    int   *came_from = (int *)malloc(sizeof(int) * size);
    char  *closed = (char *)calloc(size, 1);
    int   *heap_pos = (int *)malloc(sizeof(int) * size);
    int   *goal_dx = (int *)malloc(sizeof(int) * w);
    int   *goal_dy = (int *)malloc(sizeof(int) * h);

    if (!g_score || !came_from || !closed || !heap_pos || !goal_dx || !goal_dy) {
        free(g_score); free(came_from); free(closed); free(heap_pos);
        free(goal_dx); free(goal_dy);
        return -1;
    }

    /* Initialize g_score to infinity. */
    for (int i = 0; i < size; i++) g_score[i] = 1e30;
    g_score[start_idx] = 0.0;
    memset(came_from, -1, sizeof(int) * size);
    /*
     * Set heap positions to -1 (not in heap).  0xFF relies on two's
     * complement representation for signed integers (universal on supported
     * targets; mandated by C23).
     */
    memset(heap_pos, 0xFF, sizeof(int) * size);
    for (int x = 0; x < w; x++) {
        int dx = x - gx;
        goal_dx[x] = dx < 0 ? -dx : dx;
    }
    for (int y = 0; y < h; y++) {
        int dy = y - gy;
        goal_dy[y] = dy < 0 ? -dy : dy;
    }

    MinHeap heap;
    int initial_cap = size < 256 ? size : 256;
    if (heap_init(&heap, initial_cap) < 0) {
        free(g_score); free(came_from); free(closed); free(heap_pos);
        free(goal_dx); free(goal_dy);
        return -1;
    }
    if (heap_push_or_decrease(
            &heap,
            heap_pos,
            HEURISTIC_WEIGHT * octile_h_from_deltas(goal_dx[sx], goal_dy[sy]),
            start_idx
        ) < 0) {
        heap_free(&heap);
        free(g_score); free(came_from); free(closed); free(heap_pos);
        free(goal_dx); free(goal_dy);
        return -1;
    }

    int found = 0;

    while (heap.size > 0) {
        HeapEntry top = heap_pop(&heap, heap_pos);
        int ci = top.idx;

        if (ci == goal_idx) { found = 1; break; }
        closed[ci] = 1;

        double cg = g_score[ci];
        int cx = ci / h;
        int cy = ci % h;

        for (int d = 0; d < 8; d++) {
            int nx = cx + DX[d];
            int ny = cy + DY[d];
            if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;

            int ni = nx * h + ny;
            if (closed[ni]) continue;

            short nc = cost[ni];
            if (nc == 0) continue;

            double mult = d < 4 ? 1.0 : SQRT2;
            double tent_g = cg + (double)nc * mult;
            if (tent_g < g_score[ni]) {
                g_score[ni] = tent_g;
                came_from[ni] = ci;
                double f = tent_g + HEURISTIC_WEIGHT * octile_h_from_deltas(goal_dx[nx], goal_dy[ny]);
                if (heap_push_or_decrease(&heap, heap_pos, f, ni) < 0) {
                    heap_free(&heap);
                    free(g_score); free(came_from); free(closed); free(heap_pos);
                    free(goal_dx); free(goal_dy);
                    return -1;
                }
            }
        }
    }

    if (found) {
        /* Reconstruct path (excluding start). */
        int len = 0;
        int node = goal_idx;
        while (node != start_idx) {
            out_path[len++] = node;
            node = came_from[node];
        }
        /* Reverse in place. */
        for (int i = 0; i < len / 2; i++) {
            int tmp = out_path[i];
            out_path[i] = out_path[len - 1 - i];
            out_path[len - 1 - i] = tmp;
        }
        *out_len = len;
    }

    heap_free(&heap);
    free(g_score);
    free(came_from);
    free(closed);
    free(heap_pos);
    free(goal_dx);
    free(goal_dy);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Python interface                                                   */
/* ------------------------------------------------------------------ */

/*
 * astar(cost_array, start_x, start_y, goal_x, goal_y) -> list[tuple[int,int]]
 *
 * cost_array: numpy int16 array of shape (width, height), C-contiguous.
 * Returns list of (x, y) tuples (excluding start), or empty list if no path.
 */
PyObject *brileta_native_astar(PyObject *self, PyObject *args) {
    PyObject *cost_obj;
    int sx, sy, gx, gy;

    if (!PyArg_ParseTuple(args, "Oiiii", &cost_obj, &sx, &sy, &gx, &gy))
        return NULL;

    /* Get buffer from numpy array. */
    Py_buffer buf;
    if (PyObject_GetBuffer(cost_obj, &buf, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0)
        return NULL;

    /* Validate format: must be int16 ('h'). */
    if (buf.ndim != 2 || strcmp(buf.format, "h") != 0) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_TypeError,
                        "cost must be a 2D int16 C-contiguous array");
        return NULL;
    }

    int w = (int)buf.shape[0];
    int h = (int)buf.shape[1];
    const short *cost = (const short *)buf.buf;

    /* Bounds check. */
    if (sx < 0 || sx >= w || sy < 0 || sy >= h ||
        gx < 0 || gx >= w || gy < 0 || gy >= h) {
        PyBuffer_Release(&buf);
        PyErr_SetString(PyExc_ValueError,
                        "start or goal is out of bounds");
        return NULL;
    }

    /* Same start and goal. */
    if (sx == gx && sy == gy) {
        PyBuffer_Release(&buf);
        return PyList_New(0);
    }

    /* Allocate output buffer. */
    int *path_buf = (int *)malloc(sizeof(int) * w * h);
    if (!path_buf) {
        PyBuffer_Release(&buf);
        return PyErr_NoMemory();
    }

    int path_len;
    int rc;
    Py_BEGIN_ALLOW_THREADS
    rc = astar_search(cost, w, h, sx, sy, gx, gy, path_buf, &path_len);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&buf);

    if (rc < 0) {
        free(path_buf);
        return PyErr_NoMemory();
    }

    /* Build Python list of (x, y) tuples. */
    PyObject *result = PyList_New(path_len);
    if (!result) { free(path_buf); return NULL; }

    for (int i = 0; i < path_len; i++) {
        int idx = path_buf[i];
        int x = idx / h;
        int y = idx % h;
        PyObject *tup = Py_BuildValue("(ii)", x, y);
        if (!tup) {
            Py_DECREF(result);
            free(path_buf);
            return NULL;
        }
        PyList_SET_ITEM(result, i, tup);
    }

    free(path_buf);
    return result;
}
