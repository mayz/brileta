/*
 * Native Wave Function Collapse solver for catley.
 *
 * This file implements the `wfc_solve` callable exported by the shared
 * `catley.util._native` extension module.
 *
 * Data model:
 * - Each cell stores possible patterns as a uint8 bitmask (max 8 patterns).
 * - The wave is flattened to a 1D buffer using x * height + y indexing.
 * - Constraint propagation uses precomputed lookup tables:
 *   propagation_masks[dir][current_mask] -> valid neighbor mask.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Defined in _native.c during module initialization. */
extern PyObject *catley_native_wfc_contradiction_error;

/* ------------------------------------------------------------------ */
/* Popcount lookup                                                     */
/* ------------------------------------------------------------------ */

static uint8_t POPCOUNT_TABLE[256];
static int POPCOUNT_READY = 0;

static void ensure_popcount_table(void) {
    if (POPCOUNT_READY) return;

    for (int i = 0; i < 256; i++) {
        int v = i;
        uint8_t c = 0;
        while (v) {
            c += (uint8_t)(v & 1);
            v >>= 1;
        }
        POPCOUNT_TABLE[i] = c;
    }

    POPCOUNT_READY = 1;
}

static inline uint8_t popcount_u8(uint8_t mask) {
    return POPCOUNT_TABLE[mask];
}

/* ------------------------------------------------------------------ */
/* xoshiro128++ PRNG with SplitMix64 seeding                           */
/* ------------------------------------------------------------------ */

/*
 * WfcRng is the local RNG state used by the native solver.
 *
 * Why xoshiro128++:
 * - Fast enough for tight inner loops.
 * - Good statistical quality for game/procedural content.
 * - Small state footprint (4x32-bit).
 *
 * We seed xoshiro with SplitMix64 because:
 * - Python provides a single 64-bit seed.
 * - xoshiro needs multiple non-zero state words.
 * - SplitMix64 expands one seed into well-scrambled state values.
 */
typedef struct {
    uint32_t s[4];
} WfcRng;

static inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z;

    *state += 0x9E3779B97F4A7C15ULL;
    z = *state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_init(WfcRng *rng, uint64_t seed) {
    uint64_t sm = seed;
    uint64_t a = splitmix64_next(&sm);
    uint64_t b = splitmix64_next(&sm);

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

static inline uint32_t rng_next_u32(WfcRng *rng) {
    /* xoshiro128++ output scrambler. */
    uint32_t result = rotl32(rng->s[0] + rng->s[3], 7) + rng->s[0];
    uint32_t t = rng->s[1] << 9;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = rotl32(rng->s[3], 11);

    return result;
}

static inline double rng_next_double(WfcRng *rng) {
    /*
     * Generate a float in [0, 1) from the upper 53 random bits.
     * This mirrors common high-quality float conversion schemes and avoids
     * leaning on lower bits, which are the weakest bits for xoshiro/xoroshiro
     * family generators.
     */
    uint64_t hi = (uint64_t)(rng_next_u32(rng) >> 5);  /* 27 bits */
    uint64_t lo = (uint64_t)(rng_next_u32(rng) >> 6);  /* 26 bits */
    uint64_t mantissa = (hi << 26) | lo;               /* 53 bits */
    return (double)mantissa * (1.0 / 9007199254740992.0);  /* 2^-53 */
}

/* ------------------------------------------------------------------ */
/* Push-only min-heap with stale entry skipping                        */
/* ------------------------------------------------------------------ */

typedef struct {
    double entropy;
    uint64_t counter;
    int idx;
} HeapEntry;

typedef struct {
    HeapEntry *data;
    int size;
    int capacity;
} MinHeap;

static inline int heap_less(const HeapEntry *a, const HeapEntry *b) {
    if (a->entropy < b->entropy) return 1;
    if (a->entropy > b->entropy) return 0;
    return a->counter < b->counter;
}

static int heap_init(MinHeap *heap, int capacity) {
    if (capacity < 64) capacity = 64;

    heap->data = (HeapEntry *)malloc(sizeof(HeapEntry) * capacity);
    if (!heap->data) return -1;

    heap->size = 0;
    heap->capacity = capacity;
    return 0;
}

static void heap_free(MinHeap *heap) {
    free(heap->data);
    heap->data = NULL;
    heap->size = 0;
    heap->capacity = 0;
}

static int heap_push(MinHeap *heap, HeapEntry entry) {
    if (heap->size >= heap->capacity) {
        int new_capacity = heap->capacity * 2;
        HeapEntry *new_data = (HeapEntry *)realloc(
            heap->data, sizeof(HeapEntry) * new_capacity
        );
        if (!new_data) return -1;

        heap->data = new_data;
        heap->capacity = new_capacity;
    }

    int i = heap->size++;
    heap->data[i] = entry;

    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!heap_less(&heap->data[i], &heap->data[parent])) break;

        HeapEntry temp = heap->data[i];
        heap->data[i] = heap->data[parent];
        heap->data[parent] = temp;
        i = parent;
    }

    return 0;
}

static HeapEntry heap_pop(MinHeap *heap) {
    HeapEntry top = heap->data[0];
    heap->size--;

    if (heap->size > 0) {
        heap->data[0] = heap->data[heap->size];
        int i = 0;

        for (;;) {
            int left = 2 * i + 1;
            int right = left + 1;
            int smallest = i;

            if (
                left < heap->size
                && heap_less(&heap->data[left], &heap->data[smallest])
            ) {
                smallest = left;
            }

            if (
                right < heap->size
                && heap_less(&heap->data[right], &heap->data[smallest])
            ) {
                smallest = right;
            }

            if (smallest == i) break;

            HeapEntry temp = heap->data[i];
            heap->data[i] = heap->data[smallest];
            heap->data[smallest] = temp;
            i = smallest;
        }
    }

    return top;
}

/* ------------------------------------------------------------------ */
/* Propagation stack                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    int *data;
    int size;
    int capacity;
} IntStack;

static int stack_init(IntStack *stack, int capacity) {
    if (capacity < 64) capacity = 64;

    stack->data = (int *)malloc(sizeof(int) * capacity);
    if (!stack->data) return -1;

    stack->size = 0;
    stack->capacity = capacity;
    return 0;
}

static void stack_free(IntStack *stack) {
    free(stack->data);
    stack->data = NULL;
    stack->size = 0;
    stack->capacity = 0;
}

static int stack_push(IntStack *stack, int value) {
    if (stack->size >= stack->capacity) {
        int new_capacity = stack->capacity * 2;
        int *new_data = (int *)realloc(stack->data, sizeof(int) * new_capacity);
        if (!new_data) return -1;

        stack->data = new_data;
        stack->capacity = new_capacity;
    }

    stack->data[stack->size++] = value;
    return 0;
}

static inline int stack_pop(IntStack *stack) {
    return stack->data[--stack->size];
}

/* ------------------------------------------------------------------ */
/* WFC solver internals                                                */
/* ------------------------------------------------------------------ */

/*
 * Solver state is aggregated here so helper functions can stay simple and
 * operate on one pointer (instead of passing many independent buffers).
 */
typedef struct {
    int width;
    int height;
    int size;
    int num_patterns;

    const uint8_t *propagation_masks;
    const double *pattern_weights;
    uint8_t *wave;

    WfcRng rng;
    MinHeap heap;
    IntStack stack;
    uint8_t *in_stack;
    uint64_t heap_counter;
} WfcSolver;

static inline int wave_index(const WfcSolver *solver, int x, int y) {
    /* Flat layout matches the Python implementation: wave[x, y]. */
    return x * solver->height + y;
}

static double calculate_entropy(WfcSolver *solver, int idx) {
    uint8_t mask = solver->wave[idx];
    uint8_t count = popcount_u8(mask);

    if (count <= 1) return 0.0;

    double total_weight = 0.0;
    for (int bit = 0; bit < solver->num_patterns; bit++) {
        if (mask & (1U << bit)) {
            total_weight += solver->pattern_weights[bit];
        }
    }

    if (total_weight == 0.0) return 0.0;

    double entropy = 0.0;
    for (int bit = 0; bit < solver->num_patterns; bit++) {
        if (mask & (1U << bit)) {
            double weight = solver->pattern_weights[bit];
            if (weight > 0.0) {
                double p = weight / total_weight;
                entropy -= p * log(p);
            }
        }
    }

    /*
     * Add tiny deterministic noise to break ties. This mirrors Python behavior
     * where equal-entropy cells are randomly ordered.
     */
    entropy += rng_next_double(&solver->rng) * 0.001;
    return entropy;
}

static int push_entropy(WfcSolver *solver, int idx) {
    HeapEntry entry;
    entry.entropy = calculate_entropy(solver, idx);
    entry.counter = solver->heap_counter++;
    entry.idx = idx;
    return heap_push(&solver->heap, entry);
}

/*
 * Find the next cell to collapse.
 *
 * Returns:
 *   0: success, *out_idx set
 *   1: no candidate found (all collapsed)
 *  -1: out of memory
 *   2: contradiction (empty mask found)
 */
static int find_min_entropy_cell(WfcSolver *solver, int *out_idx) {
    while (solver->heap.size > 0) {
        HeapEntry entry = heap_pop(&solver->heap);
        uint8_t mask = solver->wave[entry.idx];
        uint8_t count = popcount_u8(mask);

        if (count == 0) return 2;
        if (count == 1) continue;

        /*
         * The heap is push-only (we never decrease-key in place), so entries can
         * become stale. Recompute entropy and skip/re-push stale candidates.
         */
        double current_entropy = calculate_entropy(solver, entry.idx);
        if (fabs(current_entropy - entry.entropy) > 0.01) {
            HeapEntry updated;
            updated.entropy = current_entropy;
            updated.counter = solver->heap_counter++;
            updated.idx = entry.idx;
            if (heap_push(&solver->heap, updated) < 0) return -1;
            continue;
        }

        *out_idx = entry.idx;
        return 0;
    }

    return 1;
}

/*
 * Choose a pattern bit from the possibility mask.
 * Returns selected bit index, or -1 on contradiction.
 */
static int weighted_choice(WfcSolver *solver, uint8_t mask) {
    if (mask == 0) return -1;

    int bits[8];
    double weights[8];
    int count = 0;

    for (int bit = 0; bit < solver->num_patterns; bit++) {
        if (mask & (1U << bit)) {
            bits[count] = bit;
            weights[count] = solver->pattern_weights[bit];
            count++;
        }
    }

    if (count == 0) return -1;

    double total = 0.0;
    for (int i = 0; i < count; i++) {
        total += weights[i];
    }

    /*
     * If all remaining weights are zero, fall back to uniform random choice.
     * This mirrors Python's fallback behavior.
     */
    if (total == 0.0) {
        int pick = (int)(rng_next_double(&solver->rng) * (double)count);
        if (pick >= count) pick = count - 1;
        return bits[pick];
    }

    double r = rng_next_double(&solver->rng) * total;
    double cumulative = 0.0;

    for (int i = 0; i < count; i++) {
        cumulative += weights[i];
        if (r <= cumulative) return bits[i];
    }

    return bits[count - 1];
}

/*
 * Propagate constraints from a starting cell.
 *
 * Returns:
 *   0: success
 *  -1: out of memory
 *   1: contradiction
 */
static int propagate(
    WfcSolver *solver,
    int start_idx,
    int *uncollapsed_cells
) {
    static const int DIR_DX[4] = {0, 1, 0, -1};
    static const int DIR_DY[4] = {-1, 0, 1, 0};

    if (stack_push(&solver->stack, start_idx) < 0) return -1;
    /* in_stack avoids duplicate entries and keeps propagation bounded. */
    solver->in_stack[start_idx] = 1;

    int iterations = 0;
    int max_iterations = solver->size * 10;

    while (solver->stack.size > 0) {
        iterations++;
        if (iterations >= max_iterations) return 1;

        int idx = stack_pop(&solver->stack);
        solver->in_stack[idx] = 0;

        int x = idx / solver->height;
        int y = idx % solver->height;
        uint8_t current_mask = solver->wave[idx];

        for (int dir = 0; dir < 4; dir++) {
            int nx = x + DIR_DX[dir];
            int ny = y + DIR_DY[dir];

            if (
                nx < 0
                || nx >= solver->width
                || ny < 0
                || ny >= solver->height
            ) {
                continue;
            }

            int nidx = wave_index(solver, nx, ny);
            uint8_t neighbor_mask = solver->wave[nidx];

            if (popcount_u8(neighbor_mask) <= 1) {
                continue;
            }

            const uint8_t *direction_table =
                solver->propagation_masks + (dir * 256);
            uint8_t valid_for_neighbor = direction_table[current_mask];
            uint8_t new_mask = neighbor_mask & valid_for_neighbor;

            if (new_mask != neighbor_mask) {
                if (new_mask == 0) return 1;

                solver->wave[nidx] = new_mask;

                /*
                 * Only uncertain cells need heap updates. Collapsed cells are
                 * terminal and can be skipped by entropy selection.
                 */
                uint8_t new_count = popcount_u8(new_mask);
                if (new_count > 1) {
                    if (push_entropy(solver, nidx) < 0) return -1;
                } else {
                    (*uncollapsed_cells)--;
                }

                if (!solver->in_stack[nidx]) {
                    if (stack_push(&solver->stack, nidx) < 0) return -1;
                    solver->in_stack[nidx] = 1;
                }
            }
        }
    }

    return 0;
}

/*
 * Run WFC solve loop in-place on solver->wave.
 *
 * Returns:
 *   0: success
 *  -1: out of memory
 *   1: contradiction
 */
static int wfc_solve_inner(WfcSolver *solver) {
    int uncollapsed_cells = 0;

    /*
     * Single initialization scan:
     * - detect contradictions early (empty masks),
     * - track how many cells still need collapsing,
     * - seed heap only with uncertain cells.
     */
    for (int idx = 0; idx < solver->size; idx++) {
        uint8_t count = popcount_u8(solver->wave[idx]);
        if (count == 0) return 1;
        if (count > 1) {
            if (push_entropy(solver, idx) < 0) return -1;
            uncollapsed_cells++;
        }
    }

    int iterations = 0;
    int max_iterations = solver->size * 2;

    while (uncollapsed_cells > 0) {
        iterations++;
        if (iterations >= max_iterations) return 1;

        int cell_idx;
        int rc = find_min_entropy_cell(solver, &cell_idx);
        if (rc == -1) return -1;
        if (rc == 2) return 1;
        if (rc == 1) break;

        uint8_t mask = solver->wave[cell_idx];
        int chosen_bit = weighted_choice(solver, mask);
        if (chosen_bit < 0) return 1;

        solver->wave[cell_idx] = (uint8_t)(1U << chosen_bit);
        uncollapsed_cells--;

        /* Collapse one cell, then propagate constraints outward. */
        rc = propagate(solver, cell_idx, &uncollapsed_cells);
        if (rc != 0) return rc;
    }

    for (int i = 0; i < solver->size; i++) {
        if (popcount_u8(solver->wave[i]) != 1) {
            return 1;
        }
    }

    return 0;
}

static int single_bit_index(uint8_t mask, int num_patterns) {
    for (int bit = 0; bit < num_patterns; bit++) {
        if (mask & (1U << bit)) {
            return bit;
        }
    }
    return -1;
}

static void set_wfc_contradiction_error(const char *message) {
    if (catley_native_wfc_contradiction_error) {
        PyErr_SetString(catley_native_wfc_contradiction_error, message);
    } else {
        PyErr_SetString(PyExc_RuntimeError, message);
    }
}

/* ------------------------------------------------------------------ */
/* Python interface                                                    */
/* ------------------------------------------------------------------ */

/*
 * wfc_solve(
 *     width,
 *     height,
 *     num_patterns,
 *     propagation_masks,
 *     pattern_weights,
 *     initial_wave,
 *     seed,
 * ) -> list[list[int]]
 */
PyObject *catley_native_wfc_solve(PyObject *self, PyObject *args) {
    int width;
    int height;
    int num_patterns;
    PyObject *propagation_obj;
    PyObject *weights_obj;
    PyObject *wave_obj;
    unsigned long long seed;

    Py_buffer propagation_buf = {0};
    Py_buffer weights_buf = {0};
    Py_buffer wave_buf = {0};

    uint8_t *wave_copy = NULL;
    uint8_t *in_stack = NULL;

    MinHeap heap = {0};
    IntStack stack = {0};

    PyObject *result = NULL;

    if (!PyArg_ParseTuple(
            args,
            "iiiOOOK",
            &width,
            &height,
            &num_patterns,
            &propagation_obj,
            &weights_obj,
            &wave_obj,
            &seed
        )) {
        return NULL;
    }

    if (width <= 0 || height <= 0) {
        PyErr_SetString(PyExc_ValueError, "width and height must be positive");
        return NULL;
    }

    if (num_patterns <= 0 || num_patterns > 8) {
        PyErr_SetString(
            PyExc_ValueError,
            "num_patterns must be in range [1, 8]"
        );
        return NULL;
    }

    if (PyObject_GetBuffer(
            propagation_obj,
            &propagation_buf,
            PyBUF_C_CONTIGUOUS | PyBUF_FORMAT
        ) < 0) {
        goto cleanup;
    }

    int propagation_ok =
        propagation_buf.ndim == 2
        && propagation_buf.itemsize == 1
        && (strcmp(propagation_buf.format, "B") == 0
            || strcmp(propagation_buf.format, "b") == 0)
        && propagation_buf.shape[0] == 4
        && propagation_buf.shape[1] == 256;

    if (!propagation_ok) {
        PyErr_SetString(
            PyExc_TypeError,
            "propagation_masks must be a 2D uint8 C-contiguous array with shape (4, 256)"
        );
        goto cleanup;
    }

    if (PyObject_GetBuffer(
            weights_obj,
            &weights_buf,
            PyBUF_C_CONTIGUOUS | PyBUF_FORMAT
        ) < 0) {
        goto cleanup;
    }

    int weights_ok =
        weights_buf.ndim == 1
        && strcmp(weights_buf.format, "d") == 0
        && weights_buf.shape[0] == num_patterns;

    if (!weights_ok) {
        PyErr_SetString(
            PyExc_TypeError,
            "pattern_weights must be a 1D float64 C-contiguous array with length num_patterns"
        );
        goto cleanup;
    }

    if (PyObject_GetBuffer(
            wave_obj,
            &wave_buf,
            PyBUF_C_CONTIGUOUS | PyBUF_FORMAT
        ) < 0) {
        goto cleanup;
    }

    int wave_ok =
        wave_buf.ndim == 2
        && wave_buf.itemsize == 1
        && (strcmp(wave_buf.format, "B") == 0 || strcmp(wave_buf.format, "b") == 0)
        && wave_buf.shape[0] == width
        && wave_buf.shape[1] == height;

    if (!wave_ok) {
        PyErr_SetString(
            PyExc_TypeError,
            "initial_wave must be a 2D uint8 C-contiguous array with shape (width, height)"
        );
        goto cleanup;
    }

    int size = width * height;
    wave_copy = (uint8_t *)malloc((size_t)size);
    if (!wave_copy) {
        PyErr_NoMemory();
        goto cleanup;
    }

    /*
     * Solve mutates wave in place. Copy the input buffer so callers keep their
     * original array untouched.
     */
    memcpy(wave_copy, wave_buf.buf, (size_t)size);

    uint8_t all_patterns_mask = (uint8_t)((1U << num_patterns) - 1U);
    for (int i = 0; i < size; i++) {
        if ((wave_copy[i] & (uint8_t)(~all_patterns_mask)) != 0) {
            PyErr_SetString(
                PyExc_ValueError,
                "initial_wave contains bits outside num_patterns"
            );
            goto cleanup;
        }
    }

    if (heap_init(&heap, size) < 0) {
        PyErr_NoMemory();
        goto cleanup;
    }

    if (stack_init(&stack, size) < 0) {
        PyErr_NoMemory();
        goto cleanup;
    }

    in_stack = (uint8_t *)calloc((size_t)size, 1);
    if (!in_stack) {
        PyErr_NoMemory();
        goto cleanup;
    }

    ensure_popcount_table();

    WfcSolver solver;
    solver.width = width;
    solver.height = height;
    solver.size = size;
    solver.num_patterns = num_patterns;
    solver.propagation_masks = (const uint8_t *)propagation_buf.buf;
    solver.pattern_weights = (const double *)weights_buf.buf;
    solver.wave = wave_copy;
    solver.heap = heap;
    solver.stack = stack;
    solver.in_stack = in_stack;
    solver.heap_counter = 0;
    rng_init(&solver.rng, (uint64_t)seed);

    int rc = wfc_solve_inner(&solver);

    heap = solver.heap;
    stack = solver.stack;

    if (rc < 0) {
        PyErr_NoMemory();
        goto cleanup;
    }

    if (rc > 0) {
        set_wfc_contradiction_error("WFC contradiction");
        goto cleanup;
    }

    result = PyList_New(width);
    if (!result) goto cleanup;

    for (int x = 0; x < width; x++) {
        PyObject *column = PyList_New(height);
        if (!column) {
            Py_DECREF(result);
            result = NULL;
            goto cleanup;
        }

        for (int y = 0; y < height; y++) {
            uint8_t mask = wave_copy[wave_index(&solver, x, y)];
            int bit_idx = single_bit_index(mask, num_patterns);
            if (bit_idx < 0 || popcount_u8(mask) != 1) {
                Py_DECREF(column);
                Py_DECREF(result);
                result = NULL;
                PyErr_SetString(PyExc_SystemError, "WFC result is not fully collapsed");
                goto cleanup;
            }

            PyObject *value = PyLong_FromLong(bit_idx);
            if (!value) {
                Py_DECREF(column);
                Py_DECREF(result);
                result = NULL;
                goto cleanup;
            }

            PyList_SET_ITEM(column, y, value);
        }

        PyList_SET_ITEM(result, x, column);
    }

cleanup:
    PyBuffer_Release(&propagation_buf);
    PyBuffer_Release(&weights_buf);
    PyBuffer_Release(&wave_buf);

    heap_free(&heap);
    stack_free(&stack);

    free(in_stack);
    free(wave_copy);

    return result;
}
