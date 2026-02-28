/*
 * Native SpatialHashGrid implementation for brileta.
 *
 * A C extension type that replaces the pure-Python SpatialHashGrid with a
 * fast hash-table-based spatial index.  Objects must have integer .x and .y
 * attributes (read via PyObject_GetAttrString).
 *
 * Internal data structures:
 *   - cell_table:  hash map from (cx, cy) cell coords -> linked list of entries
 *   - obj_table:   hash map from PyObject* id -> ObjEntry (tracks current cell)
 *
 * Both tables use open addressing with linear probing.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Interned attribute name strings (initialized once at module load)   */
/* ------------------------------------------------------------------ */

static PyObject *attr_x = NULL;
static PyObject *attr_y = NULL;
static PyObject *attr_x1 = NULL;
static PyObject *attr_y1 = NULL;
static PyObject *attr_x2 = NULL;
static PyObject *attr_y2 = NULL;

/* ------------------------------------------------------------------ */
/* Cell entry: a single object stored in a cell's linked list.        */
/* ------------------------------------------------------------------ */

typedef struct CellEntry {
    PyObject *obj; /* borrowed reference -- the obj_table owns the ref */
    struct CellEntry *next;
} CellEntry;

/* ------------------------------------------------------------------ */
/* Cell table: maps (cx, cy) -> linked list of CellEntry.             */
/* Open addressing, linear probing, tombstone-aware.                  */
/* ------------------------------------------------------------------ */

typedef struct {
    int cx, cy;      /* cell coordinates */
    CellEntry *head; /* linked list of objects in this cell */
    int occupied;    /* 1 = live, 0 = empty, -1 = tombstone */
} CellSlot;

typedef struct {
    CellSlot *slots;
    Py_ssize_t capacity;
    Py_ssize_t count;      /* number of live entries */
    Py_ssize_t tombstones; /* number of tombstoned slots */
} CellTable;

/* ------------------------------------------------------------------ */
/* Object table: maps PyObject* -> (cx, cy) cell it belongs to.       */
/* Keyed by object pointer identity (like Python id()).                */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject *obj; /* strong reference */
    int cx, cy;    /* cell coords this object currently lives in */
    int occupied;  /* 1 = live, 0 = empty, -1 = tombstone */
} ObjSlot;

typedef struct {
    ObjSlot *slots;
    Py_ssize_t capacity;
    Py_ssize_t count;      /* number of live entries */
    Py_ssize_t tombstones; /* number of tombstoned slots */
} ObjTable;

/* ------------------------------------------------------------------ */
/* Hash helpers                                                       */
/* ------------------------------------------------------------------ */

/* Hash two ints into a size_t for cell lookup. */
static inline size_t hash_cell(int cx, int cy) {
    /* Mix both coordinates with bit rotation to reduce collisions. */
    size_t h = (size_t)(unsigned int)cx;
    h = (h ^ (h >> 16)) * 0x45d9f3bU;
    h ^= (size_t)(unsigned int)cy * 0x9e3779b9U;
    h = (h ^ (h >> 13)) * 0xc2b2ae35U;
    return h ^ (h >> 16);
}

/* Hash an object pointer for the obj table. */
static inline size_t hash_ptr(PyObject *obj) {
    size_t h = (size_t)obj;
    h = (h >> 4) | (h << (sizeof(size_t) * 8 - 4)); /* shift away low alignment bits */
    h ^= h >> 16;
    h *= 0x45d9f3bU;
    return h ^ (h >> 13);
}

/* ------------------------------------------------------------------ */
/* Cell table operations                                              */
/* ------------------------------------------------------------------ */

static int cell_table_init(CellTable *t, Py_ssize_t capacity) {
    t->slots = (CellSlot *)calloc((size_t)capacity, sizeof(CellSlot));
    if (!t->slots)
        return -1;
    t->capacity = capacity;
    t->count = 0;
    t->tombstones = 0;
    return 0;
}

static void cell_table_free_entries(CellTable *t) {
    for (Py_ssize_t i = 0; i < t->capacity; i++) {
        CellEntry *e = t->slots[i].head;
        while (e) {
            CellEntry *next = e->next;
            free(e);
            e = next;
        }
    }
}

static void cell_table_destroy(CellTable *t) {
    if (t->slots) {
        cell_table_free_entries(t);
        free(t->slots);
        t->slots = NULL;
    }
    t->capacity = 0;
    t->count = 0;
    t->tombstones = 0;
}

/* Find a slot for (cx, cy). Returns the slot index. If found, the slot
 * has occupied==1. If not found, returns the first empty/tombstone slot. */
static Py_ssize_t cell_table_find(CellTable *t, int cx, int cy) {
    size_t h = hash_cell(cx, cy);
    Py_ssize_t mask = t->capacity - 1;
    Py_ssize_t idx = (Py_ssize_t)(h & (size_t)mask);
    Py_ssize_t first_tombstone = -1;

    for (Py_ssize_t i = 0; i < t->capacity; i++) {
        Py_ssize_t slot = (idx + i) & mask;
        CellSlot *s = &t->slots[slot];
        if (s->occupied == 0) {
            /* Empty slot - key not present. */
            return (first_tombstone >= 0) ? first_tombstone : slot;
        }
        if (s->occupied == -1) {
            /* Tombstone - remember it but keep searching. */
            if (first_tombstone < 0)
                first_tombstone = slot;
            continue;
        }
        if (s->cx == cx && s->cy == cy)
            return slot; /* Found. */
    }
    /* Table is full of tombstones + live entries - should not happen
     * if load factor is maintained. */
    return (first_tombstone >= 0) ? first_tombstone : 0;
}

static int cell_table_resize(CellTable *t, Py_ssize_t new_cap);

/* Get or create the cell at (cx, cy). Returns NULL on OOM. */
static CellSlot *cell_table_get_or_create(CellTable *t, int cx, int cy) {
    /* Rehash when combined load (live + tombstones) exceeds 70%.
     * If most of the load is tombstones, rehash to same capacity
     * (just cleans up); otherwise double. */
    if ((t->count + t->tombstones) * 10 > t->capacity * 7) {
        Py_ssize_t new_cap = (t->count * 5 > t->capacity * 3) ? t->capacity * 2 : t->capacity;
        if (cell_table_resize(t, new_cap) < 0)
            return NULL;
    }

    Py_ssize_t idx = cell_table_find(t, cx, cy);
    CellSlot *s = &t->slots[idx];
    if (s->occupied == 1)
        return s;

    /* New cell - slot may be empty or a tombstone. */
    if (s->occupied == -1)
        t->tombstones--;
    s->cx = cx;
    s->cy = cy;
    s->head = NULL;
    s->occupied = 1;
    t->count++;
    return s;
}

/* Look up the cell at (cx, cy). Returns NULL if not found. */
static CellSlot *cell_table_lookup(CellTable *t, int cx, int cy) {
    Py_ssize_t idx = cell_table_find(t, cx, cy);
    CellSlot *s = &t->slots[idx];
    return (s->occupied == 1 && s->cx == cx && s->cy == cy) ? s : NULL;
}

static int cell_table_resize(CellTable *t, Py_ssize_t new_cap) {
    CellSlot *old_slots = t->slots;
    Py_ssize_t old_cap = t->capacity;

    t->slots = (CellSlot *)calloc((size_t)new_cap, sizeof(CellSlot));
    if (!t->slots) {
        t->slots = old_slots;
        return -1;
    }
    t->capacity = new_cap;
    t->count = 0;
    t->tombstones = 0; /* fresh table has no tombstones */

    for (Py_ssize_t i = 0; i < old_cap; i++) {
        if (old_slots[i].occupied == 1) {
            Py_ssize_t idx = cell_table_find(t, old_slots[i].cx, old_slots[i].cy);
            t->slots[idx] = old_slots[i];
            t->slots[idx].occupied = 1;
            t->count++;
        } else {
            /* Free any orphaned entry lists from tombstoned slots. */
            CellEntry *e = old_slots[i].head;
            while (e) {
                CellEntry *next = e->next;
                free(e);
                e = next;
            }
        }
    }

    free(old_slots);
    return 0;
}

/* Remove an object from a cell's linked list. Returns 1 if the cell
 * is now empty, 0 otherwise. */
static int cell_remove_obj(CellSlot *cell, PyObject *obj) {
    CellEntry **pp = &cell->head;
    while (*pp) {
        if ((*pp)->obj == obj) {
            CellEntry *victim = *pp;
            *pp = victim->next;
            free(victim);
            return cell->head == NULL;
        }
        pp = &(*pp)->next;
    }
    return cell->head == NULL;
}

/* Mark a cell slot as a tombstone (caller has already freed entries). */
static void cell_table_remove_slot(CellTable *t, CellSlot *s) {
    s->occupied = -1;
    s->head = NULL;
    t->count--;
    t->tombstones++;
}

/* ------------------------------------------------------------------ */
/* Object table operations                                            */
/* ------------------------------------------------------------------ */

static int obj_table_init(ObjTable *t, Py_ssize_t capacity) {
    t->slots = (ObjSlot *)calloc((size_t)capacity, sizeof(ObjSlot));
    if (!t->slots)
        return -1;
    t->capacity = capacity;
    t->count = 0;
    t->tombstones = 0;
    return 0;
}

static void obj_table_destroy(ObjTable *t) {
    /* Save and zero the table BEFORE decrefing objects, so that any
     * reentrant access (e.g. from an object's __del__) sees an empty
     * table rather than a half-destroyed one. */
    ObjSlot *slots = t->slots;
    Py_ssize_t cap = t->capacity;
    t->slots = NULL;
    t->capacity = 0;
    t->count = 0;
    t->tombstones = 0;

    if (slots) {
        for (Py_ssize_t i = 0; i < cap; i++) {
            if (slots[i].occupied == 1)
                Py_DECREF(slots[i].obj);
        }
        free(slots);
    }
}

static Py_ssize_t obj_table_find(ObjTable *t, PyObject *obj) {
    size_t h = hash_ptr(obj);
    Py_ssize_t mask = t->capacity - 1;
    Py_ssize_t idx = (Py_ssize_t)(h & (size_t)mask);
    Py_ssize_t first_tombstone = -1;

    for (Py_ssize_t i = 0; i < t->capacity; i++) {
        Py_ssize_t slot = (idx + i) & mask;
        ObjSlot *s = &t->slots[slot];
        if (s->occupied == 0)
            return (first_tombstone >= 0) ? first_tombstone : slot;
        if (s->occupied == -1) {
            if (first_tombstone < 0)
                first_tombstone = slot;
            continue;
        }
        if (s->obj == obj)
            return slot;
    }
    return (first_tombstone >= 0) ? first_tombstone : 0;
}

static int obj_table_resize(ObjTable *t, Py_ssize_t new_cap) {
    ObjSlot *old_slots = t->slots;
    Py_ssize_t old_cap = t->capacity;

    t->slots = (ObjSlot *)calloc((size_t)new_cap, sizeof(ObjSlot));
    if (!t->slots) {
        t->slots = old_slots;
        return -1;
    }
    t->capacity = new_cap;
    t->count = 0;
    t->tombstones = 0; /* fresh table has no tombstones */

    for (Py_ssize_t i = 0; i < old_cap; i++) {
        if (old_slots[i].occupied == 1) {
            Py_ssize_t idx = obj_table_find(t, old_slots[i].obj);
            t->slots[idx] = old_slots[i];
            t->slots[idx].occupied = 1;
            t->count++;
        }
    }

    free(old_slots);
    return 0;
}

/* Look up obj. Returns pointer to slot if found (occupied==1), else NULL. */
static ObjSlot *obj_table_lookup(ObjTable *t, PyObject *obj) {
    Py_ssize_t idx = obj_table_find(t, obj);
    ObjSlot *s = &t->slots[idx];
    return (s->occupied == 1 && s->obj == obj) ? s : NULL;
}

/* Insert obj -> (cx, cy). Takes a new reference to obj. Returns 0 on
 * success, -1 on OOM. */
static int obj_table_insert(ObjTable *t, PyObject *obj, int cx, int cy) {
    /* Rehash when combined load (live + tombstones) exceeds 70%. */
    if ((t->count + t->tombstones) * 10 > t->capacity * 7) {
        Py_ssize_t new_cap = (t->count * 5 > t->capacity * 3) ? t->capacity * 2 : t->capacity;
        if (obj_table_resize(t, new_cap) < 0)
            return -1;
    }

    Py_ssize_t idx = obj_table_find(t, obj);
    ObjSlot *s = &t->slots[idx];
    if (s->occupied == 1) {
        /* Already present - just update cell coords. */
        s->cx = cx;
        s->cy = cy;
        return 0;
    }

    /* Slot may be empty or a tombstone. */
    if (s->occupied == -1)
        t->tombstones--;
    Py_INCREF(obj);
    s->obj = obj;
    s->cx = cx;
    s->cy = cy;
    s->occupied = 1;
    t->count++;
    return 0;
}

/* Remove obj from the table. Releases the reference. */
static void obj_table_remove(ObjTable *t, PyObject *obj) {
    Py_ssize_t idx = obj_table_find(t, obj);
    ObjSlot *s = &t->slots[idx];
    if (s->occupied == 1 && s->obj == obj) {
        Py_DECREF(s->obj);
        s->obj = NULL;
        s->occupied = -1;
        t->count--;
        t->tombstones++;
    }
}

/* ------------------------------------------------------------------ */
/* Attribute reading helpers (return 0 on success, -1 on error).      */
/* ------------------------------------------------------------------ */

/* Read a single integer attribute from a Python object. */
static int read_int_attr(PyObject *obj, PyObject *attr_name, int *out) {
    PyObject *v = PyObject_GetAttr(obj, attr_name);
    if (!v)
        return -1;
    long lv = PyLong_AsLong(v);
    Py_DECREF(v);
    if (lv == -1 && PyErr_Occurred())
        return -1;
    *out = (int)lv;
    return 0;
}

/* Read .x and .y from an object. */
static int read_obj_xy(PyObject *obj, int *out_x, int *out_y) {
    if (read_int_attr(obj, attr_x, out_x) < 0)
        return -1;
    if (read_int_attr(obj, attr_y, out_y) < 0)
        return -1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Floor division that rounds towards negative infinity (like Python). */
/* ------------------------------------------------------------------ */

static inline int floor_div(int a, int b) {
    int q = a / b;
    int r = a % b;
    /* Adjust if remainder is nonzero and signs differ. */
    if (r != 0 && ((r ^ b) < 0))
        q -= 1;
    return q;
}

/* ------------------------------------------------------------------ */
/* SpatialHashGrid Python type                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    PyObject_HEAD int cell_size;
    CellTable cells;
    ObjTable objs;
} SpatialHashGridObject;

/* ---- __init__ ---------------------------------------------------- */

static int SpatialHashGrid_init(SpatialHashGridObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"cell_size", NULL};
    int cell_size = 16;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &cell_size))
        return -1;

    if (cell_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "Cell size must be a positive integer.");
        return -1;
    }

    self->cell_size = cell_size;

    if (cell_table_init(&self->cells, 64) < 0) {
        PyErr_NoMemory();
        return -1;
    }

    if (obj_table_init(&self->objs, 64) < 0) {
        cell_table_destroy(&self->cells);
        PyErr_NoMemory();
        return -1;
    }

    return 0;
}

/* ---- dealloc / GC support ---------------------------------------- */

static void SpatialHashGrid_dealloc(SpatialHashGridObject *self) {
    PyObject_GC_UnTrack(self);
    cell_table_destroy(&self->cells);
    obj_table_destroy(&self->objs);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Visit all Python objects held in the obj table so the cyclic GC can
 * discover reference cycles involving the grid. */
static int SpatialHashGrid_traverse(SpatialHashGridObject *self, visitproc visit, void *arg) {
    for (Py_ssize_t i = 0; i < self->objs.capacity; i++) {
        if (self->objs.slots[i].occupied == 1)
            Py_VISIT(self->objs.slots[i].obj);
    }
    return 0;
}

/* Break reference cycles by releasing all held Python objects. */
static int SpatialHashGrid_tp_clear(SpatialHashGridObject *self) {
    cell_table_destroy(&self->cells);
    obj_table_destroy(&self->objs);
    return 0;
}

/* ---- helpers ----------------------------------------------------- */

/* Map world coords to cell coords. */
static inline void grid_hash(int cell_size, int wx, int wy, int *cx, int *cy) {
    *cx = floor_div(wx, cell_size);
    *cy = floor_div(wy, cell_size);
}

/* Add an object to a cell's linked list and to the obj table.
 * Returns 0 on success, -1 on error. */
static int grid_add_to_cell(SpatialHashGridObject *self, PyObject *obj, int cx, int cy) {
    CellSlot *cell = cell_table_get_or_create(&self->cells, cx, cy);
    if (!cell) {
        PyErr_NoMemory();
        return -1;
    }

    CellEntry *entry = (CellEntry *)malloc(sizeof(CellEntry));
    if (!entry) {
        PyErr_NoMemory();
        return -1;
    }
    entry->obj = obj; /* borrowed ref - obj table owns the strong ref */
    entry->next = cell->head;
    cell->head = entry;

    if (obj_table_insert(&self->objs, obj, cx, cy) < 0) {
        /* Roll back the cell entry. */
        cell->head = entry->next;
        free(entry);
        if (cell->head == NULL) {
            cell_table_remove_slot(&self->cells, cell);
        }
        PyErr_NoMemory();
        return -1;
    }

    return 0;
}

/* Remove an object from its current cell (but not from the obj table). */
static void grid_remove_from_cell(SpatialHashGridObject *self, PyObject *obj, int cx, int cy) {
    CellSlot *cell = cell_table_lookup(&self->cells, cx, cy);
    if (!cell)
        return;
    int now_empty = cell_remove_obj(cell, obj);
    if (now_empty) {
        cell_table_remove_slot(&self->cells, cell);
    }
}

/* ---- add --------------------------------------------------------- */

static PyObject *SpatialHashGrid_add(SpatialHashGridObject *self, PyObject *obj) {
    /* If already tracked, remove first to prevent duplicate cell entries. */
    ObjSlot *existing = obj_table_lookup(&self->objs, obj);
    if (existing) {
        grid_remove_from_cell(self, obj, existing->cx, existing->cy);
        obj_table_remove(&self->objs, obj);
    }

    int wx, wy;
    if (read_obj_xy(obj, &wx, &wy) < 0)
        return NULL;

    int cx, cy;
    grid_hash(self->cell_size, wx, wy, &cx, &cy);

    if (grid_add_to_cell(self, obj, cx, cy) < 0)
        return NULL;

    Py_RETURN_NONE;
}

/* ---- remove ------------------------------------------------------ */

static PyObject *SpatialHashGrid_remove(SpatialHashGridObject *self, PyObject *obj) {
    ObjSlot *os = obj_table_lookup(&self->objs, obj);
    if (!os)
        Py_RETURN_NONE; /* Not in grid - silent no-op. */

    grid_remove_from_cell(self, obj, os->cx, os->cy);
    obj_table_remove(&self->objs, obj);
    Py_RETURN_NONE;
}

/* ---- update ------------------------------------------------------ */

static PyObject *SpatialHashGrid_update(SpatialHashGridObject *self, PyObject *obj) {
    int wx, wy;
    if (read_obj_xy(obj, &wx, &wy) < 0)
        return NULL;

    int new_cx, new_cy;
    grid_hash(self->cell_size, wx, wy, &new_cx, &new_cy);

    ObjSlot *os = obj_table_lookup(&self->objs, obj);
    if (!os) {
        /* Not tracked yet - treat as fresh add. */
        if (grid_add_to_cell(self, obj, new_cx, new_cy) < 0)
            return NULL;
        Py_RETURN_NONE;
    }

    if (os->cx == new_cx && os->cy == new_cy)
        Py_RETURN_NONE; /* Same cell - nothing to do. */

    /* Prepare the new cell and entry BEFORE modifying old state, so
     * failure leaves the grid consistent. */
    CellSlot *new_cell = cell_table_get_or_create(&self->cells, new_cx, new_cy);
    if (!new_cell) {
        PyErr_NoMemory();
        return NULL;
    }

    CellEntry *entry = (CellEntry *)malloc(sizeof(CellEntry));
    if (!entry) {
        /* Clean up the empty cell we may have just created. */
        if (new_cell->head == NULL)
            cell_table_remove_slot(&self->cells, new_cell);
        PyErr_NoMemory();
        return NULL;
    }

    /* All allocations succeeded - now update state (cannot fail). */
    grid_remove_from_cell(self, obj, os->cx, os->cy);

    entry->obj = obj;
    entry->next = new_cell->head;
    new_cell->head = entry;

    /* os is still valid: grid_remove_from_cell only touches the cell
     * table, and cell_table_get_or_create cannot resize the obj table. */
    os->cx = new_cx;
    os->cy = new_cy;

    Py_RETURN_NONE;
}

/* ---- get_at_point ------------------------------------------------ */

static PyObject *SpatialHashGrid_get_at_point(SpatialHashGridObject *self, PyObject *args) {
    int wx, wy;
    if (!PyArg_ParseTuple(args, "ii", &wx, &wy))
        return NULL;

    int cx, cy;
    grid_hash(self->cell_size, wx, wy, &cx, &cy);

    PyObject *result = PyList_New(0);
    if (!result)
        return NULL;

    CellSlot *cell = cell_table_lookup(&self->cells, cx, cy);
    if (!cell)
        return result;

    for (CellEntry *e = cell->head; e; e = e->next) {
        int ox, oy;
        if (read_obj_xy(e->obj, &ox, &oy) < 0) {
            Py_DECREF(result);
            return NULL;
        }
        if (ox == wx && oy == wy) {
            if (PyList_Append(result, e->obj) < 0) {
                Py_DECREF(result);
                return NULL;
            }
        }
    }

    return result;
}

/* ---- Internal bounds query (shared by get_in_bounds / get_in_radius) */

static PyObject *query_bounds(SpatialHashGridObject *self, int x1, int y1, int x2, int y2) {
    int cx1, cy1, cx2, cy2;
    grid_hash(self->cell_size, x1, y1, &cx1, &cy1);
    grid_hash(self->cell_size, x2, y2, &cx2, &cy2);

    PyObject *result = PyList_New(0);
    if (!result)
        return NULL;

    for (int cx = cx1; cx <= cx2; cx++) {
        for (int cy = cy1; cy <= cy2; cy++) {
            CellSlot *cell = cell_table_lookup(&self->cells, cx, cy);
            if (!cell)
                continue;
            for (CellEntry *e = cell->head; e; e = e->next) {
                int ox, oy;
                if (read_obj_xy(e->obj, &ox, &oy) < 0) {
                    Py_DECREF(result);
                    return NULL;
                }
                if (ox >= x1 && ox <= x2 && oy >= y1 && oy <= y2) {
                    if (PyList_Append(result, e->obj) < 0) {
                        Py_DECREF(result);
                        return NULL;
                    }
                }
            }
        }
    }

    return result;
}

/* ---- get_in_bounds ----------------------------------------------- */

static PyObject *SpatialHashGrid_get_in_bounds(SpatialHashGridObject *self, PyObject *args) {
    int x1, y1, x2, y2;
    if (!PyArg_ParseTuple(args, "iiii", &x1, &y1, &x2, &y2))
        return NULL;

    /* Normalize so x1 <= x2, y1 <= y2 */
    if (x1 > x2) {
        int tmp = x1;
        x1 = x2;
        x2 = tmp;
    }
    if (y1 > y2) {
        int tmp = y1;
        y1 = y2;
        y2 = tmp;
    }

    return query_bounds(self, x1, y1, x2, y2);
}

/* ---- get_in_radius ----------------------------------------------- */

static PyObject *
SpatialHashGrid_get_in_radius(SpatialHashGridObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"x", "y", "radius", NULL};
    int x, y, radius;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iii", kwlist, &x, &y, &radius))
        return NULL;

    return query_bounds(self, x - radius, y - radius, x + radius, y + radius);
}

/* ---- get_in_rect ------------------------------------------------- */

static PyObject *SpatialHashGrid_get_in_rect(SpatialHashGridObject *self, PyObject *arg) {
    /* Read .x1, .y1, .x2, .y2 from a Rect-like object. */
    int x1, y1, x2, y2;
    if (read_int_attr(arg, attr_x1, &x1) < 0 || read_int_attr(arg, attr_y1, &y1) < 0 ||
        read_int_attr(arg, attr_x2, &x2) < 0 || read_int_attr(arg, attr_y2, &y2) < 0)
        return NULL;

    /* Normalize just in case. */
    if (x1 > x2) {
        int tmp = x1;
        x1 = x2;
        x2 = tmp;
    }
    if (y1 > y2) {
        int tmp = y1;
        y1 = y2;
        y2 = tmp;
    }

    return query_bounds(self, x1, y1, x2, y2);
}

/* ---- clear ------------------------------------------------------- */

static PyObject *SpatialHashGrid_clear(SpatialHashGridObject *self, PyObject *Py_UNUSED(ignored)) {
    cell_table_destroy(&self->cells);
    obj_table_destroy(&self->objs);

    /* Re-initialize with small tables. */
    if (cell_table_init(&self->cells, 64) < 0)
        return PyErr_NoMemory();
    if (obj_table_init(&self->objs, 64) < 0) {
        cell_table_destroy(&self->cells);
        return PyErr_NoMemory();
    }

    Py_RETURN_NONE;
}

/* ---- cell_size property (read-only) ------------------------------ */

static PyObject *SpatialHashGrid_get_cell_size(SpatialHashGridObject *self, void *closure) {
    (void)closure;
    return PyLong_FromLong(self->cell_size);
}

/* ---- Method and type definitions --------------------------------- */

static PyMethodDef SpatialHashGrid_methods[] = {
    {"add",
     (PyCFunction)SpatialHashGrid_add,
     METH_O,
     "add(obj) -> None\n\nAdd an object to the grid."},
    {"remove",
     (PyCFunction)SpatialHashGrid_remove,
     METH_O,
     "remove(obj) -> None\n\nRemove an object from the grid."},
    {"update",
     (PyCFunction)SpatialHashGrid_update,
     METH_O,
     "update(obj) -> None\n\nUpdate the position of an object that has moved."},
    {"get_at_point",
     (PyCFunction)SpatialHashGrid_get_at_point,
     METH_VARARGS,
     "get_at_point(x, y) -> list\n\nGet all objects at a specific tile (x, y)."},
    {"get_in_radius",
     (PyCFunction)SpatialHashGrid_get_in_radius,
     METH_VARARGS | METH_KEYWORDS,
     "get_in_radius(x, y, radius) -> list\n\n"
     "Get all objects within a Chebyshev distance (radius) of a point."},
    {"get_in_bounds",
     (PyCFunction)SpatialHashGrid_get_in_bounds,
     METH_VARARGS,
     "get_in_bounds(x1, y1, x2, y2) -> list\n\n"
     "Get all objects within a rectangular bounding box."},
    {"get_in_rect",
     (PyCFunction)SpatialHashGrid_get_in_rect,
     METH_O,
     "get_in_rect(rect) -> list\n\n"
     "Get all objects within a Rect bounds (reads .x1, .y1, .x2, .y2)."},
    {"clear",
     (PyCFunction)SpatialHashGrid_clear,
     METH_NOARGS,
     "clear() -> None\n\nRemove all objects from the index."},
    {NULL, NULL, 0, NULL}};

static PyGetSetDef SpatialHashGrid_getset[] = {{"cell_size",
                                                (getter)SpatialHashGrid_get_cell_size,
                                                NULL,
                                                "The cell size of the grid (read-only).",
                                                NULL},
                                               {NULL, NULL, NULL, NULL, NULL}};

static PyTypeObject SpatialHashGridType = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0).tp_name = "brileta.util._native.SpatialHashGrid",
    .tp_doc = "SpatialHashGrid(cell_size=16)\n\n"
              "A spatial hash grid for efficient 2D spatial queries.\n"
              "Objects must have integer .x and .y attributes.",
    .tp_basicsize = sizeof(SpatialHashGridObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SpatialHashGrid_init,
    .tp_dealloc = (destructor)SpatialHashGrid_dealloc,
    .tp_traverse = (traverseproc)SpatialHashGrid_traverse,
    .tp_clear = (inquiry)SpatialHashGrid_tp_clear,
    .tp_methods = SpatialHashGrid_methods,
    .tp_getset = SpatialHashGrid_getset,
};

/* ------------------------------------------------------------------ */
/* Module-level registration (called from _native.c PyInit__native)   */
/* ------------------------------------------------------------------ */

int brileta_native_init_spatial_type(PyObject *module) {
    /* Intern attribute name strings for fast lookups. */
    static struct {
        PyObject **slot;
        const char *name;
    } attrs[] = {
        {&attr_x, "x"},
        {&attr_y, "y"},
        {&attr_x1, "x1"},
        {&attr_y1, "y1"},
        {&attr_x2, "x2"},
        {&attr_y2, "y2"},
    };
    for (size_t i = 0; i < sizeof(attrs) / sizeof(attrs[0]); i++) {
        if (!*attrs[i].slot) {
            *attrs[i].slot = PyUnicode_InternFromString(attrs[i].name);
            if (!*attrs[i].slot)
                return -1;
        }
    }

    if (PyType_Ready(&SpatialHashGridType) < 0)
        return -1;

    if (PyModule_AddObjectRef(module, "SpatialHashGrid", (PyObject *)&SpatialHashGridType) < 0)
        return -1;

    return 0;
}
