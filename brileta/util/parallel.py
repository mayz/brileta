"""Process-based parallel execution for CPU-bound work.

Provides :func:`parallel_map`, a drop-in replacement for
``list(map(fn, ...))`` that distributes work across multiple processes.
Each worker has its own Python interpreter and GIL, so C-extension code
runs with true parallelism across cores.

A persistent process pool is created on first use and reused across calls
to amortize the per-call startup cost of ``spawn``-mode workers on macOS.

Usage::

    from brileta.util.parallel import parallel_map

    # Same semantics as list(map(fn, xs, ys)), but parallel.
    results = parallel_map(fn, xs, ys)
"""

from __future__ import annotations

import atexit
import multiprocessing
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# A PyInstaller build NEVER spawns a worker pool. A spawned worker re-execs the
# app; in a windowed macOS build every child runs the Cocoa bootloader and shows
# up as a phantom Dock/Cmd+Tab icon. Forcing the frozen binary to run serially -
# unconditionally, whatever the map size or however the world is (re)generated,
# including from the dev console - makes those icons structurally impossible: no
# pool is ever created, so no worker process ever exists. World-gen is a ~15ms
# one-shot on the shipped map; even a huge dev-console regen is a few seconds
# serially, which is fine for a developer action in a shipped binary.
_FROZEN = getattr(sys, "frozen", False)

# In the dev environment (make run), workers are plain Python processes that
# never show as icons, so the pool is free to use - but only worth it once the
# work outweighs the ~65ms macOS worker-spawn cost. Below this item count serial
# is faster and avoids spawning workers for trivial startup work. Measured
# break-even is ~1.5-2k items; 5k leaves comfortable margin.
_PARALLEL_MIN_ITEMS = 5000

# Persistent worker pool, lazily created on first parallel_map() call.
# macOS uses 'spawn' by default (Python 3.12+), so each worker must
# import all modules from scratch.  A persistent pool amortizes that
# startup cost across subsequent calls (e.g. map regeneration).
_pool: ProcessPoolExecutor | None = None


def _ensure_pool() -> ProcessPoolExecutor:
    """Return the shared process pool, creating it on first access."""
    global _pool
    if _pool is None:
        n = multiprocessing.cpu_count() or 4
        _pool = ProcessPoolExecutor(max_workers=n)
        atexit.register(shutdown_pool)
    return _pool


def parallel_map[T](
    fn: Callable[..., T],
    *iterables: Iterable[Any],
    chunksize: int = 64,
) -> list[T]:
    """Apply *fn* to argument tuples from *iterables* in parallel.

    Distributes work across a process pool, bypassing the GIL entirely.
    Behaves like ``list(map(fn, *iterables))`` but runs on multiple cores.

    *fn* must be a **module-scope** function so it can be pickled and sent
    to worker processes.  All arguments and return values must also be
    picklable (numpy arrays, ints, enums, etc. are fine).

    Args:
        fn: Callable to apply.  Must be defined at module scope.
        *iterables: Argument iterables, one per *fn* parameter.
        chunksize: Items per IPC batch.  Larger values reduce overhead
            when items are individually cheap.

    Returns:
        Results in input order, identical to ``list(map(fn, ...))``.
    """
    # Callers pass finite lists; materialize so we can size the work and reuse
    # the columns for either path.
    cols = [list(it) for it in iterables]
    n = len(cols[0]) if cols else 0
    if _FROZEN or n < _PARALLEL_MIN_ITEMS:
        return list(map(fn, *cols))
    pool = _ensure_pool()
    return list(pool.map(fn, *cols, chunksize=chunksize))


def shutdown_pool() -> None:
    """Shut down the persistent worker pool.

    Called automatically at interpreter exit via :func:`atexit.register`.
    Can also be called manually to free resources early.
    """
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False)
        _pool = None
