#!/usr/bin/env python3
"""Benchmark comparing native C A* and tcod's C implementation.

Runs both implementations on identical cost maps and prints a timing comparison
table, followed by correctness checks.

Usage:
    uv run python scripts/benchmark_pathfinding.py
"""

# ruff: noqa: E402  # Allow path setup before importing project modules

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import tcod.path

# Add the project root to Python path so running as a script works.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from catley.util._native import astar as native_astar

# ---------------------------------------------------------------------------
# Map generators
# ---------------------------------------------------------------------------


def _make_open_field(width: int, height: int) -> np.ndarray:
    """All-walkable map (worst case - maximum search space)."""
    return np.ones((width, height), dtype=np.int16)


def _make_dungeon(
    width: int,
    height: int,
    wall_fraction: float,
    seed: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Random walls to simulate a dungeon.  Positive = walkable, 0 = blocked.

    Start and goal tiles are forced walkable so the benchmark actually
    measures pathfinding work rather than an instant "no path" return.
    """
    rng = np.random.default_rng(seed)
    cost = (rng.random((width, height)) > wall_fraction).astype(np.int16)
    cost[start[0], start[1]] = 1
    cost[goal[0], goal[1]] = 1
    return cost


def _make_weighted(
    width: int,
    height: int,
    seed: int,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    """Variable-cost terrain. Costs between 1 and 10, with some blocked cells.

    Start and goal tiles are forced walkable.
    """
    rng = np.random.default_rng(seed)
    cost = rng.integers(1, 11, size=(width, height), dtype=np.int16)
    # Block ~15% of tiles
    blocked = rng.random((width, height)) < 0.15
    cost[blocked] = 0
    cost[start[0], start[1]] = 1
    cost[goal[0], goal[1]] = 1
    return cost


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _run_tcod(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Run tcod A* and return the path (excluding start)."""
    astar = tcod.path.AStar(cost=cost, diagonal=1)
    return astar.get_path(start[0], start[1], goal[0], goal[1])


def _run_native(
    cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """Run our native C A* and return the path (excluding start)."""
    c_cost = np.ascontiguousarray(cost, dtype=np.int16)
    return native_astar(c_cost, start[0], start[1], goal[0], goal[1])


def _bench(fn: object, *args: object) -> float:
    """Time *fn(*args)* and return average ms per call."""
    # Warm up
    fn(*args)  # type: ignore[operator]
    timer = timeit.Timer(lambda: fn(*args))  # type: ignore[operator]
    number, total = timer.autorange()
    return (total / number) * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    scenarios: list[tuple[str, np.ndarray, tuple[int, int], tuple[int, int]]] = [
        ("Open 120x80", _make_open_field(120, 80), (5, 5), (115, 75)),
        (
            "Dungeon 120x80 (40%)",
            _make_dungeon(120, 80, 0.40, seed=42, start=(5, 5), goal=(115, 75)),
            (5, 5),
            (115, 75),
        ),
        ("Open 256x256", _make_open_field(256, 256), (10, 10), (245, 245)),
        (
            "Dungeon 256x256 (30%)",
            _make_dungeon(256, 256, 0.30, seed=99, start=(10, 10), goal=(245, 245)),
            (10, 10),
            (245, 245),
        ),
        (
            "Weighted 120x80",
            _make_weighted(120, 80, seed=7, start=(5, 5), goal=(115, 75)),
            (5, 5),
            (115, 75),
        ),
    ]

    print("A* Benchmark: native (C) vs tcod (C)")
    print("=" * 78)
    print(f"{'Scenario':<28} {'tcod (C)':>10} {'native (C)':>12} {'native/tcod':>12}")
    print("-" * 78)

    for name, cost, start, goal in scenarios:
        tcod_ms = _bench(_run_tcod, cost, start, goal)
        native_ms = _bench(_run_native, cost, start, goal)
        ratio = native_ms / tcod_ms if tcod_ms > 0 else float("inf")
        print(f"{name:<28} {tcod_ms:>9.3f}ms {native_ms:>11.3f}ms {ratio:>11.2f}x")

    print("-" * 78)
    print("native/tcod < 1 means our native C is faster than tcod.")
    print()

    # ------------------------------------------------------------------
    # Correctness checks
    # ------------------------------------------------------------------
    print("Correctness checks...")
    start, goal = (5, 5), (115, 75)
    cost = _make_dungeon(120, 80, 0.20, seed=42, start=start, goal=goal)

    tcod_path = _run_tcod(cost, start, goal)
    native_path = _run_native(cost, start, goal)

    if not tcod_path and not native_path:
        print("  Both agree: no path exists.")
    elif not tcod_path or not native_path:
        print(
            f"  MISMATCH: tcod found {'a' if tcod_path else 'no'} path, "
            f"native found {'a' if native_path else 'no'} path."
        )
    else:
        print(f"  tcod path length:   {len(tcod_path)}")
        print(f"  native path length: {len(native_path)}")

    # Straightness check: horizontal path should not zigzag.
    straight_cost = _make_open_field(120, 80)
    straight_path = _run_native(straight_cost, (5, 40), (115, 40))
    y_vals = {y for _, y in straight_path}
    if y_vals == {40}:
        print("  Straightness check: horizontal path stays on y=40. OK!")
    else:
        print(f"  Straightness FAIL: y values deviated to {y_vals}")

    # No-path check.
    blocked_cost = np.zeros((10, 10), dtype=np.int16)
    blocked_cost[0, 0] = 1
    blocked_cost[9, 9] = 1
    tcod_none = _run_tcod(blocked_cost, (0, 0), (9, 9))
    native_none = _run_native(blocked_cost, (0, 0), (9, 9))
    if not tcod_none and not native_none:
        print("  Blocked-map check: both correctly return empty path.")
    else:
        print(
            f"  Blocked-map MISMATCH: tcod={len(tcod_none)}, native={len(native_none)}"
        )


if __name__ == "__main__":
    main()
