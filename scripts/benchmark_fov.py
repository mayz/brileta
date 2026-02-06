#!/usr/bin/env python3
"""Benchmark comparing our symmetric shadowcasting FOV against tcod's C implementation.

Runs both implementations on identical inputs and prints a timing comparison table.

Usage:
    uv run python scripts/benchmark_fov.py
"""

# ruff: noqa: E402  # Allow path setup before importing project modules

from __future__ import annotations

import sys
import timeit
from pathlib import Path

import numpy as np
import tcod.constants
import tcod.map

# Add the project root to Python path so running as a script works.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from catley.environment.fov import compute_fov


def _make_open_field(width: int, height: int) -> np.ndarray:
    """All-transparent map (worst case - maximum visible tiles)."""
    return np.ones((width, height), dtype=np.bool_)


def _make_dungeon(
    width: int, height: int, wall_fraction: float, seed: int
) -> np.ndarray:
    """Randomly scatter walls to simulate a dungeon layout."""
    rng = np.random.default_rng(seed)
    return (rng.random((width, height)) > wall_fraction).astype(np.bool_)


def _benchmark_ours(
    transparent: np.ndarray, origin: tuple[int, int], radius: int
) -> float:
    """Time our Python implementation and return average ms per call."""
    # Warm up.
    compute_fov(transparent, origin, radius)

    timer = timeit.Timer(lambda: compute_fov(transparent, origin, radius))
    # Auto-range to get a reliable measurement.
    number, total = timer.autorange()
    return (total / number) * 1000  # ms


def _benchmark_tcod(
    transparent: np.ndarray, origin: tuple[int, int], radius: int
) -> float:
    """Time tcod's C implementation and return average ms per call."""
    # Warm up.
    tcod.map.compute_fov(
        transparent,
        origin,
        radius=radius,
        light_walls=True,
        algorithm=tcod.constants.FOV_SYMMETRIC_SHADOWCAST,
    )

    def run() -> None:
        tcod.map.compute_fov(
            transparent,
            origin,
            radius=radius,
            light_walls=True,
            algorithm=tcod.constants.FOV_SYMMETRIC_SHADOWCAST,
        )

    timer = timeit.Timer(run)
    number, total = timer.autorange()
    return (total / number) * 1000  # ms


def main() -> None:
    scenarios: list[tuple[str, np.ndarray, tuple[int, int], int]] = [
        ("Open field", _make_open_field(120, 80), (60, 40), 50),
        ("Dungeon (~40% walls)", _make_dungeon(120, 80, 0.40, seed=42), (60, 40), 50),
        ("Small radius", _make_open_field(120, 80), (60, 40), 10),
    ]

    print("FOV Benchmark: ours (native C extension) vs tcod (C)")
    print("=" * 62)
    print(f"{'Scenario':<24} {'tcod (C)':>10} {'ours (native)':>14} {'ratio':>8}")
    print("-" * 62)

    for name, transparent, origin, radius in scenarios:
        tcod_ms = _benchmark_tcod(transparent, origin, radius)
        ours_ms = _benchmark_ours(transparent, origin, radius)
        ratio = ours_ms / tcod_ms if tcod_ms > 0 else float("inf")
        print(f"{name:<24} {tcod_ms:>9.3f}ms {ours_ms:>11.3f}ms {ratio:>7.1f}x")

    print("-" * 62)
    print("Lower ratio = closer to C performance.")
    print()

    # Correctness check: verify both produce the same results.
    print("Correctness check...")
    transparent = _make_dungeon(120, 80, 0.40, seed=123)
    origin = (60, 40)
    radius = 50

    ours = compute_fov(transparent, origin, radius)
    theirs = tcod.map.compute_fov(
        transparent,
        origin,
        radius=radius,
        light_walls=True,
        algorithm=tcod.constants.FOV_SYMMETRIC_SHADOWCAST,
    )

    match_count = np.sum(ours == theirs)
    total_tiles = ours.size
    mismatch_count = total_tiles - match_count
    match_pct = match_count / total_tiles * 100

    print(f"  Agreement: {match_pct:.2f}% ({match_count}/{total_tiles} tiles)")
    if mismatch_count > 0:
        print(f"  Mismatches: {mismatch_count} tiles")
        # Show a few mismatches for debugging.
        diff_coords = np.argwhere(ours != theirs)
        for coord in diff_coords[:10]:
            x, y = coord
            print(
                f"    ({x}, {y}): ours={ours[x, y]}, tcod={theirs[x, y]}, "
                f"transparent={transparent[x, y]}"
            )
    else:
        print("  Perfect match!")


if __name__ == "__main__":
    main()
