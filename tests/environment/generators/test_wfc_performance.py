"""Performance tests for the WFC solver.

These tests verify the optimized WFC solver meets performance targets.
"""

from __future__ import annotations

import random
import time

from brileta.environment.generators.pipeline.layers.terrain import (
    create_terrain_patterns,
)
from brileta.environment.generators.wfc_solver import WFCSolver


class TestWFCPerformance:
    """Performance benchmarks for WFCSolver."""

    def test_100x80_under_500ms(self) -> None:
        """WFC should solve 100x80 grid in under 500ms."""
        patterns = create_terrain_patterns()
        rng = random.Random(42)

        start = time.perf_counter()
        solver = WFCSolver(100, 80, patterns, rng)
        solver.solve()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"WFC took {elapsed:.2f}s, expected <0.5s"

    def test_determinism_preserved(self) -> None:
        """Same seed produces identical results after optimization."""
        patterns = create_terrain_patterns()

        result1 = WFCSolver(30, 30, patterns, random.Random(123)).solve()
        result2 = WFCSolver(30, 30, patterns, random.Random(123)).solve()

        assert result1 == result2

    def test_60x60_regression(self) -> None:
        """60x60 should complete in under 200ms (was ~6s before optimization)."""
        patterns = create_terrain_patterns()
        rng = random.Random(999)

        start = time.perf_counter()
        solver = WFCSolver(60, 60, patterns, rng)
        solver.solve()
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"WFC took {elapsed:.2f}s, expected <0.2s"

    def test_large_grid_150x150(self) -> None:
        """150x150 grid should complete in under 2s."""
        patterns = create_terrain_patterns()
        rng = random.Random(777)

        start = time.perf_counter()
        solver = WFCSolver(150, 150, patterns, rng)
        solver.solve()
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"WFC took {elapsed:.2f}s, expected <2.0s"
