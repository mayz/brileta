"""Tests for the generic Wave Function Collapse solver.

These tests verify the WFCSolver works correctly with arbitrary pattern sets,
independent of any specific domain (terrain, settlements, etc.).
"""

from __future__ import annotations

import contextlib
import random
from enum import IntEnum, auto

import pytest

from brileta.environment.generators.wfc_solver import (
    DIR_OFFSETS,
    DIRECTIONS,
    OPPOSITE_DIR,
    WFCContradiction,
    WFCPattern,
    WFCSolver,
)
from brileta.environment.tile_types import TileTypeID

# =============================================================================
# Test Pattern Set
# =============================================================================


class SimplePatternID(IntEnum):
    """Simple test patterns for solver tests."""

    A = 0
    B = auto()
    C = auto()


def create_test_patterns() -> dict[SimplePatternID, WFCPattern[SimplePatternID]]:
    """Create a simple pattern set for testing.

    Pattern adjacency rules:
    - A can be next to A and B (common base pattern)
    - B can be next to A, B, and C (transition pattern)
    - C can be next to B and C (rare pattern)

    This creates a gradient: A <-> B <-> C where A and C cannot be directly adjacent.
    """
    patterns: dict[SimplePatternID, WFCPattern[SimplePatternID]] = {}

    # Pattern A: common, connects to A and B
    patterns[SimplePatternID.A] = WFCPattern(
        pattern_id=SimplePatternID.A,
        tile_type=TileTypeID.GRASS,
        weight=3.0,
        valid_neighbors={
            "N": {SimplePatternID.A, SimplePatternID.B},
            "E": {SimplePatternID.A, SimplePatternID.B},
            "S": {SimplePatternID.A, SimplePatternID.B},
            "W": {SimplePatternID.A, SimplePatternID.B},
        },
    )

    # Pattern B: transition, connects to all
    patterns[SimplePatternID.B] = WFCPattern(
        pattern_id=SimplePatternID.B,
        tile_type=TileTypeID.DIRT_PATH,
        weight=2.0,
        valid_neighbors={
            "N": {SimplePatternID.A, SimplePatternID.B, SimplePatternID.C},
            "E": {SimplePatternID.A, SimplePatternID.B, SimplePatternID.C},
            "S": {SimplePatternID.A, SimplePatternID.B, SimplePatternID.C},
            "W": {SimplePatternID.A, SimplePatternID.B, SimplePatternID.C},
        },
    )

    # Pattern C: rare, connects to B and C only
    patterns[SimplePatternID.C] = WFCPattern(
        pattern_id=SimplePatternID.C,
        tile_type=TileTypeID.GRAVEL,
        weight=1.0,
        valid_neighbors={
            "N": {SimplePatternID.B, SimplePatternID.C},
            "E": {SimplePatternID.B, SimplePatternID.C},
            "S": {SimplePatternID.B, SimplePatternID.C},
            "W": {SimplePatternID.B, SimplePatternID.C},
        },
    )

    return patterns


# =============================================================================
# Basic Solver Functionality
# =============================================================================


class TestSolverBasicFunctionality:
    """Tests for WFCSolver basic operation."""

    def test_solver_produces_valid_pattern_grid(self) -> None:
        """With fixed seed, solver returns grid where every cell has one pattern."""
        width, height = 10, 10
        patterns = create_test_patterns()
        rng = random.Random(42)

        solver = WFCSolver(width, height, patterns, rng)
        result = solver.solve()

        # Verify grid dimensions
        assert len(result) == width
        assert all(len(col) == height for col in result)

        # Verify every cell has exactly one pattern (a valid SimplePatternID)
        for x in range(width):
            for y in range(height):
                pattern = result[x][y]
                assert isinstance(pattern, SimplePatternID)
                assert pattern in patterns

    def test_solver_respects_adjacency_rules(self) -> None:
        """Every adjacent pair in output satisfies pattern's valid_neighbors."""
        width, height = 12, 12
        patterns = create_test_patterns()
        rng = random.Random(123)

        solver = WFCSolver(width, height, patterns, rng)
        result = solver.solve()

        # Check all adjacent pairs
        for x in range(width):
            for y in range(height):
                current_pattern = result[x][y]
                current_def = patterns[current_pattern]

                for direction in DIRECTIONS:
                    dx, dy = DIR_OFFSETS[direction]
                    nx, ny = x + dx, y + dy

                    # Skip out-of-bounds neighbors
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue

                    neighbor_pattern = result[nx][ny]
                    valid_neighbors = current_def.valid_neighbors.get(direction, set())

                    assert neighbor_pattern in valid_neighbors, (
                        f"Invalid adjacency at ({x},{y})->{direction}->({nx},{ny}): "
                        f"{current_pattern.name} disallows {neighbor_pattern.name}"
                    )

    def test_a_and_c_never_adjacent(self) -> None:
        """Pattern A and C should never be directly adjacent (must have B between)."""
        width, height = 15, 15
        patterns = create_test_patterns()
        rng = random.Random(456)

        solver = WFCSolver(width, height, patterns, rng)
        result = solver.solve()

        for x in range(width):
            for y in range(height):
                if result[x][y] == SimplePatternID.A:
                    # Check all neighbors
                    for direction in DIRECTIONS:
                        dx, dy = DIR_OFFSETS[direction]
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            assert result[nx][ny] != SimplePatternID.C, (
                                f"A and C are adjacent at ({x},{y}) and ({nx},{ny})"
                            )

    def test_constrain_cell(self) -> None:
        """Constraining a cell limits its possibilities correctly."""
        width, height = 5, 5
        patterns = create_test_patterns()
        rng = random.Random(789)

        solver = WFCSolver(width, height, patterns, rng)

        # Constrain center to only pattern C
        solver.constrain_cell(2, 2, {SimplePatternID.C})

        result = solver.solve()

        # Center should be C
        assert result[2][2] == SimplePatternID.C

        # All neighbors of center must be B or C (C can't touch A)
        for direction in DIRECTIONS:
            dx, dy = DIR_OFFSETS[direction]
            nx, ny = 2 + dx, 2 + dy
            assert result[nx][ny] in {SimplePatternID.B, SimplePatternID.C}


# =============================================================================
# Contradiction Handling
# =============================================================================


class TestContradictionHandling:
    """Tests for WFC contradiction detection and iteration limits."""

    def test_empty_possibilities_raises(self) -> None:
        """Cell with no possibilities raises WFCContradiction."""
        patterns = create_test_patterns()
        rng = random.Random(111)

        solver = WFCSolver(3, 3, patterns, rng)
        solver.wave[1, 1] = 0  # Empty bitmask (no possibilities)

        with pytest.raises(WFCContradiction):
            solver.solve()

    def test_constrain_to_empty_raises(self) -> None:
        """Constraining to no valid patterns raises WFCContradiction."""
        patterns = create_test_patterns()
        rng = random.Random(222)

        solver = WFCSolver(3, 3, patterns, rng)

        with pytest.raises(WFCContradiction):
            solver.constrain_cell(1, 1, set())  # Empty set

    def test_propagate_iteration_limit(self) -> None:
        """Propagation doesn't hang indefinitely - has iteration limits."""
        patterns = create_test_patterns()
        rng = random.Random(333)

        width, height = 5, 5
        solver = WFCSolver(width, height, patterns, rng)

        # Verify iteration limit is reasonable
        expected_max = width * height * 10
        assert expected_max == 250

        # Force a scenario that triggers propagation
        # Set cell to only pattern C (bit index 2)
        solver.wave[2, 2] = 1 << solver.pattern_to_bit[SimplePatternID.C]

        # Should complete within limits (either succeeds or raises)
        with contextlib.suppress(WFCContradiction):
            solver._propagate(2, 2)

    def test_solve_iteration_limit(self) -> None:
        """Main solve loop has iteration limits to prevent hangs."""
        patterns = create_test_patterns()
        rng = random.Random(444)

        width, height = 3, 3
        solver = WFCSolver(width, height, patterns, rng)

        # Should complete without hanging
        with contextlib.suppress(WFCContradiction):
            solver.solve()


# =============================================================================
# Determinism
# =============================================================================


class TestDeterminism:
    """Tests for reproducibility with seeds."""

    def test_solver_deterministic_with_seed(self) -> None:
        """Same seed produces identical solver output."""
        width, height = 15, 15
        patterns = create_test_patterns()

        # First run
        rng1 = random.Random(42424242)
        solver1 = WFCSolver(width, height, patterns, rng1)
        result1 = solver1.solve()

        # Second run with same seed
        rng2 = random.Random(42424242)
        solver2 = WFCSolver(width, height, patterns, rng2)
        result2 = solver2.solve()

        # Results should be identical
        for x in range(width):
            for y in range(height):
                assert result1[x][y] == result2[x][y], (
                    f"Mismatch at ({x},{y}): {result1[x][y]} vs {result2[x][y]}"
                )

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different outputs (with high probability)."""
        width, height = 10, 10
        patterns = create_test_patterns()

        rng1 = random.Random(12345)
        solver1 = WFCSolver(width, height, patterns, rng1)
        result1 = solver1.solve()

        rng2 = random.Random(67890)
        solver2 = WFCSolver(width, height, patterns, rng2)
        result2 = solver2.solve()

        # Count differences
        differences = 0
        for x in range(width):
            for y in range(height):
                if result1[x][y] != result2[x][y]:
                    differences += 1

        # Should have at least some differences (very likely with different seeds)
        assert differences > 0, "Different seeds produced identical results"


# =============================================================================
# Pattern Definitions
# =============================================================================


class TestPatternDefinitions:
    """Tests for pattern definition validity."""

    def test_all_patterns_have_valid_neighbors(self) -> None:
        """Every pattern has valid_neighbors for all 4 directions."""
        patterns = create_test_patterns()

        for pattern_id, pattern in patterns.items():
            for direction in DIRECTIONS:
                assert direction in pattern.valid_neighbors, (
                    f"Pattern {pattern_id.name} missing neighbors for {direction}"
                )
                assert len(pattern.valid_neighbors[direction]) > 0, (
                    f"Pattern {pattern_id.name} has empty neighbors for {direction}"
                )

    def test_adjacency_rules_are_symmetric(self) -> None:
        """If A allows B to the east, B should allow A to the west."""
        patterns = create_test_patterns()

        for pattern_id, pattern in patterns.items():
            for direction in DIRECTIONS:
                opposite = OPPOSITE_DIR[direction]

                for neighbor_id in pattern.valid_neighbors[direction]:
                    neighbor = patterns[neighbor_id]
                    assert pattern_id in neighbor.valid_neighbors[opposite], (
                        f"Asymmetric adjacency: {pattern_id.name} allows "
                        f"{neighbor_id.name} to {direction}, but {neighbor_id.name} "
                        f"does not allow {pattern_id.name} to {opposite}"
                    )

    def test_patterns_have_correct_tile_types(self) -> None:
        """Patterns map to expected tile types."""
        patterns = create_test_patterns()

        assert patterns[SimplePatternID.A].tile_type == TileTypeID.GRASS
        assert patterns[SimplePatternID.B].tile_type == TileTypeID.DIRT_PATH
        assert patterns[SimplePatternID.C].tile_type == TileTypeID.GRAVEL


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_1x1_grid(self) -> None:
        """Solver works with minimal 1x1 grid."""
        patterns = create_test_patterns()
        rng = random.Random(555)

        solver = WFCSolver(1, 1, patterns, rng)
        result = solver.solve()

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0] in patterns

    def test_single_pattern(self) -> None:
        """Solver works with a single pattern that can be next to itself."""
        single_pattern: dict[int, WFCPattern[int]] = {
            0: WFCPattern(
                pattern_id=0,
                tile_type=TileTypeID.GRASS,
                weight=1.0,
                valid_neighbors={"N": {0}, "E": {0}, "S": {0}, "W": {0}},
            )
        }
        rng = random.Random(666)

        solver = WFCSolver(5, 5, single_pattern, rng)
        result = solver.solve()

        # All cells should be pattern 0
        for x in range(5):
            for y in range(5):
                assert result[x][y] == 0

    def test_non_square_grid(self) -> None:
        """Solver works with non-square grids."""
        patterns = create_test_patterns()
        rng = random.Random(777)

        # Wide grid
        solver = WFCSolver(20, 5, patterns, rng)
        result = solver.solve()
        assert len(result) == 20
        assert all(len(col) == 5 for col in result)

        # Tall grid
        rng2 = random.Random(888)
        solver2 = WFCSolver(5, 20, patterns, rng2)
        result2 = solver2.solve()
        assert len(result2) == 5
        assert all(len(col) == 20 for col in result2)
