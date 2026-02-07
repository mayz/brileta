from __future__ import annotations

import numpy as np
import pytest

from catley.util._native import WFCContradictionError, wfc_solve


def _build_propagation_masks(
    num_patterns: int,
    allowed_neighbors: dict[int, set[int]],
) -> np.ndarray:
    """Build direction-agnostic propagation tables for native WFC tests."""
    masks = np.zeros((4, 256), dtype=np.uint8)

    for direction in range(4):
        for current_mask in range(256):
            valid_mask = 0
            for bit_idx in range(num_patterns):
                if current_mask & (1 << bit_idx):
                    for neighbor_bit in allowed_neighbors[bit_idx]:
                        valid_mask |= 1 << neighbor_bit
            masks[direction, current_mask] = valid_mask

    return masks


def _validate_adjacency(
    grid: list[list[int]],
    allowed_neighbors: dict[int, set[int]],
) -> None:
    """Assert all cardinal adjacencies satisfy the allowed-neighbor mapping."""
    width = len(grid)
    height = len(grid[0]) if width else 0

    offsets = ((0, -1), (1, 0), (0, 1), (-1, 0))
    for x in range(width):
        for y in range(height):
            current = grid[x][y]
            allowed = allowed_neighbors[current]
            for dx, dy in offsets:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    assert grid[nx][ny] in allowed


def test_wfc_solve_raises_type_error_for_wrong_propagation_dtype() -> None:
    """The C extension should reject non-uint8 propagation masks."""
    width = 4
    height = 4
    num_patterns = 2

    propagation_masks = np.zeros((4, 256), dtype=np.float64)
    pattern_weights = np.ones(num_patterns, dtype=np.float64)
    initial_wave = np.full((width, height), 0b11, dtype=np.uint8)

    with pytest.raises(TypeError, match="propagation_masks"):
        wfc_solve(
            width,
            height,
            num_patterns,
            propagation_masks,
            pattern_weights,
            initial_wave,
            123,
        )


def test_wfc_solve_raises_value_error_for_out_of_range_wave_bits() -> None:
    """Input wave bits outside num_patterns should be rejected."""
    width = 3
    height = 3
    num_patterns = 2

    propagation_masks = np.zeros((4, 256), dtype=np.uint8)
    pattern_weights = np.array([1.0, 1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b100, dtype=np.uint8)

    with pytest.raises(ValueError, match="outside num_patterns"):
        wfc_solve(
            width,
            height,
            num_patterns,
            propagation_masks,
            pattern_weights,
            initial_wave,
            55,
        )


def test_wfc_solve_raises_native_contradiction_error() -> None:
    """Impossible constraints should surface as native contradiction errors."""
    width = 2
    height = 1
    num_patterns = 2

    # No pattern allows any neighbor in any direction, so propagation must fail.
    propagation_masks = np.zeros((4, 256), dtype=np.uint8)
    pattern_weights = np.array([1.0, 1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b11, dtype=np.uint8)

    with pytest.raises(WFCContradictionError, match="contradiction"):
        wfc_solve(
            width,
            height,
            num_patterns,
            propagation_masks,
            pattern_weights,
            initial_wave,
            42,
        )


def test_wfc_solve_produces_grid_that_satisfies_constraints() -> None:
    """A solved grid should satisfy all adjacency constraints."""
    width = 12
    height = 12
    num_patterns = 3

    # Gradient adjacency: 0 <-> 1 <-> 2, but 0 and 2 cannot touch.
    allowed_neighbors = {0: {0, 1}, 1: {0, 1, 2}, 2: {1, 2}}
    propagation_masks = _build_propagation_masks(num_patterns, allowed_neighbors)
    pattern_weights = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b111, dtype=np.uint8)

    result = wfc_solve(
        width,
        height,
        num_patterns,
        propagation_masks,
        pattern_weights,
        initial_wave,
        999,
    )

    assert len(result) == width
    assert all(len(column) == height for column in result)

    for column in result:
        for bit_idx in column:
            assert bit_idx in {0, 1, 2}

    _validate_adjacency(result, allowed_neighbors)


def test_wfc_solve_handles_1x1_grid() -> None:
    """Degenerate 1x1 grid should collapse to a valid pattern index."""
    width = 1
    height = 1
    num_patterns = 3

    allowed_neighbors = {0: {0, 1}, 1: {0, 1, 2}, 2: {1, 2}}
    propagation_masks = _build_propagation_masks(num_patterns, allowed_neighbors)
    pattern_weights = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b111, dtype=np.uint8)

    result = wfc_solve(
        width,
        height,
        num_patterns,
        propagation_masks,
        pattern_weights,
        initial_wave,
        1234,
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] in {0, 1, 2}


def test_wfc_solve_handles_single_pattern() -> None:
    """Single-pattern mode should produce only that pattern everywhere."""
    width = 8
    height = 5
    num_patterns = 1

    propagation_masks = np.zeros((4, 256), dtype=np.uint8)
    propagation_masks[:, 0b1] = 0b1
    pattern_weights = np.array([1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b1, dtype=np.uint8)

    result = wfc_solve(
        width,
        height,
        num_patterns,
        propagation_masks,
        pattern_weights,
        initial_wave,
        9,
    )

    assert all(cell == 0 for column in result for cell in column)


def test_wfc_solve_accepts_mixed_precollapsed_wave() -> None:
    """Native solve should handle mixed collapsed/uncollapsed initial wave cells."""
    width = 10
    height = 10
    num_patterns = 3

    allowed_neighbors = {0: {0, 1}, 1: {0, 1, 2}, 2: {1, 2}}
    propagation_masks = _build_propagation_masks(num_patterns, allowed_neighbors)
    pattern_weights = np.array([3.0, 2.0, 1.0], dtype=np.float64)
    initial_wave = np.full((width, height), 0b111, dtype=np.uint8)

    # Pre-collapse a vertical stripe to pattern 1, leaving the rest unconstrained.
    initial_wave[4, :] = 0b010

    result = wfc_solve(
        width,
        height,
        num_patterns,
        propagation_masks,
        pattern_weights,
        initial_wave,
        777,
    )

    assert all(result[4][y] == 1 for y in range(height))
    _validate_adjacency(result, allowed_neighbors)


def test_wfc_solve_completes_large_grid() -> None:
    """Native solver should complete a larger grid without raising."""
    width = 100
    height = 80
    num_patterns = 4

    allowed_neighbors = {
        0: {0, 1},
        1: {0, 1, 2, 3},
        2: {1, 2, 3},
        3: {1, 2, 3},
    }
    propagation_masks = _build_propagation_masks(num_patterns, allowed_neighbors)
    pattern_weights = np.array([4.0, 3.0, 2.0, 1.5], dtype=np.float64)
    initial_wave = np.full((width, height), 0b1111, dtype=np.uint8)

    result = wfc_solve(
        width,
        height,
        num_patterns,
        propagation_masks,
        pattern_weights,
        initial_wave,
        2026,
    )

    assert len(result) == width
    assert all(len(column) == height for column in result)
