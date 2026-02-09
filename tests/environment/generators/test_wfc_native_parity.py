"""Confidence checks for the production (native) WFC solver path.

These tests focus on production behavior:
- Outputs satisfy adjacency constraints
- Forced constraints are preserved
- Unsatisfiable inputs raise contradictions
"""

from __future__ import annotations

import random
from collections import Counter
from enum import IntEnum

import pytest

from brileta.environment.generators.pipeline.layers.terrain import (
    TerrainPatternID,
    create_terrain_patterns,
)
from brileta.environment.generators.wfc_solver import (
    DIR_OFFSETS,
    DIRECTIONS,
    WFCContradiction,
    WFCPattern,
    WFCSolver,
)
from brileta.environment.tile_types import TileTypeID


def _assert_grid_respects_adjacency(
    grid: list[list[TerrainPatternID]],
    patterns: dict[TerrainPatternID, WFCPattern[TerrainPatternID]],
) -> None:
    """Assert every cardinal neighbor pair satisfies adjacency rules."""
    width = len(grid)
    height = len(grid[0]) if width else 0

    assert width > 0
    assert height > 0

    for x in range(width):
        for y in range(height):
            pattern_id = grid[x][y]
            pattern = patterns[pattern_id]

            for direction in DIRECTIONS:
                dx, dy = DIR_OFFSETS[direction]
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                assert grid[nx][ny] in pattern.valid_neighbors[direction]


def _run_solver(
    *,
    width: int,
    height: int,
    seed: int,
    constraints: list[tuple[int, int, set[TerrainPatternID]]] | None = None,
) -> list[list[TerrainPatternID]]:
    """Solve one terrain WFC instance using the production native path."""
    patterns = create_terrain_patterns()
    solver = WFCSolver(width, height, patterns, random.Random(seed))

    if constraints:
        solver.constrain_cells(constraints)

    result = solver.solve()
    _assert_grid_respects_adjacency(result, patterns)
    return result


def _pattern_ratios(
    grids: list[list[list[TerrainPatternID]]],
) -> dict[TerrainPatternID, float]:
    """Compute normalized pattern frequencies across solved grids."""
    counts: Counter[TerrainPatternID] = Counter()
    total = 0

    for grid in grids:
        for column in grid:
            counts.update(column)
            total += len(column)

    assert total > 0
    return {pattern_id: counts[pattern_id] / total for pattern_id in TerrainPatternID}


def test_native_solver_produces_diverse_patterns_across_many_seeds() -> None:
    """Over many seeds, native solve should generate all terrain pattern types."""
    seeds = list(range(30))
    width, height = 24, 24

    native_grids = [
        _run_solver(width=width, height=height, seed=seed) for seed in seeds
    ]
    native_ratios = _pattern_ratios(native_grids)

    for pattern_id in TerrainPatternID:
        assert native_ratios[pattern_id] > 0.0, (
            f"Pattern {pattern_id.name} never appeared across {len(seeds)} runs"
        )


def test_native_solver_preserves_forced_constraints() -> None:
    """Native path should honor pre-constrained cobblestone roads."""
    width, height = 40, 20
    road_x = width // 2

    constraints: list[tuple[int, int, set[TerrainPatternID]]] = [
        (road_x, y, {TerrainPatternID.COBBLESTONE}) for y in range(height)
    ]

    native_result = _run_solver(
        width=width,
        height=height,
        seed=2026,
        constraints=constraints,
    )

    for y in range(height):
        assert native_result[road_x][y] == TerrainPatternID.COBBLESTONE


@pytest.mark.parametrize("seed", [0, 42, 2026])
def test_native_solver_is_deterministic(seed: int) -> None:
    """Same seed and constraints should produce the same solved grid."""
    width, height = 20, 20

    result_a = _run_solver(width=width, height=height, seed=seed)
    result_b = _run_solver(width=width, height=height, seed=seed)

    assert result_a == result_b


class _UnsatPatternID(IntEnum):
    A = 0
    B = 1


def _create_unsat_patterns() -> dict[_UnsatPatternID, WFCPattern[_UnsatPatternID]]:
    """Create a pattern set that is unsatisfiable for any 2-cell line."""
    no_neighbors = {direction: set() for direction in DIRECTIONS}
    return {
        _UnsatPatternID.A: WFCPattern(
            pattern_id=_UnsatPatternID.A,
            tile_type=TileTypeID.GRASS,
            valid_neighbors=no_neighbors,
        ),
        _UnsatPatternID.B: WFCPattern(
            pattern_id=_UnsatPatternID.B,
            tile_type=TileTypeID.DIRT_PATH,
            valid_neighbors=no_neighbors,
        ),
    }


def test_native_solver_raises_on_unsat_problem() -> None:
    """Unsatisfiable instances should raise WFCContradiction."""
    patterns = _create_unsat_patterns()
    native_solver = WFCSolver(2, 1, patterns, random.Random(7))
    with pytest.raises(WFCContradiction):
        native_solver.solve()
