from __future__ import annotations

import numpy as np
import pytest

from brileta.util._native import astar


def test_astar_raises_type_error_for_wrong_dtype() -> None:
    """The C extension should reject non-int16 arrays."""
    cost = np.ones((4, 4), dtype=np.float64)

    with pytest.raises(TypeError, match="2D int16 C-contiguous array"):
        astar(cost, 0, 0, 3, 3)


@pytest.mark.parametrize(
    ("sx", "sy", "gx", "gy"),
    [
        (-1, 0, 3, 3),
        (0, -1, 3, 3),
        (0, 0, 4, 3),
        (0, 0, 3, 4),
    ],
)
def test_astar_raises_value_error_for_out_of_bounds_coords(
    sx: int,
    sy: int,
    gx: int,
    gy: int,
) -> None:
    """Out-of-bounds start/goal coordinates should raise ValueError."""
    cost = np.ones((4, 4), dtype=np.int16)

    with pytest.raises(ValueError, match="start or goal is out of bounds"):
        astar(cost, sx, sy, gx, gy)


def test_astar_handles_single_column_grid() -> None:
    """A 1xN grid should reconstruct a straight vertical path."""
    cost = np.ones((1, 8), dtype=np.int16)

    path = astar(cost, 0, 0, 0, 7)

    assert path == [(0, y) for y in range(1, 8)]


def test_astar_handles_single_row_grid() -> None:
    """An Nx1 grid should reconstruct a straight horizontal path."""
    cost = np.ones((8, 1), dtype=np.int16)

    path = astar(cost, 0, 0, 7, 0)

    assert path == [(x, 0) for x in range(1, 8)]


def test_astar_returns_empty_when_start_is_blocked() -> None:
    """Blocked start tiles are impassable and should produce no path."""
    cost = np.ones((5, 5), dtype=np.int16)
    cost[0, 0] = 0
    cost[4, 4] = 1

    assert astar(cost, 0, 0, 4, 4) == []
