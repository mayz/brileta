"""Field-of-view computation using Albert Ford's symmetric shadowcasting.

Implements the algorithm described at https://www.albertford.com/shadowcasting/
with exact integer arithmetic to avoid floating-point edge cases.

Key properties of symmetric shadowcasting:
- **Symmetry**: If tile A can see tile B, then B can see A.
- **Light walls**: Opaque tiles that border a visible floor tile are themselves
  visible (the wall that blocks you is always revealed).
- **Exactness**: Slopes are tracked as integer numerator/denominator pairs with
  cross-multiplication for comparisons - same precision as ``fractions.Fraction``
  but without the object-allocation and GCD overhead.

The algorithm processes four cardinal quadrants (north, south, east, west),
each covering a 90-degree arc. Within each quadrant, it scans outward row by
row, tracking which angular sectors are still unblocked.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Quadrant transform coefficients: (col_to_x, depth_to_x, col_to_y, depth_to_y).
# For a given quadrant, world coordinates are:
#   wx = ox + col * cx + depth * dx
#   wy = oy + col * cy + depth * dy
_QUADRANT_TRANSFORMS: list[tuple[int, int, int, int]] = [
    (1, 0, 0, -1),  # North: col->x, depth->-y
    (0, 1, 1, 0),  # East:  depth->x, col->y
    (1, 0, 0, 1),  # South: col->x, depth->+y
    (0, -1, 1, 0),  # West:  depth->-x, col->y
]


def compute_fov(
    transparent: NDArray[np.bool_],
    origin: tuple[int, int],
    radius: int,
) -> NDArray[np.bool_]:
    """Compute the set of tiles visible from *origin*.

    Args:
        transparent: Boolean array shaped ``(width, height)``.
            ``True`` means the tile is see-through (floor, open door, etc.).
        origin: ``(x, y)`` position of the viewer.
        radius: Maximum sight distance. Tiles beyond this are never visible.

    Returns:
        Boolean array with the same shape as *transparent*, where ``True``
        marks a visible tile.
    """
    width, height = transparent.shape
    visible = np.zeros_like(transparent, dtype=np.bool_)
    ox, oy = origin

    # The origin tile is always visible.
    if 0 <= ox < width and 0 <= oy < height:
        visible[ox, oy] = True

    # Scan all four cardinal quadrants.
    for cx, dx, cy, dy in _QUADRANT_TRANSFORMS:
        _scan_quadrant(
            cx, dx, cy, dy, ox, oy, radius, width, height, transparent, visible
        )

    return visible


def _scan_quadrant(
    cx: int,
    dx: int,
    cy: int,
    dy: int,
    ox: int,
    oy: int,
    radius: int,
    width: int,
    height: int,
    transparent: NDArray[np.bool_],
    visible: NDArray[np.bool_],
) -> None:
    """Iteratively scan one 90-degree quadrant outward from the origin.

    Uses an explicit stack instead of recursion to avoid stack-depth issues
    on large radii.

    Slopes are represented as integer (numerator, denominator) pairs.
    Comparisons use cross-multiplication for exact rational arithmetic
    without the overhead of ``fractions.Fraction``.
    """
    # Stack entries: (row_depth, start_num, start_den, end_num, end_den).
    # Initial sector spans the full quadrant: slope -1/1 to 1/1.
    stack: list[tuple[int, int, int, int, int]] = [(1, -1, 1, 1, 1)]

    while stack:
        depth, s_num, s_den, e_num, e_den = stack.pop()

        if depth > radius:
            continue

        # Column range for this row's visible sector.
        # min_col = round_ties_up(depth * s_num / s_den)
        #         = floor(depth * s_num / s_den + 1/2)
        #         = floor((2 * depth * s_num + s_den) / (2 * s_den))
        # Python's // is floor division, which is correct for positive denominator.
        two_s_den = 2 * s_den
        min_col = (2 * depth * s_num + s_den) // two_s_den

        # max_col = round_ties_down(depth * e_num / e_den)
        #         = ceil(depth * e_num / e_den - 1/2)
        #         = ceil((2 * depth * e_num - e_den) / (2 * e_den))
        #         = -( -(2 * depth * e_num - e_den) // (2 * e_den) )
        two_e_den = 2 * e_den
        max_col = -(-(2 * depth * e_num - e_den) // two_e_den)

        prev_was_wall: bool | None = None

        for col in range(min_col, max_col + 1):
            # Inline quadrant transform.
            wx = ox + col * cx + depth * dx
            wy = oy + col * cy + depth * dy

            # Bounds check - tiles outside the map are treated as walls.
            in_bounds = 0 <= wx < width and 0 <= wy < height
            is_wall = not in_bounds or not transparent[wx, wy]
            is_floor = not is_wall

            # Reveal the tile if it's in-bounds and either:
            # - a wall (light_walls: you see the wall that blocks you), or
            # - symmetric (center falls within the visible sector).
            # Symmetric check: col >= depth * s_num/s_den AND col <= depth * e_num/e_den
            # Cross-multiply (denominators are always positive):
            #   col * s_den >= depth * s_num AND col * e_den <= depth * e_num
            if in_bounds and (
                is_wall
                or (col * s_den >= depth * s_num and col * e_den <= depth * e_num)
            ):
                visible[wx, wy] = True

            # Track transitions between floor and wall tiles to adjust the
            # visible sector for deeper rows.
            if prev_was_wall is not None:
                if prev_was_wall and is_floor:
                    # Wall-to-floor: sector re-opens. Adjust start slope to the
                    # near edge of this tile: (2*col - 1) / (2*depth).
                    s_num = 2 * col - 1
                    s_den = 2 * depth
                elif not prev_was_wall and is_wall:
                    # Floor-to-wall: a shadow begins. Push the still-visible
                    # sector for the next row depth.
                    stack.append(
                        (
                            depth + 1,
                            s_num,
                            s_den,
                            2 * col - 1,  # slope near-edge numerator
                            2 * depth,  # slope near-edge denominator
                        )
                    )

            prev_was_wall = is_wall

        # If the row ended on a floor tile, the sector continues unobstructed.
        if prev_was_wall is not None and not prev_was_wall:
            stack.append((depth + 1, s_num, s_den, e_num, e_den))
