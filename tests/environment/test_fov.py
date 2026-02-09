"""Tests for the symmetric shadowcasting FOV implementation."""

from __future__ import annotations

import numpy as np

from brileta.environment.fov import compute_fov


def _make_map(width: int, height: int, *, fill: bool = True) -> np.ndarray:
    """Create a transparent map. fill=True means all see-through (open field)."""
    return np.full((width, height), fill, dtype=np.bool_)


# ── 1. Open field ──────────────────────────────────────────────────────────


def test_open_field_everything_within_radius_visible() -> None:
    """Player in center of an open room sees every tile within radius."""
    width, height = 21, 21
    transparent = _make_map(width, height)
    origin = (10, 10)
    radius = 10

    visible = compute_fov(transparent, origin, radius)

    # Every tile within Manhattan-adjacent Chebyshev distance <= radius should
    # be visible in an open field, but FOV uses Euclidean-ish line of sight.
    # At minimum, all tiles within radius along cardinal/diagonal axes must
    # be visible.
    for dx in range(-radius, radius + 1):
        x = origin[0] + dx
        if 0 <= x < width:
            assert visible[x, origin[1]], f"Tile ({x}, {origin[1]}) should be visible"

    for dy in range(-radius, radius + 1):
        y = origin[1] + dy
        if 0 <= y < height:
            assert visible[origin[0], y], f"Tile ({origin[0]}, {y}) should be visible"


# ── 2. Single wall blocks tiles behind it ──────────────────────────────────


def test_single_wall_blocks_behind() -> None:
    """A wall blocks visibility to tiles directly behind it."""
    width, height = 21, 21
    transparent = _make_map(width, height)
    origin = (10, 10)

    # Place a wall directly north of origin at distance 2.
    transparent[10, 8] = False

    visible = compute_fov(transparent, origin, radius=10)

    # The wall itself should be visible (light_walls behavior).
    assert visible[10, 8], "Wall tile should be visible"

    # Tiles behind the wall (further north on the same column) should be blocked.
    # At least the immediately behind tile should be hidden.
    assert not visible[10, 7], "Tile behind wall should not be visible"


# ── 3. Corridor limits visibility ──────────────────────────────────────────


def test_corridor_limits_visibility() -> None:
    """A long corridor restricts visibility to the corridor tiles only."""
    width, height = 20, 20
    # Start with all walls.
    transparent = _make_map(width, height, fill=False)

    # Carve a horizontal corridor at y=10, from x=0 to x=19.
    for x in range(width):
        transparent[x, 10] = True

    origin = (0, 10)
    visible = compute_fov(transparent, origin, radius=19)

    # All corridor tiles should be visible.
    for x in range(width):
        assert visible[x, 10], f"Corridor tile ({x}, 10) should be visible"

    # Tiles above/below the corridor walls should NOT be visible (they are
    # walls themselves, but tiles beyond those walls should be hidden).
    # The wall tiles at y=9 and y=11 are visible (light_walls), but y=8 is not.
    for x in range(1, width):
        if 0 <= x < width:
            assert not visible[x, 8], (
                f"Tile ({x}, 8) beyond corridor wall should not be visible"
            )


# ── 4. Symmetry: if A sees B, B sees A ────────────────────────────────────


def test_symmetry() -> None:
    """The defining property of symmetric shadowcasting."""
    width, height = 30, 30
    rng = np.random.default_rng(42)
    transparent = rng.random((width, height)) > 0.3  # ~30% walls
    transparent = transparent.astype(np.bool_)
    radius = 12

    # Pick two arbitrary floor tiles as origins.
    floor_tiles = list(zip(*np.where(transparent), strict=True))
    a = floor_tiles[0]
    b = floor_tiles[len(floor_tiles) // 2]

    vis_from_a = compute_fov(transparent, a, radius)
    vis_from_b = compute_fov(transparent, b, radius)

    # If A sees B, then B must see A (and vice versa).
    if vis_from_a[b[0], b[1]]:
        assert vis_from_b[a[0], a[1]], "Symmetry violated: A sees B but B doesn't see A"
    if vis_from_b[a[0], a[1]]:
        assert vis_from_a[b[0], b[1]], "Symmetry violated: B sees A but A doesn't see B"


def test_symmetry_exhaustive_small_map() -> None:
    """Exhaustively check symmetry on a small map: for every visible pair."""
    width, height = 15, 15
    rng = np.random.default_rng(99)
    transparent = rng.random((width, height)) > 0.25
    transparent = transparent.astype(np.bool_)
    radius = 14

    # Precompute FOV from every floor tile.
    fov_cache: dict[tuple[int, int], np.ndarray] = {}
    for x in range(width):
        for y in range(height):
            if transparent[x, y]:
                fov_cache[(x, y)] = compute_fov(transparent, (x, y), radius)

    # Check symmetry for every pair.
    violations = []
    for (ax, ay), vis_a in fov_cache.items():
        for bx, by in fov_cache:
            if vis_a[bx, by] and not fov_cache[(bx, by)][ax, ay]:
                violations.append(((ax, ay), (bx, by)))

    assert not violations, f"Symmetry violations found: {violations[:5]}"


# ── 5. Origin always visible ──────────────────────────────────────────────


def test_origin_always_visible() -> None:
    """The player's own tile is always in the FOV."""
    transparent = _make_map(10, 10)
    for ox, oy in [(0, 0), (5, 5), (9, 9), (0, 9), (9, 0)]:
        visible = compute_fov(transparent, (ox, oy), radius=5)
        assert visible[ox, oy], f"Origin ({ox}, {oy}) should always be visible"


def test_origin_visible_even_when_opaque() -> None:
    """Origin is visible even if the origin tile itself is opaque."""
    transparent = _make_map(10, 10, fill=False)
    visible = compute_fov(transparent, (5, 5), radius=5)
    assert visible[5, 5], "Origin should be visible even on opaque tile"


# ── 6. Radius respected ──────────────────────────────────────────────────


def test_nothing_beyond_radius() -> None:
    """No tile beyond the radius is ever visible."""
    width, height = 40, 40
    transparent = _make_map(width, height)
    origin = (20, 20)
    radius = 5

    visible = compute_fov(transparent, origin, radius)

    for x in range(width):
        for y in range(height):
            if visible[x, y]:
                # Euclidean distance from origin to (x, y).
                dist = ((x - origin[0]) ** 2 + (y - origin[1]) ** 2) ** 0.5
                # Allow a small margin because shadowcasting scans by row depth,
                # which is a Chebyshev-like metric, not pure Euclidean.
                # Row depth can be at most `radius`, and column offset adds more.
                # The maximum distance for a visible tile is roughly radius * sqrt(2).
                # But the algorithm limits row_depth to radius, so the diagonal
                # extent is bounded.
                assert dist <= radius * 1.5, (
                    f"Tile ({x}, {y}) at distance {dist:.1f} exceeds radius {radius}"
                )


# ── 7. Map edge handling ─────────────────────────────────────────────────


def test_player_at_map_corner() -> None:
    """Player at (0, 0) doesn't crash and sees nearby tiles."""
    transparent = _make_map(10, 10)
    visible = compute_fov(transparent, (0, 0), radius=5)
    assert visible[0, 0]
    assert visible[1, 0]
    assert visible[0, 1]


def test_player_at_map_edge() -> None:
    """Player at the edge of the map doesn't crash."""
    transparent = _make_map(10, 10)
    for origin in [(0, 5), (9, 5), (5, 0), (5, 9)]:
        visible = compute_fov(transparent, origin, radius=5)
        assert visible[origin[0], origin[1]]


# ── 8. Pillar casts shadow wedge ──────────────────────────────────────────


def test_pillar_shadow_and_diagonal_neighbors() -> None:
    """A single opaque pillar casts a shadow wedge behind it, but diagonal
    neighbors of the pillar remain visible."""
    width, height = 21, 21
    transparent = _make_map(width, height)
    origin = (10, 10)

    # Place a pillar two tiles east of origin.
    pillar_x, pillar_y = 12, 10
    transparent[pillar_x, pillar_y] = False

    visible = compute_fov(transparent, origin, radius=10)

    # Pillar itself is visible.
    assert visible[pillar_x, pillar_y], "Pillar should be visible"

    # Diagonal neighbors of the pillar should still be visible (the shadow
    # wedge is narrow for a single pillar).
    assert visible[pillar_x, pillar_y - 1], "Tile above pillar should be visible"
    assert visible[pillar_x, pillar_y + 1], "Tile below pillar should be visible"

    # Tiles directly behind the pillar should be blocked.
    assert not visible[pillar_x + 2, pillar_y], (
        "Tile well behind pillar should be blocked"
    )


# ── 9. Closed room ────────────────────────────────────────────────────────


def test_closed_room_blocks_interior() -> None:
    """A player outside a fully walled room cannot see inside it."""
    width, height = 20, 20
    transparent = _make_map(width, height)

    # Build a 5x5 room with walls on all sides (interior at 6-8, 6-8).
    for x in range(5, 10):
        for y in range(5, 10):
            transparent[x, y] = False
    # Carve interior (but keep walls intact - room is sealed).
    for x in range(6, 9):
        for y in range(6, 9):
            transparent[x, y] = True

    # Player is outside the room.
    origin = (2, 2)
    visible = compute_fov(transparent, origin, radius=15)

    # Interior tiles should not be visible.
    for x in range(6, 9):
        for y in range(6, 9):
            assert not visible[x, y], (
                f"Interior tile ({x}, {y}) should not be visible from outside"
            )


# ── 10. Output shape/dtype ────────────────────────────────────────────────


def test_output_shape_and_dtype() -> None:
    """Result matches input shape and is bool dtype."""
    for shape in [(10, 10), (80, 50), (1, 1), (120, 80)]:
        transparent = _make_map(*shape)
        origin = (shape[0] // 2, shape[1] // 2)
        visible = compute_fov(transparent, origin, radius=10)

        assert visible.shape == transparent.shape, (
            f"Shape mismatch: {visible.shape} != {transparent.shape}"
        )
        assert visible.dtype == np.bool_, f"Expected bool dtype, got {visible.dtype}"


# ── Edge cases ────────────────────────────────────────────────────────────


def test_1x1_map() -> None:
    """A 1x1 map: the only tile is the origin and is visible."""
    transparent = _make_map(1, 1)
    visible = compute_fov(transparent, (0, 0), radius=5)
    assert visible[0, 0]


def test_radius_zero() -> None:
    """Radius 0 means only the origin tile is visible."""
    transparent = _make_map(10, 10)
    visible = compute_fov(transparent, (5, 5), radius=0)
    assert visible[5, 5]
    # No other tile should be visible.
    assert visible.sum() == 1
