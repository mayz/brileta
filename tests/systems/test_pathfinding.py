from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap
from catley.environment.tile_types import TileTypeID

if TYPE_CHECKING:
    from catley.environment.map import MapRegion
from catley.game.actors import Actor
from catley.util.pathfinding import find_local_path
from catley.util.spatial import SpatialHashGrid, SpatialIndex


@dataclass
class DummyActor:
    x: int
    y: int
    blocks_movement: bool = True


@pytest.fixture
def basic_setup() -> tuple[GameMap, SpatialHashGrid[DummyActor]]:
    tiles = np.full((10, 10), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions: dict[int, MapRegion] = {}
    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=np.full((10, 10), -1, dtype=np.int16, order="F"),
    )
    gm = GameMap(10, 10, map_data)
    gm.invalidate_property_caches()
    index: SpatialHashGrid[DummyActor] = SpatialHashGrid(cell_size=16)
    return gm, index


def test_find_path_simple_straight_line(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


def test_find_path_simple_diagonal_line(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (4, 4)
    )
    assert path == [(2, 2), (3, 3), (4, 4)]


def test_find_path_no_path_exists(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            gm.tiles[5 + dx, 5 + dy] = TileTypeID.WALL
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (5, 5)
    )
    assert path == []


def test_find_path_avoids_static_wall(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    gm.tiles[1, 3] = TileTypeID.WALL
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )
    assert path
    assert path[-1] == (1, 5)
    assert (1, 3) not in path


def test_find_path_ignores_non_blocking_actor(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    actor = DummyActor(1, 3, False)
    index.add(actor)
    path = find_local_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


def test_find_path_start_and_end_are_same(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    actor = DummyActor(3, 3)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (3, 3), (3, 3)
    )
    assert path == []


def test_find_path_target_is_impassable(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    gm.tiles[1, 5] = TileTypeID.WALL
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )
    assert path == []


def test_distant_blocking_actor_ignored(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    distant = DummyActor(8, 8)
    index.add(distant)
    path = find_local_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


def test_blocking_actors_ignored_for_pathfinding(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """Blocking actors are intentionally ignored during pathfinding.

    Collisions with moving actors are handled at movement execution time
    via the collision detection system, not during path planning.
    """
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    other = DummyActor(1, 3)
    index.add(pathing)
    index.add(other)
    path = find_local_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    # Path goes straight through - blocking actors don't affect pathfinding
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


# --- Hazard Avoidance Tests ---


def test_find_path_avoids_acid_pool(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """AI prefers paths that avoid acid pools when alternatives exist."""
    gm, index = basic_setup
    # Place acid pool on direct path
    gm.tiles[1, 3] = TileTypeID.ACID_POOL
    gm.invalidate_property_caches()

    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )

    assert path
    assert path[-1] == (1, 5)
    assert (1, 3) not in path  # Avoided the acid


def test_find_path_avoids_hot_coals(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """AI prefers paths that avoid hot coals when alternatives exist."""
    gm, index = basic_setup
    # Place hot coals on direct path
    gm.tiles[1, 3] = TileTypeID.HOT_COALS
    gm.invalidate_property_caches()

    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )

    assert path
    assert path[-1] == (1, 5)
    assert (1, 3) not in path  # Avoided the coals


def test_find_path_through_hazard_when_no_alternative(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """AI walks through hazard when it's the only path."""
    gm, index = basic_setup
    # Create a corridor with acid in the middle (no way around)
    # Block all adjacent columns to force through the hazard
    for x in range(10):
        if x != 1:
            gm.tiles[x, 3] = TileTypeID.WALL
    gm.tiles[1, 3] = TileTypeID.ACID_POOL
    gm.invalidate_property_caches()

    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )

    assert path
    assert (1, 3) in path  # Had to go through acid
    assert path[-1] == (1, 5)


@dataclass
class DummyFire:
    """A mock fire actor for testing hazard avoidance."""

    x: int
    y: int
    blocks_movement: bool = False
    damage_per_turn: int = 5


def test_find_path_avoids_fire_actor(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """AI avoids fire actors (campfires, etc.)."""
    gm, _index = basic_setup

    fire = DummyFire(1, 3)
    # Use a fresh index that can hold DummyFire
    fire_index: SpatialHashGrid[DummyFire] = SpatialHashGrid(cell_size=16)
    fire_index.add(fire)

    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], fire_index), cast(Actor, actor), (1, 1), (1, 5)
    )

    assert path
    assert path[-1] == (1, 5)
    assert (1, 3) not in path  # Avoided the fire


def test_find_path_through_fire_when_no_alternative(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    """AI walks through fire when it's the only path."""
    gm, _index = basic_setup
    # Create a corridor with fire in the middle
    for x in range(10):
        if x != 1:
            gm.tiles[x, 3] = TileTypeID.WALL
    gm.invalidate_property_caches()

    fire = DummyFire(1, 3)
    fire_index: SpatialHashGrid[DummyFire] = SpatialHashGrid(cell_size=16)
    fire_index.add(fire)

    actor = DummyActor(1, 1)
    path = find_local_path(
        gm, cast(SpatialIndex[Actor], fire_index), cast(Actor, actor), (1, 1), (1, 5)
    )

    assert path
    assert (1, 3) in path  # Had to go through fire
    assert path[-1] == (1, 5)
