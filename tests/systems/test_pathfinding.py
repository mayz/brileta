from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from catley.environment import tile_types
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap

if TYPE_CHECKING:
    from catley.environment.map import MapRegion
from catley.game.actors import Actor
from catley.util.pathfinding import find_path
from catley.util.spatial import SpatialHashGrid, SpatialIndex


@dataclass
class DummyActor:
    x: int
    y: int
    blocks_movement: bool = True


@pytest.fixture
def basic_setup() -> tuple[GameMap, SpatialHashGrid[DummyActor]]:
    tiles = np.full((10, 10), tile_types.TILE_TYPE_ID_FLOOR, dtype=np.uint8, order="F")  # type: ignore[attr-defined]
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
    path = find_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


def test_find_path_simple_diagonal_line(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    actor = DummyActor(1, 1)
    path = find_path(
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
            gm.tiles[5 + dx, 5 + dy] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (5, 5)
    )
    assert path == []


def test_find_path_avoids_static_wall(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    gm.tiles[1, 3] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (1, 1), (1, 5)
    )
    assert path
    assert path[-1] == (1, 5)
    assert (1, 3) not in path


def test_find_path_avoids_blocking_actor(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    actor = DummyActor(1, 3, True)
    index.add(actor)
    path = find_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    assert path
    assert (1, 3) not in path
    assert path[-1] == (1, 5)


def test_find_path_ignores_non_blocking_actor(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    actor = DummyActor(1, 3, False)
    index.add(actor)
    path = find_path(
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
    path = find_path(
        gm, cast(SpatialIndex[Actor], index), cast(Actor, actor), (3, 3), (3, 3)
    )
    assert path == []


def test_find_path_target_is_impassable(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    gm.tiles[1, 5] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
    gm.invalidate_property_caches()
    actor = DummyActor(1, 1)
    path = find_path(
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
    path = find_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    assert path == [(1, 2), (1, 3), (1, 4), (1, 5)]


def test_pathing_actor_is_ignored(
    basic_setup: tuple[GameMap, SpatialHashGrid[DummyActor]],
) -> None:
    gm, index = basic_setup
    pathing = DummyActor(1, 1)
    other = DummyActor(1, 3)
    index.add(pathing)
    index.add(other)
    path = find_path(
        gm,
        cast(SpatialIndex[Actor], index),
        cast(Actor, pathing),
        (1, 1),
        (1, 5),
    )
    assert path
    assert (1, 3) not in path
    assert path[-1] == (1, 5)
