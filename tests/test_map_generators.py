from __future__ import annotations

import random

import numpy as np

from catley.environment import tile_types
from catley.environment.generators import RoomsAndCorridorsGenerator


def test_room_region_creation() -> None:
    random.seed(0)
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    room_regions = [r for r in data.regions.values() if r.region_type == "room"]
    assert len(room_regions) == 4

    first_room = room_regions[0]
    room_rect = first_room.bounds[0]
    slice_x = slice(room_rect.x1 + 1, room_rect.x2)
    slice_y = slice(room_rect.y1 + 1, room_rect.y2)
    assert np.all(data.tile_to_region_id[slice_x, slice_y] == first_room.id)


def test_hallway_region_identification() -> None:
    random.seed(0)
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    hallway_regions = [r for r in data.regions.values() if r.region_type == "hallway"]
    assert len(hallway_regions) >= 1

    # Known vertical hallway for seed 0
    region = data.regions[5]
    bounds = region.bounds[0]
    slice_x = slice(bounds.x1, bounds.x1 + 1)
    slice_y = slice(bounds.y1, bounds.y2)
    assert np.all(data.tile_to_region_id[slice_x, slice_y] == region.id)


def test_door_connection_logic() -> None:
    random.seed(0)
    gen = RoomsAndCorridorsGenerator(
        20, 20, max_rooms=2, min_room_size=5, max_room_size=7
    )
    data = gen.generate()

    room_ids = [r.id for r in data.regions.values() if r.region_type == "room"]
    hallway_id = next(r.id for r in data.regions.values() if r.region_type == "hallway")

    room1_id = room_ids[0]
    door_tile = data.regions[room1_id].connections[hallway_id]
    assert data.regions[hallway_id].connections[room1_id] == door_tile


def test_no_unassigned_walkable_tiles() -> None:
    random.seed(0)
    gen = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    data = gen.generate()

    for x in range(data.tiles.shape[0]):
        for y in range(data.tiles.shape[1]):
            tile = data.tiles[x, y]
            if tile in (
                tile_types.TILE_TYPE_ID_FLOOR,  # type: ignore[attr-defined]
                tile_types.TILE_TYPE_ID_DOOR_OPEN,  # type: ignore[attr-defined]
            ):
                assert data.tile_to_region_id[x, y] != -1
