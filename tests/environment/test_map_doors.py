import random

from catley.environment import tile_types
from catley.environment.generators import RoomsAndCorridorsGenerator
from catley.environment.map import GameMap


def test_make_map_places_doors() -> None:
    random.seed(0)
    generator = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    map_data = generator.generate()
    gm = GameMap(40, 30, map_data)
    num_doors = int((gm.tiles == tile_types.TILE_TYPE_ID_DOOR_CLOSED).sum())  # type: ignore[attr-defined]
    assert num_doors > 0
