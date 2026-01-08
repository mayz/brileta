import random

from catley.environment.generators import RoomsAndCorridorsGenerator
from catley.environment.map import GameMap
from catley.environment.tile_types import TileTypeID


def test_make_map_places_doors() -> None:
    random.seed(0)
    generator = RoomsAndCorridorsGenerator(
        40, 30, max_rooms=5, min_room_size=5, max_room_size=8
    )
    map_data = generator.generate()
    gm = GameMap(40, 30, map_data)
    num_doors = int((gm.tiles == TileTypeID.DOOR_CLOSED).sum())
    assert num_doors > 0
