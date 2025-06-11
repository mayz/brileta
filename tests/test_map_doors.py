import random

from catley.environment import tile_types
from catley.environment.map import GameMap


def test_make_map_places_doors() -> None:
    random.seed(0)
    gm = GameMap(40, 30)
    gm.make_map(5, 5, 8)
    num_doors = int((gm.tiles == tile_types.TILE_TYPE_ID_DOOR_CLOSED).sum())  # type: ignore[attr-defined]
    assert num_doors > 0
