import numpy as np
import pytest

from catley.world import tile_types


def test_get_walkable_map():
    ids = np.array([[tile_types.TILE_TYPE_ID_WALL, tile_types.TILE_TYPE_ID_FLOOR]])  # type: ignore[attr-defined]
    walk = tile_types.get_walkable_map(ids)
    assert walk.tolist() == [[False, True]]


def test_get_tile_type_data_by_id():
    floor = tile_types.get_tile_type_data_by_id(tile_types.TILE_TYPE_ID_FLOOR)  # type: ignore[attr-defined]
    assert floor["walkable"]
    assert floor["transparent"]
    with pytest.raises(IndexError):
        tile_types.get_tile_type_data_by_id(99)


def test_boulder_properties() -> None:
    boulder = tile_types.get_tile_type_data_by_id(tile_types.TILE_TYPE_ID_BOULDER)  # type: ignore[attr-defined]
    assert boulder["transparent"]
    assert boulder["casts_shadows"]
    assert boulder["cover_bonus"] == 2
