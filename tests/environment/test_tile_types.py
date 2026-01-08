import numpy as np
import pytest

from catley.environment import tile_types
from catley.environment.tile_types import TileTypeID


def test_get_walkable_map():
    ids = np.array([[TileTypeID.WALL, TileTypeID.FLOOR]])
    walk = tile_types.get_walkable_map(ids)
    assert walk.tolist() == [[False, True]]


def test_get_tile_type_data_by_id():
    floor = tile_types.get_tile_type_data_by_id(TileTypeID.FLOOR)
    assert floor["walkable"]
    assert floor["transparent"]
    with pytest.raises(IndexError):
        tile_types.get_tile_type_data_by_id(99)


def test_boulder_properties() -> None:
    boulder = tile_types.get_tile_type_data_by_id(TileTypeID.BOULDER)
    assert boulder["transparent"]
    assert boulder["casts_shadows"]
    assert boulder["cover_bonus"] == 2


def test_all_tile_types_have_registered_data() -> None:
    """Ensure every TileTypeID enum value has corresponding data registered."""
    for tile_id in TileTypeID:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert data is not None, f"No data registered for {tile_id.name}"


def test_tile_type_id_works_as_numpy_index() -> None:
    """Ensure IntEnum values work directly with numpy arrays."""
    tiles = np.zeros((3, 3), dtype=np.uint8)
    tiles[1, 1] = TileTypeID.FLOOR  # Should work without casting
    assert tiles[1, 1] == TileTypeID.FLOOR
