import numpy as np
import pytest

from catley.environment import tile_types
from catley.environment.tile_types import TileTypeID, get_tile_material
from catley.game.enums import ImpactMaterial


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


def test_tile_material_field() -> None:
    """Test that tiles have a material field for impact sounds."""
    wall = tile_types.get_tile_type_data_by_id(TileTypeID.WALL)
    field_names = wall.dtype.names
    assert field_names is not None, "TileTypeData must have named fields"
    assert "material" in field_names, "TileTypeData must have material field"


def test_get_tile_material() -> None:
    """Test the get_tile_material helper function."""
    # Doors are wood
    assert get_tile_material(TileTypeID.DOOR_CLOSED) == ImpactMaterial.WOOD
    assert get_tile_material(TileTypeID.DOOR_OPEN) == ImpactMaterial.WOOD

    # Walls and floors are stone
    assert get_tile_material(TileTypeID.WALL) == ImpactMaterial.STONE
    assert get_tile_material(TileTypeID.FLOOR) == ImpactMaterial.STONE
    assert get_tile_material(TileTypeID.BOULDER) == ImpactMaterial.STONE


def test_all_tiles_have_valid_material() -> None:
    """Ensure every tile type has a valid ImpactMaterial."""
    for tile_id in TileTypeID:
        material = get_tile_material(tile_id)
        assert isinstance(material, ImpactMaterial), (
            f"Tile {tile_id.name} has invalid material"
        )
