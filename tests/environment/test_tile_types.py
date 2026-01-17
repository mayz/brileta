import numpy as np
import pytest

from catley.environment import tile_types
from catley.environment.tile_types import (
    TileTypeID,
    get_tile_hazard_info,
    get_tile_material,
)
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


# --- Hazardous Terrain Tests ---


def test_get_tile_hazard_info_safe_tiles() -> None:
    """Safe tiles should return empty dice string."""
    damage_dice, damage_type = get_tile_hazard_info(TileTypeID.FLOOR)
    assert damage_dice == ""
    assert damage_type == ""

    damage_dice, damage_type = get_tile_hazard_info(TileTypeID.WALL)
    assert damage_dice == ""
    assert damage_type == ""


def test_get_tile_hazard_info_acid_pool() -> None:
    """Acid pool should deal acid damage with dice roll."""
    damage_dice, damage_type = get_tile_hazard_info(TileTypeID.ACID_POOL)
    assert damage_dice == "1d4"
    assert damage_type == "acid"


def test_get_tile_hazard_info_hot_coals() -> None:
    """Hot coals should deal fire damage with dice roll."""
    damage_dice, damage_type = get_tile_hazard_info(TileTypeID.HOT_COALS)
    assert damage_dice == "1d6"
    assert damage_type == "fire"


def test_get_tile_hazard_info_invalid_id() -> None:
    """Invalid tile IDs should return safe values."""
    damage_dice, damage_type = get_tile_hazard_info(999)
    assert damage_dice == ""
    assert damage_type == ""


def test_hazardous_tile_properties() -> None:
    """Hazardous tiles should be walkable and transparent."""
    acid = tile_types.get_tile_type_data_by_id(TileTypeID.ACID_POOL)
    assert acid["walkable"]
    assert acid["transparent"]
    assert acid["hazard_damage"] == "1d4"
    assert acid["hazard_damage_type"] == "acid"

    coals = tile_types.get_tile_type_data_by_id(TileTypeID.HOT_COALS)
    assert coals["walkable"]
    assert coals["transparent"]
    assert coals["hazard_damage"] == "1d6"
    assert coals["hazard_damage_type"] == "fire"


def test_all_tiles_have_hazard_fields() -> None:
    """Ensure every tile type has hazard fields (even if zero)."""
    for tile_id in TileTypeID:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        field_names = data.dtype.names
        assert field_names is not None
        assert "hazard_damage" in field_names
        assert "hazard_damage_type" in field_names


# --- Shadow Casting Tests ---


def test_get_casts_shadows_map_basic() -> None:
    """Test vectorized shadow-casting lookup for basic tile types."""
    # Create a 2D array of tile IDs
    ids = np.array(
        [
            [TileTypeID.FLOOR, TileTypeID.WALL],
            [TileTypeID.BOULDER, TileTypeID.OUTDOOR_WALL],
        ]
    )
    shadows = tile_types.get_casts_shadows_map(ids)

    # Floor and Wall don't cast shadows, Boulder and Outdoor Wall do
    assert shadows.shape == ids.shape
    assert shadows[0, 0] is np.False_  # FLOOR
    assert shadows[0, 1] is np.False_  # WALL
    assert shadows[1, 0] is np.True_  # BOULDER
    assert shadows[1, 1] is np.True_  # OUTDOOR_WALL


def test_get_casts_shadows_map_all_tiles() -> None:
    """Ensure get_casts_shadows_map works for all tile types."""
    all_ids = np.array([list(TileTypeID)])
    shadows = tile_types.get_casts_shadows_map(all_ids)

    # Verify each result matches the individual lookup
    for i, tile_id in enumerate(TileTypeID):
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert shadows[0, i] == data["casts_shadows"], (
            f"Mismatch for {tile_id.name}: vectorized={shadows[0, i]}, "
            f"individual={data['casts_shadows']}"
        )


def test_get_casts_shadows_map_empty_array() -> None:
    """Empty arrays should return empty results."""
    ids = np.array([], dtype=np.uint8).reshape(0, 0)
    shadows = tile_types.get_casts_shadows_map(ids)
    assert shadows.shape == (0, 0)


def test_shadow_casting_tiles_are_correct() -> None:
    """Verify which tiles are expected to cast shadows."""
    # These tiles should cast shadows (from tile_types.py definitions)
    shadow_casters = [TileTypeID.OUTDOOR_WALL, TileTypeID.BOULDER]

    # These tiles should NOT cast shadows
    non_shadow_casters = [
        TileTypeID.FLOOR,
        TileTypeID.WALL,
        TileTypeID.OUTDOOR_FLOOR,
        TileTypeID.DOOR_CLOSED,
        TileTypeID.DOOR_OPEN,
        TileTypeID.ACID_POOL,
        TileTypeID.HOT_COALS,
    ]

    for tile_id in shadow_casters:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert data["casts_shadows"], f"{tile_id.name} should cast shadows"

    for tile_id in non_shadow_casters:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert not data["casts_shadows"], f"{tile_id.name} should not cast shadows"


# --- Tile Light Emission Tests ---


def test_get_emission_map_basic() -> None:
    """Test vectorized emission lookup for basic tile types."""
    ids = np.array([[TileTypeID.FLOOR, TileTypeID.ACID_POOL]])
    emissions = tile_types.get_emission_map(ids)

    assert not emissions[0, 0]["emits_light"]  # FLOOR doesn't emit
    assert emissions[0, 1]["emits_light"]  # ACID_POOL does emit
    assert emissions[0, 1]["light_radius"] == 2
    assert emissions[0, 1]["light_intensity"] == 0.5


def test_emission_properties_acid_pool() -> None:
    """Acid pool should emit green light."""
    data = tile_types.get_tile_type_data_by_id(TileTypeID.ACID_POOL)
    emission = data["emission"]
    assert emission["emits_light"]
    assert tuple(emission["light_color"]) == (80, 180, 80)  # Green
    assert emission["light_radius"] == 2
    assert emission["light_intensity"] == 0.5


def test_emission_properties_hot_coals() -> None:
    """Hot coals should emit orange/red light."""
    data = tile_types.get_tile_type_data_by_id(TileTypeID.HOT_COALS)
    emission = data["emission"]
    assert emission["emits_light"]
    assert tuple(emission["light_color"]) == (180, 100, 40)  # Orange
    assert emission["light_radius"] == 2
    assert emission["light_intensity"] == 0.5


def test_non_emitting_tiles() -> None:
    """Most tiles should not emit light."""
    non_emitters = [TileTypeID.FLOOR, TileTypeID.WALL, TileTypeID.DOOR_CLOSED]
    for tile_id in non_emitters:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert not data["emission"]["emits_light"], (
            f"{tile_id.name} should not emit light"
        )


def test_all_tiles_have_emission_field() -> None:
    """Ensure every tile type has emission field (even if not emitting)."""
    for tile_id in TileTypeID:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        field_names = data.dtype.names
        assert field_names is not None
        assert "emission" in field_names


def test_get_emission_map_all_tiles() -> None:
    """Ensure get_emission_map works for all tile types."""
    all_ids = np.array([list(TileTypeID)])
    emissions = tile_types.get_emission_map(all_ids)

    # Verify each result matches the individual lookup
    for i, tile_id in enumerate(TileTypeID):
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert emissions[0, i]["emits_light"] == data["emission"]["emits_light"], (
            f"Mismatch for {tile_id.name}"
        )
