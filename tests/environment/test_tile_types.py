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
    assert boulder["shadow_height"] == 2
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


def test_get_hazard_cost_safe_tiles() -> None:
    """Safe tiles should return cost of 1."""
    assert tile_types.get_hazard_cost(TileTypeID.FLOOR) == 1
    assert tile_types.get_hazard_cost(TileTypeID.WALL) == 1


def test_get_hazard_cost_hazardous_tiles() -> None:
    """Hazardous tiles should return cost > 1 based on damage."""
    # ACID_POOL does 1d4 (avg 2.5), cost = 5 + 2.5*2 = 10
    assert tile_types.get_hazard_cost(TileTypeID.ACID_POOL) == 10

    # HOT_COALS does 1d6 (avg 3.5), cost = 5 + 3.5*2 = 12
    assert tile_types.get_hazard_cost(TileTypeID.HOT_COALS) == 12


def test_get_hazard_cost_invalid_id() -> None:
    """Invalid tile IDs should return safe cost of 1."""
    assert tile_types.get_hazard_cost(999) == 1


def test_get_hazard_cost_map_vectorized() -> None:
    """Test vectorized hazard cost lookup."""
    ids = np.array(
        [
            [TileTypeID.FLOOR, TileTypeID.ACID_POOL],
            [TileTypeID.WALL, TileTypeID.HOT_COALS],
        ]
    )
    costs = tile_types.get_hazard_cost_map(ids)

    assert costs.shape == ids.shape
    assert costs[0, 0] == 1  # FLOOR
    assert costs[0, 1] == 10  # ACID_POOL
    assert costs[1, 0] == 1  # WALL
    assert costs[1, 1] == 12  # HOT_COALS


# --- Shadow Casting Tests ---


def test_get_shadow_height_map_basic() -> None:
    """Test vectorized shadow height lookup for basic tile types."""
    # Create a 2D array of tile IDs
    ids = np.array(
        [
            [TileTypeID.FLOOR, TileTypeID.WALL],
            [TileTypeID.BOULDER, TileTypeID.OUTDOOR_WALL],
        ]
    )
    heights = tile_types.get_shadow_height_map(ids)

    # Check expected shadow heights
    assert heights.shape == ids.shape
    assert heights[0, 0] == 0  # FLOOR - no shadow
    assert heights[0, 1] == 4  # WALL - tall shadow
    assert heights[1, 0] == 2  # BOULDER - short shadow
    assert heights[1, 1] == 4  # OUTDOOR_WALL - tall shadow


def test_get_shadow_height_map_all_tiles() -> None:
    """Ensure get_shadow_height_map works for all tile types."""
    all_ids = np.array([list(TileTypeID)])
    heights = tile_types.get_shadow_height_map(all_ids)

    # Verify each result matches the individual lookup
    for i, tile_id in enumerate(TileTypeID):
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert heights[0, i] == data["shadow_height"], (
            f"Mismatch for {tile_id.name}: vectorized={heights[0, i]}, "
            f"individual={data['shadow_height']}"
        )


def test_get_shadow_height_map_empty_array() -> None:
    """Empty arrays should return empty results."""
    ids = np.array([], dtype=np.uint8).reshape(0, 0)
    heights = tile_types.get_shadow_height_map(ids)
    assert heights.shape == (0, 0)


def test_shadow_height_tiles_are_correct() -> None:
    """Verify expected shadow heights for all tile types."""
    # Tiles with specific shadow heights
    expected_heights: dict[TileTypeID, int] = {
        TileTypeID.WALL: 4,
        TileTypeID.OUTDOOR_WALL: 4,
        TileTypeID.DOOR_CLOSED: 4,  # Door is set in a wall - same shadow as building
        TileTypeID.DOOR_OPEN: 4,  # Doorframe and wall above still cast building shadow
        TileTypeID.BOULDER: 2,
    }

    # These tiles should have zero shadow height
    no_shadow_tiles = [
        TileTypeID.FLOOR,
        TileTypeID.OUTDOOR_FLOOR,
        TileTypeID.ACID_POOL,
        TileTypeID.HOT_COALS,
    ]

    for tile_id, expected in expected_heights.items():
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert data["shadow_height"] == expected, (
            f"{tile_id.name} should have shadow_height={expected}, "
            f"got {data['shadow_height']}"
        )

    for tile_id in no_shadow_tiles:
        data = tile_types.get_tile_type_data_by_id(tile_id)
        assert data["shadow_height"] == 0, f"{tile_id.name} should have shadow_height=0"


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
