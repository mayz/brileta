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
            [TileTypeID.BOULDER, TileTypeID.DOOR_CLOSED],
        ]
    )
    heights = tile_types.get_shadow_height_map(ids)

    # Check expected shadow heights
    assert heights.shape == ids.shape
    assert heights[0, 0] == 0  # FLOOR - no shadow
    assert heights[0, 1] == 4  # WALL - tall shadow
    assert heights[1, 0] == 2  # BOULDER - short shadow
    assert heights[1, 1] == 4  # DOOR_CLOSED - tall shadow


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
        TileTypeID.DOOR_CLOSED: 4,  # Door is set in a wall - same shadow as building
        TileTypeID.DOOR_OPEN: 4,  # Doorframe and wall above still cast building shadow
        TileTypeID.BOULDER: 2,
    }

    # These tiles should have zero shadow height
    no_shadow_tiles = [
        TileTypeID.FLOOR,
        TileTypeID.COBBLESTONE,
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


# --- Sub-tile Noise Tests ---


def test_cobblestone_has_nonzero_sub_tile_jitter() -> None:
    """Cobblestone should have a nonzero sub-tile jitter amplitude."""
    ids = np.array([TileTypeID.COBBLESTONE])
    noise = tile_types.get_sub_tile_jitter_map(ids)
    assert noise[0] > 0.0


def test_floor_has_zero_sub_tile_jitter() -> None:
    """Floor tiles should have zero sub-tile jitter (smooth flagstone)."""
    ids = np.array([TileTypeID.FLOOR])
    jitter = tile_types.get_sub_tile_jitter_map(ids)
    assert jitter[0] == 0.0


def test_wall_has_nonzero_sub_tile_jitter() -> None:
    """Wall tiles should have subtle sub-tile jitter (stone block grain)."""
    ids = np.array([TileTypeID.WALL])
    jitter = tile_types.get_sub_tile_jitter_map(ids)
    assert jitter[0] > 0.0


# --- Terrain Decoration Tests ---


def test_decoration_deterministic() -> None:
    """Same world coords and seed always produce the same glyph and colors."""
    n = 10
    tile_ids = np.full(n, TileTypeID.GRASS, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)
    seed = 42

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        chars = np.zeros(n, dtype=np.int32)
        fg = np.full((n, 3), 100, dtype=np.uint8)
        bg = np.full((n, 3), 50, dtype=np.uint8)
        tile_types.apply_terrain_decoration(
            chars, fg, bg, tile_ids, world_x, world_y, seed
        )
        return chars.copy(), fg.copy(), bg.copy()

    ch1, fg1, bg1 = run()
    ch2, fg2, bg2 = run()
    np.testing.assert_array_equal(ch1, ch2)
    np.testing.assert_array_equal(fg1, fg2)
    np.testing.assert_array_equal(bg1, bg2)


def test_decoration_different_seed_differs() -> None:
    """Different decoration seeds produce different results."""
    n = 20
    tile_ids = np.full(n, TileTypeID.GRASS, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)

    chars_a = np.zeros(n, dtype=np.int32)
    fg_a = np.full((n, 3), 100, dtype=np.uint8)
    bg_a = np.full((n, 3), 50, dtype=np.uint8)
    tile_types.apply_terrain_decoration(
        chars_a, fg_a, bg_a, tile_ids, world_x, world_y, decoration_seed=1
    )

    chars_b = np.zeros(n, dtype=np.int32)
    fg_b = np.full((n, 3), 100, dtype=np.uint8)
    bg_b = np.full((n, 3), 50, dtype=np.uint8)
    tile_types.apply_terrain_decoration(
        chars_b, fg_b, bg_b, tile_ids, world_x, world_y, decoration_seed=9999
    )

    # At least some outputs should differ across 20 tiles with different seeds.
    assert not np.array_equal(bg_a, bg_b) or not np.array_equal(chars_a, chars_b)


def test_decoration_jitter_stays_in_bounds() -> None:
    """Output colors stay in [0, 255] even at extreme base values and max jitter."""
    n = 200
    # Use GRASS (bg_jitter=4, fg_jitter=12) for the widest jitter range.
    tile_ids = np.full(n, TileTypeID.GRASS, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)

    for base_val in (0, 3, 252, 255):
        chars = np.zeros(n, dtype=np.int32)
        fg = np.full((n, 3), base_val, dtype=np.uint8)
        bg = np.full((n, 3), base_val, dtype=np.uint8)
        tile_types.apply_terrain_decoration(
            chars, fg, bg, tile_ids, world_x, world_y, decoration_seed=123
        )
        assert fg.min() >= 0 and fg.max() <= 255
        assert bg.min() >= 0 and bg.max() <= 255


def test_decoration_skips_unregistered_tiles() -> None:
    """Tiles without a TerrainDecorationDef entry are left untouched."""
    n = 5
    # FLOOR has no decoration def (removed when defaults were all zero).
    tile_ids = np.full(n, TileTypeID.FLOOR, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)

    chars = np.full(n, ord("X"), dtype=np.int32)
    fg = np.full((n, 3), 100, dtype=np.uint8)
    bg = np.full((n, 3), 50, dtype=np.uint8)
    original_chars = chars.copy()
    original_fg = fg.copy()
    original_bg = bg.copy()

    tile_types.apply_terrain_decoration(
        chars, fg, bg, tile_ids, world_x, world_y, decoration_seed=42
    )

    np.testing.assert_array_equal(chars, original_chars)
    np.testing.assert_array_equal(fg, original_fg)
    np.testing.assert_array_equal(bg, original_bg)


def test_decoration_independent_fg_jitter() -> None:
    """When fg_jitter > 0, fg and bg offsets diverge (not locked together)."""
    n = 100
    # GRASS has fg_jitter=12, so fg gets independent hash bits.
    tile_ids = np.full(n, TileTypeID.GRASS, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)

    chars = np.zeros(n, dtype=np.int32)
    fg = np.full((n, 3), 128, dtype=np.uint8)
    bg = np.full((n, 3), 128, dtype=np.uint8)
    tile_types.apply_terrain_decoration(
        chars, fg, bg, tile_ids, world_x, world_y, decoration_seed=7
    )

    # With independent jitter, fg and bg offsets from the same base (128)
    # should differ for at least some tiles across 100 samples.
    fg_offsets = fg[:, 0].astype(np.int16) - 128
    bg_offsets = bg[:, 0].astype(np.int16) - 128
    assert not np.array_equal(fg_offsets, bg_offsets)


def test_decoration_locked_fg_when_zero_fg_jitter() -> None:
    """When fg_jitter == 0, fg gets the same offset as bg."""
    n = 50
    # COBBLESTONE has fg_jitter=0, so fg is locked to bg offset.
    tile_ids = np.full(n, TileTypeID.COBBLESTONE, dtype=np.uint8)
    world_x = np.arange(n, dtype=np.int32)
    world_y = np.zeros(n, dtype=np.int32)

    chars = np.zeros(n, dtype=np.int32)
    fg = np.full((n, 3), 128, dtype=np.uint8)
    bg = np.full((n, 3), 128, dtype=np.uint8)
    tile_types.apply_terrain_decoration(
        chars, fg, bg, tile_ids, world_x, world_y, decoration_seed=7
    )

    # fg and bg should have identical offsets from the shared base.
    fg_offsets = fg[:, 0].astype(np.int16) - 128
    bg_offsets = bg[:, 0].astype(np.int16) - 128
    np.testing.assert_array_equal(fg_offsets, bg_offsets)
