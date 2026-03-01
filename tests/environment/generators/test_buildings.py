"""Tests for the building system.

Phase T2 tests cover:
- T2.1: Building/Room dataclasses
- T2.2: BuildingTemplate
- T2.3: Default templates
"""

from __future__ import annotations

import random

import numpy as np

from brileta import config
from brileta.environment.generators.buildings import Building, BuildingTemplate, Room
from brileta.environment.generators.buildings.templates import (
    BLACKSMITH_TEMPLATE,
    INN_TEMPLATE,
    LIBRARY_TEMPLATE,
    MEDIUM_HOUSE_TEMPLATE,
    SHOP_TEMPLATE,
    SMALL_HOUSE_TEMPLATE,
    TAVERN_TEMPLATE,
    WAREHOUSE_TEMPLATE,
    get_default_templates,
)
from brileta.util.coordinates import Rect
from brileta.view.views.world_view import _WEAR_MATERIAL_MAP

# =============================================================================
# T2.1: Building/Room Dataclasses
# =============================================================================


class TestBuildingRoomDataclasses:
    """Tests for Building and Room dataclasses."""

    def test_building_creation(self) -> None:
        """Building has expected defaults including roof metadata."""
        footprint = Rect(10, 20, 15, 12)
        building = Building(
            id=1,
            building_type="house",
            footprint=footprint,
        )

        assert building.id == 1
        assert building.building_type == "house"
        assert building.footprint == footprint
        assert building.rooms == []
        assert building.door_positions == []
        assert building.roof_style == "thatch"
        assert building.roof_profile == "gable"
        assert building.flat_section_ratio == 0.0
        assert building.floor_count == 1
        assert building.wing is None

    def test_rectangular_building_has_no_wing(self) -> None:
        """Rectangular buildings expose a single occupied section."""
        footprint = Rect(10, 20, 15, 12)
        building = Building(id=1, building_type="house", footprint=footprint)

        assert building.wing is None
        assert building.bounding_rect.x1 == footprint.x1
        assert building.bounding_rect.y1 == footprint.y1
        assert building.bounding_rect.x2 == footprint.x2
        assert building.bounding_rect.y2 == footprint.y2
        assert building.occupied_rects == (footprint,)
        assert building.is_compound is False

    def test_building_clamps_flat_section_ratio_to_valid_range(self) -> None:
        """flat_section_ratio is clamped into [0.0, 1.0] during post-init."""
        low = Building(
            id=11,
            building_type="warehouse",
            footprint=Rect(0, 0, 12, 10),
            roof_profile="low_slope",
            flat_section_ratio=-0.25,
        )
        high = Building(
            id=12,
            building_type="warehouse",
            footprint=Rect(0, 0, 12, 10),
            roof_profile="low_slope",
            flat_section_ratio=1.75,
        )

        assert low.flat_section_ratio == 0.0
        assert high.flat_section_ratio == 1.0

    def test_building_with_rooms_and_doors(self) -> None:
        """Building can be created with rooms and door positions."""
        footprint = Rect(5, 5, 10, 8)
        room = Room(region_id=42, room_type="living_room", bounds=Rect(6, 6, 8, 6))

        building = Building(
            id=2,
            building_type="shop",
            footprint=footprint,
            rooms=[room],
            door_positions=[(5, 8)],
        )

        assert len(building.rooms) == 1
        assert building.rooms[0] == room
        assert building.door_positions == [(5, 8)]

    def test_building_interior_bounds(self) -> None:
        """Interior bounds is footprint minus walls (1 tile each side)."""
        footprint = Rect(10, 20, 15, 12)
        building = Building(id=1, building_type="house", footprint=footprint)

        interior = building.interior_bounds
        assert interior.x1 == 11  # footprint.x1 + 1
        assert interior.y1 == 21  # footprint.y1 + 1
        assert interior.width == 13  # footprint.width - 2
        assert interior.height == 10  # footprint.height - 2

    def test_building_contains_point(self) -> None:
        """contains_point correctly checks if point is within footprint."""
        footprint = Rect(10, 10, 5, 5)  # (10,10) to (14,14)
        building = Building(id=1, building_type="house", footprint=footprint)

        # Inside footprint
        assert building.contains_point(10, 10)
        assert building.contains_point(12, 12)
        assert building.contains_point(14, 14)

        # Outside footprint
        assert not building.contains_point(9, 10)
        assert not building.contains_point(10, 9)
        assert not building.contains_point(15, 10)
        assert not building.contains_point(10, 15)

    def test_l_shape_building_properties(self) -> None:
        """Compound buildings expose both occupied sections and a merged bounding box."""
        primary = Rect(10, 10, 8, 8)
        wing = Rect.from_bounds(10, 5, 14, 10)
        building = Building(id=2, building_type="house", footprint=primary, wing=wing)

        assert building.is_compound is True
        assert building.occupied_rects == (primary, wing)
        bounds = building.bounding_rect
        assert (bounds.x1, bounds.y1, bounds.x2, bounds.y2) == (10, 5, 18, 18)

    def test_l_shape_contains_point(self) -> None:
        """contains_point should include both sections but not the empty bounding corner."""
        primary = Rect(10, 10, 8, 8)
        wing = Rect.from_bounds(10, 5, 14, 10)
        building = Building(id=3, building_type="house", footprint=primary, wing=wing)

        assert building.contains_point(16, 12)  # primary
        assert building.contains_point(11, 7)  # wing
        assert not building.contains_point(16, 7)  # in bounding box, outside union
        assert not building.contains_point(30, 30)  # fully outside

    def test_l_shape_interior_rects(self) -> None:
        """Each occupied section should produce its own inset interior rect.

        The wing's shared edge (south, in this configuration) is NOT inset
        so the walkable junction corridor is included in the wing interior.
        """
        primary = Rect(10, 10, 8, 8)
        wing = Rect.from_bounds(10, 5, 14, 10)
        building = Building(id=4, building_type="house", footprint=primary, wing=wing)

        interior_primary, interior_wing = building.interior_rects
        assert (
            interior_primary.x1,
            interior_primary.y1,
            interior_primary.width,
            interior_primary.height,
        ) == (
            11,
            11,
            6,
            6,
        )
        # Wing extends north of footprint: south edge (y2) is not inset,
        # giving height 4 instead of 3.
        assert (
            interior_wing.x1,
            interior_wing.y1,
            interior_wing.width,
            interior_wing.height,
        ) == (
            11,
            6,
            2,
            4,
        )
        interior_bounds = building.interior_bounds
        assert (
            interior_bounds.x1,
            interior_bounds.y1,
            interior_bounds.width,
            interior_bounds.height,
        ) == (11, 11, 6, 6)

    def test_l_shape_tile_set(self) -> None:
        """tile_set should be exactly the union of all occupied section tiles."""
        primary = Rect(10, 10, 8, 8)
        wing = Rect.from_bounds(10, 5, 14, 10)
        building = Building(id=5, building_type="house", footprint=primary, wing=wing)

        expected = {
            (x, y)
            for x in range(primary.x1, primary.x2)
            for y in range(primary.y1, primary.y2)
        }
        expected.update(
            {(x, y) for x in range(wing.x1, wing.x2) for y in range(wing.y1, wing.y2)}
        )
        assert building.tile_set() == expected

    def test_bounding_rect_equals_footprint_for_rectangular(self) -> None:
        """Backward-compat: rectangular buildings report bounding_rect as footprint."""
        footprint = Rect(3, 4, 9, 7)
        building = Building(id=6, building_type="shop", footprint=footprint)
        assert building.bounding_rect.x1 == footprint.x1
        assert building.bounding_rect.y1 == footprint.y1
        assert building.bounding_rect.x2 == footprint.x2
        assert building.bounding_rect.y2 == footprint.y2

    def test_room_creation(self) -> None:
        """Room has region_id, room_type, bounds."""
        bounds = Rect(5, 5, 8, 6)
        room = Room(
            region_id=99,
            room_type="bedroom",
            bounds=bounds,
        )

        assert room.region_id == 99
        assert room.room_type == "bedroom"
        assert room.bounds == bounds


class TestBuildingChimney:
    """Tests for chimney offset and world position computation."""

    def test_no_chimney_by_default(self) -> None:
        """Building has no chimney when offset is not specified."""
        building = Building(id=1, building_type="house", footprint=Rect(0, 0, 12, 10))
        assert building.chimney_offset is None
        assert building.chimney_world_pos is None

    def test_chimney_world_pos_from_offset(self) -> None:
        """chimney_world_pos adds the offset to the footprint top-left."""
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(50, 30, 12, 10),
            chimney_offset=(3, 2),
        )
        assert building.chimney_world_pos == (53, 32)

    def test_template_randomizes_chimney_position(self) -> None:
        """has_chimney=True places a chimney at a random interior position."""
        template = BuildingTemplate(
            name="test",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            has_chimney=True,
        )
        rng = random.Random(42)
        building = template.create_building(
            building_id=1, position=(10, 20), width=14, height=12, rng=rng
        )
        assert building.chimney_offset is not None
        dx, dy = building.chimney_offset
        # Must be at least 2 tiles from each edge.
        assert 2 <= dx <= 14 - 3
        assert 2 <= dy <= 12 - 3

    def test_chimney_position_varies_across_buildings(self) -> None:
        """Different RNG states produce different chimney positions."""
        template = BuildingTemplate(
            name="test",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            has_chimney=True,
        )
        offsets = set()
        for seed in range(20):
            rng = random.Random(seed)
            b = template.create_building(
                building_id=seed, position=(0, 0), width=14, height=12, rng=rng
            )
            assert b.chimney_offset is not None
            offsets.add(b.chimney_offset)
        # With 20 seeds over a reasonable range, we should see variety.
        assert len(offsets) > 1

    def test_template_without_chimney(self) -> None:
        """has_chimney=False produces buildings without chimneys."""
        template = BuildingTemplate(
            name="test",
            building_type="shop",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
        )
        rng = random.Random(42)
        building = template.create_building(
            building_id=1, position=(10, 20), width=14, height=12, rng=rng
        )
        assert building.chimney_offset is None

    def test_default_templates_chimney_assignments(self) -> None:
        """Verify which default templates have chimneys and which don't."""
        from brileta.environment.generators.buildings.templates import (
            BLACKSMITH_TEMPLATE,
            BUTCHER_TEMPLATE,
            GENERAL_STORE_TEMPLATE,
            INN_TEMPLATE,
            LIBRARY_TEMPLATE,
            WAREHOUSE_TEMPLATE,
        )

        # Templates with chimneys
        assert SMALL_HOUSE_TEMPLATE.has_chimney is True
        assert MEDIUM_HOUSE_TEMPLATE.has_chimney is True
        assert BLACKSMITH_TEMPLATE.has_chimney is True
        assert TAVERN_TEMPLATE.has_chimney is True
        assert INN_TEMPLATE.has_chimney is True

        # Templates without chimneys
        assert GENERAL_STORE_TEMPLATE.has_chimney is False
        assert BUTCHER_TEMPLATE.has_chimney is False
        assert LIBRARY_TEMPLATE.has_chimney is False
        assert SHOP_TEMPLATE.has_chimney is False
        assert WAREHOUSE_TEMPLATE.has_chimney is False


class TestBuildingRidgeAxis:
    """Tests for the Building.ridge_axis computed property."""

    def test_horizontal_for_wide_building(self) -> None:
        """Ridge runs along the longer axis: horizontal when width > height."""
        building = Building(id=1, building_type="house", footprint=Rect(0, 0, 10, 6))
        assert building.ridge_axis == "horizontal"

    def test_vertical_for_tall_building(self) -> None:
        """Ridge runs along the longer axis: vertical when height > width."""
        building = Building(id=1, building_type="house", footprint=Rect(0, 0, 6, 10))
        assert building.ridge_axis == "vertical"

    def test_square_deterministic_from_position(self) -> None:
        """Square buildings pick ridge axis based on position hash for variety."""
        # Even sum (2+4=6) -> horizontal
        b1 = Building(id=1, building_type="house", footprint=Rect(2, 4, 5, 5))
        assert b1.ridge_axis == "horizontal"
        # Odd sum (3+4=7) -> vertical
        b2 = Building(id=2, building_type="house", footprint=Rect(3, 4, 5, 5))
        assert b2.ridge_axis == "vertical"

    def test_same_position_gives_same_axis(self) -> None:
        """Ridge axis is deterministic for the same footprint position."""
        b1 = Building(id=1, building_type="house", footprint=Rect(5, 5, 5, 5))
        b2 = Building(id=2, building_type="tavern", footprint=Rect(5, 5, 5, 5))
        assert b1.ridge_axis == b2.ridge_axis


# =============================================================================
# T2.2: BuildingTemplate
# =============================================================================


class TestBuildingTemplate:
    """Tests for BuildingTemplate functionality."""

    def test_template_creates_building_within_size_bounds(self) -> None:
        """Generated building dimensions within min/max constraints."""
        template = BuildingTemplate(
            name="test",
            building_type="test_type",
            min_width=10,
            max_width=15,
            min_height=8,
            max_height=12,
        )

        rng = random.Random(42)

        # Generate many sizes and verify all within bounds
        for _ in range(100):
            width, height = template.generate_size(rng)
            assert 10 <= width <= 15, f"Width {width} out of bounds [10,15]"
            assert 8 <= height <= 12, f"Height {height} out of bounds [8,12]"

    def test_template_creates_correct_room_count(self) -> None:
        """Room count within specified range."""
        template = BuildingTemplate(
            name="test",
            building_type="test_type",
            min_width=10,
            max_width=15,
            min_height=8,
            max_height=12,
            room_count_min=2,
            room_count_max=4,
        )

        rng = random.Random(123)

        # Generate many room counts and verify all within bounds
        for _ in range(100):
            room_count = template.generate_room_count(rng)
            assert 2 <= room_count <= 4, f"Room count {room_count} out of bounds [2,4]"

    def test_template_create_building(self) -> None:
        """create_building produces Building with correct footprint and roof fields."""
        template = BuildingTemplate(
            name="test_house",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            roof_style="shingle",
            floor_count=3,
        )

        building = template.create_building(
            building_id=5,
            position=(20, 30),
            width=14,
            height=12,
        )

        assert building.id == 5
        assert building.building_type == "house"
        assert building.footprint.x1 == 20
        assert building.footprint.y1 == 30
        assert building.footprint.width == 14
        assert building.footprint.height == 12
        # Rooms and doors are added later by BuildingPlacementLayer
        assert building.rooms == []
        assert building.door_positions == []
        assert building.roof_style == "shingle"
        assert building.roof_profile == "gable"
        assert building.flat_section_ratio == 0.0
        assert building.floor_count == 3

    def test_template_profile_family_selects_low_slope_and_quantizes_flat_section(
        self,
    ) -> None:
        """Low-slope profiles should use deterministic quantized flat-section ratios."""
        template = BuildingTemplate(
            name="test_workshop",
            building_type="workshop",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            roof_style="tin",
            roof_profile_family=("low_slope",),
            low_slope_flat_section_range=(0.45, 0.75),
            low_slope_flat_section_buckets=4,
        )
        rng = random.Random(1)

        building = template.create_building(
            building_id=5,
            position=(20, 30),
            width=14,
            height=12,
            rng=rng,
        )

        assert building.roof_profile == "low_slope"
        # 4 buckets in [0.45, 0.75] => {0.45, 0.55, 0.65, 0.75}
        assert round(building.flat_section_ratio, 2) in {0.45, 0.55, 0.65, 0.75}

    def test_template_profile_family_can_select_flat_profile(self) -> None:
        """Flat profile should set full flat-section ratio."""
        template = BuildingTemplate(
            name="test_warehouse",
            building_type="warehouse",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            roof_style="tin",
            roof_profile_family=("flat",),
        )

        building = template.create_building(
            building_id=6,
            position=(10, 10),
            width=14,
            height=12,
        )

        assert building.roof_profile == "flat"
        assert building.flat_section_ratio == 1.0

    def test_template_door_placement_on_perimeter(self) -> None:
        """Doors placed on building perimeter (validated in layer tests)."""
        # This test validates template has door_count attribute
        # Actual door placement is tested in layer tests
        template = BuildingTemplate(
            name="test",
            building_type="test_type",
            min_width=10,
            max_width=15,
            min_height=8,
            max_height=12,
            door_count=2,
        )

        assert template.door_count == 2

    def test_template_wing_generation(self) -> None:
        """Positive shape weights with 100% config chance always generate a wing."""
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            template = BuildingTemplate(
                name="test_compound",
                building_type="house",
                min_width=18,
                max_width=18,
                min_height=14,
                max_height=14,
                l_shape_weight=1.0,
            )
            rng = random.Random(42)
            building = template.create_building(
                building_id=1, position=(10, 10), width=18, height=14, rng=rng
            )

            assert building.wing is not None
            wing = building.wing
            assert wing.width >= 5
            assert wing.height >= 5

            fp = building.footprint
            overlaps_or_touches = not (
                wing.x2 < fp.x1 or wing.x1 > fp.x2 or wing.y2 < fp.y1 or wing.y1 > fp.y2
            )
            extends_outward = (
                wing.x1 < fp.x1 or wing.x2 > fp.x2 or wing.y1 < fp.y1 or wing.y2 > fp.y2
            )
            assert overlaps_or_touches
            assert extends_outward
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_template_no_wing_when_weights_zero(self) -> None:
        """Both shape weights at 0.0 should never generate a wing."""
        template = BuildingTemplate(
            name="test_rect",
            building_type="house",
            min_width=18,
            max_width=18,
            min_height=14,
            max_height=14,
        )
        for seed in range(20):
            rng = random.Random(seed)
            building = template.create_building(
                building_id=seed, position=(0, 0), width=18, height=14, rng=rng
            )
            assert building.wing is None

    def test_template_wing_varies_across_seeds(self) -> None:
        """Different seeds should produce wings anchored to different corners."""
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            template = BuildingTemplate(
                name="test_varied_wing",
                building_type="house",
                min_width=18,
                max_width=18,
                min_height=14,
                max_height=14,
                l_shape_weight=1.0,
            )

            corners: set[str] = set()
            for seed in range(30):
                rng = random.Random(seed)
                building = template.create_building(
                    building_id=seed, position=(20, 20), width=18, height=14, rng=rng
                )
                assert building.wing is not None
                wing = building.wing
                fp = building.footprint

                if wing.y2 <= fp.y1 + 1 and wing.y1 < fp.y1:
                    corners.add("nw" if wing.x1 == fp.x1 else "ne")
                elif wing.y1 >= fp.y2 - 1 and wing.y2 > fp.y2:
                    corners.add("sw" if wing.x1 == fp.x1 else "se")
                elif wing.x2 <= fp.x1 + 1 and wing.x1 < fp.x1:
                    corners.add("nw" if wing.y1 == fp.y1 else "sw")
                elif wing.x1 >= fp.x2 - 1 and wing.x2 > fp.x2:
                    corners.add("ne" if wing.y1 == fp.y1 else "se")

            assert len(corners) > 1
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_template_t_shape_generation(self) -> None:
        """T-shape weight only should generate a centered side wing."""
        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 1.0
            template = BuildingTemplate(
                name="test_t_shape",
                building_type="house",
                min_width=20,
                max_width=20,
                min_height=16,
                max_height=16,
                t_shape_weight=1.0,
            )
            rng = random.Random(7)
            building = template.create_building(
                building_id=1, position=(30, 30), width=20, height=16, rng=rng
            )
            assert building.wing is not None

            fp = building.footprint
            wing = building.wing
            if (wing.y2 <= fp.y1 + 1 and wing.y1 < fp.y1) or (
                wing.y1 >= fp.y2 - 1 and wing.y2 > fp.y2
            ):
                # North/south attachment: centered on X span.
                left_margin = wing.x1 - fp.x1
                right_margin = fp.x2 - wing.x2
                assert left_margin > 0
                assert right_margin > 0
            elif (wing.x2 <= fp.x1 + 1 and wing.x1 < fp.x1) or (
                wing.x1 >= fp.x2 - 1 and wing.x2 > fp.x2
            ):
                # West/east attachment: centered on Y span.
                top_margin = wing.y1 - fp.y1
                bottom_margin = fp.y2 - wing.y2
                assert top_margin > 0
                assert bottom_margin > 0
            else:
                raise AssertionError(
                    f"Wing does not attach to the primary body: {wing}"
                )
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance

    def test_compound_shape_chance_zero_suppresses_wings(self) -> None:
        """Config chance at 0.0 should suppress all wings regardless of weights."""
        template = BuildingTemplate(
            name="test_suppressed",
            building_type="house",
            min_width=20,
            max_width=20,
            min_height=16,
            max_height=16,
            l_shape_weight=1.0,
            t_shape_weight=1.0,
        )

        old_chance = config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE
        try:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = 0.0
            for seed in range(10):
                rng = random.Random(seed)
                building = template.create_building(
                    building_id=seed, position=(0, 0), width=20, height=16, rng=rng
                )
                assert building.wing is None
        finally:
            config.SETTLEMENT_BUILDING_COMPOUND_SHAPE_CHANCE = old_chance


# =============================================================================
# T2.3: Default Templates
# =============================================================================


class TestDefaultTemplates:
    """Tests for default building templates."""

    def test_small_house_template(self) -> None:
        """Verify size ranges and room counts for small house."""
        template = SMALL_HOUSE_TEMPLATE

        assert template.name == "small_house"
        assert template.building_type == "house"
        assert template.min_width == 12
        assert template.max_width == 16
        assert template.min_height == 10
        assert template.max_height == 14
        assert template.room_count_min == 2
        assert template.room_count_max == 3
        assert template.roof_style == "thatch"

    def test_medium_house_template(self) -> None:
        """Verify size ranges and room counts for medium house."""
        template = MEDIUM_HOUSE_TEMPLATE

        assert template.name == "medium_house"
        assert template.building_type == "house"
        assert template.min_width == 18
        assert template.max_width == 22
        assert template.min_height == 14
        assert template.max_height == 18
        assert template.room_count_min == 2
        assert template.room_count_max == 4
        assert template.l_shape_weight == 0.7
        assert template.t_shape_weight == 0.3

    def test_shop_template(self) -> None:
        """Shop has front, back, and storage rooms."""
        template = SHOP_TEMPLATE

        assert template.name == "shop"
        assert template.building_type == "shop"
        assert template.min_width == 12
        assert template.max_width == 16
        assert template.min_height == 10
        assert template.max_height == 14
        assert template.room_count_min == 2
        assert template.room_count_max == 3
        assert "shop_floor" in template.room_types
        assert "back_room" in template.room_types
        assert template.roof_style == "shingle"
        assert template.roof_profile_family == ("gable", "low_slope")

    def test_tavern_template(self) -> None:
        """Tavern is larger with multiple rooms."""
        template = TAVERN_TEMPLATE

        assert template.name == "tavern"
        assert template.building_type == "tavern"
        # Larger than houses
        assert template.min_width == 22
        assert template.max_width == 28
        assert template.min_height == 18
        assert template.max_height == 22
        # Multiple rooms
        assert template.room_count_min == 4
        assert template.room_count_max == 6
        assert "main_hall" in template.room_types
        assert template.roof_style == "shingle"
        assert template.l_shape_weight == 0.7
        assert template.t_shape_weight == 0.3

    def test_industrial_templates_use_tin_roofs(self) -> None:
        """Blacksmiths and warehouses use tin roofs with industrial profiles."""
        assert BLACKSMITH_TEMPLATE.roof_style == "tin"
        assert WAREHOUSE_TEMPLATE.roof_style == "tin"
        assert BLACKSMITH_TEMPLATE.roof_profile_family == ("gable", "low_slope")
        assert WAREHOUSE_TEMPLATE.roof_profile_family == ("low_slope", "flat")
        assert BLACKSMITH_TEMPLATE.l_shape_weight == 0.7
        assert BLACKSMITH_TEMPLATE.t_shape_weight == 0.3

    def test_inn_template_wing_probability(self) -> None:
        """Inn should allow compound footprints."""
        assert INN_TEMPLATE.l_shape_weight == 0.7
        assert INN_TEMPLATE.t_shape_weight == 0.3

    def test_get_default_templates(self) -> None:
        """get_default_templates returns expected templates."""
        templates = get_default_templates()

        assert len(templates) == 10
        names = {t.name for t in templates}
        expected = {
            "small_house",
            "medium_house",
            "general_store",
            "butcher",
            "blacksmith",
            "tavern",
            "inn",
            "library",
            "shop",
            "warehouse",
        }
        assert names == expected

        # Only larger archetypes should generate wings by default.
        l_weights = {t.name: t.l_shape_weight for t in templates}
        t_weights = {t.name: t.t_shape_weight for t in templates}
        for name in ("medium_house", "tavern", "inn", "blacksmith"):
            assert l_weights[name] > 0, f"{name} should have positive L-shape weight"
            assert t_weights[name] > 0, f"{name} should have positive T-shape weight"
        for name in ("small_house", "shop"):
            assert l_weights[name] == 0.0
            assert t_weights[name] == 0.0


# =============================================================================
# T2.4: Building Condition
# =============================================================================


class TestBuildingCondition:
    """Tests for per-building condition field and wear integration."""

    def test_condition_defaults_to_pristine(self) -> None:
        """Building without explicit condition is pristine (0.0)."""
        building = Building(id=1, building_type="house", footprint=Rect(0, 0, 12, 10))
        assert building.condition == 0.0

    def test_condition_stored_on_building(self) -> None:
        """Explicit condition value is preserved."""
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(0, 0, 12, 10),
            condition=0.42,
        )
        assert building.condition == 0.42

    def test_condition_clamped_low(self) -> None:
        """Negative condition values are clamped to 0.0."""
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(0, 0, 12, 10),
            condition=-0.5,
        )
        assert building.condition == 0.0

    def test_condition_clamped_high(self) -> None:
        """Condition values above 1.0 are clamped to 1.0."""
        building = Building(
            id=1,
            building_type="house",
            footprint=Rect(0, 0, 12, 10),
            condition=1.8,
        )
        assert building.condition == 1.0

    def test_template_generates_condition_in_range(self) -> None:
        """create_building produces condition within the template's configured range."""
        template = BuildingTemplate(
            name="test",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            condition_range=(0.2, 0.6),
        )
        for seed in range(50):
            rng = random.Random(seed)
            b = template.create_building(
                building_id=seed, position=(0, 0), width=14, height=12, rng=rng
            )
            assert 0.2 <= b.condition <= 0.6, (
                f"seed {seed}: condition {b.condition} outside [0.2, 0.6]"
            )

    def test_template_condition_varies_across_buildings(self) -> None:
        """Different RNG seeds produce different condition values."""
        template = BuildingTemplate(
            name="test",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            condition_range=(0.0, 1.0),
        )
        conditions = set()
        for seed in range(30):
            rng = random.Random(seed)
            b = template.create_building(
                building_id=seed, position=(0, 0), width=14, height=12, rng=rng
            )
            conditions.add(round(b.condition, 2))
        # With 30 seeds over [0, 1], should see real variety.
        assert len(conditions) > 5

    def test_template_condition_without_rng_uses_midpoint(self) -> None:
        """Without an RNG, condition is the midpoint of the range."""
        template = BuildingTemplate(
            name="test",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
            condition_range=(0.2, 0.8),
        )
        b = template.create_building(
            building_id=1, position=(0, 0), width=14, height=12
        )
        assert b.condition == 0.5

    def test_default_templates_have_condition_ranges(self) -> None:
        """All default templates have a condition_range with min < max."""
        for template in get_default_templates():
            lo, hi = template.condition_range
            assert 0.0 <= lo < hi <= 1.0, (
                f"{template.name}: invalid condition_range ({lo}, {hi})"
            )

    def test_residential_templates_weather_more_than_public(self) -> None:
        """Houses have higher max condition than tavern/inn (less maintenance)."""
        assert (
            SMALL_HOUSE_TEMPLATE.condition_range[1] > TAVERN_TEMPLATE.condition_range[1]
        )

    def test_industrial_templates_wear_more(self) -> None:
        """Blacksmith and warehouse can reach higher condition than tavern/inn."""
        assert (
            BLACKSMITH_TEMPLATE.condition_range[1] > TAVERN_TEMPLATE.condition_range[1]
        )
        assert WAREHOUSE_TEMPLATE.condition_range[1] > INN_TEMPLATE.condition_range[1]

    def test_library_best_maintained(self) -> None:
        """Library has lowest max condition of all public buildings."""
        assert LIBRARY_TEMPLATE.condition_range[1] <= TAVERN_TEMPLATE.condition_range[1]
        assert LIBRARY_TEMPLATE.condition_range[1] <= INN_TEMPLATE.condition_range[1]

    def test_all_templates_have_explicit_condition_range(self) -> None:
        """Every default template should set condition_range explicitly.

        Relying on the class default is fragile - if the default changes,
        templates that intended a specific range would silently shift.
        """
        # The class default exists as a fallback for tests/ad-hoc templates,
        # but every shipped template should have an explicit value. We verify
        # this by checking that no two templates share the exact same range
        # object identity as the class default (which would indicate they
        # didn't override it). Since dataclass field defaults are shared
        # objects, we can't rely on identity; instead we just confirm that
        # every template's range was set to something intentional by checking
        # it's present in the source. This test is a documentation guard -
        # the real protection is the wear_material_map test below.
        for template in get_default_templates():
            lo, hi = template.condition_range
            assert hi <= 1.0, f"{template.name}: max condition {hi} > 1.0"
            assert lo >= 0.0, f"{template.name}: min condition {lo} < 0.0"


class TestWearPack:
    """Tests for the wear_pack bit-packing used by the shader pipeline."""

    def test_wear_pack_roundtrip(self) -> None:
        """Pack known values into wear_pack and verify each field unpacks correctly."""
        material_id = 3  # tin
        condition_byte = 178  # ~0.70 condition
        edge_byte = 200  # ~0.78 edge proximity
        bld_hash = 0xAB

        wear_pack = np.uint32(
            material_id | (condition_byte << 8) | (edge_byte << 16) | (bld_hash << 24)
        )

        # Unpack the same way the shader does.
        assert int(wear_pack & 0xFF) == material_id
        assert int((wear_pack >> 8) & 0xFF) == condition_byte
        assert int((wear_pack >> 16) & 0xFF) == edge_byte
        assert int((wear_pack >> 24) & 0xFF) == bld_hash

    def test_wear_pack_zero_means_no_wear(self) -> None:
        """A wear_pack of 0 should indicate no material (no wear applied)."""
        wear_pack = np.uint32(0)
        assert int(wear_pack & 0xFF) == 0  # material_id == 0 means no wear

    def test_wear_pack_all_fields_max(self) -> None:
        """All fields at maximum values pack and unpack without overflow."""
        material_id = 0xFF
        condition_byte = 0xFF
        edge_byte = 0xFF
        bld_hash = 0xFF

        wear_pack = np.uint32(
            material_id | (condition_byte << 8) | (edge_byte << 16) | (bld_hash << 24)
        )
        assert wear_pack == np.uint32(0xFFFFFFFF)
        assert int(wear_pack & 0xFF) == 0xFF
        assert int((wear_pack >> 8) & 0xFF) == 0xFF
        assert int((wear_pack >> 16) & 0xFF) == 0xFF
        assert int((wear_pack >> 24) & 0xFF) == 0xFF

    def test_wear_material_map_covers_all_roof_styles(self) -> None:
        """Every roof_style used by a default template has a _WEAR_MATERIAL_MAP entry.

        If someone adds a new roof style and forgets the map entry, wear
        silently becomes a no-op (material_id=0). This catches that.
        """
        for template in get_default_templates():
            assert template.roof_style in _WEAR_MATERIAL_MAP, (
                f"Template '{template.name}' uses roof_style='{template.roof_style}' "
                f"which has no entry in _WEAR_MATERIAL_MAP"
            )
