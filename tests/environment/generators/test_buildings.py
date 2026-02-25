"""Tests for the building system.

Phase T2 tests cover:
- T2.1: Building/Room dataclasses
- T2.2: BuildingTemplate
- T2.3: Default templates
"""

from __future__ import annotations

import random

from brileta.environment.generators.buildings import Building, BuildingTemplate, Room
from brileta.environment.generators.buildings.templates import (
    MEDIUM_HOUSE_TEMPLATE,
    SHOP_TEMPLATE,
    SMALL_HOUSE_TEMPLATE,
    TAVERN_TEMPLATE,
    get_default_templates,
)
from brileta.util.coordinates import Rect

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
        assert building.floor_count == 1

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
        assert building.floor_count == 3

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
