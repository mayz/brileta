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
        """Building has id, type, footprint, rooms, door_positions."""
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
        """create_building produces Building with correct footprint."""
        template = BuildingTemplate(
            name="test_house",
            building_type="house",
            min_width=12,
            max_width=16,
            min_height=10,
            max_height=14,
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
