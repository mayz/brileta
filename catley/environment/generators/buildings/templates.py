"""Building templates for procedural building generation.

Templates define the constraints and properties for generating buildings.
Each template specifies size ranges, room counts, and building-specific
configuration.

Building variety is controlled via:
- weight: Higher weight = more likely to be picked (houses are common)
- max_per_settlement: Caps unique buildings (only 1 tavern, 1 blacksmith, etc.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from catley.util.coordinates import Rect

from .building import Building


@dataclass
class BuildingTemplate:
    """Template defining constraints for building generation.

    Templates specify the size ranges, room configuration, and other
    properties for generating a specific type of building.

    Attributes:
        name: Human-readable name for this template.
        building_type: The semantic type to assign to generated buildings.
        min_width: Minimum building width (including walls).
        max_width: Maximum building width (including walls).
        min_height: Minimum building height (including walls).
        max_height: Maximum building height (including walls).
        room_count_min: Minimum number of rooms.
        room_count_max: Maximum number of rooms.
        door_count: Number of exterior doors (default 1).
        room_types: List of room type names to use for rooms.
        weight: Selection weight (higher = more common). Default 1.0.
        max_per_settlement: Max instances per settlement. None = unlimited.
    """

    name: str
    building_type: str
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    room_count_min: int = 1
    room_count_max: int = 1
    door_count: int = 1
    room_types: tuple[str, ...] = ("main",)
    weight: float = 1.0
    max_per_settlement: int | None = None

    def generate_size(self, rng: random.Random) -> tuple[int, int]:
        """Generate random dimensions within the template's constraints.

        Args:
            rng: Random number generator.

        Returns:
            A tuple of (width, height) for the building.
        """
        width = rng.randint(self.min_width, self.max_width)
        height = rng.randint(self.min_height, self.max_height)
        return width, height

    def generate_room_count(self, rng: random.Random) -> int:
        """Generate random room count within the template's constraints.

        Args:
            rng: Random number generator.

        Returns:
            Number of rooms to generate.
        """
        return rng.randint(self.room_count_min, self.room_count_max)

    def create_building(
        self,
        building_id: int,
        position: tuple[int, int],
        width: int,
        height: int,
    ) -> Building:
        """Create a Building object with the given position and size.

        This creates the Building shell - rooms and doors are added later
        by the BuildingPlacementLayer.

        Args:
            building_id: Unique ID for this building.
            position: (x, y) position of the top-left corner.
            width: Width of the building.
            height: Height of the building.

        Returns:
            A new Building object with the specified footprint.
        """
        x, y = position
        footprint = Rect(x, y, width, height)

        return Building(
            id=building_id,
            building_type=self.building_type,
            footprint=footprint,
        )


# =============================================================================
# Residential Buildings
# =============================================================================

SMALL_HOUSE_TEMPLATE = BuildingTemplate(
    name="small_house",
    building_type="house",
    min_width=12,
    max_width=16,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("living_room", "bedroom", "kitchen"),
    weight=2.0,  # Common - many small houses in a settlement
)

MEDIUM_HOUSE_TEMPLATE = BuildingTemplate(
    name="medium_house",
    building_type="house",
    min_width=18,
    max_width=22,
    min_height=14,
    max_height=18,
    room_count_min=2,
    room_count_max=4,
    room_types=("living_room", "bedroom", "kitchen", "storage"),
    weight=2.0,  # Less common than small houses
)

# =============================================================================
# Commercial Buildings
# =============================================================================

GENERAL_STORE_TEMPLATE = BuildingTemplate(
    name="general_store",
    building_type="general_store",
    min_width=14,
    max_width=18,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("shop_floor", "storage", "office"),
    weight=1.0,
    max_per_settlement=1,  # Only one general store per town
)

BUTCHER_TEMPLATE = BuildingTemplate(
    name="butcher",
    building_type="butcher",
    min_width=10,
    max_width=14,
    min_height=8,
    max_height=12,
    room_count_min=2,
    room_count_max=2,
    room_types=("shop_floor", "cold_storage"),
    weight=0.8,
    max_per_settlement=1,
)

BLACKSMITH_TEMPLATE = BuildingTemplate(
    name="blacksmith",
    building_type="blacksmith",
    min_width=14,
    max_width=18,
    min_height=12,
    max_height=16,
    room_count_min=2,
    room_count_max=3,
    room_types=("forge", "workshop", "storage"),
    weight=0.8,
    max_per_settlement=1,
)

# =============================================================================
# Public Buildings
# =============================================================================

TAVERN_TEMPLATE = BuildingTemplate(
    name="tavern",
    building_type="tavern",
    min_width=22,
    max_width=28,
    min_height=18,
    max_height=22,
    room_count_min=4,
    room_count_max=6,
    room_types=("main_hall", "kitchen", "storage", "bedroom", "cellar"),
    weight=1.0,
    max_per_settlement=1,  # One tavern per settlement
)

INN_TEMPLATE = BuildingTemplate(
    name="inn",
    building_type="inn",
    min_width=20,
    max_width=26,
    min_height=16,
    max_height=20,
    room_count_min=4,
    room_count_max=6,
    room_types=("common_room", "kitchen", "guest_room", "guest_room", "storage"),
    weight=0.8,
    max_per_settlement=1,  # One inn per settlement
)

LIBRARY_TEMPLATE = BuildingTemplate(
    name="library",
    building_type="library",
    min_width=16,
    max_width=22,
    min_height=14,
    max_height=18,
    room_count_min=2,
    room_count_max=4,
    room_types=("reading_room", "stacks", "study", "archive"),
    weight=0.5,  # Rare - not every town has a library
    max_per_settlement=1,
)

# =============================================================================
# Small Commercial (can have multiples)
# =============================================================================

SHOP_TEMPLATE = BuildingTemplate(
    name="shop",
    building_type="shop",
    min_width=12,
    max_width=16,
    min_height=10,
    max_height=14,
    room_count_min=2,
    room_count_max=3,
    room_types=("shop_floor", "back_room", "storage"),
    weight=1.5,  # Generic shops are fairly common
    max_per_settlement=3,  # Can have a few different shops
)

WAREHOUSE_TEMPLATE = BuildingTemplate(
    name="warehouse",
    building_type="warehouse",
    min_width=16,
    max_width=24,
    min_height=12,
    max_height=18,
    room_count_min=2,
    room_count_max=3,
    room_types=("storage", "loading_area", "office"),
    weight=0.6,
    max_per_settlement=2,
)


def get_default_templates() -> list[BuildingTemplate]:
    """Get the default set of building templates.

    Returns:
        A list of BuildingTemplate objects for common building types.
    """
    return [
        # Residential (common)
        SMALL_HOUSE_TEMPLATE,
        MEDIUM_HOUSE_TEMPLATE,
        # Commercial (unique)
        GENERAL_STORE_TEMPLATE,
        BUTCHER_TEMPLATE,
        BLACKSMITH_TEMPLATE,
        # Public (unique)
        TAVERN_TEMPLATE,
        INN_TEMPLATE,
        LIBRARY_TEMPLATE,
        # Small commercial (can have multiples)
        SHOP_TEMPLATE,
        WAREHOUSE_TEMPLATE,
    ]
