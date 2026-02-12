"""Building and Room dataclasses for semantic building representation.

Buildings wrap regions and provide a higher-level semantic layer. Each building
contains rooms, which correspond to MapRegions. This allows querying buildings
semantically while still using the existing region system for pathfinding and
lighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from brileta.types import WorldTilePos
from brileta.util.coordinates import Rect


@dataclass
class Room:
    """A single room within a building.

    Each room corresponds to a MapRegion and has a semantic type that
    describes its purpose (e.g., "main_hall", "bedroom", "shop_floor").

    Attributes:
        region_id: The ID of the MapRegion for this room.
        room_type: Semantic type describing the room's purpose.
        bounds: The rectangular bounds of the room.
    """

    region_id: int
    room_type: str
    bounds: Rect


@dataclass
class Building:
    """A building consisting of one or more rooms.

    Buildings provide semantic grouping of rooms and track shared properties
    like the building type and door positions. Each building has a footprint
    (the outer bounds including walls) and a list of rooms inside.

    Attributes:
        id: Unique identifier for this building.
        building_type: Semantic type (e.g., "house", "shop", "tavern").
        footprint: The outer bounds of the building including walls.
        rooms: List of Room objects inside this building.
        door_positions: List of (x, y) positions where doors are placed.
    """

    id: int
    building_type: str
    footprint: Rect
    rooms: list[Room] = field(default_factory=list)
    door_positions: list[WorldTilePos] = field(default_factory=list)

    @property
    def interior_bounds(self) -> Rect:
        """Get the interior bounds (footprint minus walls).

        Returns:
            A Rect representing the walkable interior area.
        """
        return Rect(
            self.footprint.x1 + 1,
            self.footprint.y1 + 1,
            self.footprint.width - 2,
            self.footprint.height - 2,
        )

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this building's footprint.

        Args:
            x: X coordinate to check.
            y: Y coordinate to check.

        Returns:
            True if the point is within the building's footprint.
        """
        return (
            self.footprint.x1 <= x < self.footprint.x2
            and self.footprint.y1 <= y < self.footprint.y2
        )
