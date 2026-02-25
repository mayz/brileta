"""Building and Room dataclasses for semantic building representation.

Buildings wrap regions and provide a higher-level semantic layer. Each building
contains rooms, which correspond to MapRegions. This allows querying buildings
semantically while still using the existing region system for pathfinding and
lighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from brileta.types import FootprintOffset, WorldTilePos
from brileta.util.coordinates import Rect


class ChimneyType(Enum):
    """Visual style of a chimney. Only STONE is rendered for now.

    Future variants (BRICK, STOVEPIPE, etc.) can be added here and
    wired into the roof substitution renderer when needed.
    """

    STONE = "stone"


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
        roof_style: Visual roof style used by render-time roof substitution.
        floor_count: Number of floors (stored for future multi-story phases).
        chimney_offset: Optional (dx, dy) offset from footprint top-left where a
            chimney sits. None means no chimney on this building.
        chimney_type: Visual style of the chimney (only STONE for now).
    """

    id: int
    building_type: str
    footprint: Rect
    rooms: list[Room] = field(default_factory=list)
    door_positions: list[WorldTilePos] = field(default_factory=list)
    roof_style: str = "thatch"
    floor_count: int = 1
    chimney_offset: FootprintOffset | None = None
    chimney_type: ChimneyType = ChimneyType.STONE

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

    @property
    def chimney_world_pos(self) -> WorldTilePos | None:
        """Absolute world position of the chimney, or None if no chimney."""
        if self.chimney_offset is None:
            return None
        dx, dy = self.chimney_offset
        return (self.footprint.x1 + dx, self.footprint.y1 + dy)

    @property
    def ridge_axis(self) -> str:
        """Ridge line axis derived from footprint dimensions.

        The ridge runs along the longer axis of the building: horizontal for
        wide buildings, vertical for tall buildings. Square buildings use a
        position-based hash for visual variety across the map.
        """
        if self.footprint.width > self.footprint.height:
            return "horizontal"
        if self.footprint.height > self.footprint.width:
            return "vertical"
        # Square: deterministic pick based on position for variety.
        return (
            "horizontal"
            if (self.footprint.x1 + self.footprint.y1) % 2 == 0
            else "vertical"
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
