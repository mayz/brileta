"""Building and Room dataclasses for semantic building representation.

Buildings wrap regions and provide a higher-level semantic layer. Each building
contains rooms, which correspond to MapRegions. This allows querying buildings
semantically while still using the existing region system for pathfinding and
lighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from brileta.types import FootprintOffset, WorldTilePos, saturate
from brileta.util.coordinates import Rect

# Perspective offset constants for pseudo-3D building rendering.
# The roof shifts north by this many tile units per floor, exposing a
# south-facing wall strip. Capped at MAX_PERSPECTIVE_FLOORS so very
# tall buildings don't push the roof off-screen.
WALL_HEIGHT_PER_FLOOR: float = 1.5
MAX_PERSPECTIVE_FLOORS: int = 4

# Chimney perspective rendering constants.
# Projected height: visible south face of the chimney in the 3/4 view, in
# tile units.  Smaller than the physical height because foreshortening
# compresses the vertical dimension.  Per-building height is randomized
# between MIN and MAX for visual variety.
CHIMNEY_MIN_PROJECTED_HEIGHT: float = 0.2
CHIMNEY_MAX_PROJECTED_HEIGHT: float = 1.0
# Shadow height: physical height of the chimney above the roof surface, in
# tile units.  Used to compute shadow length together with the sun
# elevation.  Larger than the projected height because the real chimney
# extends well above the roofline.  Coupled to projected height via a
# per-building height factor so taller-looking chimneys cast longer shadows.
CHIMNEY_MIN_SHADOW_HEIGHT: float = 1.5
CHIMNEY_MAX_SHADOW_HEIGHT: float = 4.0
# CHIMNEY_SHADOW_INTENSITY: peak darkening factor applied to roof tiles
# directly adjacent to the chimney.  Fades linearly toward the shadow tip.
CHIMNEY_SHADOW_INTENSITY: float = 0.15


class ChimneyType(Enum):
    """Visual style of a chimney. Only STONE is rendered for now.

    Future variants (BRICK, STOVEPIPE, etc.) can be added here and
    wired into the roof substitution renderer when needed.
    """

    STONE = "stone"


RoofProfile = Literal["gable", "low_slope", "flat"]
RoofStyle = Literal["thatch", "shingle", "tin"]


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
    (the primary body rectangle including walls), an optional wing rectangle,
    and a list of rooms inside.

    Attributes:
        id: Unique identifier for this building.
        building_type: Semantic type (e.g., "house", "shop", "tavern").
        footprint: Primary body bounds including walls.
        wing: Optional secondary bounds for compound footprints.
        rooms: List of Room objects inside this building.
        door_positions: List of (x, y) positions where doors are placed.
        roof_style: Visual roof style used by render-time roof substitution
            (e.g., "thatch", "shingle", "tin").
        roof_profile: Roof shape profile ("gable", "low_slope", or "flat").
        flat_section_ratio: Fraction of the short roof axis treated as a flat
            center section for "low_slope" profiles. Ignored for "gable" and
            "flat" profiles.
        floor_count: Number of floors (stored for future multi-story phases).
        chimney_offset: Optional (dx, dy) offset from footprint top-left where a
            chimney sits. None means no chimney on this building.
        chimney_type: Visual style of the chimney (only STONE for now).
        chimney_projected_height: Visible south face of the chimney in 3/4
            perspective, in tile units.  Randomized per building.
        chimney_shadow_height: Physical height above the roof for shadow
            computation, in tile units.  Coupled to projected height.
        condition: Weathering/degradation state from 0.0 (pristine) to 1.0
            (dilapidated). Set during generation and used by the roof
            substitution renderer to paint material-specific weathering
            effects (stain patches, moss, rust, etc.).
    """

    id: int
    building_type: str
    footprint: Rect
    rooms: list[Room] = field(default_factory=list)
    door_positions: list[WorldTilePos] = field(default_factory=list)
    roof_style: RoofStyle = "thatch"
    roof_profile: RoofProfile = "gable"
    flat_section_ratio: float = 0.0
    floor_count: int = 1
    chimney_offset: FootprintOffset | None = None
    chimney_type: ChimneyType = ChimneyType.STONE
    chimney_projected_height: float = CHIMNEY_MIN_PROJECTED_HEIGHT
    chimney_shadow_height: float = CHIMNEY_MIN_SHADOW_HEIGHT
    condition: float = 0.0
    """Weathering/degradation state from 0.0 (pristine) to 1.0 (dilapidated).

    Set during generation based on building type. Drives visual weathering
    on roofs: stain patches, moss/lichen, thinning, rust. Higher values
    produce more patches, darker patches, wider coverage.
    """
    wing: Rect | None = None
    """Optional secondary footprint rectangle for compound L/T-shaped buildings."""

    # Cached perspective values - computed once in __post_init__ from
    # floor_count (which is immutable after building generation).
    perspective_north_offset: float = field(init=False, repr=False)
    perspective_ceil_offset: int = field(init=False, repr=False)
    perspective_floor_offset: int = field(init=False, repr=False)
    perspective_frac: float = field(init=False, repr=False)
    perspective_has_frac: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute perspective values from floor_count."""
        self.flat_section_ratio = saturate(float(self.flat_section_ratio))
        self.condition = saturate(float(self.condition))

        pno = min(self.floor_count, MAX_PERSPECTIVE_FLOORS) * WALL_HEIGHT_PER_FLOOR
        self.perspective_north_offset = pno
        self.perspective_ceil_offset = math.ceil(pno)
        self.perspective_floor_offset = math.floor(pno)
        frac = pno - math.floor(pno)
        self.perspective_frac = frac
        self.perspective_has_frac = frac > 0.001

    @property
    def is_compound(self) -> bool:
        """Return True when this building has a secondary wing rectangle."""
        return self.wing is not None

    @property
    def occupied_rects(self) -> tuple[Rect, ...]:
        """Get the rectangles whose union forms the building shape."""
        if self.wing is None:
            return (self.footprint,)
        return (self.footprint, self.wing)

    @property
    def bounding_rect(self) -> Rect:
        """Get the axis-aligned bounding rectangle of the full building."""
        rects = self.occupied_rects
        return Rect.from_bounds(
            min(rect.x1 for rect in rects),
            min(rect.y1 for rect in rects),
            max(rect.x2 for rect in rects),
            max(rect.y2 for rect in rects),
        )

    @property
    def interior_rects(self) -> tuple[Rect, ...]:
        """Get interior rectangles (each occupied section inset by one tile).

        For compound buildings the wing's shared edge with the footprint is
        NOT inset.  This keeps the walkable junction corridor inside the
        wing's interior rect so every interior floor tile is covered by a
        room region.  Rectangular buildings are simply inset by 1 on all
        sides as before.
        """
        fp = self.footprint
        fp_interior = Rect(fp.x1 + 1, fp.y1 + 1, fp.width - 2, fp.height - 2)

        if self.wing is None:
            return (fp_interior,)

        wing = self.wing

        # Default: inset every edge by 1.
        w_x1 = wing.x1 + 1
        w_y1 = wing.y1 + 1
        w_x2 = wing.x2 - 1
        w_y2 = wing.y2 - 1

        # Detect which edge of the wing is attached to the footprint and
        # skip the inset on that edge so the junction corridor is included.
        if wing.y1 < fp.y1:
            # Wing extends north - shared on wing's south edge.
            w_y2 = wing.y2
        elif wing.y2 > fp.y2:
            # Wing extends south - shared on wing's north edge.
            w_y1 = wing.y1
        elif wing.x1 < fp.x1:
            # Wing extends west - shared on wing's east edge.
            w_x2 = wing.x2
        elif wing.x2 > fp.x2:
            # Wing extends east - shared on wing's west edge.
            w_x1 = wing.x1

        wing_interior = Rect.from_bounds(w_x1, w_y1, w_x2, w_y2)
        return (fp_interior, wing_interior)

    @property
    def interior_bounds(self) -> Rect:
        """Get the interior bounds (footprint minus walls).

        Returns:
            A Rect representing the walkable interior area.
        """
        # Backward-compat accessor: the primary section's interior.
        return self.interior_rects[0]

    @property
    def chimney_world_pos(self) -> WorldTilePos | None:
        """Absolute world position of the chimney, or None if no chimney."""
        if self.chimney_offset is None:
            return None
        dx, dy = self.chimney_offset
        return (self.footprint.x1 + dx, self.footprint.y1 + dy)

    @property
    def ridge_axis(self) -> str:
        """Ridge line axis derived from overall building bounds.

        The ridge runs along the longer axis of the building: horizontal for
        wide buildings, vertical for tall buildings. Square buildings use a
        position-based hash for visual variety across the map.
        """
        bounds = self.bounding_rect
        if bounds.width > bounds.height:
            return "horizontal"
        if bounds.height > bounds.width:
            return "vertical"
        # Square: deterministic pick based on position for variety.
        return "horizontal" if (bounds.x1 + bounds.y1) % 2 == 0 else "vertical"

    def tile_set(self) -> set[WorldTilePos]:
        """Get all world tiles occupied by the building's shape union."""
        tiles: set[WorldTilePos] = set()
        for rect in self.occupied_rects:
            for x in range(rect.x1, rect.x2):
                for y in range(rect.y1, rect.y2):
                    tiles.add((x, y))
        return tiles

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this building's occupied shape.

        Args:
            x: X coordinate to check.
            y: Y coordinate to check.

        Returns:
            True if the point is within any occupied section.
        """
        return any(
            rect.x1 <= x < rect.x2 and rect.y1 <= y < rect.y2
            for rect in self.occupied_rects
        )
