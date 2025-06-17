from __future__ import annotations

"""
Type definitions and conversion functions for dealing with coordinate systems.

COORDINATE SYSTEM DOCUMENTATION:
===============================

All coordinate systems except PixelPos refer to TILES (discrete integer positions).
Only PixelPos refers to actual screen pixels.

TILE-BASED COORDINATES (Always integers):
- WorldTilePos: Integer tile positions on the game map (e.g., Actor.x=5, Actor.y=3)
- ViewportTilePos: Integer tile positions relative to viewport (what's visible)
- RootConsoleTilePos: Integer tile positions on the UI console
- TileCoord: Individual tile coordinate component (always int)
- Rect: Rectangle/bounding box in tile coordinates.

PIXEL-BASED COORDINATES (Can be float in SDL3):
- PixelPos: Raw SDL pixel coordinates from mouse/events
  (can be float for sub-pixel precision)
- PixelCoord: Individual pixel coordinate component (int | float for SDL3)
"""

# =============================================================================
# TILE-BASED COORDINATE SYSTEMS (Always integers)
# =============================================================================


TileCoord = int  # Always integer tile position

# Game world coordinates - absolute positions on the game map
WorldTileCoord = TileCoord  # Example: x=5, y=3
WorldTilePos = tuple[
    WorldTileCoord, WorldTileCoord
]  # Example: (5, 3) = tile 5,3 on map

# Viewport coordinates - relative to the visible area
ViewportTileCoord = TileCoord  # Example: vp_x=0, vp_y=0
ViewportTilePos = tuple[
    ViewportTileCoord, ViewportTileCoord
]  # Example: (0, 0) = top-left of viewport

# Root console coordinates - UI panel positioning
RootConsoleTileCoord = TileCoord  # Example: root_x=10, root_y=5
RootConsoleTilePos = tuple[
    RootConsoleTileCoord, RootConsoleTileCoord
]  # Example: (10, 5) = tile 10,5 on UI

# =============================================================================
# PIXEL-BASED COORDINATE SYSTEMS (Can be float in SDL3)
# =============================================================================

# Raw SDL pixel coordinates (SDL3-ready for float precision)
PixelCoord = int | float  # Example: px_x=123.5
PixelPos = tuple[PixelCoord, PixelCoord]  # Example: (123.5, 456.7)

# =============================================================================
# UTILITY TYPES
# =============================================================================

# Individual tile dimensions in pixels (for working with tilesets)
TileDimensions = tuple[int, int]  # Example: (16, 16) = 16x16 pixel tiles

# =============================================================================
# NAMING CONVENTION EXAMPLES
# =============================================================================

"""
CORRECT variable naming examples:

# World coordinates (game map tiles)
world_tile_x: WorldTileCoord = 5
world_tile_y: WorldTileCoord = 3
actor_world_pos: WorldTilePos = (5, 3)

# Viewport coordinates (visible area tiles)
vp_tile_x: ViewportTileCoord = 2
vp_tile_y: ViewportTileCoord = 1
tile_in_viewport: ViewportTilePos = (2, 1)

# Root console coordinates (UI tiles)
root_tile_x: RootConsoleTileCoord = 10
root_tile_y: RootConsoleTileCoord = 15
menu_position: RootConsoleTilePos = (10, 15)

# Pixel coordinates (SDL screen pixels - can be float!)
mouse_px_x: PixelCoord = 123.5      # SDL3 float precision
mouse_px_y: PixelCoord = 456.7
mouse_pixel_pos: PixelPos = (123.5, 456.7)

# Tile dimensions
tile_width_px: int = 16              # Tile width in pixels
tile_height_px: int = 16             # Tile height in pixels
tile_size: TileDimensions = (16, 16) # Tile dimensions in pixels

AVOID these ambiguous names:
- x, y (unclear coordinate system)
- mouse_x, mouse_y (unclear if pixels or tiles)
- screen_x, screen_y (could be viewport or root console)
- pos, position (unclear coordinate system)
"""


class Rect:
    """Rectangle/bounding box in tile coordinates."""

    def __init__(self, x: TileCoord, y: TileCoord, w: TileCoord, h: TileCoord) -> None:
        self.x1: TileCoord = x
        self.y1: TileCoord = y
        self.x2: TileCoord = x + w
        self.y2: TileCoord = y + h

    @classmethod
    def from_bounds(
        cls, x1: TileCoord, y1: TileCoord, x2: TileCoord, y2: TileCoord
    ) -> Rect:
        """Create a Rect from corner coordinates (x1, y1, x2, y2)."""
        width = x2 - x1
        height = y2 - y1
        return cls(x1, y1, width, height)

    @property
    def width(self) -> TileCoord:
        return self.x2 - self.x1

    @property
    def height(self) -> TileCoord:
        return self.y2 - self.y1

    def center(self) -> tuple[int, int]:
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    def intersects(self, other: Rect) -> bool:
        return (
            self.x1 <= other.x2
            and self.x2 >= other.x1
            and self.y1 <= other.y2
            and self.y2 >= other.y1
        )

    def __repr__(self) -> str:
        return f"Rect(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2})"


# =============================================================================
# BOUNDS CHECKING HELPERS
# =============================================================================


def is_valid_world_tile_pos(
    pos: WorldTilePos, map_width: TileCoord, map_height: TileCoord
) -> bool:
    """Check if world tile position is within map bounds."""
    x, y = pos
    return 0 <= x < map_width and 0 <= y < map_height


def is_valid_pixel_pos(
    pos: PixelPos, screen_width: PixelCoord, screen_height: PixelCoord
) -> bool:
    """Check if pixel position is within screen bounds."""
    px_x, px_y = pos
    return 0 <= px_x < screen_width and 0 <= px_y < screen_height


class CoordinateConverter:
    """Handles pixel-to-tile coordinate conversion with SDL3 float support."""

    def __init__(
        self,
        console_width_in_tiles: TileCoord,
        console_height_in_tiles: TileCoord,
        tile_width_px: PixelCoord,
        tile_height_px: PixelCoord,
        renderer_width: PixelCoord,
        renderer_height: PixelCoord,
    ) -> None:
        """Initialize coordinate converter with console and renderer dimensions."""
        self.console_width_in_tiles = console_width_in_tiles
        self.console_height_in_tiles = console_height_in_tiles
        self.tile_width_px = tile_width_px
        self.tile_height_px = tile_height_px

        expected_width_px: PixelCoord = console_width_in_tiles * tile_width_px
        expected_height_px: PixelCoord = console_height_in_tiles * tile_height_px

        console_aspect: float = expected_width_px / expected_height_px
        window_aspect: float = renderer_width / renderer_height

        if console_aspect > window_aspect:
            # Letterboxed vertically - top/bottom bars
            scaled_height = renderer_width / console_aspect
            self.offset_x = 0
            self.offset_y = (renderer_height - scaled_height) / 2
            self.tile_width_screen_px = renderer_width / console_width_in_tiles
            self.tile_height_screen_px = scaled_height / console_height_in_tiles
        else:
            # Letterboxed horizontally - left/right bars
            scaled_width = renderer_height * console_aspect
            self.offset_x = (renderer_width - scaled_width) / 2
            self.offset_y = 0
            self.tile_width_screen_px = scaled_width / console_width_in_tiles
            self.tile_height_screen_px = renderer_height / console_height_in_tiles

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Convert pixel coordinates to root console tile coordinates.

        Return:
            A RootConsoleTilePos.
        """
        # Adjust for letterboxing offsets
        adj_x: float = pixel_x - self.offset_x
        adj_y: float = pixel_y - self.offset_y

        tile_x: TileCoord = int(adj_x // self.tile_width_screen_px)
        tile_y: TileCoord = int(adj_y // self.tile_height_screen_px)

        # Clamp to valid range
        tile_x = max(0, min(tile_x, self.console_width_in_tiles - 1))
        tile_y = max(0, min(tile_y, self.console_height_in_tiles - 1))

        return (tile_x, tile_y)
