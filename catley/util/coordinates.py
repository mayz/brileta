from __future__ import annotations

from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    TileCoord,
    WorldTilePos,
)

"""Conversion functions for dealing with coordinate systems."""


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
        renderer_scaled_width: PixelCoord,
        renderer_scaled_height: PixelCoord,
    ) -> None:
        """Initializes converter with final rendered console dimensions."""
        self.console_width_in_tiles = console_width_in_tiles
        self.console_height_in_tiles = console_height_in_tiles

        # The renderer now handles the offset calculation. This converter just needs
        # to know the final size of the rendered area to calculate tile sizes.
        self.offset_x = 0
        self.offset_y = 0

        # Calculate tile size based on the final scaled dimensions
        self.tile_width_screen_px = renderer_scaled_width / console_width_in_tiles
        self.tile_height_screen_px = renderer_scaled_height / console_height_in_tiles

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
