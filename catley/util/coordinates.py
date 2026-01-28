from __future__ import annotations

from typing import TYPE_CHECKING

from catley.types import (
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    TileCoord,
    WorldTilePos,
)

if TYPE_CHECKING:
    from catley.view.render.effects.particles import SubTileParticleSystem
    from catley.view.render.graphics import GraphicsContext
    from catley.view.render.viewport import ViewportSystem

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
        """Initializes converter with final rendered console dimensions.

        Note: This converter assumes pixel coordinates are already adjusted for
        letterboxing. The graphics backends handle letterboxing offset before
        calling this converter.
        """
        self.console_width_in_tiles = console_width_in_tiles
        self.console_height_in_tiles = console_height_in_tiles
        self.tile_width_screen_px = renderer_scaled_width / console_width_in_tiles
        self.tile_height_screen_px = renderer_scaled_height / console_height_in_tiles

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Convert pixel coordinates to root console tile coordinates.

        Assumes pixel coordinates are already adjusted for letterboxing offset.
        Use graphics.pixel_to_tile() for raw screen coordinates.

        Return:
            A RootConsoleTilePos.
        """
        tile_x: TileCoord = int(pixel_x // self.tile_width_screen_px)
        tile_y: TileCoord = int(pixel_y // self.tile_height_screen_px)

        # Clamp to valid range
        tile_x = max(0, min(tile_x, self.console_width_in_tiles - 1))
        tile_y = max(0, min(tile_y, self.console_height_in_tiles - 1))

        return (tile_x, tile_y)


# =============================================================================
# PARTICLE COORDINATE CONVERSION
# =============================================================================


def convert_particle_to_screen_coords(
    particle_system: SubTileParticleSystem,
    particle_index: int,
    viewport_bounds: Rect,
    view_offset: RootConsoleTilePos,
    graphics: GraphicsContext,
    viewport_system: ViewportSystem | None = None,
) -> tuple[float, float] | None:
    """Convert particle index to screen coordinates, or None if off-screen.

    This shared function consolidates the particle coordinate conversion logic
    that was previously duplicated across all graphics backends.

    Args:
        particle_system: The particle system containing the particle
        particle_index: Index of the particle to convert
        viewport_bounds: Viewport bounds for culling
        view_offset: Root console tile position offset
        graphics: Graphics context for final coordinate conversion
        viewport_system: Optional viewport system for world-to-viewport conversion

    Returns:
        Screen pixel coordinates (x, y) or None if particle is off-screen
    """
    if not (0 <= particle_index < particle_system.active_count):
        return None

    # Access data directly from the particle_system object
    sub_x, sub_y = particle_system.positions[particle_index]

    # Convert particle's sub-tile coordinates to world coordinates
    world_x = sub_x / particle_system.subdivision
    world_y = sub_y / particle_system.subdivision

    # Use viewport system to convert world coordinates to viewport coordinates
    if viewport_system is not None:
        # Check if the world position is visible (tile containing particle)
        tile_x = int(world_x) if world_x >= 0 else int(world_x) - 1
        tile_y = int(world_y) if world_y >= 0 else int(world_y) - 1
        if not viewport_system.is_visible(tile_x, tile_y):
            return None

        # Convert tile coordinates to viewport coordinates (like actors do)
        vp_x, vp_y = viewport_system.world_to_screen(tile_x, tile_y)
    else:
        # Fallback to old behavior if no viewport system provided
        vp_x, vp_y = world_x, world_y

    # Cull particles outside the viewport bounds (viewport coordinates)
    if (
        vp_x < viewport_bounds.x1
        or vp_x > viewport_bounds.x2
        or vp_y < viewport_bounds.y1
        or vp_y > viewport_bounds.y2
    ):
        return None

    # Convert viewport tile coordinates to root console tile coordinates
    root_x = view_offset[0] + vp_x
    root_y = view_offset[1] + vp_y

    # Use the graphics context's own converter to get final pixel coordinates
    return graphics.console_to_screen_coords(root_x, root_y)
