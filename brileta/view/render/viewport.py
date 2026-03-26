"""
Viewport System for efficient rendering of large game worlds.

This module provides a camera that tracks a position in world space and a viewport
that defines the visible area on the screen. It's the foundation for efficient
rendering, allowing the game to handle large maps by only processing and drawing
what the player can see.

--- COORDINATE SYSTEM DOCUMENTATION ---

There are four distinct coordinate systems. Understanding them is key to avoiding bugs.
A strict naming convention is used to make the code self-documenting.

1.  World Coordinates (`world_x`, `world_y`)
    -   Description: The absolute, integer-based coordinate on the main game map.
        This is the "source of truth" for all game logic
        (actor positions, map features).
    -   Origin: (0, 0) is the top-left corner of the entire game map.
    -   Example: `player.x`, `gw.game_map.tiles[world_x, world_y]`

2.  Viewport Coordinates (`vp_x`, `vp_y`)
    -   Description: The coordinate relative to the top-left of the visible
        rendering area (the `GameWorldView`'s console). Also referred to as "screen"
        or "view" coordinates in the context of the game world view.
    -   Origin: (0, 0) is the top-left tile of the `game_map_console`.
    -   Example: `game_map_console.rgb["ch"][vp_x, vp_y] = ord('@')`

3.  Root Console Coordinates (`root_x`, `root_y`)
    -   Description: The coordinate on the main application's root console.
        This is used for positioning UI views and handling mouse events that
        could be over any view, not just the game world.
    -   Origin: (0, 0) is the top-left corner of the game window.
    -   Example: `root_console.print(x=root_x, y=root_y, text="HP: 10")`

4.  Pixel Coordinates (`px_x`, `px_y`)
    -   Description: The raw pixel coordinate from the OS/SDL, used for
        mouse events and low-level SDL drawing (like the custom cursor).
    -   Origin: (0, 0) is the top-left pixel of the game window.
    -   Example: `event.position` in an `input_events.MouseState` event.

--- TRANSFORMATION FLOW ---

-   Input (Mouse Click):
    `Pixel` -> `Root Console` (via `coordinate_converter`)
    `Root Console` -> `Viewport` (by subtracting view offset)
    `Viewport` -> `World` (via `viewport_to_world`)

-   Output (Rendering an Actor):
    `World` -> `Viewport` (via `world_to_viewport`)
    `Viewport` -> `Root Console` (via view composition and `GraphicsContext` draw calls)
    `Root Console` -> `Pixel` (via backend presentation in `GraphicsContext.present_frame`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from brileta.environment.map import TileCoord
from brileta.types import (
    ViewportTilePos,
    WorldTileCoord,
    WorldTilePos,
)
from brileta.util.coordinates import Rect

if TYPE_CHECKING:
    from brileta.game.actors import Actor


@dataclass
class Camera:
    """Camera that tracks a position in world space."""

    # Keep the type as float instead of WorldTileCoord for
    # smooth interpolation between tiles
    world_x: float = 0.0  # Camera center in world coordinates
    world_y: float = 0.0

    # Dead zone parameters - player can move within this area without camera movement
    dead_zone_width: float = 20.0  # Width of dead zone in tiles (1/3 of viewport)
    dead_zone_height: float = 12.0  # Height of dead zone in tiles (1/3 of viewport)

    def follow_actor(self, actor: Actor, smoothing: float = 0.15) -> None:
        """Follow an actor with dead zone to reduce jarring camera movement."""
        target_x = float(actor.x)
        target_y = float(actor.y)

        # Calculate distance from camera center to player
        dx = target_x - self.world_x
        dy = target_y - self.world_y

        # Dead zone boundaries (half-width/height from center)
        dead_zone_half_w = self.dead_zone_width / 2.0
        dead_zone_half_h = self.dead_zone_height / 2.0

        # Only move camera if player is outside the dead zone
        new_camera_x = self.world_x
        new_camera_y = self.world_y

        if abs(dx) > dead_zone_half_w:
            # Player is outside dead zone horizontally
            if dx > 0:
                # Player is to the right, move camera right
                new_camera_x = target_x - dead_zone_half_w
            else:
                # Player is to the left, move camera left
                new_camera_x = target_x + dead_zone_half_w

        if abs(dy) > dead_zone_half_h:
            # Player is outside dead zone vertically
            if dy > 0:
                # Player is below, move camera down
                new_camera_y = target_y - dead_zone_half_h
            else:
                # Player is above, move camera up
                new_camera_y = target_y + dead_zone_half_h

        # Apply smooth movement to new camera position
        self.world_x += (new_camera_x - self.world_x) * smoothing
        self.world_y += (new_camera_y - self.world_y) * smoothing

    def set_position(self, world_x: float, world_y: float) -> None:
        """Instantly move camera to a specific world position."""
        self.world_x = world_x
        self.world_y = world_y


@dataclass
class Viewport:
    """Defines the visible area of the game world and handles coordinate transforms."""

    width_tiles: TileCoord
    height_tiles: TileCoord
    offset_x: TileCoord = 0
    offset_y: TileCoord = 0
    map_width: TileCoord | None = None
    map_height: TileCoord | None = None

    # Per-frame cache for get_world_bounds. The result is pure (depends only on
    # camera position and viewport dimensions), and is called ~230 times per
    # frame with the same inputs.
    _cached_bounds: Rect | None = field(default=None, init=False, repr=False)
    _cached_bounds_key: object = field(default=None, init=False, repr=False)

    def get_world_bounds(self, camera: Camera) -> Rect:
        """Return the clamped world-space bounds currently visible."""
        key = (
            camera.world_x,
            camera.world_y,
            self.width_tiles,
            self.height_tiles,
            self.map_width,
            self.map_height,
        )
        if self._cached_bounds_key == key:
            return self._cached_bounds  # ty: ignore[invalid-return-type]

        half_w: TileCoord = self.width_tiles // 2
        half_h: TileCoord = self.height_tiles // 2
        center_x = round(camera.world_x)
        center_y = round(camera.world_y)
        left = center_x - half_w
        top = center_y - half_h

        if self.map_width is not None:
            max_left = max(0, self.map_width - self.width_tiles)
            left = min(max(left, 0), max_left)
        if self.map_height is not None:
            max_top = max(0, self.map_height - self.height_tiles)
            top = min(max(top, 0), max_top)

        right = left + self.width_tiles - 1
        bottom = top + self.height_tiles - 1
        result = Rect.from_bounds(left, top, right, bottom)
        self._cached_bounds = result
        self._cached_bounds_key = key
        return result

    def world_to_viewport(
        self, world_x: WorldTileCoord, world_y: WorldTileCoord, camera: Camera
    ) -> ViewportTilePos:
        """Converts world coordinates to viewport (view-relative) coordinates."""
        bounds = self.get_world_bounds(camera)
        left, top = bounds.x1, bounds.y1
        return world_x - left + self.offset_x, world_y - top + self.offset_y

    def viewport_to_world(
        self, vp_x: float, vp_y: float, camera: Camera
    ) -> tuple[float, float]:
        """Converts viewport (view-relative) coordinates to world coordinates."""
        bounds = self.get_world_bounds(camera)
        left, top = bounds.x1, bounds.y1
        return vp_x - self.offset_x + left, vp_y - self.offset_y + top

    def resize(self, new_width: TileCoord, new_height: TileCoord) -> None:
        """Updates the viewport's dimensions."""
        self.width_tiles = new_width
        self.height_tiles = new_height
        self._cached_bounds_key = None  # Invalidate bounds cache


class ViewportSystem:
    """Manages the Camera and Viewport to provide a clean API for rendering."""

    def __init__(self, viewport_width: TileCoord, viewport_height: TileCoord) -> None:
        self.camera = Camera()
        self.viewport = Viewport(viewport_width, viewport_height)
        self._display_width_tiles: float = float(viewport_width)
        self._display_height_tiles: float = float(viewport_height)
        # Per-frame cache for get_display_scale_factors
        self._cached_scale: tuple[float, float] | None = None
        self._cached_scale_key: object = None
        # Pre-computed scalars for world_to_screen_float (set by update_camera).
        # Avoids method dispatch and cache-key construction in the hottest path.
        self._wtsf_left: float = 0.0
        self._wtsf_top: float = 0.0
        self._wtsf_ox: float = 0.0
        self._wtsf_oy: float = 0.0
        self._wtsf_sx: float = 1.0
        self._wtsf_sy: float = 1.0

    def set_display_size(
        self, display_width_tiles: TileCoord, display_height_tiles: TileCoord
    ) -> None:
        """Set the fixed on-screen viewport size in root-console tile units."""
        self._display_width_tiles = float(max(1, display_width_tiles))
        self._display_height_tiles = float(max(1, display_height_tiles))
        self._cached_scale_key = None  # Invalidate scale cache

    def get_display_scale_factors(self) -> tuple[float, float]:
        """Return root-console tiles per visible world tile for this viewport."""
        key = (
            self._display_width_tiles,
            self._display_height_tiles,
            self.viewport.width_tiles,
            self.viewport.height_tiles,
        )
        if self._cached_scale_key == key:
            return self._cached_scale  # ty: ignore[invalid-return-type]
        width_scale = self._display_width_tiles / max(1, self.viewport.width_tiles)
        height_scale = self._display_height_tiles / max(1, self.viewport.height_tiles)
        result = (width_scale, height_scale)
        self._cached_scale = result
        self._cached_scale_key = key
        return result

    def update_camera(self, player: Actor, map_width: int, map_height: int) -> None:
        """Update camera to follow the player and compute viewport offsets."""
        self.viewport.map_width = map_width
        self.viewport.map_height = map_height

        vp_w = self.viewport.width_tiles
        vp_h = self.viewport.height_tiles

        # Keep the dead zone proportional to the current visible area so zooming
        # in does not make it cover the whole viewport (and vice versa).
        self.camera.dead_zone_width = vp_w / 3.0
        self.camera.dead_zone_height = vp_h / 3.0

        # This should be the ONLY line that updates the camera's raw position.
        self.camera.follow_actor(player)

        # Clamp camera to world boundaries and calculate viewport offsets.
        half_vp_w = vp_w / 2.0
        half_vp_h = vp_h / 2.0

        if map_width < vp_w:
            # Center map in viewport if map is smaller than viewport
            self.camera.world_x = map_width / 2.0 - 0.5
            self.viewport.offset_x = (vp_w - map_width) // 2
        else:
            # Clamp camera to map edges. The camera position is rounded in
            # get_world_bounds, so we use half_vp_w - 0.5 as minimum (rounds up
            # to show column 0) and map_width - half_vp_w as maximum (shows the
            # last column). Python's banker's rounding means 19.5 rounds to 20
            # but 22.5 rounds to 22, so we avoid .5 at the max end.
            self.camera.world_x = max(
                half_vp_w - 0.5,
                min(self.camera.world_x, map_width - half_vp_w),
            )
            self.viewport.offset_x = 0

        if map_height < vp_h:
            # Center map in viewport if map is smaller than viewport
            self.camera.world_y = map_height / 2.0 - 0.5
            self.viewport.offset_y = (vp_h - map_height) // 2
        else:
            # Clamp camera to map edges. Same logic as X axis - avoid .5 at max
            # to prevent banker's rounding from cutting off the last row.
            self.camera.world_y = max(
                half_vp_h - 0.5,
                min(self.camera.world_y, map_height - half_vp_h),
            )
            self.viewport.offset_y = 0

        # Pre-compute values used by world_to_screen_float so the hot path
        # reduces to pure arithmetic with no method calls or cache lookups.
        bounds = self.viewport.get_world_bounds(self.camera)
        sx, sy = self.get_display_scale_factors()
        self._wtsf_left = float(bounds.x1)
        self._wtsf_top = float(bounds.y1)
        self._wtsf_ox = float(self.viewport.offset_x)
        self._wtsf_oy = float(self.viewport.offset_y)
        self._wtsf_sx = sx
        self._wtsf_sy = sy

    def get_visible_bounds(self) -> Rect:
        """Get the world coordinates that are currently visible."""
        return self.viewport.get_world_bounds(self.camera)

    def is_visible(self, world_x: WorldTileCoord, world_y: WorldTileCoord) -> bool:
        """Check if a world position is currently inside the viewport."""
        bounds = self.get_visible_bounds()
        return bounds.x1 <= world_x <= bounds.x2 and bounds.y1 <= world_y <= bounds.y2

    def world_to_screen(
        self, world_x: WorldTileCoord, world_y: WorldTileCoord
    ) -> ViewportTilePos:
        """Converts world coordinates to viewport/screen coordinates.

        Uses pre-computed scalars set by update_camera, same as
        world_to_screen_float but returns rounded ints.
        """
        return (
            round((world_x - self._wtsf_left + self._wtsf_ox) * self._wtsf_sx),
            round((world_y - self._wtsf_top + self._wtsf_oy) * self._wtsf_sy),
        )

    def world_to_screen_float(
        self, world_x: float, world_y: float
    ) -> tuple[float, float]:
        """Converts world coordinates to viewport/screen coordinates (float).

        Uses pre-computed scalars set by update_camera to avoid method dispatch
        and cache-key construction on every call (~757K calls/session).
        """
        return (
            (world_x - self._wtsf_left + self._wtsf_ox) * self._wtsf_sx,
            (world_y - self._wtsf_top + self._wtsf_oy) * self._wtsf_sy,
        )

    def screen_to_world(self, vp_x: float, vp_y: float) -> WorldTilePos:
        """Converts viewport/screen coordinates to world coordinates."""
        scale_x, scale_y = self.get_display_scale_factors()
        world_x, world_y = self.viewport.viewport_to_world(
            vp_x / scale_x,
            vp_y / scale_y,
            self.camera,
        )
        return (math.floor(world_x), math.floor(world_y))

    def get_camera_fractional_offset(self) -> tuple[float, float]:
        """Return the fractional part of the camera position for smooth scrolling.

        This is the difference between the actual camera position and the
        rounded position used for tile selection. When the background is rendered
        with padding tiles, this offset can be applied at presentation time
        to achieve smooth sub-tile scrolling.

        Returns:
            (offset_x, offset_y) in tile units. Positive values mean the camera
            is past the tile center, requiring a negative presentation offset.
        """
        return (
            self.camera.world_x - round(self.camera.world_x),
            self.camera.world_y - round(self.camera.world_y),
        )

    def get_display_camera_fractional_offset(self) -> tuple[float, float]:
        """Return camera fractional offset scaled to root-console tile units."""
        frac_x, frac_y = self.get_camera_fractional_offset()
        scale_x, scale_y = self.get_display_scale_factors()
        return (frac_x * scale_x, frac_y * scale_y)

    @property
    def offset_x(self) -> int:
        return self.viewport.offset_x

    @property
    def offset_y(self) -> int:
        return self.viewport.offset_y
