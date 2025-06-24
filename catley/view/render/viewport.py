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
    -   Example: `event.position` in a `tcod.event.MouseState` event.

--- TRANSFORMATION FLOW ---

-   Input (Mouse Click):
    `Pixel` -> `Root Console` (via `coordinate_converter`)
    `Root Console` -> `Viewport` (by subtracting view offset)
    `Viewport` -> `World` (via `viewport_to_world`)

-   Output (Rendering an Actor):
    `World` -> `Viewport` (via `world_to_viewport`)
    `Viewport` -> `Root Console` (via `tcod.console.blit`)
    `Root Console` -> `Pixel` (via `context.present`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from catley.environment.map import TileCoord
from catley.util.coordinates import (
    Rect,
    ViewportTileCoord,
    ViewportTilePos,
    WorldTileCoord,
    WorldTilePos,
)

if TYPE_CHECKING:
    from catley.game.actors import Actor


@dataclass
class Camera:
    """Camera that tracks a position in world space."""

    # Keep the type as float instead of WorldTileCoord for
    # smooth interpolation between tiles
    world_x: float = 0.0  # Camera center in world coordinates
    world_y: float = 0.0

    def follow_actor(self, actor: Actor, smoothing: float = 0.15) -> None:
        """Smoothly follow an actor with camera lag to reduce jitter."""
        target_x = float(actor.x)
        target_y = float(actor.y)
        self.world_x += (target_x - self.world_x) * smoothing
        self.world_y += (target_y - self.world_y) * smoothing

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

    def get_world_bounds(self, camera: Camera) -> Rect:
        """Return the clamped world-space bounds currently visible."""
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
        return Rect.from_bounds(left, top, right, bottom)

    def world_to_viewport(
        self, world_x: WorldTileCoord, world_y: WorldTileCoord, camera: Camera
    ) -> ViewportTilePos:
        """Converts world coordinates to viewport (view-relative) coordinates."""
        bounds = self.get_world_bounds(camera)
        left, top = bounds.x1, bounds.y1
        return world_x - left + self.offset_x, world_y - top + self.offset_y

    def viewport_to_world(
        self, vp_x: ViewportTileCoord, vp_y: ViewportTileCoord, camera: Camera
    ) -> WorldTilePos:
        """Converts viewport (view-relative) coordinates to world coordinates."""
        bounds = self.get_world_bounds(camera)
        left, top = bounds.x1, bounds.y1
        return vp_x - self.offset_x + left, vp_y - self.offset_y + top


class ViewportSystem:
    """Manages the Camera and Viewport to provide a clean API for rendering."""

    def __init__(self, viewport_width: TileCoord, viewport_height: TileCoord) -> None:
        self.camera = Camera()
        self.viewport = Viewport(viewport_width, viewport_height)

    def update_camera(self, player: Actor, map_width: int, map_height: int) -> None:
        """Update camera to follow the player and compute viewport offsets."""
        # This should be the ONLY line that updates the camera's raw position.
        self.camera.follow_actor(player)

        self.viewport.map_width = map_width
        self.viewport.map_height = map_height

        vp_w = self.viewport.width_tiles
        vp_h = self.viewport.height_tiles

        # Clamp camera to world boundaries and calculate viewport offsets.
        half_vp_w = vp_w / 2.0
        half_vp_h = vp_h / 2.0

        if map_width < vp_w:
            # Center map in viewport if map is smaller than viewport
            self.camera.world_x = map_width / 2.0 - 0.5
            self.viewport.offset_x = (vp_w - map_width) // 2
        else:
            # Clamp camera to map edges
            self.camera.world_x = max(
                half_vp_w - 0.5,
                min(self.camera.world_x, map_width - half_vp_w - 0.5),
            )
            self.viewport.offset_x = 0

        if map_height < vp_h:
            # Center map in viewport if map is smaller than viewport
            self.camera.world_y = map_height / 2.0 - 0.5
            self.viewport.offset_y = (vp_h - map_height) // 2
        else:
            # Clamp camera to map edges
            self.camera.world_y = max(
                half_vp_h - 0.5,
                min(self.camera.world_y, map_height - half_vp_h - 0.5),
            )
            self.viewport.offset_y = 0

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
        """Converts world coordinates to viewport/screen coordinates."""
        return self.viewport.world_to_viewport(world_x, world_y, self.camera)

    def world_to_screen_float(
        self, world_x: float, world_y: float
    ) -> tuple[float, float]:
        """Converts world coordinates to viewport/screen coordinates."""
        # Same math as world_to_screen but accepts/returns floats
        bounds = self.viewport.get_world_bounds(self.camera)
        left, top = bounds.x1, bounds.y1
        return (
            world_x - left + self.viewport.offset_x,
            world_y - top + self.viewport.offset_y,
        )

    def screen_to_world(
        self, vp_x: ViewportTileCoord, vp_y: ViewportTileCoord
    ) -> WorldTilePos:
        """Converts viewport/screen coordinates to world coordinates."""
        return self.viewport.viewport_to_world(vp_x, vp_y, self.camera)

    @property
    def offset_x(self) -> int:
        return self.viewport.offset_x

    @property
    def offset_y(self) -> int:
        return self.viewport.offset_y
