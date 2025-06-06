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
        rendering area (the `GameWorldPanel`'s console). Also referred to as "screen"
        or "panel" coordinates in the context of the game world view.
    -   Origin: (0, 0) is the top-left tile of the `game_map_console`.
    -   Example: `game_map_console.rgb["ch"][vp_x, vp_y] = ord('@')`

3.  Root Console Coordinates (`root_x`, `root_y`)
    -   Description: The coordinate on the main application's root console.
        This is used for positioning UI panels and handling mouse events that
        could be over any panel, not just the game world.
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
    `Root Console` -> `Viewport` (by subtracting panel offset)
    `Viewport` -> `World` (via `viewport_to_world`)

-   Output (Rendering an Actor):
    `World` -> `Viewport` (via `world_to_viewport`)
    `Viewport` -> `Root Console` (via `tcod.console.blit`)
    `Root Console` -> `Pixel` (via `context.present`)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.game.actors import Actor


@dataclass
class Camera:
    """Camera that tracks a position in world space."""

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

    width_tiles: int
    height_tiles: int

    def get_world_bounds(self, camera: Camera) -> tuple[int, int, int, int]:
        """Get the integer world coordinates of the viewport's corners."""
        half_width = self.width_tiles / 2.0
        half_height = self.height_tiles / 2.0
        left = int(camera.world_x - half_width)
        right = int(camera.world_x + half_width)
        top = int(camera.world_y - half_height)
        bottom = int(camera.world_y + half_height)
        return left, right, top, bottom

    def world_to_viewport(
        self, world_x: int, world_y: int, camera: Camera
    ) -> tuple[int, int]:
        """Converts world coordinates to viewport (panel-relative) coordinates."""
        vp_x = world_x - int(camera.world_x - self.width_tiles / 2.0)
        vp_y = world_y - int(camera.world_y - self.height_tiles / 2.0)
        return vp_x, vp_y

    def viewport_to_world(
        self, vp_x: int, vp_y: int, camera: Camera
    ) -> tuple[int, int]:
        """Converts viewport (panel-relative) coordinates to world coordinates."""
        world_x = vp_x + int(camera.world_x - self.width_tiles / 2.0)
        world_y = vp_y + int(camera.world_y - self.height_tiles / 2.0)
        return world_x, world_y


class ViewportSystem:
    """Manages the Camera and Viewport to provide a clean API for rendering."""

    def __init__(self, viewport_width: int, viewport_height: int) -> None:
        self.camera = Camera()
        self.viewport = Viewport(viewport_width, viewport_height)

    def update_camera(self, player: Actor, map_width: int, map_height: int) -> None:
        """Update camera to follow the player, clamping to map boundaries."""
        self.camera.follow_actor(player)
        half_vp_width = self.viewport.width_tiles / 2.0
        half_vp_height = self.viewport.height_tiles / 2.0
        self.camera.world_x = max(
            half_vp_width - 0.5,
            min(map_width - half_vp_width - 0.5, self.camera.world_x),
        )
        self.camera.world_y = max(
            half_vp_height - 0.5,
            min(map_height - half_vp_height - 0.5, self.camera.world_y),
        )

    def get_visible_bounds(self) -> tuple[int, int, int, int]:
        """Get the world coordinates that are currently visible."""
        return self.viewport.get_world_bounds(self.camera)

    def is_visible(
        self, world_x: int, world_y: int, map_width: int, map_height: int
    ) -> bool:
        """Check if a world position is currently inside the viewport."""
        left, right, top, bottom = self.get_visible_bounds()
        return left <= world_x <= right and top <= world_y <= bottom

    def world_to_screen(self, world_x: int, world_y: int) -> tuple[int, int]:
        """Converts world coordinates to viewport/screen coordinates."""
        return self.viewport.world_to_viewport(world_x, world_y, self.camera)

    def screen_to_world(self, vp_x: int, vp_y: int) -> tuple[int, int]:
        """Converts viewport/screen coordinates to world coordinates."""
        return self.viewport.viewport_to_world(vp_x, vp_y, self.camera)
