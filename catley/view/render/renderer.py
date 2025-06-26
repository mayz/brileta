"""
Defines the abstract interface for all rendering engines in the game.

This module provides the `Renderer` abstract base class, which defines a
stable contract for all low-level rendering operations. Its primary
responsibility is to provide a backend-agnostic API for drawing game elements
to the screen.

Key Principles:
- A Renderer knows *how* to draw things (e.g., sprites, text, shapes), but not
  *what* or *when* to draw them. That is the responsibility of higher-level
  systems like the `FrameManager` and `View`s.
- The interface provides high-level drawing commands (e.g., `draw_actor_smooth`,
  `render_particles`) that are independent of the underlying graphics library.
- It is the final stage in the rendering pipeline, handling the conversion of
  game-state data into pixels on the screen.
- It is the single source of truth for screen dimensions and coordinate-space
  transformations.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

from catley import colors
from catley.game.enums import BlendMode
from catley.util.coordinates import (
    PixelCoord,
    PixelPos,
    Rect,
    RootConsoleTilePos,
    TileDimensions,
)
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class Renderer(abc.ABC):
    """Abstract base class for drawing graphics to the screen."""

    @property
    @abc.abstractmethod
    def tile_dimensions(self) -> TileDimensions:
        pass

    @property
    @abc.abstractmethod
    def console_width_tiles(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def console_height_tiles(self) -> int:
        pass

    @abc.abstractmethod
    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: float = 1.0,
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates.

        Args:
            char: Character to draw (e.g., '@', 'g')
            color: Base actor color in 0-255 RGB format
            screen_x: Screen X coordinate in pixels (can be fractional)
            screen_y: Screen Y coordinate in pixels (can be fractional)
            light_intensity: RGB lighting multipliers in 0.0-1.0 range
            alpha: Interpolation factor between previous and current state (0.0-1.0)
                  Used for smooth movement between fixed timestep updates
        """
        pass

    @abc.abstractmethod
    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draws the active cursor using data from the cursor manager."""
        pass

    @abc.abstractmethod
    def draw_tile_highlight(
        self,
        root_x: int,
        root_y: int,
        color: colors.Color,
        alpha: float,
    ) -> None:
        """Draws a semi-transparent highlight over a single tile."""
        pass

    @abc.abstractmethod
    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        """
        Renders all particles from a SubTileParticleSystem for a specific layer.

        This method is responsible for iterating through the active particles,
        culling those outside the viewport, calculating their final screen
        positions, and drawing them.

        Args:
            particle_system: The data source containing all particle states.
            layer: The specific layer to render (e.g., UNDER_ACTORS or OVER_ACTORS).
            viewport_bounds: The currently visible area in the renderer's internal
                             tile coordinates
                             (e.g., world_view.width x world_view.height).
            view_offset: The offset of the viewport in root console coordinates,
                         used for calculating final screen positions.
        """
        pass

    @abc.abstractmethod
    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        """Apply a circular environmental effect using cached textures."""
        pass

    @abc.abstractmethod
    def prepare_to_present(self) -> None:
        """Converts the root console to a texture and copies it to the backbuffer."""
        pass

    @abc.abstractmethod
    def finalize_present(self) -> None:
        """Presents the backbuffer to the screen."""
        pass

    @abc.abstractmethod
    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        pass

    @abc.abstractmethod
    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """
        Converts root console coordinates to final, hardware-rendered
        screen pixel coordinates.

        This critical transformation step accounts for the current window size,
        aspect ratio, and any letterboxing or pillarboxing applied by the renderer.

        Args:
            console_x: The x-coordinate on the root console grid. Must be a float
                    to represent sub-tile positions for smooth rendering.
            console_y: The y-coordinate on the root console grid. Must be a float.

        Returns:
            A PixelPos tuple (px_x, px_y) representing the top-left corner of the
            given console coordinate on the final rendered screen.
        """
        pass

    @abc.abstractmethod
    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Converts final screen pixel coordinates to root console tile coordinates."""
        pass

    @abc.abstractmethod
    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Creates a backend-specific texture from a raw NumPy RGBA pixel array."""
        pass

    @abc.abstractmethod
    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Presents a pre-rendered, backend-specific texture object to the screen at
        a given tile location."""
        pass

    @abc.abstractmethod
    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        pass

    @abc.abstractmethod
    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draws a raw, unfilled rectangle directly to the screen for debugging."""
        pass
