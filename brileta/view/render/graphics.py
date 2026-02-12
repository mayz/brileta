"""
Defines the abstract interface for a graphics backend in the game.

This module provides the `GraphicsContext` abstract base class, which defines a
stable contract for all low-level rendering operations. Its primary responsibility
is to provide a backend-agnostic API for drawing game elements to the screen.

Key Principles:
- A GraphicsContext knows *how* to draw things (e.g., sprites, text, shapes), but not
  *what* or *when* to draw them. That is the responsibility of higher-level systems
  like the `FrameManager` and `View`s.
- The interface provides high-level drawing commands (e.g., `draw_actor_smooth`,
  `render_particles`) that are independent of the underlying graphics library.
- It is the final stage in the rendering pipeline, handling the conversion of
  game-state data into pixels on the screen.
- It is the single source of truth for screen dimensions and coordinate-space
  transformations.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from brileta import colors
from brileta.game.enums import BlendMode
from brileta.types import (
    ColorRGBf,
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    PixelPos,
    PixelRect,
    RootConsoleTilePos,
    TileDimensions,
    ViewOffset,
    WorldTilePos,
)
from brileta.util.coordinates import CoordinateConverter, Rect
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.view.render.effects.decals import DecalSystem
from brileta.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from brileta.view.render.viewport import ViewportSystem

if TYPE_CHECKING:
    from brileta.backends.wgpu.resource_manager import WGPUResourceManager
    from brileta.view.ui.cursor_manager import CursorManager


class GraphicsContext(abc.ABC):
    """Abstract base class for drawing graphics to the screen."""

    # Backends without explicit resource managers inherit the default None value.
    resource_manager: WGPUResourceManager | None = None

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
    def render_glyph_buffer_to_texture(
        self,
        glyph_buffer: GlyphBuffer,
        buffer_override: Any = None,
        secondary_override: Any = None,
        cache_key_suffix: str = "",
    ) -> Any:
        """
        Takes a GlyphBuffer scene description and renders it to a new,
        backend-specific texture object.

        This is potentially an expensive operation, and the result should be
        cached by the caller if it will be used across multiple frames.

        Returns:
            An opaque texture handle of a backend-specific type.
        """
        pass

    @abc.abstractmethod
    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: ColorRGBf = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        world_pos: WorldTilePos | None = None,
        tile_bg: colors.Color | None = None,
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates.

        Args:
            char: Character to draw (e.g., '@', 'g')
            color: Base actor color in 0-255 RGB format
            screen_x: Screen X coordinate in pixels (can be fractional)
            screen_y: Screen Y coordinate in pixels (can be fractional)
            light_intensity: RGB lighting multipliers in 0.0-1.0 range
            interpolation_alpha: Interpolation factor between previous and current
                state (0.0-1.0). Used for smooth movement between updates.
            scale_x: Horizontal scale factor (1.0 = normal width).
            scale_y: Vertical scale factor (1.0 = normal height).
            world_pos: Optional world tile coordinates used by backends that
                sample lighting directly on GPU.
            tile_bg: Optional tile background RGB in 0-255 format. Backends that
                perform actor contrast checks in shader space can use this value.
        """
        pass

    @abc.abstractmethod
    def draw_actor_shadow(
        self,
        char: str,
        screen_x: float,
        screen_y: float,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_pixels: float,
        shadow_alpha: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        fade_tip: bool = True,
    ) -> None:
        """Draw a projected glyph shadow as a stretched parallelogram quad."""
        pass

    @abc.abstractmethod
    def draw_actor_outline(
        self,
        char: str,
        screen_x: float,
        screen_y: float,
        color: colors.Color,
        alpha: Opacity,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw an outlined version of a character glyph for combat targeting.

        Renders a 1-pixel outline tracing the glyph shape. The outline is
        pre-generated in an outlined tileset and tinted with the specified color.

        Args:
            char: Character to draw (e.g., 'g' for goblin)
            screen_x: Screen X coordinate in pixels (can be fractional)
            screen_y: Screen Y coordinate in pixels (can be fractional)
            color: RGB color to tint the outline
            alpha: Opacity of the outline (0.0-1.0), used for shimmer effect
            scale_x: Horizontal scale factor (1.0 = normal width).
            scale_y: Vertical scale factor (1.0 = normal height).
        """
        pass

    @abc.abstractmethod
    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get the (x_scale, y_scale) factor for high-DPI displays.

        Returns (1.0, 1.0) for standard DPI displays.
        Returns higher values (e.g., 2.0, 2.0) for high-DPI displays.

        This is used to properly scale UI elements like cursors on high-DPI displays.
        """
        pass

    @abc.abstractmethod
    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draws the active cursor using data from the cursor manager."""
        pass

    @abc.abstractmethod
    def draw_tile_highlight(
        self,
        root_x: float,
        root_y: float,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draws a semi-transparent highlight over a single tile."""
        pass

    @abc.abstractmethod
    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem | None = None,
    ) -> None:
        """
        Renders all particles from a SubTileParticleSystem for a specific layer.

        This method is responsible for iterating through the active particles,
        culling those outside the viewport, calculating their final screen
        positions, and drawing them.

        Args:
            particle_system: The data source containing all particle states.
            layer: The specific layer to render (e.g., UNDER_ACTORS or OVER_ACTORS).
            viewport_bounds: The currently visible area in the GraphicsContext's
                             internal tile coordinates
                             (e.g., world_view.width x world_view.height).
            view_offset: The offset of the viewport in root console coordinates,
                         used for calculating final screen positions.
        """
        pass

    @abc.abstractmethod
    def render_decals(
        self,
        decal_system: DecalSystem,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem,
        game_time: float,
    ) -> None:
        """
        Renders all visible decals from a DecalSystem.

        Decals are persistent tile-based visual marks (like blood splatters)
        that should be rendered after the background but before actors and
        particles.

        Args:
            decal_system: The data source containing all decal states.
            viewport_bounds: The currently visible area in tile coordinates.
            view_offset: The offset of the viewport in root console coordinates.
            viewport_system: The viewport system for coordinate conversion.
            game_time: Current game time for calculating decal alpha (fade-out).
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
    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        pass

    @abc.abstractmethod
    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """
        Converts root console coordinates to final, hardware-rendered
        screen pixel coordinates.

        This critical transformation step accounts for the current window size,
        aspect ratio, and any letterboxing or pillarboxing applied by the
        GraphicsContext.

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
    def draw_texture_alpha(
        self,
        texture: Any,
        screen_x: PixelCoord,
        screen_y: PixelCoord,
        alpha: Opacity,
    ) -> None:
        """Draw a texture at pixel coordinates with alpha modulation.

        Used for effects like floating text that need sub-tile positioning
        and fade effects.

        Args:
            texture: Backend-specific texture object
            screen_x: X position in screen pixels
            screen_y: Y position in screen pixels
            alpha: Overall opacity (0.0 to 1.0)
        """
        pass

    @abc.abstractmethod
    def draw_background(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
        offset_x_pixels: float = 0.0,
        offset_y_pixels: float = 0.0,
    ) -> None:
        """Draws the main world background texture. This is an immediate draw call
        that should happen before other rendering.

        Args:
            texture: The texture to draw.
            x_tile: X position in tiles.
            y_tile: Y position in tiles.
            width_tiles: Width in tiles.
            height_tiles: Height in tiles.
            offset_x_pixels: Optional sub-pixel X offset for screen shake.
            offset_y_pixels: Optional sub-pixel Y offset for screen shake.
        """
        pass

    @abc.abstractmethod
    def draw_rect_outline(
        self,
        px_x: int,
        px_y: int,
        px_w: int,
        px_h: int,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draws an unfilled rectangle outline directly to the screen."""
        pass

    @property
    @abc.abstractmethod
    def coordinate_converter(self) -> CoordinateConverter:
        """Provides access to the coordinate converter for converting between
        console and screen coordinates."""
        pass

    @abc.abstractmethod
    def create_canvas(self, transparent: bool = True) -> Any:
        """Creates a backend-appropriate canvas for drawing operations."""
        pass

    @abc.abstractmethod
    def release_texture(self, texture: Any) -> None:
        """Release a texture's GPU resources.

        Call this when done with a texture created via texture_from_numpy().
        Each backend handles cleanup appropriately for its texture type.

        Args:
            texture: The backend-specific texture object to release.
        """
        pass

    @abc.abstractmethod
    def draw_debug_tile_grid(
        self,
        view_origin: WorldTilePos,
        view_size: tuple[int, int],
        offset_pixels: ViewOffset,
    ) -> None:
        """Render a debug tile grid overlay for the given view bounds.

        Args:
            view_origin: Top-left corner of the view in tile coordinates.
            view_size: Size of the view in tiles (width, height).
            offset_pixels: Pixel offset for screen shake effects.
        """
        pass

    def set_atmospheric_layer(
        self,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: Any | None,
        explored_texture: Any | None,
        visible_texture: Any | None,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        strength: float,
        tint_color: colors.Color,
        drift_offset: ViewOffset,
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        blend_mode: str,
        pixel_bounds: PixelRect,
    ) -> None:
        """Queue or ignore atmospheric layer data.

        Backends that do not support atmospheric rendering intentionally do nothing.
        """
        return

    def set_noise_seed(self, seed: int) -> None:
        """Set the noise seed for sub-tile brightness variation.

        Called once per frame before render_glyph_buffer_to_texture() so the
        fragment shader can produce deterministic per-pixel brightness noise.
        Backends that do not support this intentionally do nothing.
        """
        return

    def set_noise_tile_offset(self, offset_x: int, offset_y: int) -> None:
        """Set the world-space tile offset for stable noise hashing.

        Converts buffer-space tile indices to world coordinates so the noise
        pattern stays anchored to world tiles regardless of camera scrolling.
        Backends that do not support this intentionally do nothing.
        """
        return

    def cleanup(self) -> None:
        """Release backend resources before shutdown.

        Backends that do not own explicit resources intentionally do nothing.
        """
        return

    @contextmanager
    def shadow_pass(self) -> Iterator[None]:
        """Context manager that brackets shadow geometry in the vertex buffer.

        The default implementation is a no-op.  Backends override this to record
        vertex-buffer offsets so that shadow quads can be drawn in a separate
        render pass with different blend state.
        """
        yield
