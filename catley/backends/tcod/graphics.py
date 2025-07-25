"""
A concrete implementation of the GraphicsContext interface using the TCOD library.

This module provides the `TCODGraphicsContext`, which handles all low-level drawing
operations by interfacing directly with TCOD's SDL backend. It translates the abstract
drawing commands from the `GraphicsContext` interface into specific TCOD/SDL API calls.

Key Implementation Details:
- It manages a TCOD `Context` and a root `Console`.
- It uses TCOD's `console_render` to convert `tcod.Console` objects (used by
  some views and canvases) into drawable textures.
- It leverages direct SDL calls via `tcod.sdl.render` for high-performance,
  specialized drawing tasks like smooth actor movement and particle rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import tcod.render
import tcod.sdl.render
from tcod.console import Console

from catley import colors
from catley.game.enums import BlendMode
from catley.input_handler import PixelPos
from catley.types import (
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    RootConsoleTilePos,
    TileDimensions,
)
from catley.util.caching import ResourceCache
from catley.util.coordinates import CoordinateConverter, Rect
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from catley.view.render.graphics import GraphicsContext

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


@dataclass
class LetterboxGeometry:
    """Holds the calculated geometry for letterboxing/pillarboxing."""

    scaled_w: int
    scaled_h: int
    offset_x: int
    offset_y: int
    dest_rect: tuple[int, int, int, int]


class TCODGraphicsContext(GraphicsContext):
    """Low-level graphics primitives and SDL/TCOD operations."""

    def __init__(
        self,
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: TileDimensions,
    ) -> None:
        # Extract SDL components from context
        sdl_renderer = context.sdl_renderer
        assert sdl_renderer is not None

        sdl_atlas = context.sdl_atlas
        assert sdl_atlas is not None

        self.context = context
        self.sdl_renderer: tcod.sdl.render.Renderer = sdl_renderer
        self.console_render = tcod.render.SDLConsoleRender(sdl_atlas)
        self.root_console = root_console
        self._tile_dimensions = tile_dimensions

        # Set up coordinate conversion
        self._coordinate_converter = self._create_coordinate_converter()

        # Set up smooth actor rendering
        self._actor_texture_cache = {}
        self._effect_cache_limit = 20
        self._effect_texture_cache = ResourceCache[int, tcod.sdl.render.Texture](
            name="EffectTextureCache", max_size=self._effect_cache_limit
        )
        self._tileset = context.sdl_atlas.tileset if context.sdl_atlas else None

        # The cache for temporary TCOD consoles.
        self._temp_console_cache: dict[tuple[int, int], Console] = {}

    def _get_temp_console(self, width: int, height: int) -> Console:
        """
        Retrieves a reusable tcod.Console from a cache.

        If a console of the specified size is not in the cache, a new one
        is created and stored for future use. This avoids repeated memory
        allocations each frame for views of a consistent size.
        """
        size = (width, height)
        if size not in self._temp_console_cache:
            # Cache miss: create and store a new console.
            self._temp_console_cache[size] = Console(width, height, order="F")

        # Return the cached or newly created console.
        return self._temp_console_cache[size]

    def get_render_destination_rect(self) -> tuple[int, int, int, int]:
        """
        Returns the destination rectangle for the main console texture,
        accounting for letterboxing or pillarboxing.
        """
        return self._get_letterbox_geometry().dest_rect

    def _get_letterbox_geometry(self) -> LetterboxGeometry:
        """Calculates letterbox/pillarbox geometry (single source of truth)."""
        renderer_width, renderer_height = self.sdl_renderer.output_size
        console_aspect = self.root_console.width / self.root_console.height
        window_aspect = renderer_width / renderer_height

        if console_aspect > window_aspect:
            # Letterboxed vertically (top/bottom bars)
            scaled_w = renderer_width
            scaled_h = int(renderer_width / console_aspect)
            offset_x = 0
            offset_y = (renderer_height - scaled_h) // 2
        else:
            # Letterboxed horizontally (left/right bars)
            scaled_h = renderer_height
            scaled_w = int(renderer_height * console_aspect)
            offset_x = (renderer_width - scaled_w) // 2
            offset_y = 0

        dest_rect = (offset_x, offset_y, scaled_w, scaled_h)

        return LetterboxGeometry(scaled_w, scaled_h, offset_x, offset_y, dest_rect)

    @property
    def coordinate_converter(self) -> CoordinateConverter:
        """Provides access to the coordinate converter for converting between
        console and screen coordinates."""
        return self._coordinate_converter

    @property
    def tile_dimensions(self) -> TileDimensions:
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        return self.root_console.width

    @property
    def console_height_tiles(self) -> int:
        return self.root_console.height

    def render_glyph_buffer_to_texture(
        self, glyph_buffer: GlyphBuffer, *args, **kwargs
    ) -> Any:
        """
        Takes a GlyphBuffer scene description and renders it to a new,
        backend-specific texture object.

        This is potentially an expensive operation, and the result should be
        cached by the caller if it will be used across multiple frames.

        Returns:
            An opaque texture handle of a backend-specific type.
        """
        # Adapts the generic GlyphBuffer data structure to the
        # TCOD-specific rendering pipeline.

        # Get a reusable tcod.Console from the cache.
        temp_console = self._get_temp_console(glyph_buffer.width, glyph_buffer.height)

        # Perform the fast, vectorized data copy.
        # Both GlyphBuffer and TCOD console use (width, height) indexing
        temp_console.rgba[0 : glyph_buffer.width, 0 : glyph_buffer.height] = (
            glyph_buffer.data
        )

        # 3. Use TCOD's native renderer on its native data structure.
        return self._texture_from_console(temp_console)

    def clear_console(self, console: Console) -> None:
        """Clear a console."""
        console.clear()

    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates with interpolation.

        Args:
            char: Character to draw (e.g., '@', 'g')
            color: Base actor color in 0-255 RGB format
            screen_x: Screen X coordinate in pixels (can be fractional)
            screen_y: Screen Y coordinate in pixels (can be fractional)
            light_intensity: RGB lighting multipliers in 0.0-1.0 range
            alpha: Interpolation factor (0.0 = previous state, 1.0 = current state)
        """
        if not self._tileset:
            return

        # Interpolate the screen coordinates based on alpha
        # screen_x and screen_y are already interpolated by the caller (WorldView)
        # This method just uses them as is.

        char_code = ord(char)
        tile_pixels = self._tileset.get_tile(char_code)

        # Get or create cached texture
        if char_code not in self._actor_texture_cache:
            # Ensure RGBA format
            if tile_pixels.shape[2] == 3:
                rgba_pixels = np.zeros((*tile_pixels.shape[:2], 4), dtype=np.uint8)
                rgba_pixels[:, :, :3] = tile_pixels
                rgba_pixels[:, :, 3] = 255
                tile_pixels = rgba_pixels

            texture = self.sdl_renderer.upload_texture(tile_pixels)
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
            self._actor_texture_cache[char_code] = texture

        texture = self._actor_texture_cache[char_code]

        # Apply lighting and color
        final_color = (
            int(color[0] * light_intensity[0]),
            int(color[1] * light_intensity[1]),
            int(color[2] * light_intensity[2]),
        )
        texture.color_mod = final_color

        # Draw at precise position with scaling
        shape = cast(tuple[int, int, int], tile_pixels.shape)
        texture_h = shape[0]
        texture_w = shape[1]

        # Scale texture proportionally to current tile size
        tile_w, tile_h = self.tile_dimensions
        scale = min(tile_w / texture_w, tile_h / texture_h)
        scaled_w = int(texture_w * scale)
        scaled_h = int(texture_h * scale)

        dest_rect = (int(screen_x), int(screen_y), scaled_w, scaled_h)
        self.sdl_renderer.copy(texture, dest=dest_rect)

        # Reset for next use
        texture.color_mod = (255, 255, 255)

    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draws the active cursor using data from the cursor manager."""
        cursor_data = cursor_manager.cursors.get(cursor_manager.active_cursor_type)
        if not cursor_data:
            return

        texture: tcod.sdl.render.Texture

        # LAZY TEXTURE CREATION:
        # Check if the texture has been created and cached on the CursorData object.
        if cursor_data.texture is None:
            # If not, create it now using the renderer-specific function.
            texture = self.sdl_renderer.upload_texture(cursor_data.pixels)
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
            # Cache it on the object for next time.
            cursor_data.texture = texture

        # From here, the logic is the same as before.
        texture = cursor_data.texture
        hotspot_x, hotspot_y = cursor_data.hotspot

        # Get display scale factor for high-DPI displays
        scale_x, scale_y = self.get_display_scale_factor()

        # Apply scaling to cursor size and hotspot (coordinates scaled in input handler)
        scaled_width = int(texture.width * scale_x)
        scaled_height = int(texture.height * scale_y)
        scaled_hotspot_x = int(hotspot_x * scale_x)
        scaled_hotspot_y = int(hotspot_y * scale_y)

        dest_rect = (
            cursor_manager.mouse_pixel_x - scaled_hotspot_x,
            cursor_manager.mouse_pixel_y - scaled_hotspot_y,
            scaled_width,
            scaled_height,
        )
        self.sdl_renderer.copy(texture, dest=dest_rect)

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        # The main loop logic is preserved from the original implementation.
        for i in range(particle_system.active_count):
            if particle_system.layers[i] != layer.value:
                continue

            coords = self._convert_particle_to_screen_coords(
                particle_system, i, viewport_bounds, view_offset
            )
            if coords is None:
                continue

            screen_x, screen_y = coords

            # Calculate alpha based on lifetime or flash intensity
            if not np.isnan(particle_system.flash_intensity[i]):
                alpha = particle_system.flash_intensity[i]
            else:
                alpha = particle_system.lifetimes[i] / particle_system.max_lifetimes[i]

            # Call the renderer's own low-level drawing primitive.
            self._draw_particle_smooth(
                particle_system.chars[i],
                tuple(particle_system.colors[i]),
                screen_x,
                screen_y,
                Opacity(alpha),
            )

    def _draw_particle_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        alpha: Opacity,
    ) -> None:
        """Draw a particle with alpha blending."""
        if not self._tileset:
            return

        char_code = ord(char) if char else ord(" ")
        tile_pixels = self._tileset.get_tile(char_code)

        if char_code not in self._actor_texture_cache:
            if tile_pixels.shape[2] == 3:
                rgba_pixels = np.zeros((*tile_pixels.shape[:2], 4), dtype=np.uint8)
                rgba_pixels[:, :, :3] = tile_pixels
                rgba_pixels[:, :, 3] = 255
                tile_pixels = rgba_pixels
            texture = self.sdl_renderer.upload_texture(tile_pixels)
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
            self._actor_texture_cache[char_code] = texture

        texture = self._actor_texture_cache[char_code]
        texture.color_mod = color
        texture.alpha_mod = int(max(0.0, min(1.0, alpha)) * 255)

        shape = cast(tuple[int, int, int], tile_pixels.shape)
        texture_h = shape[0]
        texture_w = shape[1]

        tile_w, tile_h = self.tile_dimensions
        scale = min(tile_w / texture_w, tile_h / texture_h)
        scaled_w = int(texture_w * scale)
        scaled_h = int(texture_h * scale)

        dest_rect = (int(screen_x), int(screen_y), scaled_w, scaled_h)
        self.sdl_renderer.copy(texture, dest=dest_rect)

        texture.color_mod = (255, 255, 255)
        texture.alpha_mod = 255

    def _convert_particle_to_screen_coords(
        self,
        particle_system: SubTileParticleSystem,
        particle_index: int,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> tuple[float, float] | None:
        """Convert particle index to screen coordinates, or None if off-screen."""
        if not (0 <= particle_index < particle_system.active_count):
            return None

        # Access data directly from the particle_system object
        sub_x, sub_y = particle_system.positions[particle_index]

        # Convert particle's sub-tile coordinates to viewport's tile coordinates
        vp_x = sub_x / particle_system.subdivision
        vp_y = sub_y / particle_system.subdivision

        # Cull particles outside the viewport bounds
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

        # Use the renderer's own converter to get final pixel coordinates
        return self.console_to_screen_coords(root_x, root_y)

    def draw_tile_highlight(
        self,
        root_x: int,
        root_y: int,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draws a semi-transparent highlight over a single tile."""
        if not self.sdl_renderer:
            return

        # Convert tile coordinates to screen pixel coordinates
        px_x1, px_y1 = self.console_to_screen_coords(root_x, root_y)
        px_x2, px_y2 = self.console_to_screen_coords(root_x + 1, root_y + 1)
        px_w = px_x2 - px_x1
        px_h = px_y2 - px_y1

        r, g, b = color
        a = int(255 * alpha)

        # Create reusable highlight texture on first use
        if not hasattr(self, "_highlight_texture"):
            # Create a 1x1 white pixel texture
            pixel_array = np.ones((1, 1, 4), dtype=np.uint8) * 255
            self._highlight_texture = self.sdl_renderer.upload_texture(pixel_array)
            self._highlight_texture.blend_mode = tcod.sdl.render.BlendMode.BLEND

        # Modify the reusable texture for this highlight
        self._highlight_texture.color_mod = (r, g, b)
        self._highlight_texture.alpha_mod = a

        # Draw the highlight rectangle by stretching the 1x1 texture
        dest_rect = (int(px_x1), int(px_y1), int(px_w), int(px_h))
        self.sdl_renderer.copy(self._highlight_texture, dest=dest_rect)

        # Reset the texture for next use
        self._highlight_texture.color_mod = (255, 255, 255)
        self._highlight_texture.alpha_mod = 255

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        """Apply a circular environmental effect using cached textures."""
        if radius <= 0 or intensity <= 0:
            return

        px_radius = max(1, int(radius * self.tile_dimensions[0]))

        texture = self._effect_texture_cache.get(px_radius)
        if texture is None:
            size = px_radius * 2 + 1
            y, x = np.ogrid[-px_radius : px_radius + 1, -px_radius : px_radius + 1]
            dist = np.sqrt(x * x + y * y)
            alpha = np.clip(1.0 - dist / px_radius, 0.0, 1.0)
            arr = np.zeros((size, size, 4), dtype=np.uint8)
            arr[..., :3] = 255
            arr[..., 3] = (alpha * 255).astype(np.uint8)

            texture = self.sdl_renderer.upload_texture(arr)
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
            self._effect_texture_cache.store(px_radius, texture)

        texture.color_mod = (
            int(tint_color[0] * intensity),
            int(tint_color[1] * intensity),
            int(tint_color[2] * intensity),
        )
        texture.alpha_mod = int(255 * intensity)

        if blend_mode is BlendMode.TINT:
            texture.blend_mode = tcod.sdl.render.BlendMode.MOD
        else:
            texture.blend_mode = tcod.sdl.render.BlendMode.BLEND

        screen_x, screen_y = self.console_to_screen_coords(*position)
        size = px_radius * 2 + 1
        dest = (int(screen_x - px_radius), int(screen_y - px_radius), size, size)
        self.sdl_renderer.copy(texture, dest=dest)

        texture.color_mod = (255, 255, 255)
        texture.alpha_mod = 255

    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        renderer_width, renderer_height = self.sdl_renderer.output_size

        # Determine tile scaling based on current window size
        tile_width = max(1, renderer_width // self.root_console.width)
        tile_height = max(1, renderer_height // self.root_console.height)
        self._tile_dimensions = (tile_width, tile_height)

        self._coordinate_converter = self._create_coordinate_converter()

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """Converts root console coordinates to final screen pixel coordinates."""
        geometry = self._get_letterbox_geometry()

        screen_x = geometry.offset_x + console_x * (
            geometry.scaled_w / self.root_console.width
        )
        screen_y = geometry.offset_y + console_y * (
            geometry.scaled_h / self.root_console.height
        )

        return screen_x, screen_y

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Converts final screen pixel coordinates to root console tile coordinates."""
        geometry = self._get_letterbox_geometry()

        # Adjust for letterboxing offsets before passing to the converter
        adjusted_pixel_x = pixel_x - geometry.offset_x
        adjusted_pixel_y = pixel_y - geometry.offset_y

        return self._coordinate_converter.pixel_to_tile(
            adjusted_pixel_x, adjusted_pixel_y
        )

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get the (x_scale, y_scale) factor for high-DPI displays."""
        sdl_window = self.context.sdl_window
        if sdl_window:
            # Get logical window size (what the OS reports)
            logical_w, logical_h = sdl_window.size
            # Get actual renderer output size (high-DPI aware)
            physical_w, physical_h = self.sdl_renderer.output_size

            scale_x = physical_w / logical_w if logical_w > 0 else 1.0
            scale_y = physical_h / logical_h if logical_h > 0 else 1.0
        else:
            scale_x = scale_y = 1.0

        return scale_x, scale_y

    def _create_coordinate_converter(self) -> CoordinateConverter:
        """Create a coordinate converter with current dimensions."""
        # Get the final, letterboxed geometry
        geometry = self._get_letterbox_geometry()

        # Initialize the converter with the SCALED dimensions, not the full
        # renderer dimensions. This simplifies the converter's logic.
        return CoordinateConverter(
            console_width_in_tiles=self.root_console.width,
            console_height_in_tiles=self.root_console.height,
            renderer_scaled_width=geometry.scaled_w,
            renderer_scaled_height=geometry.scaled_h,
        )

    def _texture_from_console(
        self, console: Console, transparent: bool = True
    ) -> tcod.sdl.render.Texture:
        """Convert a tcod Console to an SDL texture."""
        texture = self.console_render.render(console)
        # All lighting/UI textures use pre-multiplied alpha blending.
        texture.blend_mode = tcod.sdl.render.BlendMode.BLEND

        return texture

    def texture_from_numpy(
        self, pixels: np.ndarray, transparent: bool = True
    ) -> tcod.sdl.render.Texture:
        """Convert a numpy array to an SDL texture."""
        texture = self.sdl_renderer.upload_texture(pixels)
        texture.blend_mode = tcod.sdl.render.BlendMode.BLEND
        return texture

    def present_texture(
        self,
        texture: tcod.sdl.render.Texture,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Present a texture at the specified tile coordinates."""
        left, top = self.console_to_screen_coords(x_tile, y_tile)
        right, bottom = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )
        dest_width = right - left
        dest_height = bottom - top

        dest_rect = (int(left), int(top), int(dest_width), int(dest_height))
        self.sdl_renderer.copy(texture, dest=dest_rect)

    def draw_background(
        self,
        texture: tcod.sdl.render.Texture,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Draws the main world background texture. This is an immediate draw call
        that should happen before other rendering."""
        # For TCOD, draw_background is just an alias for present_texture
        self.present_texture(texture, x_tile, y_tile, width_tiles, height_tiles)

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draws a raw, unfilled rectangle directly to the screen for debugging."""
        if not self.sdl_renderer:
            return

        # Create a small 1x1 pixel texture for drawing the outline
        if not hasattr(self, "_debug_pixel_texture"):
            # Create a 1x1 white pixel texture
            pixel_array = np.ones((1, 1, 4), dtype=np.uint8) * 255
            self._debug_pixel_texture = self.sdl_renderer.upload_texture(pixel_array)
            self._debug_pixel_texture.blend_mode = tcod.sdl.render.BlendMode.BLEND

        # Set the color for the texture
        self._debug_pixel_texture.color_mod = color

        # Draw the four sides of the rectangle using the 1x1 texture
        # Top edge
        self.sdl_renderer.copy(self._debug_pixel_texture, dest=(px_x, px_y, px_w, 1))
        # Bottom edge
        self.sdl_renderer.copy(
            self._debug_pixel_texture, dest=(px_x, px_y + px_h - 1, px_w, 1)
        )
        # Left edge
        self.sdl_renderer.copy(self._debug_pixel_texture, dest=(px_x, px_y, 1, px_h))
        # Right edge
        self.sdl_renderer.copy(
            self._debug_pixel_texture, dest=(px_x + px_w - 1, px_y, 1, px_h)
        )

        # Reset color mod for next use
        self._debug_pixel_texture.color_mod = (255, 255, 255)

    def create_canvas(self, transparent: bool = True) -> Any:
        """Creates a TCOD canvas that uses Console."""
        from catley.backends.tcod.canvas import TCODConsoleCanvas

        return TCODConsoleCanvas(self, transparent)
