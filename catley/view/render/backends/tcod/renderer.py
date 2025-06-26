"""
A concrete implementation of the Renderer interface using the TCOD library.

This module provides the `TCODRenderer`, which handles all low-level drawing
operations by interfacing directly with TCOD's SDL backend. It translates the
abstract drawing commands from the `Renderer` interface into specific TCOD/SDL
API calls.

Key Implementation Details:
- It manages a TCOD `Context` and a root `Console`.
- It uses TCOD's `console_render` to convert `tcod.Console` objects (used by
  some views and canvases) into drawable textures.
- It leverages direct SDL calls via `tcod.sdl.render` for high-performance,
  specialized drawing tasks like smooth actor movement and particle rendering.
- It is responsible for handling the main window, fullscreen toggling, and
  presenting the final frame to the screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import tcod.render
import tcod.sdl.render
from tcod.console import Console

from catley import colors
from catley.game.enums import BlendMode
from catley.input_handler import PixelPos
from catley.util.caching import ResourceCache
from catley.util.coordinates import (
    CoordinateConverter,
    PixelCoord,
    Rect,
    RootConsoleTilePos,
    TileDimensions,
)
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from catley.view.render.renderer import Renderer

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class TCODRenderer(Renderer):
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
        self.coordinate_converter = self._create_coordinate_converter()

        # Set up smooth actor rendering
        self._actor_texture_cache = {}
        self._effect_cache_limit = 20
        self._effect_texture_cache = ResourceCache[int, tcod.sdl.render.Texture](
            name="EffectTextureCache", max_size=self._effect_cache_limit
        )
        self._tileset = context.sdl_atlas.tileset if context.sdl_atlas else None

    @property
    def tile_dimensions(self) -> TileDimensions:
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        return self.root_console.width

    @property
    def console_height_tiles(self) -> int:
        return self.root_console.height

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
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates.

        Args:
            char: Character to draw (e.g., '@', 'g')
            color: Base actor color in 0-255 RGB format
            screen_x: Screen X coordinate in pixels (can be fractional)
            screen_y: Screen Y coordinate in pixels (can be fractional)
            light_intensity: RGB lighting multipliers in 0.0-1.0 range
        """
        if not self._tileset:
            return

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

        dest_rect = (
            cursor_manager.mouse_pixel_x - hotspot_x,
            cursor_manager.mouse_pixel_y - hotspot_y,
            texture.width,
            texture.height,
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
                alpha,
            )

    def _draw_particle_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        alpha: float,
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
        alpha: float,
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

    def prepare_to_present(self) -> None:
        """Converts the root console to a texture and copies it to the backbuffer."""
        self.root_console.clear()

        console_texture = self.console_render.render(self.root_console)
        self.sdl_renderer.clear()

        renderer_width, renderer_height = self.sdl_renderer.output_size
        console_aspect = self.root_console.width / self.root_console.height
        window_aspect = renderer_width / renderer_height

        if console_aspect > window_aspect:
            # Letterboxing (black bars on top/bottom)
            dest_w = renderer_width
            dest_h = int(renderer_width / console_aspect)
            dest = (0, (renderer_height - dest_h) // 2, renderer_width, dest_h)
        else:
            # Pillarboxing (black bars on left/right)
            dest_w = int(renderer_height * console_aspect)
            dest_h = renderer_height
            dest = ((renderer_width - dest_w) // 2, 0, dest_w, renderer_height)

        self.sdl_renderer.copy(console_texture, dest=dest)

    def finalize_present(self) -> None:
        """Presents the backbuffer to the screen."""
        self.sdl_renderer.present()

    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        renderer_width, renderer_height = self.sdl_renderer.output_size

        # Determine tile scaling based on current window size
        tile_width = max(1, renderer_width // self.root_console.width)
        tile_height = max(1, renderer_height // self.root_console.height)
        self._tile_dimensions = (tile_width, tile_height)

        self.coordinate_converter = self._create_coordinate_converter()

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
        renderer_width, renderer_height = self.sdl_renderer.output_size
        console_aspect = self.root_console.width / self.root_console.height
        window_aspect = renderer_width / renderer_height

        if console_aspect > window_aspect:
            # Console is letterboxed vertically
            scaled_height = int(renderer_width / console_aspect)
            offset_y = (renderer_height - scaled_height) // 2
            screen_x = console_x * (renderer_width / self.root_console.width)
            screen_y = offset_y + console_y * (scaled_height / self.root_console.height)
        else:
            # Console is letterboxed horizontally
            scaled_width = int(renderer_height * console_aspect)
            offset_x = (renderer_width - scaled_width) // 2
            screen_x = offset_x + console_x * (scaled_width / self.root_console.width)
            screen_y = console_y * (renderer_height / self.root_console.height)

        return screen_x, screen_y

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Converts final screen pixel coordinates to root console tile coordinates."""
        return self.coordinate_converter.pixel_to_tile(pixel_x, pixel_y)

    def _create_coordinate_converter(self) -> CoordinateConverter:
        """Create a coordinate converter with current dimensions."""
        renderer_width, renderer_height = self.sdl_renderer.output_size

        return CoordinateConverter(
            console_width_in_tiles=self.root_console.width,
            console_height_in_tiles=self.root_console.height,
            tile_width_px=self.tile_dimensions[0],
            tile_height_px=self.tile_dimensions[1],
            renderer_width=renderer_width,
            renderer_height=renderer_height,
        )

    def texture_from_console(
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

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        self.context.present(self.root_console, keep_aspect=True)

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
