"""Low-level renderer bridging tcod and SDL.

Instantiate with a tcod Context and root Console.
Use clear_console(), draw_text(), blit_console(),
and present_frame() to control the frame pipeline."""

from __future__ import annotations

from collections import OrderedDict
from typing import cast

import numpy as np
import tcod.render
import tcod.sdl.render
from tcod.console import Console

from catley import colors
from catley.game.enums import BlendMode
from catley.util.coordinates import CoordinateConverter, TileDimensions


class Renderer:
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
        self.tile_dimensions = tile_dimensions

        # Set up coordinate conversion
        self.coordinate_converter = self._create_coordinate_converter()

        # Set up smooth actor rendering
        self._actor_texture_cache = {}
        self._effect_texture_cache: OrderedDict[int, tcod.sdl.render.Texture] = (
            OrderedDict()
        )
        self._effect_cache_limit = 20
        self._tileset = context.sdl_atlas.tileset if context.sdl_atlas else None

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

    def draw_particle_smooth(
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
            self._effect_texture_cache[px_radius] = texture

            if len(self._effect_texture_cache) > self._effect_cache_limit:
                old_radius, old_texture = self._effect_texture_cache.popitem(last=False)
                if hasattr(old_texture, "destroy"):
                    old_texture.destroy()  # pyright: ignore[reportAttributeAccessIssue]

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

    def draw_text(
        self, x: int, y: int, text: str, fg: colors.Color = colors.WHITE
    ) -> None:
        """Draw text at a specific position with a given color."""
        self.root_console.print(x=x, y=y, text=text, fg=fg)

    def blit_console(
        self,
        source: Console,
        dest: Console,
        dest_x: int = 0,
        dest_y: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Blit one console onto another."""
        source.blit(
            dest=dest,
            dest_x=dest_x,
            dest_y=dest_y,
            width=width or source.width,
            height=height or source.height,
        )

    def prepare_to_present(self) -> None:
        """Converts the root console to a texture and copies it to the backbuffer."""
        # Convert TCOD console to SDL texture
        console_texture = self.console_render.render(self.root_console)

        left_x, top_y = self.console_to_screen_coords(0, 0)
        right_x, bottom_y = self.console_to_screen_coords(
            self.root_console.width, self.root_console.height
        )

        width = right_x - left_x
        height = bottom_y - top_y

        dest = (int(left_x), int(top_y), int(width), int(height))

        self.sdl_renderer.clear()
        self.sdl_renderer.copy(console_texture, dest=dest)

    def finalize_present(self) -> None:
        """Presents the backbuffer to the screen."""
        self.sdl_renderer.present()

    def present_frame(self) -> None:
        """Presents the final composited frame via SDL. (Now a convenience method)"""
        self.prepare_to_present()
        self.finalize_present()

    def update_dimensions(self) -> None:
        """Update coordinate converter when window dimensions change."""
        renderer_width, renderer_height = self.sdl_renderer.output_size

        # Determine tile scaling based on current window size
        tile_width = max(1, renderer_width // self.root_console.width)
        tile_height = max(1, renderer_height // self.root_console.height)
        self.tile_dimensions = (tile_width, tile_height)

        self.coordinate_converter = self._create_coordinate_converter()

    def console_to_screen_coords(
        self, console_x: float, console_y: float
    ) -> tuple[float, float]:
        """Convert console coordinates to final screen pixel coordinates,
        accounting for letterboxing."""
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
        # Use additive blending for transparent overlays, normal for opaque menus
        texture.blend_mode = (
            tcod.sdl.render.BlendMode.ADD
            if transparent
            else tcod.sdl.render.BlendMode.BLEND
        )
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
        tile_width, tile_height = self.tile_dimensions
        dest_x = x_tile * tile_width
        dest_y = y_tile * tile_height
        dest_width = width_tiles * tile_width
        dest_height = height_tiles * tile_height

        dest_rect = (dest_x, dest_y, dest_width, dest_height)
        self.sdl_renderer.copy(texture, dest=dest_rect)
