from typing import Any

import moderngl
import numpy as np
from PIL import Image as PILImage

from catley import colors, config
from catley.backends.gl_window import GLWindow
from catley.game.enums import BlendMode
from catley.types import (
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    TileDimensions,
)
from catley.util.coordinates import CoordinateConverter, Rect
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from catley.view.render.graphics import GraphicsContext

from .screen_renderer import VERTEX_DTYPE, ScreenRenderer
from .texture_renderer import TextureRenderer


class ModernGLGraphicsContext(GraphicsContext):
    """
    A high-performance GraphicsContext that uses ModernGL. It uses a two-shader
    architecture to correctly handle different rendering contexts.
    """

    def __init__(self, window: GLWindow, context: moderngl.Context) -> None:
        self.window = window
        self.mgl_context = context
        self.mgl_context.enable(moderngl.BLEND)
        self.mgl_context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # --- Shader and Texture Setup ---
        self.atlas_texture = self._load_atlas_texture()
        self.uv_map = self._precalculate_uv_map()

        # Calculate tile dimensions for TextureRenderer
        self._tile_dimensions = (
            self.atlas_texture.width // config.TILESET_COLUMNS,
            self.atlas_texture.height // config.TILESET_ROWS,
        )

        # --- ScreenRenderer for main screen rendering ---
        self.screen_renderer = ScreenRenderer(self.mgl_context, self.atlas_texture)

        # --- TextureRenderer for UI rendering ---
        self.texture_renderer = TextureRenderer(
            self.mgl_context, self.atlas_texture, self._tile_dimensions, self.uv_map
        )

        self.SOLID_BLOCK_CHAR = 219
        self.letterbox_geometry = (0, 0, 0, 0)

        # Set initial viewport and update dimensions
        window_width, window_height = self.window.get_size()
        self.mgl_context.viewport = (0, 0, window_width, window_height)
        self.update_dimensions()

    # --- Initialization Helpers ---

    def _load_atlas_texture(self) -> moderngl.Texture:
        """Loads the tileset PNG and creates a ModernGL texture."""
        img = PILImage.open(str(config.TILESET_PATH)).convert("RGBA")
        pixels = np.array(img, dtype="u1")
        magenta_mask = (
            (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 255)
        )
        pixels[magenta_mask, 3] = 0
        texture = self.mgl_context.texture(img.size, 4, pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return texture

    def _precalculate_uv_map(self) -> np.ndarray:
        """Pre-calculates UV coordinates for all 256 possible characters."""
        uv_map = np.zeros((256, 4), dtype="f4")
        for i in range(256):
            col = i % config.TILESET_COLUMNS
            row = i // config.TILESET_COLUMNS
            u1 = col / config.TILESET_COLUMNS
            v1 = row / config.TILESET_ROWS
            u2 = (col + 1) / config.TILESET_COLUMNS
            v2 = (row + 1) / config.TILESET_ROWS
            uv_map[i] = [u1, v1, u2, v2]
        return uv_map

    # --- Frame Lifecycle & Main Rendering ---

    def prepare_to_present(self) -> None:
        """Clears the screen and resets the vertex buffer for a new frame."""
        self.mgl_context.clear(0.0, 0.0, 0.0)
        self.screen_renderer.begin_frame()

    def finalize_present(self) -> None:
        """Uploads all vertex data and draws the scene in a single batch."""
        self.screen_renderer.render_to_screen(
            self.window.get_size(), letterbox_geometry=self.letterbox_geometry
        )

    # --- GraphicsContext ABC Implementation ---

    @property
    def tile_dimensions(self) -> TileDimensions:
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        return config.SCREEN_WIDTH

    @property
    def console_height_tiles(self) -> int:
        return config.SCREEN_HEIGHT

    @property
    def coordinate_converter(self) -> CoordinateConverter:
        return self._coordinate_converter

    @property
    def program(self) -> moderngl.Program:
        """Backward compatibility: returns the screen program."""
        return self.screen_renderer.screen_program

    def render_glyph_buffer_to_texture(self, glyph_buffer: GlyphBuffer) -> Any:
        """Renders a GlyphBuffer to a new, correctly oriented texture."""
        return self.texture_renderer.render(glyph_buffer)

    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Adds a pre-rendered texture to the main screen vertex buffer."""
        if not isinstance(texture, moderngl.Texture):
            return

        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Calculate dimensions (unused but kept for potential future use)
        # px_w = px_x2 - px_x1
        # px_h = px_y2 - px_y1

        # This is now a batched operation. We need to flush the current buffer,
        # render it, then set up to render this texture, render it, and then
        # restore the state for subsequent batched rendering.
        # This is complex. A simpler approach is to add its vertices to a
        # separate buffer
        # or handle it in a separate pass. For now, let's keep it immediate.

        # Flush any pending vertex buffer data first
        if self.screen_renderer.vertex_count > 0:
            self.finalize_present()

        # Set up for immediate rendering with the custom texture
        self.screen_renderer.screen_program[
            "u_letterbox"
        ].value = self.letterbox_geometry
        texture.use(location=0)

        # Create vertices for this single texture quad
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Use non-flipped UVs
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        temp_vbo = self.mgl_context.buffer(vertices.tobytes())
        temp_vao = self.mgl_context.vertex_array(
            self.screen_renderer.screen_program,
            [(temp_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )
        temp_vao.render(moderngl.TRIANGLES, vertices=6)

        temp_vao.release()
        temp_vbo.release()

        # Restore atlas texture for other rendering
        self.atlas_texture.use(location=0)

    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
    ) -> None:
        """Draws a single character by adding its quad to the vertex buffer."""
        final_color = (
            color[0] / 255.0 * light_intensity[0],
            color[1] / 255.0 * light_intensity[1],
            color[2] / 255.0 * light_intensity[2],
            alpha,
        )
        uv_coords = self.uv_map[ord(char)]

        # Calculate scaled tile dimensions based on letterbox scaling
        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry
        w = scaled_w / self.console_width_tiles
        h = scaled_h / self.console_height_tiles

        self.screen_renderer.add_quad(screen_x, screen_y, w, h, uv_coords, final_color)

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        """Render particles by adding each one to the vertex buffer."""
        # Loop through active particles, same logic as TCOD implementation
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

            # Render this particle using the same approach as draw_actor_smooth
            self._draw_particle_to_buffer(
                particle_system.chars[i],
                tuple(particle_system.colors[i]),
                screen_x,
                screen_y,
                alpha,
            )

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

    def _draw_particle_to_buffer(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        alpha: float,
    ) -> None:
        """Draw a single particle by adding its quad to the vertex buffer."""
        # Convert color to 0-1 range and apply alpha
        final_color = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            max(0.0, min(1.0, alpha)),  # Clamp alpha to valid range
        )

        # Get UV coordinates for the particle character
        char_code = ord(char) if char else ord(" ")
        uv_coords = self.uv_map[char_code]
        w, h = self.tile_dimensions

        # Add the particle quad to the vertex buffer
        self.screen_renderer.add_quad(screen_x, screen_y, w, h, uv_coords, final_color)

    # --- Coordinate and Dimension Management ---

    def update_dimensions(self) -> None:
        """Recalculates letterboxing and updates the coordinate converter."""
        window_width, window_height = self.window.get_size()
        console_width, console_height = (
            self.console_width_tiles,
            self.console_height_tiles,
        )
        console_aspect = console_width / console_height
        window_aspect = window_width / window_height

        if console_aspect > window_aspect:
            scaled_w = window_width
            scaled_h = int(window_width / console_aspect)
            offset_x = 0
            offset_y = (window_height - scaled_h) // 2
        else:
            scaled_h = window_height
            scaled_w = int(window_height * console_aspect)
            offset_x = (window_width - scaled_w) // 2
            offset_y = 0

        self.letterbox_geometry = (offset_x, offset_y, scaled_w, scaled_h)
        self._coordinate_converter = self._create_coordinate_converter()

        # Set OpenGL viewport to full window size
        self.mgl_context.viewport = (0, 0, window_width, window_height)

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """Converts root console coordinates to final screen pixel coordinates."""
        self._ensure_coord_cache_valid()
        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry
        screen_x = offset_x + console_x * (scaled_w / self.console_width_tiles)
        screen_y = offset_y + console_y * (scaled_h / self.console_height_tiles)
        return (screen_x, screen_y)

    def _ensure_coord_cache_valid(self):
        # This can be removed or simplified if update_dimensions is called on
        # every resize
        pass

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        return self.coordinate_converter.pixel_to_tile(pixel_x, pixel_y)

    def _create_coordinate_converter(self) -> CoordinateConverter:
        _, _, scaled_w, scaled_h = self.letterbox_geometry
        return CoordinateConverter(
            console_width_in_tiles=self.console_width_tiles,
            console_height_in_tiles=self.console_height_tiles,
            renderer_scaled_width=scaled_w,
            renderer_scaled_height=scaled_h,
        )

    # --- Internal Helpers ---

    def draw_mouse_cursor(self, cursor_manager: Any) -> None:
        """Draws the active cursor using data from the cursor manager."""
        cursor_data = cursor_manager.cursors.get(cursor_manager.active_cursor_type)
        if not cursor_data:
            return

        # Create texture if not already cached
        if cursor_data.texture is None:
            # Convert numpy array to ModernGL texture
            pixels = cursor_data.pixels
            height, width = pixels.shape[:2]
            texture = self.mgl_context.texture((width, height), 4, pixels.tobytes())
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            cursor_data.texture = texture

        texture = cursor_data.texture
        hotspot_x, hotspot_y = cursor_data.hotspot

        # Calculate position adjusted for hotspot
        dest_x = cursor_manager.mouse_pixel_x - hotspot_x
        dest_y = cursor_manager.mouse_pixel_y - hotspot_y
        dest_w = texture.width
        dest_h = texture.height

        # Use white color (no tint) and full alpha
        color_rgba = (1.0, 1.0, 1.0, 1.0)
        u1, v1, u2, v2 = 0.0, 1.0, 1.0, 0.0  # Flipped V for PIL-based textures

        # Create vertex data for a single quad
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)

        # Triangle 1 (vertices 0, 1, 2)
        vertices[0] = ((dest_x, dest_y), (u1, v1), color_rgba)
        vertices[1] = ((dest_x + dest_w, dest_y), (u2, v1), color_rgba)
        vertices[2] = ((dest_x, dest_y + dest_h), (u1, v2), color_rgba)

        # Triangle 2 (vertices 3, 4, 5)
        vertices[3] = ((dest_x + dest_w, dest_y), (u2, v1), color_rgba)
        vertices[4] = ((dest_x, dest_y + dest_h), (u1, v2), color_rgba)
        vertices[5] = ((dest_x + dest_w, dest_y + dest_h), (u2, v2), color_rgba)

        # Flush any pending vertex buffer data first
        if self.screen_renderer.vertex_count > 0:
            self.finalize_present()

        # Set up for immediate rendering with the custom texture
        self.screen_renderer.screen_program[
            "u_letterbox"
        ].value = self.letterbox_geometry
        texture.use(location=0)

        # Create temporary VBO and render immediately
        temp_vbo = self.mgl_context.buffer(vertices.tobytes())
        temp_vao = self.mgl_context.vertex_array(
            self.screen_renderer.screen_program,
            [(temp_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )
        temp_vao.render(moderngl.TRIANGLES, vertices=6)

        # Clean up temporary objects
        temp_vao.release()
        temp_vbo.release()

        # Restore atlas texture for other rendering
        self.atlas_texture.use(location=0)

    def draw_tile_highlight(
        self, root_x: int, root_y: int, color: colors.Color, alpha: Opacity
    ) -> None:
        px_x, px_y = self.console_to_screen_coords(root_x, root_y)
        w, h = self.tile_dimensions
        uv = self.uv_map[self.SOLID_BLOCK_CHAR]
        color_rgba = (color[0] / 255, color[1] / 255, color[2] / 255, alpha)
        self.screen_renderer.add_quad(px_x, px_y, w, h, uv, color_rgba)

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        """Apply a circular environmental effect like lighting or fog."""
        if radius <= 0 or intensity <= 0:
            return

        # Convert tile coordinates to screen coordinates
        screen_x, screen_y = self.console_to_screen_coords(*position)

        # Calculate pixel radius based on tile dimensions
        px_radius = max(1, int(radius * self.tile_dimensions[0]))

        # Create a circular gradient texture for the effect
        size = px_radius * 2 + 1
        center = px_radius

        # Create gradient using numpy
        y_coords, x_coords = np.ogrid[:size, :size]
        distances = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)

        # Create alpha gradient that fades from center to edge
        alpha = np.clip(1.0 - distances / px_radius, 0.0, 1.0)

        # Create RGBA texture data
        effect_data = np.zeros((size, size, 4), dtype=np.uint8)
        effect_data[:, :, :3] = 255  # White base color
        effect_data[:, :, 3] = (alpha * 255).astype(np.uint8)

        # Create temporary texture for this effect
        effect_texture = self.mgl_context.texture(
            (size, size), 4, effect_data.tobytes()
        )
        effect_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Calculate final color with intensity
        final_color = (
            tint_color[0] / 255.0 * intensity,
            tint_color[1] / 255.0 * intensity,
            tint_color[2] / 255.0 * intensity,
            intensity,
        )

        # Position the effect centered on the given position
        dest_x = screen_x - px_radius
        dest_y = screen_y - px_radius
        uv_coords = (0.0, 0.0, 1.0, 1.0)  # Use full texture

        # Temporarily bind the effect texture
        current_atlas = self.atlas_texture
        effect_texture.use(location=0)
        self.screen_renderer.add_quad(
            dest_x, dest_y, size, size, uv_coords, final_color
        )
        # Restore atlas texture
        current_atlas.use(location=0)

        # Clean up temporary texture
        effect_texture.release()

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draws a raw, unfilled rectangle outline for debugging."""
        color_rgba = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)
        uv_coords = self.uv_map[self.SOLID_BLOCK_CHAR]

        # Top edge
        self.screen_renderer.add_quad(px_x, px_y, px_w, 1, uv_coords, color_rgba)
        # Bottom edge
        self.screen_renderer.add_quad(
            px_x, px_y + px_h - 1, px_w, 1, uv_coords, color_rgba
        )
        # Left edge
        self.screen_renderer.add_quad(px_x, px_y, 1, px_h, uv_coords, color_rgba)
        # Right edge
        self.screen_renderer.add_quad(
            px_x + px_w - 1, px_y, 1, px_h, uv_coords, color_rgba
        )

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get the (x_scale, y_scale) factor for high-DPI displays."""
        # For ModernGL backend, we don't have direct access to DPI info
        # Return standard scaling for now
        return (1.0, 1.0)

    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Creates a backend-specific texture from a raw NumPy RGBA pixel array."""
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("Expected RGBA or RGB image array")

        height, width = pixels.shape[:2]
        components = pixels.shape[2]

        # Ensure we have RGBA data
        if components == 3:
            rgba_pixels = np.zeros((height, width, 4), dtype=pixels.dtype)
            rgba_pixels[:, :, :3] = pixels
            rgba_pixels[:, :, 3] = 255 if not transparent else 0
            pixels = rgba_pixels

        texture = self.mgl_context.texture((width, height), 4, pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return texture

    def create_canvas(self, transparent: bool = True) -> Any:
        """Creates a ModernGL canvas that uses GlyphBuffer."""
        from catley.backends.moderngl.canvas import ModernGLCanvas

        return ModernGLCanvas(self, transparent)
