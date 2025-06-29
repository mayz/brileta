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

# Maximum number of quads (2 triangles per quad) to draw per frame.
# This should be sized to handle the entire screen + all dynamic elements.
# A 80x50 screen has 4000 cells. BG+FG = 8000 quads. Add buffer for particles etc.
MAX_QUADS = 10000

VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)

VERTEX_COMPONENTS = 8  # 2 for pos, 2 for uv, 4 for color


class ModernGLGraphicsContext(GraphicsContext):
    """
    A high-performance GraphicsContext that uses ModernGL for rendering
    via a "direct-to-buffer" approach.
    """

    def __init__(self, window: GLWindow, context: moderngl.Context) -> None:
        self.window = window
        self.mgl_context = context
        self.mgl_context.enable(moderngl.BLEND)
        self.mgl_context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # --- Shader and Texture Setup ---
        self.atlas_texture = self._load_atlas_texture()
        self.program = self._create_shader_program()
        self.uv_map = self._precalculate_uv_map()

        # --- Vertex Buffer Setup ---
        self.cpu_vertex_buffer = np.zeros(MAX_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0
        self.vbo = self.mgl_context.buffer(
            reserve=self.cpu_vertex_buffer.nbytes, dynamic=True
        )
        self.vao = self.mgl_context.vertex_array(
            self.program,
            [(self.vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )

        self._tile_dimensions = (
            self.atlas_texture.width // config.TILESET_COLUMNS,
            self.atlas_texture.height // config.TILESET_ROWS,
        )
        self.SOLID_BLOCK_CHAR = 219  # A solid block character for drawing rects
        self.letterbox_geometry = (
            0,
            0,
            0,
            0,
        )  # Initialize before calling update_dimensions
        self.update_dimensions()  # Set proper letterbox_geometry and converter

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

    def _create_shader_program(self) -> moderngl.Program:
        """Creates the GLSL program for rendering textured, colored quads."""
        vertex_shader = """
            #version 330
            in vec2 in_vert;
            in vec2 in_uv;
            in vec4 in_color;
            out vec2 v_uv;
            out vec4 v_color;
            uniform vec2 u_screen_size;
            void main() {
                v_uv = in_uv;
                v_color = in_color;
                float x = (in_vert.x / u_screen_size.x) * 2.0 - 1.0;
                float y = (in_vert.y / u_screen_size.y) * 2.0 - 1.0;
                gl_Position = vec4(x, y, 0.0, 1.0);
            }
        """
        fragment_shader = """
            #version 330
            in vec2 v_uv;
            in vec4 v_color;
            out vec4 f_color;
            uniform sampler2D u_atlas;
            void main() {
                vec4 tex_color = texture(u_atlas, v_uv);
                f_color = tex_color * v_color;
            }
        """
        program = self.mgl_context.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )
        program["u_atlas"].value = 0  # type: ignore
        return program

    def _precalculate_uv_map(self) -> np.ndarray:
        """Pre-calculates UV coordinates for all 256 possible characters."""
        uv_map = np.zeros((256, 4), dtype="f4")
        for i in range(256):
            col = i % config.TILESET_COLUMNS
            row = i // config.TILESET_COLUMNS
            u1 = col / config.TILESET_COLUMNS
            v1 = 1.0 - (row + 1) / config.TILESET_ROWS
            u2 = (col + 1) / config.TILESET_COLUMNS
            v2 = 1.0 - row / config.TILESET_ROWS
            uv_map[i] = [u1, v1, u2, v2]
        return uv_map

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
        """Provides access to the coordinate converter."""
        return self._coordinate_converter

    def prepare_to_present(self) -> None:
        """Clears the screen and resets the vertex buffer for a new frame."""
        self.mgl_context.clear(0.0, 0.0, 0.0)
        self.vertex_count = 0

    def finalize_present(self) -> None:
        """Uploads all vertex data and draws the scene in a single batch."""
        if self.vertex_count == 0:
            return

        self.program["u_screen_size"].value = self.window.get_size()  # type: ignore
        self.atlas_texture.use(location=0)

        self.vbo.write(self.cpu_vertex_buffer[: self.vertex_count].tobytes())
        self.vao.render(moderngl.TRIANGLES, vertices=self.vertex_count)

    def render_glyph_buffer_to_texture(self, glyph_buffer: GlyphBuffer) -> Any:
        """
        Takes a GlyphBuffer scene description and renders it to a new,
        backend-specific texture object.

        This is potentially an expensive operation, and the result should be
        cached by the caller if it will be used across multiple frames.

        Returns:
            An opaque texture handle of a backend-specific type.
        """
        width_px = glyph_buffer.width * self._tile_dimensions[0]
        height_px = glyph_buffer.height * self._tile_dimensions[1]
        if width_px == 0 or height_px == 0:
            return None

        dest_texture = self.mgl_context.texture((width_px, height_px), 4)
        fbo = self.mgl_context.framebuffer(color_attachments=[dest_texture])

        fbo.use()
        fbo.clear()

        # This method renders a glyph buffer in one pass.
        # It's an internal helper used by this method and potentially others.
        self._render_glyph_buffer(
            glyph_buffer, offset=(0, 0), screen_size=(width_px, height_px)
        )

        # Restore default framebuffer (may be None in standalone context)
        try:
            if self.mgl_context.screen:
                self.mgl_context.screen.use()
        except AttributeError:
            # In standalone context, there may be no screen
            pass
        fbo.release()
        return dest_texture

    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Draws a pre-rendered texture (like a UI view) to the screen."""
        if not isinstance(texture, moderngl.Texture):
            return

        px_x, px_y = self.console_to_screen_coords(x_tile, y_tile)
        px_w = width_tiles * self.tile_dimensions[0]
        px_h = height_tiles * self.tile_dimensions[1]

        # For presenting pre-baked textures, we can use a temporary, separate shader/VAO
        # or add a mode to the main shader. For simplicity, we'll just add the quad
        # to the main vertex buffer.

        # We add a single quad for the whole texture.
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Use the full texture
        color_rgba = (1.0, 1.0, 1.0, 1.0)  # White, no tint

        self._add_quad_to_buffer(px_x, px_y, px_w, px_h, (u1, v1, u2, v2), color_rgba)

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
        w, h = self.tile_dimensions

        self._add_quad_to_buffer(screen_x, screen_y, w, h, uv_coords, final_color)

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
        self._add_quad_to_buffer(screen_x, screen_y, w, h, uv_coords, final_color)

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

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry

        px_x = offset_x + console_x * (scaled_w / self.console_width_tiles)
        px_y = offset_y + console_y * (scaled_h / self.console_height_tiles)
        return px_x, px_y

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

    def _add_quad_to_buffer(self, x, y, w, h, uv_coords, color_rgba):
        """A low-level helper to add 6 vertices for a quad to the CPU buffer."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            print(
                f"WARNING: Vertex buffer full. Cannot render "
                f"quad at ({x}, {y}). Consider increasing MAX_QUADS."
            )
            return

        u1, v1, u2, v2 = uv_coords

        # OpenGL Y-coordinate is bottom-up
        y_gl = self.window.get_size()[1] - y - h

        # Create structured vertex data for the 6 vertices (2 triangles)
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)

        # Triangle 1 (vertices 0, 1, 2)
        vertices[0] = ((x, y_gl), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y_gl), (u2, v1), color_rgba)
        vertices[2] = ((x, y_gl + h), (u1, v2), color_rgba)

        # Triangle 2 (vertices 3, 4, 5)
        vertices[3] = ((x + w, y_gl), (u2, v1), color_rgba)
        vertices[4] = ((x, y_gl + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y_gl + h), (u2, v2), color_rgba)

        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

    def _render_glyph_buffer(self, glyph_buffer, offset, screen_size):
        """Internal helper to render a full glyph buffer at once."""
        vertex_data, vertex_count = self._encode_glyph_buffer_to_vertices(glyph_buffer)

        if vertex_count > 0:
            self.program["u_screen_size"].value = screen_size  # type: ignore
            self.atlas_texture.use(location=0)

            self.vbo.write(vertex_data.tobytes())
            self.vao.render(moderngl.TRIANGLES, vertices=vertex_count)

    def _encode_glyph_buffer_to_vertices(
        self, glyph_buffer: GlyphBuffer
    ) -> tuple[np.ndarray, int]:
        """
        Translates a GlyphBuffer into a NumPy array of vertex data.
        Uses a simpler approach that builds the vertex data iteratively.
        """
        h, w = glyph_buffer.data.shape
        num_cells = h * w
        if num_cells == 0:
            return np.array([], dtype=VERTEX_DTYPE), 0

        # We need 2 quads (12 vertices) per cell: one for BG, one for FG.
        num_vertices = num_cells * 12
        vertex_data = np.zeros(num_vertices, dtype=VERTEX_DTYPE)

        tile_w, tile_h = self._tile_dimensions
        vertex_idx = 0

        # Iterate through each cell and build BG + FG quads
        for y in range(h):
            for x in range(w):
                cell = glyph_buffer.data[y, x]
                char, fg_color, bg_color = cell

                # Screen position (bottom-left origin for OpenGL)
                screen_x = x * tile_w
                screen_y = (h - 1 - y) * tile_h

                # UV coordinates
                fg_uv = self.uv_map[char]
                bg_uv = self.uv_map[self.SOLID_BLOCK_CHAR]

                # Colors (normalized to 0-1)
                bg_color_norm = tuple(c / 255.0 for c in bg_color)
                fg_color_norm = tuple(c / 255.0 for c in fg_color)

                # Add BG quad (6 vertices)
                bg_vertices = np.zeros(6, dtype=VERTEX_DTYPE)
                u1, v1, u2, v2 = bg_uv
                bg_vertices[0] = ((screen_x, screen_y), (u1, v1), bg_color_norm)
                bg_vertices[1] = (
                    (screen_x + tile_w, screen_y),
                    (u2, v1),
                    bg_color_norm,
                )
                bg_vertices[2] = (
                    (screen_x, screen_y + tile_h),
                    (u1, v2),
                    bg_color_norm,
                )
                bg_vertices[3] = (
                    (screen_x + tile_w, screen_y),
                    (u2, v1),
                    bg_color_norm,
                )
                bg_vertices[4] = (
                    (screen_x, screen_y + tile_h),
                    (u1, v2),
                    bg_color_norm,
                )
                bg_vertices[5] = (
                    (screen_x + tile_w, screen_y + tile_h),
                    (u2, v2),
                    bg_color_norm,
                )

                vertex_data[vertex_idx : vertex_idx + 6] = bg_vertices
                vertex_idx += 6

                # Add FG quad (6 vertices)
                fg_vertices = np.zeros(6, dtype=VERTEX_DTYPE)
                u1, v1, u2, v2 = fg_uv
                fg_vertices[0] = ((screen_x, screen_y), (u1, v1), fg_color_norm)
                fg_vertices[1] = (
                    (screen_x + tile_w, screen_y),
                    (u2, v1),
                    fg_color_norm,
                )
                fg_vertices[2] = (
                    (screen_x, screen_y + tile_h),
                    (u1, v2),
                    fg_color_norm,
                )
                fg_vertices[3] = (
                    (screen_x + tile_w, screen_y),
                    (u2, v1),
                    fg_color_norm,
                )
                fg_vertices[4] = (
                    (screen_x, screen_y + tile_h),
                    (u1, v2),
                    fg_color_norm,
                )
                fg_vertices[5] = (
                    (screen_x + tile_w, screen_y + tile_h),
                    (u2, v2),
                    fg_color_norm,
                )

                vertex_data[vertex_idx : vertex_idx + 6] = fg_vertices
                vertex_idx += 6

        return vertex_data, num_vertices

    # All other abstract methods from GraphicsContext would need to be implemented here.
    # For example:
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
        uv_coords = (0.0, 0.0, 1.0, 1.0)  # Use full texture

        # Temporarily bind the cursor texture and add to vertex buffer
        current_atlas = self.atlas_texture
        texture.use(location=0)
        self._add_quad_to_buffer(dest_x, dest_y, dest_w, dest_h, uv_coords, color_rgba)
        # Restore atlas texture for other rendering
        current_atlas.use(location=0)

    def draw_tile_highlight(
        self, root_x: int, root_y: int, color: colors.Color, alpha: Opacity
    ) -> None:
        px_x, px_y = self.console_to_screen_coords(root_x, root_y)
        w, h = self.tile_dimensions
        uv = self.uv_map[self.SOLID_BLOCK_CHAR]
        color_rgba = (color[0] / 255, color[1] / 255, color[2] / 255, alpha)
        self._add_quad_to_buffer(px_x, px_y, w, h, uv, color_rgba)

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
        self._add_quad_to_buffer(dest_x, dest_y, size, size, uv_coords, final_color)
        # Restore atlas texture
        current_atlas.use(location=0)

        # Clean up temporary texture
        effect_texture.release()

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draws a raw, unfilled rectangle outline for debugging."""
        # Convert color to 0-1 range and add full alpha
        color_rgba = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)

        # Use solid block character for drawing the outline
        uv_coords = self.uv_map[self.SOLID_BLOCK_CHAR]

        # Draw the four edges of the rectangle using thin lines
        # Top edge
        self._add_quad_to_buffer(px_x, px_y, px_w, 1, uv_coords, color_rgba)
        # Bottom edge
        self._add_quad_to_buffer(px_x, px_y + px_h - 1, px_w, 1, uv_coords, color_rgba)
        # Left edge
        self._add_quad_to_buffer(px_x, px_y, 1, px_h, uv_coords, color_rgba)
        # Right edge
        self._add_quad_to_buffer(px_x + px_w - 1, px_y, 1, px_h, uv_coords, color_rgba)

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
