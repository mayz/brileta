import moderngl
import numpy as np

from catley.util.glyph_buffer import GlyphBuffer
from catley.util.tilesets import unicode_to_cp437

# Vertex structure for rendering
VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class TextureRenderer:
    """
    A specialized class for rendering GlyphBuffer objects into moderngl.Texture objects.
    This isolates the logic for UI rendering from the main game world rendering.
    """

    # Maximum dimensions we expect for glyph buffers - can be adjusted if needed
    MAX_BUFFER_WIDTH = 200
    MAX_BUFFER_HEIGHT = 100

    def __init__(
        self,
        mgl_context: moderngl.Context,
        atlas_texture: moderngl.Texture,
        tile_dimensions: tuple[int, int],
        uv_map: np.ndarray,
    ) -> None:
        """
        Initialize the TextureRenderer.

        Args:
            mgl_context: The ModernGL context
            atlas_texture: The main atlas texture for glyphs
            tile_dimensions: (width, height) of each tile in pixels
            uv_map: Pre-calculated UV coordinates for all 256 characters
        """
        self.mgl_context = mgl_context
        self.atlas_texture = atlas_texture
        self.tile_dimensions = tile_dimensions
        self.uv_map = uv_map
        self.SOLID_BLOCK_CHAR = 9608  # Unicode █ character

        # Create dedicated shader program for texture rendering
        self.texture_program = self._create_texture_shader_program()

        # Pre-allocate vertex buffer for maximum expected size
        max_cells = self.MAX_BUFFER_WIDTH * self.MAX_BUFFER_HEIGHT
        max_vertices = max_cells * 12  # 2 quads * 6 vertices per quad
        self.cpu_vertex_buffer = np.zeros(max_vertices, dtype=VERTEX_DTYPE)

        # Create persistent VBO with dynamic updates
        self.vbo = self.mgl_context.buffer(
            reserve=max_vertices * VERTEX_DTYPE.itemsize, dynamic=True
        )
        self.vao = self.mgl_context.vertex_array(
            self.texture_program,
            [(self.vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )

    def _create_texture_shader_program(self) -> moderngl.Program:
        """
        Creates the GLSL program for rendering to an off-screen texture (FBO).
        This shader normalizes coordinates to the texture's own size and does NOT
        flip the Y-axis.
        """
        vertex_shader = """
            #version 330
            // Renders vertices to a texture, normalizing to the texture's own
            // dimensions.
            // IMPORTANT: This shader does NOT flip the Y-axis. The vertices are
            // pre-flipped.
            in vec2 in_vert;       // Input vertex position in PRE-FLIPPED pixels
            in vec2 in_uv;
            in vec4 in_color;

            out vec2 v_uv;
            out vec4 v_color;

            uniform vec2 u_texture_size; // The size of the TARGET TEXTURE in pixels

            void main() {
                v_uv = in_uv;
                v_color = in_color;

                // Normalize to clip space (-1 to 1) for the texture
                float x = (in_vert.x / u_texture_size.x) * 2.0 - 1.0;
                float y = (in_vert.y / u_texture_size.y) * 2.0 - 1.0;

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
        program["u_atlas"].value = 0
        return program

    def _encode_glyph_buffer_to_vertices(
        self, glyph_buffer: GlyphBuffer
    ) -> tuple[np.ndarray, int]:
        """
        Translates a GlyphBuffer into the pre-allocated vertex buffer.

        Args:
            glyph_buffer: The GlyphBuffer to encode

        Returns:
            Tuple of (vertex_data slice, vertex_count) for compatibility
        """
        w, h = glyph_buffer.data.shape
        num_cells = h * w
        if num_cells == 0:
            return np.array([], dtype=VERTEX_DTYPE), 0

        # Verify buffer size is sufficient
        max_vertices_needed = num_cells * 12  # 2 quads * 6 vertices per quad
        if max_vertices_needed > len(self.cpu_vertex_buffer):
            raise ValueError(
                f"GlyphBuffer too large: needs {max_vertices_needed} vertices, "
                f"buffer has {len(self.cpu_vertex_buffer)}"
            )

        tile_w, tile_h = self.tile_dimensions
        vertex_idx = 0

        # Pre-normalize colors for all cells using vectorized operations
        fg_colors_raw = glyph_buffer.data["fg"]
        bg_colors_raw = glyph_buffer.data["bg"]
        fg_colors_norm = fg_colors_raw.astype(np.float32) / 255.0
        bg_colors_norm = bg_colors_raw.astype(np.float32) / 255.0

        # Pre-cache background UV coordinates (solid block character)
        bg_char = unicode_to_cp437(self.SOLID_BLOCK_CHAR)
        bg_uv = self.uv_map[bg_char]
        bg_u1, bg_v1, bg_u2, bg_v2 = bg_uv

        # Pre-calculate screen coordinates to avoid per-cell multiplication
        screen_x_coords = np.arange(w) * tile_w
        screen_y_coords = np.arange(h) * tile_h
        screen_x2_coords = screen_x_coords + tile_w
        screen_y2_coords = screen_y_coords + tile_h

        for y_console in range(h):
            # Cache screen Y coordinates for this row
            screen_y = screen_y_coords[y_console]
            screen_y2 = screen_y2_coords[y_console]

            for x_console in range(w):
                cell = glyph_buffer.data[x_console, y_console]
                char = cell["ch"]

                # Use pre-calculated screen coordinates
                screen_x = screen_x_coords[x_console]
                screen_x2 = screen_x2_coords[x_console]

                # Get pre-normalized colors
                fg_color_norm = fg_colors_norm[x_console, y_console]
                bg_color_norm = bg_colors_norm[x_console, y_console]

                # Convert Unicode character to CP437 and get UV coordinates
                fg_char = unicode_to_cp437(char)
                fg_uv = self.uv_map[fg_char]
                fg_u1, fg_v1, fg_u2, fg_v2 = fg_uv

                # Write BG quad as structured array slice (6 vertices)
                self.cpu_vertex_buffer[vertex_idx : vertex_idx + 6] = [
                    ((screen_x, screen_y), (bg_u1, bg_v1), bg_color_norm),
                    ((screen_x2, screen_y), (bg_u2, bg_v1), bg_color_norm),
                    ((screen_x, screen_y2), (bg_u1, bg_v2), bg_color_norm),
                    ((screen_x2, screen_y), (bg_u2, bg_v1), bg_color_norm),
                    ((screen_x, screen_y2), (bg_u1, bg_v2), bg_color_norm),
                    ((screen_x2, screen_y2), (bg_u2, bg_v2), bg_color_norm),
                ]
                vertex_idx += 6

                # Write FG quad as structured array slice (6 vertices)
                self.cpu_vertex_buffer[vertex_idx : vertex_idx + 6] = [
                    ((screen_x, screen_y), (fg_u1, fg_v1), fg_color_norm),
                    ((screen_x2, screen_y), (fg_u2, fg_v1), fg_color_norm),
                    ((screen_x, screen_y2), (fg_u1, fg_v2), fg_color_norm),
                    ((screen_x2, screen_y), (fg_u2, fg_v1), fg_color_norm),
                    ((screen_x, screen_y2), (fg_u1, fg_v2), fg_color_norm),
                    ((screen_x2, screen_y2), (fg_u2, fg_v2), fg_color_norm),
                ]
                vertex_idx += 6

        return self.cpu_vertex_buffer[:vertex_idx], vertex_idx

    def render(
        self, glyph_buffer: GlyphBuffer, target_fbo: moderngl.Framebuffer
    ) -> None:
        """
        Renders a GlyphBuffer into the provided target framebuffer.

        Args:
            glyph_buffer: The GlyphBuffer to render
            target_fbo: The pre-allocated framebuffer to render into
        """
        # Calculate the pixel dimensions for the shader uniforms
        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        if width_px == 0 or height_px == 0:
            return  # Nothing to render

        # Set up target FBO for rendering
        target_fbo.use()
        target_fbo.clear()

        # Encode vertices directly into pre-allocated buffer
        vertex_data, vertex_count = self._encode_glyph_buffer_to_vertices(glyph_buffer)

        if vertex_count > 0:
            # Update the persistent VBO with new vertex data
            self.vbo.write(vertex_data.tobytes())

            # Set uniforms and render using persistent VAO
            self.texture_program["u_texture_size"].value = (width_px, height_px)
            self.atlas_texture.use(location=0)

            self.vao.render(moderngl.TRIANGLES, vertices=vertex_count)

        # Restore default framebuffer
        if self.mgl_context.screen:
            self.mgl_context.screen.use()

    def release(self) -> None:
        """Clean up GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.texture_program.release()
