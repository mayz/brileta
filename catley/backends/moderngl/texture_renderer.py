import moderngl
import numpy as np

from catley.util.glyph_buffer import GlyphBuffer

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
        self.SOLID_BLOCK_CHAR = 219

        # Create dedicated shader program for texture rendering
        self.texture_program = self._create_texture_shader_program()

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
        Translates a GlyphBuffer into a NumPy array of vertex data, PRE-FLIPPING
        the Y-coordinates for the texture shader.

        Args:
            glyph_buffer: The GlyphBuffer to encode

        Returns:
            Tuple of (vertex_data, vertex_count)
        """
        w, h = glyph_buffer.data.shape
        num_cells = h * w
        if num_cells == 0:
            return np.array([], dtype=VERTEX_DTYPE), 0

        num_vertices = num_cells * 12  # 2 quads * 6 vertices per quad
        vertex_data = np.zeros(num_vertices, dtype=VERTEX_DTYPE)
        tile_w, tile_h = self.tile_dimensions
        vertex_idx = 0

        for y_console in range(h):
            for x_console in range(w):
                cell = glyph_buffer.data[x_console, y_console]
                char = cell["ch"]
                fg_color_rgba = tuple(cell["fg"])
                bg_color_rgba = tuple(cell["bg"])

                # Pre-flip the Y-coordinate for the texture render pass
                screen_x = x_console * tile_w
                screen_y = (h - 1 - y_console) * tile_h

                fg_uv = self.uv_map[char]
                bg_uv = self.uv_map[self.SOLID_BLOCK_CHAR]

                bg_color_norm = tuple(c / 255.0 for c in bg_color_rgba)
                fg_color_norm = tuple(c / 255.0 for c in fg_color_rgba)

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

        return vertex_data, vertex_idx

    def render(self, glyph_buffer: GlyphBuffer) -> moderngl.Texture:
        """
        Renders a GlyphBuffer to a new, correctly oriented moderngl.Texture.

        Args:
            glyph_buffer: The GlyphBuffer to render

        Returns:
            A new moderngl.Texture containing the rendered glyph buffer
        """
        # Calculate the pixel dimensions of the output texture
        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        if width_px == 0 or height_px == 0:
            # Return a minimal 1x1 transparent texture for empty buffers
            empty_data = np.zeros((1, 1, 4), dtype=np.uint8)
            return self.mgl_context.texture((1, 1), 4, empty_data.tobytes())

        # Create destination texture and FBO
        dest_texture = self.mgl_context.texture((width_px, height_px), 4)
        fbo = self.mgl_context.framebuffer(color_attachments=[dest_texture])

        # Set up FBO for rendering
        fbo.use()
        fbo.clear()

        # Get vertex data with pre-flipped Y-coordinates
        vertex_data, vertex_count = self._encode_glyph_buffer_to_vertices(glyph_buffer)

        if vertex_count > 0:
            # Create temporary VBO and VAO for this render
            temp_vbo = self.mgl_context.buffer(vertex_data.tobytes())
            temp_vao = self.mgl_context.vertex_array(
                self.texture_program,
                [(temp_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
            )

            # Set uniforms and render
            self.texture_program["u_texture_size"].value = (width_px, height_px)
            self.atlas_texture.use(location=0)

            temp_vao.render(moderngl.TRIANGLES, vertices=vertex_count)

            # Clean up temporary GL objects
            temp_vao.release()
            temp_vbo.release()

        # Restore default framebuffer and clean up
        if self.mgl_context.screen:
            self.mgl_context.screen.use()
        fbo.release()

        return dest_texture
