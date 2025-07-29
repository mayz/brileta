import moderngl
import numpy as np

from catley.types import PixelPos

from .shader_manager import ShaderManager

# Maximum number of quads (2 triangles per quad) to draw per frame.
MAX_QUADS = 10000

VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class ScreenRenderer:
    """
    Specialized class for batch-rendering all game world objects that use the
    main tileset atlas to the screen.
    """

    def __init__(
        self, mgl_context: moderngl.Context, atlas_texture: moderngl.Texture
    ) -> None:
        self.mgl_context = mgl_context
        self.atlas_texture = atlas_texture
        self.shader_manager = ShaderManager(mgl_context)

        # Create shader program
        self.screen_program = self._create_screen_shader_program()

        # Main Vertex Buffer for Screen Rendering
        self.cpu_vertex_buffer = np.zeros(MAX_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0

        # Initialize VBO with clean zero data to prevent garbage memory artifacts
        initial_data = np.zeros_like(self.cpu_vertex_buffer)
        self.vbo = self.mgl_context.buffer(initial_data.tobytes(), dynamic=True)

        self.vao = self.mgl_context.vertex_array(
            self.screen_program,
            [(self.vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )

    def _create_screen_shader_program(self) -> moderngl.Program:
        """Creates the GLSL program for rendering to the main screen with
        letterboxing support.

        Note: This shader does NOT flip the Y-axis, consistent with TextureRenderer.
        Coordinates are expected to already be in the correct coordinate system."""
        program = self.shader_manager.create_program(
            "glsl/screen/main.vert", "glsl/screen/main.frag", "screen_renderer"
        )
        program["u_atlas"].value = 0
        return program

    def begin_frame(self) -> None:
        """Reset the internal vertex count to zero at the start of a frame."""
        self.vertex_count = 0

    def add_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        color_rgba: tuple[float, float, float, float],
    ) -> None:
        """Add a quad to the vertex buffer."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return

        u1, v1, u2, v2 = uv_coords
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y), (u2, v1), color_rgba)
        vertices[2] = ((x, y + h), (u1, v2), color_rgba)
        vertices[3] = ((x + w, y), (u2, v1), color_rgba)
        vertices[4] = ((x, y + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y + h), (u2, v2), color_rgba)
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

    def render_to_screen(
        self,
        window_size: PixelPos,
        letterbox_geometry: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Main drawing method that renders all batched vertex data to the screen.

        Args:
            window_size: Full window dimensions
            letterbox_geometry: (offset_x, offset_y, scaled_w, scaled_h) for letterbox
        """
        if self.vertex_count == 0:
            return

        if letterbox_geometry is not None:
            # Pass letterbox geometry to shader for proper coordinate transformation
            self.screen_program["u_letterbox"].value = letterbox_geometry
        else:
            # No letterboxing - use full screen
            self.screen_program["u_letterbox"].value = (
                0,
                0,
                int(window_size[0]),
                int(window_size[1]),
            )

        self.atlas_texture.use(location=0)

        self.vbo.write(self.cpu_vertex_buffer[: self.vertex_count].tobytes())
        self.vao.render(moderngl.TRIANGLES, vertices=self.vertex_count)
