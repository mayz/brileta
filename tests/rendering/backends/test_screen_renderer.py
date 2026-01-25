"""
Unit tests for the ScreenRenderer class.

These tests use unittest.mock to completely mock the moderngl library,
allowing us to test the ScreenRenderer logic without needing a real OpenGL context.
"""

from unittest.mock import Mock

import moderngl
import numpy as np
import pytest

from catley.backends.moderngl.screen_renderer import (
    MAX_QUADS,
    VERTEX_DTYPE,
    ScreenRenderer,
)


class TestScreenRenderer:
    """Test suite for ScreenRenderer using mocked moderngl."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked moderngl context and related objects."""
        # Mock the moderngl context
        self.mock_mgl_context = Mock()

        # Mock atlas texture
        self.mock_atlas_texture = Mock()

        # Mock the shader program creation with proper subscripting support
        self.mock_program = Mock()
        self.mock_uniform = Mock(spec=moderngl.Uniform)
        # Configure the program to support subscripting like program["u_screen_size"]
        self.mock_program.__getitem__ = Mock(return_value=self.mock_uniform)
        self.mock_mgl_context.program.return_value = self.mock_program

        # Mock VBO and VAO
        self.mock_vbo = Mock()
        self.mock_vao = Mock()
        self.mock_mgl_context.buffer.return_value = self.mock_vbo
        self.mock_mgl_context.vertex_array.return_value = self.mock_vao

    def test_initialization(self):
        """Test that ScreenRenderer's __init__ method correctly creates its
        shader program, VBO, and VAO."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Verify initialization
        assert renderer.mgl_context == self.mock_mgl_context
        assert renderer.atlas_texture == self.mock_atlas_texture
        assert renderer.vertex_count == 0

        # Verify shader program was created
        self.mock_mgl_context.program.assert_called_once()
        assert renderer.screen_program == self.mock_program

        # Verify atlas texture sampler was bound
        self.mock_program.__getitem__.assert_called_with("u_atlas")
        assert self.mock_uniform.value == 0

        # Verify VBO was created with initialized data (not reserve)
        self.mock_mgl_context.buffer.assert_called_once()
        call_args = self.mock_mgl_context.buffer.call_args
        assert call_args.kwargs["dynamic"] is True
        # Verify the buffer was created with actual data (not reserve)
        assert len(call_args.args) > 0  # Should have data argument, not reserve
        assert renderer.vbo == self.mock_vbo

        # Verify VAO was created with correct format
        self.mock_mgl_context.vertex_array.assert_called_once_with(
            self.mock_program,
            [(self.mock_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )
        assert renderer.vao == self.mock_vao

        # Verify CPU vertex buffer was created
        assert renderer.cpu_vertex_buffer.dtype == VERTEX_DTYPE
        assert len(renderer.cpu_vertex_buffer) == MAX_QUADS * 6

    def test_begin_frame(self):
        """Test that calling begin_frame() successfully resets the internal
        vertex_count to zero."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Set vertex count to some non-zero value
        renderer.vertex_count = 42

        # Call begin_frame
        renderer.begin_frame()

        # Verify vertex count was reset to zero
        assert renderer.vertex_count == 0

    def test_add_quad(self):
        """Test that calling add_quad(...) correctly populates the internal
        cpu_vertex_buffer with exactly 6 vertices and increments the
        vertex_count by 6."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Test data
        x, y = 100.0, 200.0
        w, h = 32.0, 32.0
        uv_coords = (0.1, 0.2, 0.3, 0.4)  # u1, v1, u2, v2
        color_rgba = (1.0, 0.5, 0.0, 0.8)

        # Verify initial state
        initial_vertex_count = renderer.vertex_count
        assert initial_vertex_count == 0

        # Call add_quad
        renderer.add_quad(x, y, w, h, uv_coords, color_rgba)

        # Verify vertex count incremented by 6
        assert renderer.vertex_count == initial_vertex_count + 6

        # Verify vertex data was written correctly
        vertex_data = renderer.cpu_vertex_buffer[
            initial_vertex_count : renderer.vertex_count
        ]
        assert len(vertex_data) == 6

        # Verify all vertices have the correct color
        for vertex in vertex_data:
            np.testing.assert_array_almost_equal(vertex["color"], color_rgba)

        # Verify position data for specific vertices
        u1, v1, u2, v2 = uv_coords

        # Triangle 1: vertices 0, 1, 2
        expected_positions = [
            (x, y),  # vertex 0: bottom-left
            (x + w, y),  # vertex 1: bottom-right
            (x, y + h),  # vertex 2: top-left
            (x + w, y),  # vertex 3: bottom-right (repeated)
            (x, y + h),  # vertex 4: top-left (repeated)
            (x + w, y + h),  # vertex 5: top-right
        ]

        expected_uvs = [
            (u1, v1),  # vertex 0
            (u2, v1),  # vertex 1
            (u1, v2),  # vertex 2
            (u2, v1),  # vertex 3
            (u1, v2),  # vertex 4
            (u2, v2),  # vertex 5
        ]

        for i, (expected_pos, expected_uv) in enumerate(
            zip(expected_positions, expected_uvs, strict=False)
        ):
            vertex = vertex_data[i]
            np.testing.assert_array_almost_equal(vertex["position"], expected_pos)
            np.testing.assert_array_almost_equal(vertex["uv"], expected_uv)

    def test_add_quad_buffer_full(self):
        """Test that fills the vertex buffer to capacity and then tries to add
        one more quad. Assert that the vertex_count does not increase beyond
        the buffer's limit and that no error is raised."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Fill the buffer to near capacity (leave room for exactly one more quad)
        max_quads = MAX_QUADS
        renderer.vertex_count = (max_quads - 1) * 6

        # Test data
        x, y = 0.0, 0.0
        w, h = 32.0, 32.0
        uv_coords = (0.0, 0.0, 1.0, 1.0)
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        # This should succeed (buffer not full yet)
        initial_count = renderer.vertex_count
        renderer.add_quad(x, y, w, h, uv_coords, color_rgba)
        assert renderer.vertex_count == initial_count + 6

        # Now the buffer should be full. This call should fail gracefully
        full_count = renderer.vertex_count
        renderer.add_quad(x, y, w, h, uv_coords, color_rgba)

        # Vertex count should not have increased beyond the buffer limit
        assert renderer.vertex_count == full_count
        assert renderer.vertex_count == max_quads * 6

        # No exception should have been raised

    def test_render_to_screen(self):
        """Test that render_to_screen correctly performs the final render sequence."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Add some vertices to render
        renderer.add_quad(
            x=100.0,
            y=200.0,
            w=32.0,
            h=32.0,
            uv_coords=(0.0, 0.0, 1.0, 1.0),
            color_rgba=(1.0, 1.0, 1.0, 1.0),
        )

        window_size = (800, 600)

        # Call render_to_screen
        renderer.render_to_screen(window_size)

        # Verify uniform was set
        self.mock_program.__getitem__.assert_called_with("u_letterbox")
        # Should set letterbox to full screen when no letterbox_geometry provided
        assert self.mock_uniform.value == (0, 0, window_size[0], window_size[1])

        # Verify atlas texture was used
        self.mock_atlas_texture.use.assert_called_once_with(location=0)

        # Verify VBO write was called with correct data
        self.mock_vbo.write.assert_called_once()
        call_args = self.mock_vbo.write.call_args[0]
        written_data = call_args[0]

        # The written data should be the bytes of the vertex buffer up to vertex_count
        expected_data = renderer.cpu_vertex_buffer[: renderer.vertex_count].tobytes()
        assert written_data == expected_data

        # Verify VAO render was called with correct parameters
        self.mock_vao.render.assert_called_once()
        render_call_args = self.mock_vao.render.call_args
        # moderngl.TRIANGLES should be the first argument
        assert render_call_args.kwargs["vertices"] == renderer.vertex_count

    def test_render_to_screen_empty_buffer(self):
        """Test where begin_frame() is called but no quads are added. Assert
        that render_to_screen() does not call vbo.write() or vao.render()
        when the vertex count is zero."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Reset mocks after initialization (since atlas uniform is set during init)
        self.mock_program.__getitem__.reset_mock()
        self.mock_atlas_texture.use.reset_mock()

        # Ensure buffer is empty
        renderer.begin_frame()
        assert renderer.vertex_count == 0

        window_size = (800, 600)

        # Call render_to_screen with empty buffer
        renderer.render_to_screen(window_size)

        # Verify VBO write was NOT called
        self.mock_vbo.write.assert_not_called()

        # Verify VAO render was NOT called
        self.mock_vao.render.assert_not_called()

        # Verify uniform and texture use were also NOT called
        self.mock_program.__getitem__.assert_not_called()
        self.mock_atlas_texture.use.assert_not_called()

    def test_shader_program_creation(self):
        """Test that the shader program is created with correct parameters."""
        ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Verify program creation was called
        self.mock_mgl_context.program.assert_called_once()

        # Get the call arguments
        call_args = self.mock_mgl_context.program.call_args
        assert "vertex_shader" in call_args.kwargs
        assert "fragment_shader" in call_args.kwargs

        # Verify vertex shader contains expected content
        vertex_shader = call_args.kwargs["vertex_shader"]
        assert "#version 330" in vertex_shader
        assert "in vec2 in_vert" in vertex_shader
        assert "in vec2 in_uv" in vertex_shader
        assert "in vec4 in_color" in vertex_shader
        assert "uniform vec4 u_letterbox" in vertex_shader
        assert "gl_Position = vec4(x, y, 0.0, 1.0)" in vertex_shader

        # Verify fragment shader contains expected content
        fragment_shader = call_args.kwargs["fragment_shader"]
        assert "#version 330" in fragment_shader
        assert "uniform sampler2D u_atlas" in fragment_shader
        assert "texture(u_atlas, v_uv)" in fragment_shader

    def test_multiple_quads(self):
        """Test adding multiple quads to verify batching behavior."""
        renderer = ScreenRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
        )

        # Add three quads
        quad_data = [
            (10.0, 20.0, 16.0, 20.0, (0.0, 0.0, 0.1, 0.1), (1.0, 0.0, 0.0, 1.0)),  # Red
            (
                26.0,
                20.0,
                16.0,
                20.0,
                (0.1, 0.0, 0.2, 0.1),
                (0.0, 1.0, 0.0, 1.0),
            ),  # Green
            (
                42.0,
                20.0,
                16.0,
                20.0,
                (0.2, 0.0, 0.3, 0.1),
                (0.0, 0.0, 1.0, 1.0),
            ),  # Blue
        ]

        for x, y, w, h, uv_coords, color_rgba in quad_data:
            renderer.add_quad(x, y, w, h, uv_coords, color_rgba)

        # Should have 18 vertices total (3 quads * 6 vertices each)
        assert renderer.vertex_count == 18

        # Verify each quad's color data
        for i, (_, _, _, _, _, expected_color) in enumerate(quad_data):
            quad_start = i * 6
            quad_vertices = renderer.cpu_vertex_buffer[quad_start : quad_start + 6]

            for vertex in quad_vertices:
                np.testing.assert_array_almost_equal(vertex["color"], expected_color)

        # Render and verify all vertices are processed
        window_size = (1024, 768)
        renderer.render_to_screen(window_size)

        # Verify render was called with correct vertex count
        self.mock_vao.render.assert_called_once()
        render_call_args = self.mock_vao.render.call_args
        assert render_call_args.kwargs["vertices"] == 18
