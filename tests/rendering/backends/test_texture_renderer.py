"""
Unit tests for the TextureRenderer class.

These tests use unittest.mock to completely mock the moderngl library,
allowing us to test the TextureRenderer logic without needing a real OpenGL context.
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from catley.backends.moderngl.texture_renderer import VERTEX_DTYPE, TextureRenderer
from catley.util.glyph_buffer import GlyphBuffer


class TestTextureRenderer:
    """Test suite for TextureRenderer using mocked moderngl."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocked moderngl context and related objects."""
        # Mock the moderngl context
        self.mock_mgl_context = Mock()

        # Mock atlas texture
        self.mock_atlas_texture = Mock()

        # Mock tile dimensions
        self.tile_dimensions = (16, 20)  # 16px wide, 20px tall

        # Mock UV map (256 characters with 4 UV coordinates each)
        self.mock_uv_map = np.zeros((256, 4), dtype="f4")
        # Set some known UV coordinates for testing
        self.mock_uv_map[ord("A")] = [0.0, 0.0, 0.1, 0.1]  # Character 'A'
        self.mock_uv_map[ord("B")] = [0.1, 0.0, 0.2, 0.1]  # Character 'B'
        self.mock_uv_map[219] = [0.9, 0.9, 1.0, 1.0]  # Solid block character

        # Mock the shader program creation with proper subscripting support
        self.mock_program = Mock()
        self.mock_uniform = Mock()
        # Configure the program to support subscripting like program["u_atlas"]
        self.mock_program.__getitem__ = Mock(return_value=self.mock_uniform)
        self.mock_mgl_context.program.return_value = self.mock_program

        # Mock texture and framebuffer
        self.mock_texture = Mock()
        self.mock_fbo = Mock()
        self.mock_mgl_context.texture.return_value = self.mock_texture
        self.mock_mgl_context.framebuffer.return_value = self.mock_fbo

        # Mock VBO and VAO
        self.mock_vbo = Mock()
        self.mock_vao = Mock()
        self.mock_mgl_context.buffer.return_value = self.mock_vbo
        self.mock_mgl_context.vertex_array.return_value = self.mock_vao

        # Mock screen context
        self.mock_screen = Mock()
        self.mock_mgl_context.screen = self.mock_screen

    def test_initialization(self):
        """Test that TextureRenderer can be instantiated and creates its
        texture_program."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # Verify initialization
        assert renderer.mgl_context == self.mock_mgl_context
        assert renderer.atlas_texture == self.mock_atlas_texture
        assert renderer.tile_dimensions == self.tile_dimensions
        assert np.array_equal(renderer.uv_map, self.mock_uv_map)
        assert renderer.SOLID_BLOCK_CHAR == 9608  # Unicode █

        # Verify shader program was created
        self.mock_mgl_context.program.assert_called_once()
        assert renderer.texture_program == self.mock_program

        # Verify atlas texture sampler was bound
        self.mock_program.__getitem__.assert_called_with("u_atlas")
        assert self.mock_uniform.value == 0

        # Verify persistent VBO and VAO were created during initialization
        self.mock_mgl_context.buffer.assert_called_once()
        self.mock_mgl_context.vertex_array.assert_called_once()
        assert renderer.vbo == self.mock_vbo
        assert renderer.vao == self.mock_vao

    def test_render_to_fbo_calls(self):
        """Test the public render() method makes correct OpenGL calls."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # Reset mock call counts after initialization
        self.mock_mgl_context.buffer.reset_mock()
        self.mock_mgl_context.vertex_array.reset_mock()
        self.mock_mgl_context.texture.reset_mock()
        self.mock_mgl_context.framebuffer.reset_mock()

        # Create a 10x5 GlyphBuffer for testing
        glyph_buffer = GlyphBuffer(10, 5)
        glyph_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(1, 1, ord("B"), (0, 255, 0, 255), (128, 128, 128, 255))

        # Expected pixel dimensions (not used in new API but kept for documentation)
        # expected_width_px = 10 * self.tile_dimensions[0]  # 10 * 16 = 160
        # expected_height_px = 5 * self.tile_dimensions[1]  # 5 * 20 = 100

        # Create a mock target FBO
        mock_fbo = MagicMock()

        # Call render with target FBO
        renderer.render(glyph_buffer, mock_fbo)

        # TextureRenderer no longer creates textures/FBOs - they are provided
        # Assert FBO operations
        mock_fbo.use.assert_called_once()
        mock_fbo.clear.assert_called_once()

        # Assert that no new VBO/VAO were created during render (they're persistent)
        self.mock_mgl_context.buffer.assert_not_called()
        self.mock_mgl_context.vertex_array.assert_not_called()

        # Assert that VBO was written to (not created - it's persistent)
        # For first render, complete VBO is updated once (not per tile)
        self.mock_vbo.write.assert_called_once()

        # Assert rendering happened
        self.mock_vao.render.assert_called_once()

        # TextureRenderer no longer releases FBOs (they are reused)
        # Assert atlas texture was used
        self.mock_atlas_texture.use.assert_called_with(location=0)

        # Assert screen context restoration
        self.mock_screen.use.assert_called_once()

    def test_change_detection_optimization(self):
        """Test that subsequent renders only update dirty tiles."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # Create a buffer and render it once
        glyph_buffer = GlyphBuffer(3, 2)  # Small buffer for testing
        glyph_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(1, 0, ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))

        mock_fbo = MagicMock()

        # First render - should update complete VBO
        renderer.render(glyph_buffer, mock_fbo)

        # Reset mock to track second render
        self.mock_vbo.write.reset_mock()
        mock_fbo.use.reset_mock()
        mock_fbo.clear.reset_mock()

        # Second render with no changes - should not render at all
        renderer.render(glyph_buffer, mock_fbo)

        # Assert no VBO writes or FBO operations for unchanged buffer
        self.mock_vbo.write.assert_not_called()
        mock_fbo.use.assert_not_called()
        mock_fbo.clear.assert_not_called()

        # Reset mock again
        self.mock_vbo.write.reset_mock()
        mock_fbo.use.reset_mock()
        mock_fbo.clear.reset_mock()

        # Third render with one tile changed
        glyph_buffer.put_char(2, 1, ord("C"), (0, 0, 255, 255), (128, 128, 128, 255))
        renderer.render(glyph_buffer, mock_fbo)

        # Buffer has changed, so partial VBO update is performed
        # Should have exactly 1 VBO write for only the dirty tile (with offset)
        self.mock_vbo.write.assert_called_once()
        # Verify that write was called with an offset (partial update)
        call_args = self.mock_vbo.write.call_args
        assert call_args.kwargs.get("offset") is not None, (
            "VBO write should use offset for partial updates"
        )
        mock_fbo.use.assert_called_once()
        mock_fbo.clear.assert_called_once()

    def test_buffer_dimension_changes(self):
        """Test that dimension changes trigger complete VBO update."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # First render with small buffer
        glyph_buffer = GlyphBuffer(2, 2)
        glyph_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))

        mock_fbo = MagicMock()
        renderer.render(glyph_buffer, mock_fbo)

        # Reset mock to track second render
        self.mock_vbo.write.reset_mock()

        # Second render with different dimensions - should trigger complete update
        larger_buffer = GlyphBuffer(5, 3)
        larger_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))

        renderer.render(larger_buffer, mock_fbo)

        # Should have exactly 1 VBO write for complete buffer update (no offset)
        self.mock_vbo.write.assert_called_once()
        # Verify that write was called without offset (full update)
        call_args = self.mock_vbo.write.call_args
        assert call_args.kwargs.get("offset") is None, (
            "VBO write should not use offset for full updates"
        )

    def test_render_empty_buffer(self):
        """Test render() with buffer that results in zero pixel dimensions."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=(0, 20),  # Zero width tiles will result in 0 pixel width
            uv_map=self.mock_uv_map,
        )

        # Create minimal buffer
        glyph_buffer = GlyphBuffer(1, 1)

        # Mock empty texture return
        empty_texture = Mock()
        self.mock_mgl_context.texture.return_value = empty_texture

        # Create a mock target FBO
        mock_fbo = MagicMock()

        # Call render with target FBO
        renderer.render(glyph_buffer, mock_fbo)

        # TextureRenderer no longer creates textures - returns early for empty buffers
        # With non-zero buffer dimensions, it should still render

    def test_vertex_encoding_and_coordinate_mapping(self):
        """Test the core vertex encoding logic and coordinate mapping."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # Create a 2x2 GlyphBuffer for precise testing
        glyph_buffer = GlyphBuffer(2, 2)

        # Put specific characters at known positions
        glyph_buffer.put_char(
            0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255)
        )  # Top-left
        glyph_buffer.put_char(
            1, 0, ord("B"), (0, 255, 0, 255), (64, 64, 64, 255)
        )  # Top-right
        glyph_buffer.put_char(
            0, 1, ord("C"), (0, 0, 255, 255), (128, 128, 128, 255)
        )  # Bottom-left
        glyph_buffer.put_char(
            1, 1, ord("D"), (255, 255, 0, 255), (192, 192, 192, 255)
        )  # Bottom-right

        # Call the method directly
        vertex_data, vertex_count = renderer._encode_glyph_buffer_to_vertices(
            glyph_buffer
        )

        # Assert correct vertex count: 2x2 cells * 6 vertices/cell
        # (single quad with both fg and bg colors)
        expected_vertex_count = 2 * 2 * 6
        assert vertex_count == expected_vertex_count
        assert len(vertex_data) == expected_vertex_count

        # Verify the vertex data structure
        assert vertex_data.dtype == VERTEX_DTYPE

        # Test Y-coordinate handling (no pre-flipping)
        _tile_w, _tile_h = self.tile_dimensions

        # For a 2-tile high buffer with no Y-flipping:
        # y_console=0 should map to screen_y = 0 * tile_h = 0
        # y_console=1 should map to screen_y = 1 * tile_h = 20

        # Check vertices for character at (0,0) - should be at screen_y = 0
        char_00_start_idx = 0  # First character vertices
        char_00_vertices = vertex_data[char_00_start_idx : char_00_start_idx + 6]

        # All vertices for this character should have screen_y = 0 or 0 + tile_h
        for vertex in char_00_vertices:
            y_coord = vertex["position"][1]
            assert y_coord == 0.0 or y_coord == 20.0  # 0 or 0 + tile_h

        # Check vertices for character at (0,1) - should be at screen_y = 20
        char_01_start_idx = 2 * 6  # Third character (0,1) vertices
        char_01_vertices = vertex_data[char_01_start_idx : char_01_start_idx + 6]

        # All vertices for this character should have screen_y = 20 or 20 + tile_h
        for vertex in char_01_vertices:
            y_coord = vertex["position"][1]
            assert y_coord == 20.0 or y_coord == 40.0  # 20 or 20 + tile_h

        # Test X-coordinates (should not be flipped)
        # Character at (0,0) should have screen_x = 0
        # Character at (1,0) should have screen_x = 16
        char_10_start_idx = 1 * 6  # Second character (1,0) vertices
        char_10_vertices = vertex_data[char_10_start_idx : char_10_start_idx + 6]

        for vertex in char_10_vertices:
            x_coord = vertex["position"][0]
            assert x_coord == 16.0 or x_coord == 32.0  # 16 or 16 + tile_w

        # Test UV coordinates for known characters
        # Vertices should use foreground character 'A' UV (single quad approach)
        expected_fg_uv = self.mock_uv_map[ord("A")]  # [0.0, 0.0, 0.1, 0.1]

        first_vertex = char_00_vertices[0]  # First vertex
        actual_uv = first_vertex["uv"]
        assert np.allclose(actual_uv, expected_fg_uv[:2])  # First UV pair

        # Test color normalization (should be converted from 0-255 to 0.0-1.0)
        expected_fg_color = (255 / 255.0, 0 / 255.0, 0 / 255.0, 255 / 255.0)  # Red
        expected_bg_color = (0 / 255.0, 0 / 255.0, 0 / 255.0, 255 / 255.0)  # Black
        actual_fg_color = first_vertex["fg_color"]
        actual_bg_color = first_vertex["bg_color"]
        assert np.allclose(actual_fg_color, expected_fg_color)
        assert np.allclose(actual_bg_color, expected_bg_color)

    def test_vertex_encoding_empty_buffer(self):
        """Test vertex encoding with empty buffer."""
        renderer = TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
        )

        # Create buffer and manually modify its data to simulate empty area
        glyph_buffer = GlyphBuffer(1, 1)

        # Manually set dimensions to 0 to simulate empty buffer for testing
        glyph_buffer.width = 0
        glyph_buffer.height = 0
        glyph_buffer.data = np.zeros((0, 0), dtype=glyph_buffer.data.dtype)

        vertex_data, vertex_count = renderer._encode_glyph_buffer_to_vertices(
            glyph_buffer
        )

        assert vertex_count == 0
        assert len(vertex_data) == 0

    def test_shader_program_creation(self):
        """Test that the shader program is created with correct parameters."""
        TextureRenderer(
            mgl_context=self.mock_mgl_context,
            atlas_texture=self.mock_atlas_texture,
            tile_dimensions=self.tile_dimensions,
            uv_map=self.mock_uv_map,
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
        assert "in vec4 in_fg_color" in vertex_shader
        assert "in vec4 in_bg_color" in vertex_shader
        assert "uniform vec2 u_texture_size" in vertex_shader

        # Verify fragment shader contains expected content
        fragment_shader = call_args.kwargs["fragment_shader"]
        assert "#version 330" in fragment_shader
        assert "uniform sampler2D u_atlas" in fragment_shader
        assert "texture(u_atlas, v_uv)" in fragment_shader
        assert "mix(v_bg_color, v_fg_color, char_alpha)" in fragment_shader
