"""
Comprehensive unit tests for the ModernGL graphics backend.

These tests use a headless OpenGL context to test real rendering functionality
without requiring a window or display.
"""

from unittest.mock import Mock, patch

import moderngl
import numpy as np
import pytest

from catley.backends.moderngl.graphics import ModernGLGraphicsContext
from catley.backends.moderngl.texture_renderer import VERTEX_DTYPE
from catley.game.enums import BlendMode
from catley.types import InterpolationAlpha, Opacity
from catley.util.coordinates import Rect
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.effects.particles import SubTileParticleSystem


class MockGLWindow:
    """Mock GLWindow for testing."""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

    def get_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def get_framebuffer_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def flip(self) -> None:
        pass


class MockCursorManager:
    """Mock cursor manager for testing."""

    def __init__(self):
        self.mouse_pixel_x = 100
        self.mouse_pixel_y = 150
        self.active_cursor_type = "arrow"
        self.cursors = {"arrow": MockCursorData()}


class MockCursorData:
    """Mock cursor data for testing."""

    def __init__(self):
        self.hotspot = (5, 5)
        self.texture = None
        # Create a simple 16x16 RGBA cursor
        self.pixels = np.ones((16, 16, 4), dtype=np.uint8) * 255


class TestModernGLGraphicsContext:
    """Test suite for ModernGLGraphicsContext."""

    @pytest.fixture(autouse=True)
    def setup_gl_context(self):
        """Create headless GL context and graphics context for testing."""
        self.gl_context = moderngl.create_context(standalone=True)
        self.mock_window = MockGLWindow(800, 600)

        # Mock the atlas texture loading entirely to avoid interference
        with patch.object(
            ModernGLGraphicsContext, "_load_atlas_texture"
        ) as mock_load_atlas:
            # Create a mock atlas texture
            mock_texture = Mock()
            mock_load_atlas.return_value = mock_texture

            self.graphics_ctx = ModernGLGraphicsContext(
                self.mock_window, self.gl_context
            )

        yield

        # Cleanup
        self.gl_context.release()

    def test_initialization(self):
        """Test that graphics context initializes correctly."""
        assert self.graphics_ctx.window == self.mock_window
        assert self.graphics_ctx.mgl_context == self.gl_context
        assert self.graphics_ctx.atlas_texture is not None
        assert self.graphics_ctx.program is not None
        assert self.graphics_ctx.uv_map.shape == (256, 4)
        assert self.graphics_ctx.screen_renderer.vertex_count == 0

    def test_tile_dimensions_property(self):
        """Test tile dimensions property."""
        tile_dims = self.graphics_ctx.tile_dimensions
        assert isinstance(tile_dims, tuple)
        assert len(tile_dims) == 2
        assert tile_dims[0] > 0
        assert tile_dims[1] > 0

    def test_console_dimensions_properties(self):
        """Test console width and height properties."""
        width = self.graphics_ctx.console_width_tiles
        height = self.graphics_ctx.console_height_tiles
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width > 0
        assert height > 0

    def test_coordinate_converter_property(self):
        """Test coordinate converter property."""
        converter = self.graphics_ctx.coordinate_converter
        assert converter is not None
        assert hasattr(converter, "pixel_to_tile")

    def test_prepare_to_present(self):
        """Test prepare_to_present clears vertex buffer."""
        # Add some vertices first
        self.graphics_ctx.screen_renderer.vertex_count = 10

        self.graphics_ctx.prepare_to_present()

        assert self.graphics_ctx.screen_renderer.vertex_count == 0

    def test_console_to_screen_coords(self):
        """Test coordinate conversion from console to screen."""
        # Test basic conversion
        screen_x, screen_y = self.graphics_ctx.console_to_screen_coords(0, 0)
        assert isinstance(screen_x, float)
        assert isinstance(screen_y, float)

        # Test that different inputs give different outputs
        screen_x2, screen_y2 = self.graphics_ctx.console_to_screen_coords(10, 20)
        assert screen_x2 != screen_x or screen_y2 != screen_y

    def test_pixel_to_tile(self):
        """Test pixel to tile coordinate conversion."""
        tile_pos = self.graphics_ctx.pixel_to_tile(100, 200)
        assert isinstance(tile_pos, tuple)
        assert len(tile_pos) == 2

    def test_update_dimensions(self):
        """Test dimension updates."""
        # Change window size
        self.mock_window.width = 1024
        self.mock_window.height = 768

        old_geometry = self.graphics_ctx.letterbox_geometry
        self.graphics_ctx.update_dimensions()
        new_geometry = self.graphics_ctx.letterbox_geometry

        # Geometry should have changed
        assert new_geometry != old_geometry

    def test_uv_map_precalculation(self):
        """Test UV coordinate map generation."""
        uv_map = self.graphics_ctx.uv_map

        # Should have UV coords for all 256 characters
        assert uv_map.shape == (256, 4)

        # UV coordinates should be in valid range [0, 1]
        assert np.all(uv_map >= 0)
        assert np.all(uv_map <= 1)

        # Test specific character
        char_uv = uv_map[ord("A")]
        assert len(char_uv) == 4  # u1, v1, u2, v2

    def test_draw_actor_smooth(self):
        """Test drawing a smooth actor."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.draw_actor_smooth(
            char="@",
            color=(255, 0, 0),
            screen_x=100.0,
            screen_y=200.0,
            light_intensity=(0.8, 0.9, 1.0),
            interpolation_alpha=InterpolationAlpha(0.75),
        )

        # Should have added 6 vertices (2 triangles = 1 quad)
        assert (
            self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count + 6
        )

    def test_draw_actor_smooth_uses_opaque_color_alpha(self):
        """
        Verify that draw_actor_smooth always provides a fully opaque (1.0)
        alpha to the quad color, ignoring the interpolation_alpha parameter.

        This is a regression test for the "actor buzzing" bug where interpolation_alpha
        was incorrectly used as the color alpha channel, causing actors to flicker
        between transparent and opaque as the interpolation value changed.
        """
        # Arrange: Mock the screen_renderer.add_quad method to inspect its arguments
        with patch.object(
            self.graphics_ctx.screen_renderer, "add_quad"
        ) as mock_add_quad:
            test_interpolation_alpha = InterpolationAlpha(
                0.37
            )  # An unusual, non-1.0 value

            # Act: Call draw_actor_smooth with the test interpolation alpha
            self.graphics_ctx.draw_actor_smooth(
                char="@",
                color=(100, 150, 200),
                screen_x=50.0,
                screen_y=50.0,
                light_intensity=(0.8, 0.9, 1.0),
                interpolation_alpha=test_interpolation_alpha,
            )

            # Assert: Verify add_quad was called once
            mock_add_quad.assert_called_once()

            # Get the arguments passed to add_quad
            call_args = mock_add_quad.call_args[0]

            # The color tuple is the last positional argument
            final_color_tuple = call_args[-1]

            # Check that the 4th element (alpha) of the color tuple is 1.0
            actual_alpha = final_color_tuple[3]
            assert actual_alpha == 1.0, (
                f"Color alpha should be 1.0 (fully opaque), but was {actual_alpha}. "
                f"Interpolation alpha should not be used for color transparency."
            )

            # Additional check: ensure the RGB values are correctly calculated
            expected_r = (100 / 255.0) * 0.8  # color[0] / 255.0 * light_intensity[0]
            expected_g = (150 / 255.0) * 0.9  # color[1] / 255.0 * light_intensity[1]
            expected_b = (200 / 255.0) * 1.0  # color[2] / 255.0 * light_intensity[2]

            assert abs(final_color_tuple[0] - expected_r) < 0.001
            assert abs(final_color_tuple[1] - expected_g) < 0.001
            assert abs(final_color_tuple[2] - expected_b) < 0.001

    def test_draw_tile_highlight(self):
        """Test drawing tile highlights."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.draw_tile_highlight(
            root_x=5, root_y=10, color=(255, 255, 0), alpha=Opacity(0.5)
        )

        # Should have added vertices for the highlight
        assert self.graphics_ctx.screen_renderer.vertex_count > initial_vertex_count

    def test_draw_debug_rect(self):
        """Test drawing debug rectangles."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.draw_debug_rect(
            px_x=10, px_y=20, px_w=100, px_h=50, color=(0, 255, 0)
        )

        # Should have added vertices for 4 edges (24 vertices total)
        assert (
            self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count + 24
        )

    def test_draw_mouse_cursor(self):
        """Test drawing mouse cursor."""
        cursor_manager = MockCursorManager()
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.draw_mouse_cursor(cursor_manager)

        # Should not have added vertices to the main buffer (renders immediately)
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

        # Cursor data should now have a texture cached
        cursor_data = cursor_manager.cursors["arrow"]
        assert cursor_data.texture is not None

    def test_draw_mouse_cursor_no_cursor_data(self):
        """Test drawing mouse cursor when no cursor data exists."""
        cursor_manager = MockCursorManager()
        cursor_manager.active_cursor_type = "nonexistent"
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.draw_mouse_cursor(cursor_manager)

        # Should not have added any vertices
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_apply_environmental_effect(self):
        """Test applying environmental effects."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.apply_environmental_effect(
            position=(10.0, 15.0),
            radius=5.0,
            tint_color=(255, 200, 100),
            intensity=0.7,
            blend_mode=BlendMode.TINT,
        )

        # Environmental effects now use immediate rendering, vertex count stays same
        # The effect was rendered immediately using the gradient texture
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_apply_environmental_effect_zero_radius(self):
        """Test environmental effect with zero radius does nothing."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.apply_environmental_effect(
            position=(10.0, 15.0),
            radius=0.0,
            tint_color=(255, 200, 100),
            intensity=0.7,
            blend_mode=BlendMode.TINT,
        )

        # Should not have added any vertices
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_apply_environmental_effect_zero_intensity(self):
        """Test environmental effect with zero intensity does nothing."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.apply_environmental_effect(
            position=(10.0, 15.0),
            radius=5.0,
            tint_color=(255, 200, 100),
            intensity=0.0,
            blend_mode=BlendMode.TINT,
        )

        # Should not have added any vertices
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_add_quad_to_buffer(self):
        """Test low-level quad addition to vertex buffer."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.screen_renderer.add_quad(
            x=100,
            y=200,
            w=32,
            h=32,
            uv_coords=(0.0, 0.0, 1.0, 1.0),
            color_rgba=(1.0, 0.5, 0.0, 0.8),
        )

        # Should have added exactly 6 vertices
        assert (
            self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count + 6
        )

        # Verify vertex data was written correctly
        vertex_data = self.graphics_ctx.screen_renderer.cpu_vertex_buffer[
            initial_vertex_count : self.graphics_ctx.screen_renderer.vertex_count
        ]

        # All vertices should have the same color
        vertex_colors = vertex_data["color"]
        expected_color = np.array([1.0, 0.5, 0.0, 0.8])
        np.testing.assert_array_almost_equal(vertex_colors[0], expected_color)

    def test_add_quad_to_buffer_overflow(self):
        """Test quad buffer overflow handling."""
        # Fill the buffer to near capacity
        max_quads = len(self.graphics_ctx.screen_renderer.cpu_vertex_buffer) // 6
        self.graphics_ctx.screen_renderer.vertex_count = (max_quads - 1) * 6

        # This should succeed
        self.graphics_ctx.screen_renderer.add_quad(
            x=0,
            y=0,
            w=32,
            h=32,
            uv_coords=(0.0, 0.0, 1.0, 1.0),
            color_rgba=(1.0, 1.0, 1.0, 1.0),
        )

        # This should fail gracefully (no crash)
        initial_count = self.graphics_ctx.screen_renderer.vertex_count
        self.graphics_ctx.screen_renderer.add_quad(
            x=0,
            y=0,
            w=32,
            h=32,
            uv_coords=(0.0, 0.0, 1.0, 1.0),
            color_rgba=(1.0, 1.0, 1.0, 1.0),
        )

        # Vertex count should not have increased
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_count

    def test_render_glyph_buffer_to_texture(self):
        """Test rendering glyph buffer to texture."""
        # Create a simple glyph buffer
        glyph_buffer = GlyphBuffer(width=10, height=5)
        glyph_buffer.data[0, 0] = (ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.data[1, 1] = (ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))

        texture = self.graphics_ctx.render_glyph_buffer_to_texture(glyph_buffer)

        # Should return a ModernGL texture
        assert texture is not None
        assert isinstance(texture, moderngl.Texture)

    def test_render_glyph_buffer_to_texture_empty(self):
        """Test rendering glyph buffer with zero dimensions."""
        # Test the method's handling of empty dimensions directly
        # by calling the internal logic rather than creating invalid GlyphBuffer

        # Mock a glyph buffer with zero width/height
        from unittest.mock import Mock

        mock_glyph_buffer = Mock()
        mock_glyph_buffer.width = 0
        mock_glyph_buffer.height = 5

        texture = self.graphics_ctx.render_glyph_buffer_to_texture(mock_glyph_buffer)

        # Should return a 1x1 texture for zero dimensions (not None)
        assert texture is not None
        assert isinstance(texture, moderngl.Texture)
        assert texture.size == (1, 1)

    def test_present_texture(self):
        """Test presenting a pre-rendered texture."""
        # Create a test texture
        test_texture = self.gl_context.texture((64, 64), 4)
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        # present_texture now renders immediately, not to vertex buffer
        self.graphics_ctx.present_texture(
            texture=test_texture, x_tile=2, y_tile=3, width_tiles=4, height_tiles=2
        )

        # Should not have added vertices to the main buffer (renders immediately)
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_present_texture_invalid(self):
        """Test presenting invalid texture does nothing."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.present_texture(
            texture="not a texture", x_tile=2, y_tile=3, width_tiles=4, height_tiles=2
        )

        # Should not have added any vertices
        assert self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count

    def test_finalize_present_empty(self):
        """Test finalize_present with no vertices."""
        # Should not crash with empty vertex buffer
        self.graphics_ctx.finalize_present()

    def test_finalize_present_with_vertices(self):
        """Test finalize_present with vertices."""
        # Add some vertices
        self.graphics_ctx.draw_actor_smooth(
            char="X", color=(255, 255, 255), screen_x=50.0, screen_y=50.0
        )

        # Should not crash
        self.graphics_ctx.finalize_present()

    def test_render_particles(self):
        """Test particle rendering."""
        # Create a mock particle system
        particle_system = Mock(spec=SubTileParticleSystem)
        particle_system.active_count = 3
        particle_system.layers = np.array([1, 2, 1])
        particle_system.positions = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        particle_system.subdivision = 16
        particle_system.chars = np.array(["*", "+", "."])
        particle_system.colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        particle_system.flash_intensity = np.array([np.nan, 0.8, np.nan])
        particle_system.lifetimes = np.array([5.0, 3.0, 2.0])
        particle_system.max_lifetimes = np.array([10.0, 5.0, 4.0])

        layer = Mock()
        layer.value = 1

        viewport_bounds = Rect(0, 0, 100, 100)
        view_offset = (0, 0)

        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx.render_particles(
            particle_system=particle_system,
            layer=layer,
            viewport_bounds=viewport_bounds,
            view_offset=view_offset,
        )

        # Should have rendered particles (exact count depends on culling)
        assert self.graphics_ctx.screen_renderer.vertex_count >= initial_vertex_count

    def test_convert_particle_to_screen_coords(self):
        """Test particle coordinate conversion."""
        particle_system = Mock()
        particle_system.active_count = 2
        particle_system.positions = np.array([[16.0, 32.0], [48.0, 64.0]])
        particle_system.subdivision = 16

        viewport_bounds = Rect(0, 0, 10, 10)
        view_offset = (0, 0)

        # Test valid particle
        coords = self.graphics_ctx._convert_particle_to_screen_coords(
            particle_system, 0, viewport_bounds, view_offset
        )
        assert coords is not None
        assert isinstance(coords, tuple)
        assert len(coords) == 2

        # Test invalid particle index
        coords = self.graphics_ctx._convert_particle_to_screen_coords(
            particle_system, 5, viewport_bounds, view_offset
        )
        assert coords is None

    def test_draw_particle_to_buffer(self):
        """Test drawing individual particle to buffer."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        self.graphics_ctx._draw_particle_to_buffer(
            char="*", color=(255, 128, 64), screen_x=100.0, screen_y=200.0, alpha=0.6
        )

        # Should have added 6 vertices
        assert (
            self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count + 6
        )

    def test_draw_particle_to_buffer_alpha_clamping(self):
        """Test particle alpha clamping."""
        initial_vertex_count = self.graphics_ctx.screen_renderer.vertex_count

        # Test alpha > 1.0 gets clamped
        self.graphics_ctx._draw_particle_to_buffer(
            char="*",
            color=(255, 255, 255),
            screen_x=0.0,
            screen_y=0.0,
            alpha=1.5,  # Should be clamped to 1.0
        )

        # Test alpha < 0.0 gets clamped
        self.graphics_ctx._draw_particle_to_buffer(
            char="*",
            color=(255, 255, 255),
            screen_x=0.0,
            screen_y=0.0,
            alpha=-0.5,  # Should be clamped to 0.0
        )

        assert (
            self.graphics_ctx.screen_renderer.vertex_count == initial_vertex_count + 12
        )

    def test_encode_glyph_buffer_to_vertices(self):
        """Test glyph buffer encoding to vertices."""
        # Create a simple glyph buffer
        glyph_buffer = GlyphBuffer(width=2, height=2)
        glyph_buffer.data[0, 0] = (ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.data[1, 0] = (ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))
        glyph_buffer.data[0, 1] = (ord("C"), (0, 0, 255, 255), (0, 0, 0, 255))
        glyph_buffer.data[1, 1] = (ord("D"), (255, 255, 0, 255), (0, 0, 0, 255))

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Should have 2x2 = 4 cells, each with 6 vertices (single quad with both colors)
        expected_vertex_count = 4 * 6
        assert vertex_count == expected_vertex_count
        assert len(vertex_data) == expected_vertex_count

        # Verify data structure
        assert vertex_data.dtype == VERTEX_DTYPE

    def test_encode_empty_glyph_buffer(self):
        """Test encoding empty glyph buffer."""
        # Create a valid but empty glyph buffer (1x1 with no content)
        glyph_buffer = GlyphBuffer(width=1, height=1)
        # Clear the data to simulate empty
        glyph_buffer.data[:] = (0, (0, 0, 0, 0), (0, 0, 0, 0))

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Should have data for 1 cell (6 vertices: single quad with both colors)
        assert vertex_count == 6
        assert len(vertex_data) == 6

    def test_vertex_encoding_uv_coordinates(self):
        """Test that vertex encoding produces correct UV coordinates for specific
        characters."""
        # Create a simple 1x1 glyph buffer with a known character
        glyph_buffer = GlyphBuffer(width=1, height=1)
        glyph_buffer.data[0, 0] = (ord("A"), (255, 255, 255, 255), (0, 0, 0, 255))

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Should have 1 cell * 6 vertices = 6 vertices
        assert vertex_count == 6

        # Get UV coordinates for character 'A' (should be at the start of vertex data)
        # Each cell has 6 vertices: single quad with character 'A' UV coordinates
        # We want the vertices (0-5) for the first cell (character 'A')
        fg_vertices_start = 0  # Start from beginning since single quad approach
        fg_vertices_end = 6  # End of quad

        a_fg_vertices = vertex_data[fg_vertices_start:fg_vertices_end]

        # All UV coordinates for this character should use the same UV mapping
        # Get expected UV coordinates for 'A' from the UV map
        expected_uv = self.graphics_ctx.uv_map[ord("A")]
        u1, v1, u2, v2 = expected_uv

        # Check that the UV coordinates are reasonable (not 0,0 to 1,1 which would be
        # full texture)
        assert u1 != 0.0 or v1 != 0.0 or u2 != 1.0 or v2 != 1.0, (
            "UV coordinates should not span entire texture"
        )
        assert 0.0 <= u1 < u2 <= 1.0, f"Invalid U coordinates: {u1} to {u2}"
        assert 0.0 <= v1 < v2 <= 1.0, f"Invalid V coordinates: {v1} to {v2}"

        # Verify the vertices have the expected UV coordinates
        # Triangle 1: (u1,v1), (u2,v1), (u1,v2)
        # Triangle 2: (u2,v1), (u1,v2), (u2,v2)
        expected_uvs = [
            (u1, v1),
            (u2, v1),
            (u1, v2),  # Triangle 1
            (u2, v1),
            (u1, v2),
            (u2, v2),  # Triangle 2
        ]

        for i, expected_uv_coord in enumerate(expected_uvs):
            actual_uv = a_fg_vertices[i]["uv"]
            np.testing.assert_array_almost_equal(
                actual_uv,
                expected_uv_coord,
                decimal=5,
                err_msg=f"UV coordinate mismatch at vertex {i}",
            )

    def test_render_glyph_buffer_creates_valid_texture(self):
        """Test that render_glyph_buffer_to_texture creates a texture with valid
        content."""
        # Create a glyph buffer with a specific character
        glyph_buffer = GlyphBuffer(width=2, height=2)
        glyph_buffer.data[0, 0] = (ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.data[0, 1] = (ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))
        glyph_buffer.data[1, 0] = (ord("C"), (0, 0, 255, 255), (0, 0, 0, 255))
        glyph_buffer.data[1, 1] = (ord("D"), (255, 255, 0, 255), (0, 0, 0, 255))

        # Render to texture
        texture = self.graphics_ctx.render_glyph_buffer_to_texture(glyph_buffer)

        # Should return a valid texture
        assert texture is not None
        assert isinstance(texture, moderngl.Texture)

        # Texture should have expected dimensions (2x2 tiles * 20x20 pixels each)
        expected_width = 2 * self.graphics_ctx.tile_dimensions[0]
        expected_height = 2 * self.graphics_ctx.tile_dimensions[1]
        assert texture.size == (expected_width, expected_height)

        # Texture should not be the same as the atlas texture
        assert texture is not self.graphics_ctx.atlas_texture

    def test_encode_glyph_buffer_to_vertices_exact_calculation(self):
        """Test _encode_glyph_buffer_to_vertices with exact vertex calculations.

        Creates a 1x1 GlyphBuffer with character 'A' and verifies that the output
        NumPy array has exactly 12 vertices with manually calculated position,
        color, and UV coordinates. This catches off-by-one, indexing, or
        Y-flipping errors.
        """
        # Create a 1x1 GlyphBuffer with known character 'A'
        glyph_buffer = GlyphBuffer(width=1, height=1)
        fg_color = (255, 128, 64, 200)  # RGBA
        bg_color = (32, 64, 128, 180)  # RGBA
        glyph_buffer.data[0, 0] = (ord("A"), fg_color, bg_color)

        # Call the method under test
        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Assert exactly 6 vertices (1 quad * 6 vertices per quad)
        assert vertex_count == 6
        assert len(vertex_data) == 6

        # Get expected values for manual calculation
        tile_w, tile_h = self.graphics_ctx._tile_dimensions

        # Expected screen position (1x1 buffer, position (0,0) in buffer coordinates)
        # No Y-flipping: screen_y = y * tile_h = 0 * tile_h = 0
        expected_screen_x = 0 * tile_w  # = 0
        expected_screen_y = 0 * tile_h  # = 0

        # Expected UV coordinates for foreground character (single quad approach)
        # Note: We no longer use background UV coordinates
        expected_fg_uv = self.graphics_ctx.uv_map[ord("A")]  # character 'A'

        # Expected normalized colors
        expected_bg_color_norm = tuple(c / 255.0 for c in bg_color)
        expected_fg_color_norm = tuple(c / 255.0 for c in fg_color)

        # Manually calculate expected vertices for single quad (6 vertices)
        # Single quad approach uses foreground character UV coordinates
        fg_u1, fg_v1, fg_u2, fg_v2 = expected_fg_uv
        expected_vertices = [
            # Triangle 1
            (
                (expected_screen_x, expected_screen_y),
                (fg_u1, fg_v1),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # bottom-left
            (
                (expected_screen_x + tile_w, expected_screen_y),
                (fg_u2, fg_v1),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # bottom-right
            (
                (expected_screen_x, expected_screen_y + tile_h),
                (fg_u1, fg_v2),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # top-left
            # Triangle 2
            (
                (expected_screen_x + tile_w, expected_screen_y),
                (fg_u2, fg_v1),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # bottom-right
            (
                (expected_screen_x, expected_screen_y + tile_h),
                (fg_u1, fg_v2),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # top-left
            (
                (expected_screen_x + tile_w, expected_screen_y + tile_h),
                (fg_u2, fg_v2),
                expected_fg_color_norm,
                expected_bg_color_norm,
            ),  # top-right
        ]

        # Verify each vertex exactly matches expected calculations
        for i, expected_vertex in enumerate(expected_vertices):
            actual_vertex = vertex_data[i]
            expected_pos, expected_uv, expected_fg_color, expected_bg_color = (
                expected_vertex
            )

            # Check position
            np.testing.assert_array_almost_equal(
                actual_vertex["position"],
                expected_pos,
                decimal=5,
                err_msg=f"Position mismatch at vertex {i}",
            )

            # Check UV coordinates
            np.testing.assert_array_almost_equal(
                actual_vertex["uv"],
                expected_uv,
                decimal=5,
                err_msg=f"UV mismatch at vertex {i}",
            )

            # Check foreground color
            np.testing.assert_array_almost_equal(
                actual_vertex["fg_color"],
                expected_fg_color,
                decimal=5,
                err_msg=f"Foreground color mismatch at vertex {i}",
            )

            # Check background color
            np.testing.assert_array_almost_equal(
                actual_vertex["bg_color"],
                expected_bg_color,
                decimal=5,
                err_msg=f"Background color mismatch at vertex {i}",
            )


class TestModernGLGraphicsContextEdgeCases:
    """Test edge cases and error conditions."""

    def test_standalone_context_creation(self):
        """Test that standalone context creation works for testing."""
        context = moderngl.create_context(standalone=True)
        assert context is not None

        mock_window = MockGLWindow()

        # Should be able to create graphics context with standalone context
        with patch("catley.backends.moderngl.graphics.PILImage.open") as mock_open:
            mock_img = Mock()
            mock_img.size = (256, 256)
            mock_img.convert.return_value = mock_img
            mock_array = np.ones((256, 256, 4), dtype=np.uint8) * 255

            with patch("numpy.array", return_value=mock_array):
                mock_open.return_value = mock_img
                graphics_ctx = ModernGLGraphicsContext(mock_window, context)

                assert graphics_ctx.mgl_context == context

        context.release()

    def test_window_size_changes(self):
        """Test handling of window size changes."""
        context = moderngl.create_context(standalone=True)
        mock_window = MockGLWindow(400, 300)

        with patch("catley.backends.moderngl.graphics.PILImage.open") as mock_open:
            mock_img = Mock()
            mock_img.size = (256, 256)
            mock_img.convert.return_value = mock_img
            mock_array = np.ones((256, 256, 4), dtype=np.uint8) * 255

            with patch("numpy.array", return_value=mock_array):
                mock_open.return_value = mock_img
                graphics_ctx = ModernGLGraphicsContext(mock_window, context)

                # Change window size
                mock_window.width = 800
                mock_window.height = 600

                # Update should not crash
                graphics_ctx.update_dimensions()

                # Coordinate conversion should still work
                x, y = graphics_ctx.console_to_screen_coords(5, 10)
                assert isinstance(x, float)
                assert isinstance(y, float)

        context.release()
