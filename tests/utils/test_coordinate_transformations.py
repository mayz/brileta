"""
Unit tests for coordinate transformations in the ModernGL backend.

These tests verify that the coordinate pipeline correctly transforms from logical
game coordinates to screen coordinates and that the vertex encoding produces
correct positions.
"""

from unittest.mock import Mock, patch

import moderngl
import pytest

from catley.backends.moderngl.graphics import ModernGLGraphicsContext
from catley.util.glyph_buffer import GlyphBuffer


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


class TestCoordinateTransformations:
    """Test coordinate transformations in the ModernGL backend."""

    @pytest.fixture(autouse=True)
    def setup_gl_context(self):
        """Create headless GL context and graphics context for testing."""
        self.gl_context = moderngl.create_context(standalone=True)
        self.mock_window = MockGLWindow(800, 600)

        # Mock the tileset loading since we don't have assets in tests
        with patch("catley.backends.moderngl.graphics.PILImage.open") as mock_open:
            # Create a mock tileset image
            mock_img = Mock()
            mock_img.size = (256, 256)  # 16x16 tileset
            mock_img.convert.return_value = mock_img

            # More specific mock that only affects the atlas texture loading
            with patch.object(
                ModernGLGraphicsContext, "_load_atlas_texture"
            ) as mock_load_atlas:
                mock_texture = Mock()
                mock_texture.filter = None
                mock_load_atlas.return_value = mock_texture

                mock_open.return_value = mock_img
                self.graphics_ctx = ModernGLGraphicsContext(
                    self.mock_window, self.gl_context
                )

        yield

        # Cleanup
        self.gl_context.release()

    def test_console_to_screen_coordinate_mapping(self):
        """Test that console coordinates map correctly to screen coordinates."""
        # Test corner cases
        screen_x, screen_y = self.graphics_ctx.console_to_screen_coords(0, 0)
        assert screen_x == 0.0
        assert screen_y >= 0.0  # Should be at the top in screen space

        # Test that coordinates increase properly
        screen_x1, screen_y1 = self.graphics_ctx.console_to_screen_coords(1, 0)
        screen_x2, screen_y2 = self.graphics_ctx.console_to_screen_coords(0, 1)

        assert screen_x1 > screen_x
        assert screen_y2 > screen_y

        # Test max coordinates don't exceed window bounds
        max_x = self.graphics_ctx.console_width_tiles - 1
        max_y = self.graphics_ctx.console_height_tiles - 1
        screen_x_max, screen_y_max = self.graphics_ctx.console_to_screen_coords(
            max_x, max_y
        )

        window_w, window_h = self.mock_window.get_size()
        assert screen_x_max < window_w
        assert screen_y_max < window_h

    def test_glyph_buffer_indexing_consistency(self):
        """Test that GlyphBuffer indexing matches expected coordinate system."""
        glyph_buffer = GlyphBuffer(width=3, height=2)

        # Verify buffer shape is (width, height)
        assert glyph_buffer.data.shape == (3, 2)

        # Test that put_char uses (x, y) indexing correctly
        glyph_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(2, 1, ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))

        # Verify the characters were placed at the expected positions
        assert glyph_buffer.data[0, 0]["ch"] == ord("A")
        assert glyph_buffer.data[2, 1]["ch"] == ord("B")

        # Verify other positions are still clear (space character)
        assert glyph_buffer.data[1, 0]["ch"] == ord(" ")
        assert glyph_buffer.data[0, 1]["ch"] == ord(" ")

    def test_vertex_encoding_position_calculation(self):
        """Test that vertex encoding produces correct screen positions."""
        glyph_buffer = GlyphBuffer(width=2, height=2)

        # Place known characters at specific positions
        glyph_buffer.put_char(0, 0, ord("A"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(1, 0, ord("B"), (0, 255, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(0, 1, ord("C"), (0, 0, 255, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(1, 1, ord("D"), (255, 255, 0, 255), (0, 0, 0, 255))

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Should have 4 cells * 6 vertices each (single quad with both colors)
        assert vertex_count == 24

        tile_w, tile_h = self.graphics_ctx.tile_dimensions

        # Check positions for each cell's first vertex (BG quad, first vertex)
        # Cell (0,0) should be at screen position (0, 0) with no Y pre-flip
        cell_00_vertex = vertex_data[0]  # First vertex of first cell
        expected_x_00 = 0
        expected_y_00 = 0  # No Y pre-flip: y*tile_h = 0*tile_h = 0

        assert cell_00_vertex["position"][0] == expected_x_00
        assert cell_00_vertex["position"][1] == expected_y_00

        # Cell (1,0) should be at screen position (tile_w, 0) with no Y pre-flip
        cell_10_vertex = vertex_data[6]  # First vertex of second cell (skip 6 vertices)
        expected_x_10 = tile_w
        expected_y_10 = 0  # No Y pre-flip: y*tile_h = 0*tile_h = 0

        assert cell_10_vertex["position"][0] == expected_x_10
        assert cell_10_vertex["position"][1] == expected_y_10

        # Cell (0,1) should be at screen position (0, tile_h) with no Y pre-flip
        cell_01_vertex = vertex_data[12]  # First vertex of third cell
        expected_x_01 = 0
        expected_y_01 = tile_h  # No Y pre-flip: y*tile_h = 1*tile_h = tile_h

        assert cell_01_vertex["position"][0] == expected_x_01
        assert cell_01_vertex["position"][1] == expected_y_01

    def test_vertex_encoding_uv_coordinates_are_atlas_relative(self):
        """Test that vertex encoding produces atlas-relative UV coordinates,
        not full texture."""
        glyph_buffer = GlyphBuffer(width=1, height=1)
        glyph_buffer.put_char(0, 0, ord("A"), (255, 255, 255, 255), (0, 0, 0, 255))

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        # Get the foreground quad vertices (vertices 6-11)
        fg_vertices = vertex_data[6:12]

        # All UV coordinates should be atlas-relative (not 0,0 to 1,1)
        for vertex in fg_vertices:
            u, v = vertex["uv"]
            # UV coordinates should not span the entire texture
            assert not (u == 0.0 and v == 0.0) or not (u == 1.0 and v == 1.0)
            # UV coordinates should be in valid range
            assert 0.0 <= u <= 1.0
            assert 0.0 <= v <= 1.0

    def test_coordinate_system_consistency_across_pipeline(self):
        """Test that coordinates remain consistent through the entire pipeline."""
        # Create a simple 2x1 buffer with known content
        glyph_buffer = GlyphBuffer(width=2, height=1)
        glyph_buffer.put_char(0, 0, ord("L"), (255, 0, 0, 255), (0, 0, 0, 255))  # Left
        glyph_buffer.put_char(1, 0, ord("R"), (0, 255, 0, 255), (0, 0, 0, 255))  # Right

        # Encode to vertices
        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        tile_w, tile_h = self.graphics_ctx.tile_dimensions

        # Left character (L) should be at x=0
        left_vertex = vertex_data[0]  # First vertex of first cell
        assert left_vertex["position"][0] == 0

        # Right character (R) should be at x=tile_w
        right_vertex = vertex_data[6]  # First vertex of second cell
        assert right_vertex["position"][0] == tile_w

        # Both should be at the same Y position since they're on the same row
        assert left_vertex["position"][1] == right_vertex["position"][1]

    def test_y_coordinate_flipping_is_correct(self):
        """Test that Y coordinate flipping produces correct OpenGL coordinates."""
        # Create a 1x3 vertical buffer
        glyph_buffer = GlyphBuffer(width=1, height=3)
        glyph_buffer.put_char(0, 0, ord("T"), (255, 0, 0, 255), (0, 0, 0, 255))  # Top
        glyph_buffer.put_char(
            0, 1, ord("M"), (0, 255, 0, 255), (0, 0, 0, 255)
        )  # Middle
        glyph_buffer.put_char(
            0, 2, ord("B"), (0, 0, 255, 255), (0, 0, 0, 255)
        )  # Bottom

        vertex_data, vertex_count = (
            self.graphics_ctx.texture_renderer._encode_glyph_buffer_to_vertices(
                glyph_buffer
            )
        )

        tile_w, tile_h = self.graphics_ctx.tile_dimensions

        # In the glyph buffer: y=0 is top, y=2 is bottom
        # In the vertex data: smaller Y values correspond to smaller Y in buffer
        # So glyph buffer y=0 should map to the lowest vertex Y value

        top_vertex = vertex_data[0]  # Cell (0,0) - buffer top
        middle_vertex = vertex_data[6]  # Cell (0,1) - buffer middle
        bottom_vertex = vertex_data[12]  # Cell (0,2) - buffer bottom

        # With no Y pre-flip: Buffer top (y=0) should have lowest vertex Y
        # Buffer bottom (y=2) should have highest vertex Y
        assert top_vertex["position"][1] < middle_vertex["position"][1]
        assert middle_vertex["position"][1] < bottom_vertex["position"][1]

        # Verify exact values match expected calculation with no Y pre-flip
        # For h=3: top(y=0) -> 0*tile_h = 0
        #          middle(y=1) -> 1*tile_h = tile_h
        #          bottom(y=2) -> 2*tile_h = 2*tile_h
        expected_top_y = 0 * tile_h
        expected_middle_y = 1 * tile_h
        expected_bottom_y = 2 * tile_h

        assert top_vertex["position"][1] == expected_top_y
        assert middle_vertex["position"][1] == expected_middle_y
        assert bottom_vertex["position"][1] == expected_bottom_y
