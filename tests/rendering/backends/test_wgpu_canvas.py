"""
Unit tests for the WGPU canvas implementation.

Tests the WGPUCanvas class for Canvas interface compliance and basic functionality.
"""

from unittest.mock import Mock

import pytest

from catley import colors
from catley.util.coordinates import TileCoord

# Try to import WGPU - if not available, skip all tests
try:
    import wgpu

    from catley.backends.wgpu.canvas import WGPUCanvas
    from catley.backends.wgpu.graphics import WGPUGraphicsContext

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None
    WGPUCanvas = None
    WGPUGraphicsContext = None


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
class TestWGPUCanvas:
    """Test suite for WGPUCanvas."""

    @pytest.fixture(autouse=True)
    def setup_canvas(self):
        """Create mock renderer and canvas for testing."""
        # Create a mock renderer that satisfies the GraphicsContext interface
        self.mock_renderer = Mock()
        self.mock_renderer.tile_dimensions = (20, 20)

        # Create canvas instances for testing
        self.mock_resource_manager = Mock()
        self.canvas = WGPUCanvas(
            self.mock_renderer, self.mock_resource_manager, transparent=True
        )  # type: ignore[misc]
        self.opaque_canvas = WGPUCanvas(
            self.mock_renderer, self.mock_resource_manager, transparent=False
        )  # type: ignore[misc]

    def test_initialization(self):
        """Test canvas initialization."""
        assert self.canvas.renderer is self.mock_renderer
        assert self.canvas.transparent is True
        assert self.opaque_canvas.transparent is False

        # Inherited Canvas attributes
        assert self.canvas.drawing_offset_x == 0
        assert self.canvas.drawing_offset_y == 0
        assert self.canvas.width == 0
        assert self.canvas.height == 0

    def test_artifact_type_property(self):
        """Test artifact type property."""
        assert self.canvas.artifact_type == "glyph_buffer"

    def test_get_text_metrics(self):
        """Test text metrics method."""
        metrics = self.canvas.get_text_metrics("Hello")
        assert isinstance(metrics, tuple)
        assert len(metrics) == 3
        width, height, line_height = metrics
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(line_height, int)

        # With tile-based implementation: len("Hello") * 20 = 100 (tile_width=20)
        assert width == 100
        assert height == 20
        assert line_height == 20

    def test_get_text_metrics_with_font_size(self):
        """Test text metrics method with font size parameter."""
        metrics = self.canvas.get_text_metrics("Test", font_size=16)
        assert isinstance(metrics, tuple)
        assert len(metrics) == 3
        # Font size parameter is currently ignored
        width, height, line_height = metrics
        assert width == 80  # len("Test") * 20 (tile_width=20)

    def test_wrap_text(self):
        """Test text wrapping method."""
        text = "This is a long line of text"
        wrapped = self.canvas.wrap_text(text, max_width=100)
        assert isinstance(wrapped, list)
        # With tile_width=20, max_width=100 allows 5 chars per line
        # "This is a long line of text" should wrap into multiple lines
        assert len(wrapped) > 1
        assert all(len(line) <= 5 for line in wrapped)

    def test_wrap_text_with_font_size(self):
        """Test text wrapping with font size parameter."""
        text = "Test text"
        wrapped = self.canvas.wrap_text(text, max_width=50, font_size=14)
        assert isinstance(wrapped, list)
        # With tile_width=20, max_width=50 allows 2 chars per line
        # "Test text" (9 chars) should wrap
        assert len(wrapped) > 1

    def test_get_effective_line_height(self):
        """Test effective line height method."""
        line_height = self.canvas.get_effective_line_height()
        assert isinstance(line_height, int)
        assert line_height == 20  # Placeholder value

    def test_configure_dimensions(self):
        """Test dimension configuration."""
        self.canvas.configure_dimensions(800, 600)
        assert self.canvas.width == 800
        assert self.canvas.height == 600

    def test_configure_scaling(self):
        """Test scaling configuration."""
        # Should not raise exception
        self.canvas.configure_scaling(24)

    def test_configure_drawing_offset(self):
        """Test drawing offset configuration."""
        self.canvas.configure_drawing_offset(10, 15)
        assert self.canvas.drawing_offset_x == 10
        assert self.canvas.drawing_offset_y == 15

    def test_begin_frame(self):
        """Test frame beginning."""
        # Should not raise exception
        self.canvas.begin_frame()
        # Frame operations should be cleared
        assert self.canvas._frame_ops == []

    def test_end_frame(self):
        """Test frame ending."""
        self.canvas.begin_frame()
        result = self.canvas.end_frame()
        # With placeholder implementation, should return None
        assert result is None

    def test_draw_text_operation_recording(self):
        """Test that draw_text records operations."""
        self.canvas.begin_frame()
        self.canvas.draw_text(10, 20, "Hello", colors.WHITE)

        # Operation should be recorded
        assert len(self.canvas._frame_ops) == 1
        op = self.canvas._frame_ops[0]
        assert op[1] == 10  # pixel_x
        assert op[2] == 20  # pixel_y
        assert op[3] == "Hello"  # text
        assert op[4] == colors.WHITE  # color

    def test_draw_rect_operation_recording(self):
        """Test that draw_rect records operations."""
        self.canvas.begin_frame()
        self.canvas.draw_rect(5, 10, 100, 50, colors.RED, fill=True)

        # Operation should be recorded
        assert len(self.canvas._frame_ops) == 1
        op = self.canvas._frame_ops[0]
        assert op[1] == 5  # pixel_x
        assert op[2] == 10  # pixel_y
        assert op[3] == 100  # width
        assert op[4] == 50  # height
        assert op[5] == colors.RED  # color
        assert op[6] is True  # fill

    def test_draw_frame_operation_recording(self):
        """Test that draw_frame records operations."""
        self.canvas.begin_frame()
        self.canvas.draw_frame(
            TileCoord(2),
            TileCoord(3),
            TileCoord(10),
            TileCoord(8),
            colors.BLUE,
            colors.BLACK,
        )

        # Operation should be recorded
        assert len(self.canvas._frame_ops) == 1
        op = self.canvas._frame_ops[0]
        assert op[1] == 2  # tile_x
        assert op[2] == 3  # tile_y
        assert op[3] == 10  # width
        assert op[4] == 8  # height
        assert op[5] == colors.BLUE  # fg
        assert op[6] == colors.BLACK  # bg

    def test_create_texture_implementation(self):
        """Test that create_texture works with GlyphBuffer artifacts."""
        from catley.util.glyph_buffer import GlyphBuffer

        # Mock the renderer's render_glyph_buffer_to_texture method
        self.mock_renderer.render_glyph_buffer_to_texture = Mock(
            return_value="mock_texture"
        )

        # Test with GlyphBuffer artifact
        glyph_buffer = GlyphBuffer(5, 3)
        result = self.canvas.create_texture(self.mock_renderer, glyph_buffer)
        assert result == "mock_texture"
        self.mock_renderer.render_glyph_buffer_to_texture.assert_called_once_with(
            glyph_buffer, None
        )

        # Test with non-GlyphBuffer artifact
        result = self.canvas.create_texture(self.mock_renderer, "not_a_glyph_buffer")
        assert result is None

    def test_prepare_for_rendering(self):
        """Test prepare for rendering method."""
        # Should return False when dimensions are 0
        result = self.canvas._prepare_for_rendering()
        assert result is False

        # Should return True when dimensions are set
        self.canvas.configure_dimensions(100, 80)
        result = self.canvas._prepare_for_rendering()
        assert result is True
        assert self.canvas.private_glyph_buffer is not None

    def test_canvas_interface_compliance(self):
        """Test that WGPUCanvas properly implements Canvas interface."""
        # Test that all required Canvas methods exist and are callable
        from catley.view.render.canvas import Canvas

        # Verify WGPUCanvas is a proper subclass
        assert issubclass(WGPUCanvas, Canvas)  # type: ignore[arg-type]
        assert isinstance(self.canvas, Canvas)

        # Test that required abstract methods are implemented
        required_methods = [
            "get_text_metrics",
            "wrap_text",
            "create_texture",
            "_update_scaling_internal",
            "get_effective_line_height",
            "_prepare_for_rendering",
            "_render_text_op",
            "_render_rect_op",
            "_render_frame_op",
            "_create_artifact_from_rendered_content",
        ]

        for method_name in required_methods:
            assert hasattr(self.canvas, method_name)
            assert callable(getattr(self.canvas, method_name))

    def test_font_metrics_method(self):
        """Test get_font_metrics method."""
        ascent, descent = self.canvas.get_font_metrics()
        assert isinstance(ascent, int)
        assert isinstance(descent, int)
        assert ascent == 0  # Default implementation
        assert descent == 0  # Default implementation


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_canvas_creation_parameters():
    """Test canvas creation with different parameters."""
    mock_renderer = Mock()
    mock_renderer.tile_dimensions = (16, 16)
    mock_resource_manager = Mock()

    # Test transparent canvas (default)
    transparent_canvas = WGPUCanvas(mock_renderer, mock_resource_manager)  # type: ignore[misc]
    assert transparent_canvas.transparent is True
    assert transparent_canvas.renderer is mock_renderer

    # Test opaque canvas
    opaque_canvas = WGPUCanvas(mock_renderer, mock_resource_manager, transparent=False)  # type: ignore[misc]
    assert opaque_canvas.transparent is False
    assert opaque_canvas.renderer is mock_renderer


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_canvas_operation_caching():
    """Test canvas operation caching behavior."""
    mock_renderer = Mock()
    mock_renderer.tile_dimensions = (20, 20)
    mock_resource_manager = Mock()
    canvas = WGPUCanvas(mock_renderer, mock_resource_manager)  # type: ignore[misc]

    # Begin frame and add operations
    canvas.begin_frame()
    canvas.draw_text(0, 0, "Test", colors.WHITE)

    # First end_frame - should process operations
    result1 = canvas.end_frame()

    # Begin new frame with same operations
    canvas.begin_frame()
    canvas.draw_text(0, 0, "Test", colors.WHITE)

    # Second end_frame - should use cached result if cache logic is implemented
    result2 = canvas.end_frame()

    # Both should return None with current placeholder implementation
    assert result1 is None
    assert result2 is None
