"""
Unit tests for the WGPU graphics backend.

These tests attempt to use real WGPU resources when available (headless GPU),
falling back to mocks only when GPU resources are unavailable.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from catley import colors
from catley.game.enums import BlendMode
from catley.util.glyph_buffer import GlyphBuffer

# Try to import WGPU - if not available, skip all tests
try:
    import wgpu

    from catley.backends.wgpu.graphics import WGPUGraphicsContext

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None
    WGPUGraphicsContext = None


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
        self.image = Mock()
        self.hotspot_x = 0
        self.hotspot_y = 0


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
class TestWGPUGraphicsContext:
    """Test suite for WGPUGraphicsContext."""

    @pytest.fixture(autouse=True)
    def setup_wgpu_context(self):
        """Create WGPU context for testing."""
        # Create mock GLWindow that simulates GlfwWindow
        from catley.backends.glfw.window import GlfwWindow

        mock_glfw_window_handle = Mock()
        mock_window = GlfwWindow(mock_glfw_window_handle)
        mock_window.get_size = Mock(return_value=(800, 600))
        mock_window.get_framebuffer_size = Mock(return_value=(800, 600))
        mock_window.flip = Mock()

        # Try to create real WGPU context, fall back to mock if unavailable
        try:
            # Attempt real WGPU context creation (defer init for testing)
            self.graphics_ctx = WGPUGraphicsContext(mock_window, _defer_init=True)  # type: ignore[misc]
            self.real_wgpu = True
        except Exception:
            # Fall back to mocked WGPU for environments without GPU
            with patch("wgpu.gpu") as mock_wgpu:
                mock_adapter = Mock()
                mock_device = Mock()
                mock_queue = Mock()
                mock_surface = Mock()

                mock_wgpu.request_adapter_sync.return_value = mock_adapter
                mock_adapter.request_device_sync.return_value = mock_device
                mock_device.queue = mock_queue
                mock_wgpu.create_surface_from_window.return_value = mock_surface

                self.graphics_ctx = WGPUGraphicsContext(mock_window, _defer_init=True)  # type: ignore[misc]
                self.real_wgpu = False

        # Add mock resource manager for testing
        mock_resource_manager = Mock()
        self.graphics_ctx.resource_manager = mock_resource_manager

        yield

        # Cleanup WGPU resources to prevent segfaults
        if hasattr(self, "graphics_ctx") and self.graphics_ctx:
            # No explicit cleanup needed for mocked contexts
            pass

    def test_initialization(self):
        """Test that graphics context initializes correctly."""
        assert self.graphics_ctx.window is not None
        assert (
            self.graphics_ctx.device is None
        )  # Not initialized until initialize() called
        assert self.graphics_ctx.queue is None
        assert self.graphics_ctx.wgpu_context is None
        assert self.graphics_ctx._coordinate_converter is None

    def test_tile_dimensions_property(self):
        """Test tile dimensions property."""
        tile_dims = self.graphics_ctx.tile_dimensions
        assert isinstance(tile_dims, tuple)
        assert len(tile_dims) == 2
        assert tile_dims == (20, 20)  # Default size

    def test_console_dimensions_properties(self):
        """Test console width and height properties."""
        width = self.graphics_ctx.console_width_tiles
        height = self.graphics_ctx.console_height_tiles
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width == 80  # Default placeholder
        assert height == 50  # Default placeholder

    def test_coordinate_converter_property(self):
        """Test coordinate converter property creates instance when accessed."""
        # Initially None
        assert self.graphics_ctx._coordinate_converter is None

        # Accessing property creates instance
        converter = self.graphics_ctx.coordinate_converter
        assert converter is not None
        assert self.graphics_ctx._coordinate_converter is converter

        # Subsequent access returns same instance
        converter2 = self.graphics_ctx.coordinate_converter
        assert converter2 is converter

    def test_console_to_screen_coords_fallback(self):
        """Test console to screen coordinate conversion fallback."""
        # Test fallback behavior when coordinate converter not initialized
        result = self.graphics_ctx.console_to_screen_coords(10.0, 5.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # With 20x20 tiles: (10 * 20, 5 * 20) = (200, 100)
        assert result == (200, 100)

    def test_pixel_to_tile_fallback(self):
        """Test pixel to tile coordinate conversion fallback."""
        # Test fallback behavior when coordinate converter not initialized
        result = self.graphics_ctx.pixel_to_tile(240, 120)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # With 20x20 tiles: (240 // 20, 120 // 20) = (12, 6)
        assert result == (12, 6)

    def test_get_display_scale_factor(self):
        """Test display scale factor method."""
        scale = self.graphics_ctx.get_display_scale_factor()
        assert isinstance(scale, tuple)
        assert len(scale) == 2
        assert scale == (1.0, 1.0)  # Default implementation

    def test_update_dimensions(self):
        """Test update dimensions method."""
        # Should not raise exception (currently a no-op)
        self.graphics_ctx.update_dimensions()

    @pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
    def test_wgpu_initialization_real(self):
        """Test WGPU initialization with real GPU resources if available."""
        if not self.real_wgpu:
            pytest.skip("Real WGPU not available, using mocks")

        # Suppress GLFW initialization warning during tests
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*GLFW library is not initialized.*",
            )

            # Mock the actual WGPU calls since we don't want to create real windows
            with (
                patch("wgpu.gpu.request_adapter_sync") as mock_adapter,
                patch(
                    "catley.backends.wgpu.window_wrapper.get_glfw_present_methods"
                ) as mock_present_methods,
                patch(
                    "catley.backends.wgpu.window_wrapper.WGPUWindowWrapper.get_context"
                ) as mock_get_context,
            ):
                mock_device = Mock()
                mock_queue = Mock()
                mock_device.queue = mock_queue

                mock_adapter_instance = Mock()
                mock_adapter_instance.request_device_sync.return_value = mock_device
                mock_adapter.return_value = mock_adapter_instance
                mock_present_methods.return_value = {
                    "screen": {"platform": "test", "window": 123}
                }

                mock_context = Mock()
                mock_context.get_preferred_format.return_value = "bgra8unorm"
                mock_get_context.return_value = mock_context

                # Mock window framebuffer size for update_dimensions() call
                self.graphics_ctx.window.get_framebuffer_size.return_value = (
                    1600,
                    1000,
                )

                # Test initialization (using private method for testing)
                self.graphics_ctx._initialize()

                # Verify WGPU initialization completed successfully
                assert self.graphics_ctx.device is mock_device
                assert self.graphics_ctx.queue is mock_queue

                # Verify initialization completed successfully with required components
                assert self.graphics_ctx.screen_renderer is not None
                assert self.graphics_ctx.ui_texture_renderer is not None
                assert self.graphics_ctx.letterbox_geometry is not None
                assert self.graphics_ctx.atlas_texture is not None
                assert self.graphics_ctx.uv_map is not None
                assert self.graphics_ctx._coordinate_converter is not None

    def test_not_implemented_methods(self):
        """Test that not-yet-implemented methods raise NotImplementedError."""
        # These methods are now implemented, so test that they don't crash
        # when called with minimal/None renderers

        # draw_actor_smooth should handle None screen_renderer gracefully
        self.graphics_ctx.draw_actor_smooth("@", colors.WHITE, 10.0, 10.0)

        # draw_mouse_cursor should handle None ui_texture_renderer gracefully
        mock_cursor = MockCursorManager()
        self.graphics_ctx.draw_mouse_cursor(mock_cursor)  # type: ignore[arg-type]

        # render_particles should handle None screen_renderer gracefully
        self.graphics_ctx.render_particles(Mock(), Mock(), Mock(), Mock())

        # apply_environmental_effect is now implemented
        self.graphics_ctx.apply_environmental_effect(
            (10.0, 10.0), 5.0, colors.BLUE, 0.8, BlendMode.TINT
        )

        # present_texture and draw_background are now implemented
        self.graphics_ctx.present_texture(Mock(), 0, 0, 10, 10)
        self.graphics_ctx.draw_background(Mock(), 0, 0, 10, 10)

        self.graphics_ctx.draw_debug_rect(0, 0, 100, 100, colors.GREEN)

        # create_canvas should now work and return a WGPUCanvas
        canvas = self.graphics_ctx.create_canvas()
        assert canvas is not None
        assert hasattr(canvas, "artifact_type")
        assert canvas.artifact_type == "glyph_buffer"

    def test_texture_from_numpy_implementation(self):
        """Test that texture_from_numpy is now implemented and validates input."""
        import numpy as np

        # Test invalid input validation
        with pytest.raises(ValueError, match="Expected RGBA or RGB image array"):
            # 2D array should fail
            invalid_array = np.ones((10, 10), dtype=np.uint8)
            self.graphics_ctx.texture_from_numpy(invalid_array)

        with pytest.raises(ValueError, match="Expected RGBA or RGB image array"):
            # Wrong number of channels should fail
            invalid_array = np.ones((10, 10, 2), dtype=np.uint8)
            self.graphics_ctx.texture_from_numpy(invalid_array)

        # Test that method fails gracefully when resource manager is not initialized
        # Temporarily set resource manager to None to test the error case
        original_resource_manager = self.graphics_ctx.resource_manager
        self.graphics_ctx.resource_manager = None
        try:
            with pytest.raises(
                AssertionError, match="Resource manager not initialized"
            ):
                # Valid RGBA array
                valid_array = np.ones((10, 10, 4), dtype=np.uint8)
                self.graphics_ctx.texture_from_numpy(valid_array)
        finally:
            # Restore the mock resource manager
            self.graphics_ctx.resource_manager = original_resource_manager

    def test_interpolation_alpha_default_handling(self):
        """Test that draw_actor_smooth handles None interpolation_alpha correctly."""
        # This method is now implemented - test that it handles None gracefully
        self.graphics_ctx.draw_actor_smooth(
            "@", colors.WHITE, 10.0, 10.0, interpolation_alpha=None
        )

    def test_render_glyph_buffer_to_texture(self):
        """Test rendering a GlyphBuffer to texture."""
        # Mock the texture renderer directly (can't initialize WGPU in tests)
        mock_texture_renderer = Mock()
        mock_texture = Mock()  # Mock WGPU texture
        mock_texture_renderer.render = Mock(return_value=mock_texture)
        self.graphics_ctx.texture_renderer = mock_texture_renderer

        # Create a small glyph buffer for testing
        glyph_buffer = GlyphBuffer(5, 3)

        # Add some test content
        glyph_buffer.put_char(0, 0, ord("H"), (255, 0, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(1, 0, ord("e"), (0, 255, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(2, 0, ord("l"), (0, 0, 255, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(3, 0, ord("l"), (255, 255, 0, 255), (0, 0, 0, 255))
        glyph_buffer.put_char(4, 0, ord("o"), (255, 0, 255, 255), (0, 0, 0, 255))

        # Add a character with background
        glyph_buffer.put_char(
            2, 1, ord("@"), (255, 255, 255, 255), (128, 128, 128, 255)
        )

        # Test that render_glyph_buffer_to_texture returns the mock texture
        result = self.graphics_ctx.render_glyph_buffer_to_texture(glyph_buffer)
        assert result is mock_texture

        # Verify that the texture renderer was called with expected parameters
        # Note: cpu_buffer_override will be created as a temporary buffer
        call_args = mock_texture_renderer.render.call_args
        assert call_args[0][0] is glyph_buffer  # First arg is glyph_buffer
        assert (
            call_args[0][1] is not None
        )  # Second arg is cpu_buffer_override (created)
        assert call_args[1]["buffer_override"] is None  # buffer_override is None

    def test_add_tile_to_screen(self):
        """Test adding tiles to the screen renderer."""
        # Mock the screen renderer directly
        mock_screen_renderer = Mock()
        mock_screen_renderer.add_quad = Mock()
        self.graphics_ctx.screen_renderer = mock_screen_renderer
        self.graphics_ctx.uv_map = np.zeros((256, 4), dtype="f4")  # Mock UV map
        # Set UV coordinates for character 'A' (65)
        self.graphics_ctx.uv_map[65] = [0.0, 0.0, 0.1, 0.1]

        # Test adding a tile
        self.graphics_ctx.add_tile_to_screen(ord("A"), 10, 5, colors.RED, colors.BLUE)

        # Verify add_quad was called
        mock_screen_renderer.add_quad.assert_called_once()

        # Check the arguments
        call_args = mock_screen_renderer.add_quad.call_args
        assert call_args is not None
        kwargs = call_args.kwargs
        assert kwargs["x"] == 200  # 10 * 20 (tile width)
        assert kwargs["y"] == 100  # 5 * 20 (tile height)
        assert kwargs["w"] == 20
        assert kwargs["h"] == 20
        assert kwargs["color_rgba"] == (1.0, 0.0, 0.0, 1.0)  # Red normalized


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_wgpu_graphics_context_window_parameters():
    """Test WGPUGraphicsContext window parameter handling."""
    from catley.backends.glfw.window import GlfwWindow

    mock_glfw_window_handle = Mock()
    mock_window = GlfwWindow(mock_glfw_window_handle)
    mock_window.get_size = Mock(return_value=(1024, 768))
    mock_window.get_framebuffer_size = Mock(return_value=(1024, 768))
    mock_window.flip = Mock()

    ctx = WGPUGraphicsContext(mock_window, _defer_init=True)  # type: ignore[misc]

    # Verify window is properly stored
    assert ctx.window is mock_window


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_wgpu_graphics_context_default_parameters():
    """Test WGPUGraphicsContext default parameter handling."""
    from catley.backends.glfw.window import GlfwWindow

    mock_glfw_window_handle = Mock()
    mock_window = GlfwWindow(mock_glfw_window_handle)
    mock_window.get_size = Mock(return_value=(800, 600))
    mock_window.get_framebuffer_size = Mock(return_value=(800, 600))
    mock_window.flip = Mock()

    ctx = WGPUGraphicsContext(mock_window, _defer_init=True)  # type: ignore[misc]

    # Verify window is properly stored
    assert ctx.window is mock_window
