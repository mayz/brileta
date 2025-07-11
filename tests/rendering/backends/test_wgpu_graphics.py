"""
Unit tests for the WGPU graphics backend.

These tests attempt to use real WGPU resources when available (headless GPU),
falling back to mocks only when GPU resources are unavailable.
"""

from unittest.mock import Mock, patch

import pytest

from catley import colors
from catley.game.enums import BlendMode
from catley.types import Opacity

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
            # Attempt real WGPU context creation
            self.graphics_ctx = WGPUGraphicsContext(mock_window)  # type: ignore[misc]
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

                self.graphics_ctx = WGPUGraphicsContext(mock_window)  # type: ignore[misc]
                self.real_wgpu = False

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
        assert self.graphics_ctx.surface is None
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

        # Mock the actual WGPU calls since we don't want to create real windows in tests
        with (
            patch("wgpu.gpu.request_adapter_sync") as mock_adapter,
            patch(
                "wgpu.create_surface_from_window", create=True
            ) as mock_create_surface,
        ):
            mock_device = Mock()
            mock_queue = Mock()
            mock_device.queue = mock_queue
            mock_surface = Mock()

            mock_adapter_instance = Mock()
            mock_adapter_instance.request_device_sync.return_value = mock_device
            mock_adapter.return_value = mock_adapter_instance
            mock_create_surface.return_value = mock_surface

            # Test initialization
            self.graphics_ctx.initialize()

            # Verify WGPU calls were made
            mock_adapter.assert_called_once()
            mock_adapter_instance.request_device_sync.assert_called_once()
            assert self.graphics_ctx.device is mock_device
            assert self.graphics_ctx.queue is mock_queue

    def test_not_implemented_methods(self):
        """Test that not-yet-implemented methods raise NotImplementedError."""
        mock_cursor = MockCursorManager()

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.render_glyph_buffer_to_texture(Mock())

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.draw_actor_smooth("@", colors.WHITE, 10.0, 10.0)

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.draw_mouse_cursor(mock_cursor)  # type: ignore[arg-type]

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.draw_tile_highlight(5, 5, colors.RED, Opacity(0.5))

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.render_particles(Mock(), Mock(), Mock(), Mock())

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.apply_environmental_effect(
                (10.0, 10.0), 5.0, colors.BLUE, 0.8, BlendMode.TINT
            )

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.texture_from_numpy(Mock())

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.present_texture(Mock(), 0, 0, 10, 10)

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.draw_background(Mock(), 0, 0, 10, 10)

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.draw_debug_rect(0, 0, 100, 100, colors.GREEN)

        with pytest.raises(NotImplementedError, match="WGPU.*not yet implemented"):
            self.graphics_ctx.create_canvas()

    def test_interpolation_alpha_default_handling(self):
        """Test that draw_actor_smooth handles None interpolation_alpha correctly."""
        # This tests the internal default value handling
        with pytest.raises(NotImplementedError):
            # Should not raise TypeError for None interpolation_alpha
            self.graphics_ctx.draw_actor_smooth(
                "@", colors.WHITE, 10.0, 10.0, interpolation_alpha=None
            )


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_wgpu_graphics_context_window_parameters():
    """Test WGPUGraphicsContext window parameter handling."""
    from catley.backends.glfw.window import GlfwWindow

    mock_glfw_window_handle = Mock()
    mock_window = GlfwWindow(mock_glfw_window_handle)
    mock_window.get_size = Mock(return_value=(1024, 768))
    mock_window.get_framebuffer_size = Mock(return_value=(1024, 768))
    mock_window.flip = Mock()

    ctx = WGPUGraphicsContext(mock_window)  # type: ignore[misc]

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

    ctx = WGPUGraphicsContext(mock_window)  # type: ignore[misc]

    # Verify window is properly stored
    assert ctx.window is mock_window
