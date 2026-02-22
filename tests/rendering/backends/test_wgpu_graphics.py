"""
Unit tests for the WGPU graphics backend.

These tests attempt to use real WGPU resources when available (headless GPU),
falling back to mocks only when GPU resources are unavailable.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from brileta import colors, config
from brileta.game.enums import BlendMode
from brileta.types import Opacity
from brileta.util.glyph_buffer import GlyphBuffer

# Try to import WGPU - if not available, skip all tests
try:
    import wgpu

    from brileta.backends.wgpu.graphics import (
        WGPUGraphicsContext,
        _infer_compose_tile_dimensions,
    )

    WGPU_AVAILABLE = True
except ImportError:
    WGPU_AVAILABLE = False
    wgpu = None
    WGPUGraphicsContext = None
    _infer_compose_tile_dimensions = None


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
        from brileta.backends.glfw.window import GlfwWindow

        mock_glfw_window_handle = Mock()
        mock_window = GlfwWindow(mock_glfw_window_handle)
        mock_window.get_size = Mock(return_value=(800, 600))
        mock_window.get_framebuffer_size = Mock(return_value=(800, 600))
        mock_window.flip = Mock()

        # Try to create real WGPU context, fall back to mock if unavailable
        try:
            # Attempt real WGPU context creation (defer init for testing)
            self.graphics_ctx = WGPUGraphicsContext(mock_window, _defer_init=True)
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

                self.graphics_ctx = WGPUGraphicsContext(mock_window, _defer_init=True)
                self.real_wgpu = False

        # Add mock resource manager for testing
        mock_resource_manager = Mock()
        self.graphics_ctx.resource_manager = mock_resource_manager

        # Patch glfw.get_window_content_scale to avoid a mock cascade.
        # When passed a Mock window handle, GLFW's ctypes layer creates 86K+
        # child mock objects before raising -- costing ~50ms per call.
        with patch("glfw.get_window_content_scale", return_value=(1.0, 1.0)):
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
        assert width > 0
        assert height > 0

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

    def test_update_dimensions_syncs_glyph_renderer_tile_dimensions(self, monkeypatch):
        """Resize updates off-screen renderer tile dimensions."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        mock_glyph_renderer = Mock()
        self.graphics_ctx.glyph_renderer = mock_glyph_renderer
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1680, 1050)

        self.graphics_ctx.update_dimensions()

        mock_glyph_renderer.set_tile_dimensions.assert_called_with((40, 40))

    def test_letterbox_geometry_calculation_is_pure(self):
        """Letterbox geometry helper should not mutate console dimension state."""
        self.graphics_ctx._console_width_tiles = 77
        self.graphics_ctx._console_height_tiles = 55

        geometry = self.graphics_ctx._calculate_letterbox_geometry(800, 600, 20, 20)

        assert geometry == (0, 0, 800, 600)
        assert self.graphics_ctx.console_width_tiles == 77
        assert self.graphics_ctx.console_height_tiles == 55

    def test_content_scale_auto_detection_retina_ratio(self, monkeypatch):
        """A 2x framebuffer/window ratio should produce a 2x content scale."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1200)

        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        assert self.graphics_ctx.scale_factor == 2
        assert self.graphics_ctx.tile_dimensions == (40, 40)
        assert self.graphics_ctx.console_width_tiles == 40
        assert self.graphics_ctx.console_height_tiles == 30

    def test_content_scale_rounds_fractional_ratio(self, monkeypatch):
        """A 1.5x framebuffer/window ratio should round to 2x scale."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1200, 900)

        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        assert self.graphics_ctx.scale_factor == 2
        assert self.graphics_ctx.tile_dimensions == (40, 40)
        assert self.graphics_ctx.console_width_tiles == 30
        assert self.graphics_ctx.console_height_tiles == 22
        assert self.graphics_ctx.letterbox_geometry == (0, 10, 1200, 880)

    def test_content_scale_uses_glfw_scale_in_fullscreen_like_sizes(self, monkeypatch):
        """Content scale should honor GLFW DPI scale even if size ratio is 1:1."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)
        self.graphics_ctx.window.get_size.return_value = (1600, 1000)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1000)
        monkeypatch.setattr(
            self.graphics_ctx,
            "get_display_scale_factor",
            lambda: (2.0, 2.0),
        )

        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        assert self.graphics_ctx.scale_factor == 2
        assert self.graphics_ctx.tile_dimensions == (40, 40)
        assert self.graphics_ctx.console_width_tiles == 40
        assert self.graphics_ctx.console_height_tiles == 25

    def test_content_scale_stays_stable_when_signals_drop_after_resize(
        self, monkeypatch
    ):
        """Keep high-DPI scale stable across mode toggles with inconsistent signals."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)

        # First pass: detect high-DPI scale from window/ratio signals.
        self.graphics_ctx.window.get_size.return_value = (800, 500)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1000)
        monkeypatch.setattr(
            self.graphics_ctx,
            "get_display_scale_factor",
            lambda: (2.0, 2.0),
        )
        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()
        assert self.graphics_ctx.scale_factor == 2

        # Second pass: fullscreen-like inconsistent signals both report 1x.
        self.graphics_ctx.window.get_size.return_value = (1600, 1000)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1000)
        monkeypatch.setattr(
            self.graphics_ctx,
            "get_display_scale_factor",
            lambda: (1.0, 1.0),
        )
        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        # Sticky hint should preserve readable tile scale.
        assert self.graphics_ctx.scale_factor == 2
        assert self.graphics_ctx.tile_dimensions == (40, 40)
        assert self.graphics_ctx.console_width_tiles == 40
        assert self.graphics_ctx.console_height_tiles == 25

    def test_content_scale_relocks_after_stable_monitor_scale_change(
        self, monkeypatch
    ) -> None:
        """Locked scale should update after repeated, consistent new DPI signals."""
        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)

        # Lock initial scale to 2x.
        self.graphics_ctx.window.get_size.return_value = (800, 500)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1000)
        monkeypatch.setattr(
            self.graphics_ctx,
            "get_display_scale_factor",
            lambda: (2.0, 2.0),
        )
        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()
        assert self.graphics_ctx.locked_content_scale == 2

        # Observe stable 1x for enough samples to relock.
        self.graphics_ctx.window.get_size.return_value = (1600, 1000)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (1600, 1000)
        monkeypatch.setattr(
            self.graphics_ctx,
            "get_display_scale_factor",
            lambda: (1.0, 1.0),
        )
        for _ in range(self.graphics_ctx._CONTENT_SCALE_RELOCK_STABLE_SAMPLES):
            self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        assert self.graphics_ctx.locked_content_scale == 1
        assert self.graphics_ctx.tile_dimensions == (20, 20)

    def test_tile_zoom_scales_tiles_when_console_budget_allows(self, monkeypatch):
        """TILE_ZOOM should enlarge display tiles and reduce console size."""
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 1)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 1)
        self.graphics_ctx.window.get_size.return_value = (3200, 2400)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (3200, 2400)

        monkeypatch.setattr(config, "TILE_ZOOM", 1)
        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()
        zoom1_dims = self.graphics_ctx.tile_dimensions
        zoom1_console = (
            self.graphics_ctx.console_width_tiles,
            self.graphics_ctx.console_height_tiles,
        )

        monkeypatch.setattr(config, "TILE_ZOOM", 2)
        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()
        zoom2_dims = self.graphics_ctx.tile_dimensions
        zoom2_console = (
            self.graphics_ctx.console_width_tiles,
            self.graphics_ctx.console_height_tiles,
        )

        assert zoom1_dims == (20, 20)
        assert zoom2_dims == (40, 40)
        assert zoom1_console == (160, 120)
        assert zoom2_console == (80, 60)

    def test_zoom_safety_cap_reduces_zoom_when_console_would_be_too_small(
        self, monkeypatch
    ):
        """Renderer should step zoom down until console minimums are respected."""
        monkeypatch.setattr(config, "TILE_ZOOM", 3)
        monkeypatch.setattr(config, "MIN_CONSOLE_WIDTH", 60)
        monkeypatch.setattr(config, "MIN_CONSOLE_HEIGHT", 40)
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (800, 600)

        self.graphics_ctx._calculate_letterbox_geometry_and_tiles()

        # TILE_ZOOM=3 would make 60x60 tiles and a tiny console here.
        # The safety cap steps it down to effective zoom 1.
        assert self.graphics_ctx.tile_dimensions == (20, 20)
        assert self.graphics_ctx.scale_factor == 1
        assert self.graphics_ctx.console_width_tiles == 40
        assert self.graphics_ctx.console_height_tiles == 30

    def test_finalize_present_returns_false_when_framebuffer_is_zero(self):
        """Presentation should be skipped while minimized (0-sized framebuffer)."""
        self.graphics_ctx.screen_renderer = Mock()
        self.graphics_ctx.wgpu_context = Mock()
        self.graphics_ctx.resource_manager = Mock()
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (0, 0)

        rendered = self.graphics_ctx.finalize_present()

        assert rendered is False

    def test_finalize_present_returns_false_when_texture_acquire_fails(self):
        """Surface acquire failures should skip this frame instead of crashing."""
        self.graphics_ctx.screen_renderer = Mock()
        self.graphics_ctx.wgpu_context = Mock()
        self.graphics_ctx.resource_manager = Mock()
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (800, 600)
        self.graphics_ctx._last_window_size = (800, 600)
        self.graphics_ctx._last_framebuffer_size = (800, 600)
        self.graphics_ctx.wgpu_context.get_current_texture.side_effect = RuntimeError(
            "surface lost"
        )

        rendered = self.graphics_ctx.finalize_present()

        assert rendered is False

    def test_finalize_present_logs_first_exception_in_failure_streak(self):
        """Present failures should be visible without spamming logs every frame."""
        self.graphics_ctx.screen_renderer = Mock()
        self.graphics_ctx.wgpu_context = Mock()
        self.graphics_ctx.resource_manager = Mock()
        self.graphics_ctx.window.get_size.return_value = (800, 600)
        self.graphics_ctx.window.get_framebuffer_size.return_value = (800, 600)
        self.graphics_ctx._last_window_size = (800, 600)
        self.graphics_ctx._last_framebuffer_size = (800, 600)
        self.graphics_ctx.wgpu_context.get_current_texture.side_effect = RuntimeError(
            "surface lost"
        )

        with patch("brileta.backends.wgpu.graphics.logger.exception") as log_exception:
            assert self.graphics_ctx.finalize_present() is False
            assert self.graphics_ctx.finalize_present() is False

        log_exception.assert_called_once()

    def test_infer_compose_tile_dimensions(self):
        """Compose tile size should be inferred from texture/mask dimensions."""
        assert _infer_compose_tile_dimensions is not None

        assert _infer_compose_tile_dimensions(1640, 1240, 82, 62) == (20, 20)
        assert _infer_compose_tile_dimensions(1680, 1050, 80, 50) == (21, 21)
        assert _infer_compose_tile_dimensions(100, 100, 0, 10) is None
        assert _infer_compose_tile_dimensions(100, 100, 10, 0) is None
        assert _infer_compose_tile_dimensions(101, 100, 10, 10) is None
        assert _infer_compose_tile_dimensions(100, 103, 10, 10) is None

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
                    "brileta.backends.wgpu.window_wrapper.create_wgpu_context"
                ) as mock_create_ctx,
            ):
                mock_device = Mock()
                mock_queue = Mock()
                mock_device.queue = mock_queue

                mock_adapter_instance = Mock()
                mock_adapter_instance.request_device_sync.return_value = mock_device
                mock_adapter.return_value = mock_adapter_instance

                mock_context = Mock()
                mock_context.get_preferred_format.return_value = "bgra8unorm"
                mock_create_ctx.return_value = mock_context

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
                assert self.graphics_ctx.ui_renderer is not None
                assert self.graphics_ctx.letterbox_geometry is not None
                assert self.graphics_ctx.atlas_texture is not None
                assert self.graphics_ctx.uv_map is not None
                assert self.graphics_ctx._coordinate_converter is not None

    def test_not_implemented_methods(self):
        """Test that not-yet-implemented methods raise NotImplementedError."""
        # These methods are now implemented, so test that they don't crash
        # when called with minimal/None renderers

        # draw_actor should handle None screen_renderer gracefully
        self.graphics_ctx.draw_actor("@", colors.WHITE, 10.0, 10.0)

        # draw_mouse_cursor should handle None ui_renderer gracefully
        mock_cursor = MockCursorManager()
        self.graphics_ctx.draw_mouse_cursor(mock_cursor)  # type: ignore[arg-type]

        # render_particles should handle None screen_renderer gracefully
        self.graphics_ctx.render_particles(Mock(), Mock(), Mock(), Mock())  # type: ignore[arg-type]

        # apply_environmental_effect is now implemented
        self.graphics_ctx.apply_environmental_effect(
            (10.0, 10.0), 5.0, colors.BLUE, 0.8, BlendMode.TINT
        )

        # present_texture and draw_background are now implemented
        self.graphics_ctx.present_texture(Mock(), 0, 0, 10, 10)
        self.graphics_ctx.draw_background(Mock(), 0, 0, 10, 10)

        self.graphics_ctx.draw_rect_outline(0, 0, 100, 100, colors.GREEN, Opacity(1.0))

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
        """Test that draw_actor handles None interpolation_alpha correctly."""
        # This method is now implemented - test that it handles None gracefully
        self.graphics_ctx.draw_actor(
            "@", colors.WHITE, 10.0, 10.0, interpolation_alpha=None
        )

    def test_render_glyph_buffer_to_texture(self):
        """Test rendering a GlyphBuffer to texture."""
        # Mock the glyph renderer directly (can't initialize WGPU in tests)
        mock_glyph_renderer = Mock()
        mock_texture = Mock()  # Mock WGPU texture
        mock_glyph_renderer.render = Mock(return_value=mock_texture)
        self.graphics_ctx.glyph_renderer = mock_glyph_renderer

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

        # Verify that the glyph renderer was called with expected parameters
        # Note: secondary_override (cpu buffer) will be created as a temporary buffer
        call_args = mock_glyph_renderer.render.call_args
        assert call_args[0][0] is glyph_buffer  # First arg is glyph_buffer
        assert (
            call_args[0][1] is not None
        )  # Second arg is cpu buffer (created internally)
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
    from brileta.backends.glfw.window import GlfwWindow

    mock_glfw_window_handle = Mock()
    mock_window = GlfwWindow(mock_glfw_window_handle)
    mock_window.get_size = Mock(return_value=(1024, 768))
    mock_window.get_framebuffer_size = Mock(return_value=(1024, 768))
    mock_window.flip = Mock()

    ctx = WGPUGraphicsContext(mock_window, _defer_init=True)
    # Verify window is properly stored
    assert ctx.window is mock_window


@pytest.mark.skipif(not WGPU_AVAILABLE, reason="WGPU not available")
def test_wgpu_graphics_context_default_parameters():
    """Test WGPUGraphicsContext default parameter handling."""
    from brileta.backends.glfw.window import GlfwWindow

    mock_glfw_window_handle = Mock()
    mock_window = GlfwWindow(mock_glfw_window_handle)
    mock_window.get_size = Mock(return_value=(800, 600))
    mock_window.get_framebuffer_size = Mock(return_value=(800, 600))
    mock_window.flip = Mock()

    ctx = WGPUGraphicsContext(mock_window, _defer_init=True)
    # Verify window is properly stored
    assert ctx.window is mock_window
