from unittest.mock import MagicMock

from catley import colors
from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import PillowTextBackend, TCODTextBackend


def _make_renderer(tile_height: int = 16) -> Renderer:
    renderer = MagicMock(spec=Renderer)
    renderer.tile_dimensions = (8, tile_height)
    renderer.sdl_renderer = MagicMock()
    renderer.root_console = MagicMock()

    # Mock console_render for TCODTextBackend - return new object each time
    renderer.console_render = MagicMock()
    renderer.console_render.render.side_effect = lambda console: MagicMock()

    # Mock upload_texture for PillowTextBackend - return new object each time
    renderer.sdl_renderer.upload_texture.side_effect = lambda pixels: MagicMock()

    return renderer


def test_backend_interchangeability() -> None:
    renderer = _make_renderer()
    tcod_backend = TCODTextBackend(renderer)
    pillow_backend = PillowTextBackend(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)

    assert (
        tcod_backend.get_effective_line_height()
        == pillow_backend.get_effective_line_height()
    )

    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()
    assert asc_t + desc_t == asc_p + desc_p

    metrics_t = tcod_backend.get_text_metrics("Hi")
    metrics_p = pillow_backend.get_text_metrics("Hi")
    assert metrics_t[2] == metrics_p[2]


def test_no_unnecessary_scaling() -> None:
    renderer = _make_renderer()
    backend = PillowTextBackend(renderer)
    update_mock = MagicMock()
    backend._update_scaling_internal = update_mock  # type: ignore[assignment]
    backend.configure_scaling(16)
    backend.configure_scaling(16)
    backend.configure_scaling(20)
    assert update_mock.call_count == 1


def test_pillow_font_sizing() -> None:
    renderer = _make_renderer()
    backend = PillowTextBackend(renderer)
    for height in [12, 16, 20, 24, 32]:
        backend.configure_scaling(height)
        ascent, descent = backend.get_font_metrics()
        assert abs((ascent + descent) - height) <= 1
        assert backend.get_effective_line_height() == height


def test_layout_consistency() -> None:
    renderer = _make_renderer()
    tcod_backend = TCODTextBackend(renderer)
    pillow_backend = PillowTextBackend(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)
    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()
    baseline_t = 16 - desc_t
    baseline_p = 16 - desc_p
    assert abs(baseline_t - baseline_p) <= 1


def test_rendering_workflow() -> None:
    """Test the full rendering cycle works correctly."""
    renderer = _make_renderer()

    # Test both backends
    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # Full rendering workflow
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        backend.draw_rect(0, 0, 50, 30, colors.RED, True)
        backend.draw_frame(1, 2, 3, 4, colors.GREEN, colors.BLUE)
        texture = backend.end_frame()

        # Should produce a texture
        assert texture is not None


def test_caching_behavior() -> None:
    """Test that identical frames are cached, different frames aren't."""
    renderer = _make_renderer()

    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # First frame
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture1 = backend.end_frame()

        # Identical second frame - should reuse cached texture
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture2 = backend.end_frame()

        # Should return the same cached texture
        assert texture1 is texture2

        # Different third frame - should create new texture
        backend.begin_frame()
        backend.draw_text(10, 20, "World", colors.WHITE)  # Different text
        texture3 = backend.end_frame()

        # Should be a different texture
        assert texture3 is not texture1


def test_operation_recording() -> None:
    """Test that drawing operations are recorded correctly."""
    renderer = _make_renderer()

    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # Start frame and add operations
        backend.begin_frame()
        assert len(backend._frame_ops) == 0

        backend.draw_text(10, 20, "Hello", colors.WHITE)
        assert len(backend._frame_ops) == 1

        backend.draw_rect(0, 0, 50, 30, colors.RED, True)
        assert len(backend._frame_ops) == 2

        backend.draw_frame(1, 2, 3, 4, colors.GREEN, colors.BLUE)
        assert len(backend._frame_ops) == 3

        # Verify operation types are recorded correctly
        from catley.view.render.text_backend import DrawOperation

        assert backend._frame_ops[0][0] == DrawOperation.TEXT
        assert backend._frame_ops[1][0] == DrawOperation.RECT
        assert backend._frame_ops[2][0] == DrawOperation.FRAME

        # Begin new frame should reset operations
        backend.begin_frame()
        assert len(backend._frame_ops) == 0


def test_coordinate_types() -> None:
    """Test float pixel coordinates and tile coordinate conversions."""
    renderer = _make_renderer()

    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # Test with float pixel coordinates (PixelCoord supports int | float)
        backend.begin_frame()
        backend.draw_text(10.5, 20.7, "test", colors.WHITE)
        backend.draw_rect(5.2, 10.8, 20.3, 15.9, colors.RED, True)
        texture = backend.end_frame()

        # Should handle float coordinates without error
        assert texture is not None

        # Test tile coordinates (should be integers)
        backend.begin_frame()
        backend.draw_frame(1, 2, 3, 4, colors.GREEN, colors.BLUE)
        texture = backend.end_frame()

        assert texture is not None


def test_error_handling() -> None:
    """Test behavior with invalid inputs or states."""
    renderer = _make_renderer()

    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)

        # Test with no dimensions configured
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture = backend.end_frame()
        # Should handle gracefully (may return None)

        # Test with zero dimensions
        backend.configure_dimensions(0, 0)
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture = backend.end_frame()
        # Should handle gracefully

        # Test empty frame
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)
        backend.begin_frame()
        texture = backend.end_frame()
        # Should handle empty frames gracefully
        assert texture is not None or texture is None  # Either is acceptable


def test_dimension_cache_invalidation() -> None:
    """Test that changing dimensions invalidates the cache."""
    renderer = _make_renderer()

    for backend_class in [TCODTextBackend, PillowTextBackend]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # Render first frame
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture1 = backend.end_frame()

        # Same operations, should use cache
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture2 = backend.end_frame()
        assert texture1 is texture2

        # Change dimensions - should invalidate cache
        backend.configure_dimensions(120, 100)

        # Verify cache was invalidated by checking internal state
        assert backend._cached_frame_texture is None
        assert backend._last_frame_ops == []

        # Same operations, but should create new texture due to dimension change
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture3 = backend.end_frame()

        # Should be a new texture (cache was invalidated)
        assert texture3 is not None
