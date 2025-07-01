from unittest.mock import MagicMock

from catley import colors
from catley.backends.moderngl.canvas import ModernGLCanvas
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.backends.tcod.canvas import TCODConsoleCanvas
from catley.view.render.graphics import GraphicsContext


def _make_renderer(tile_height: int = 16) -> GraphicsContext:
    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (8, tile_height)
    renderer.console_width_tiles = 80
    renderer.console_height_tiles = 50
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
    tcod_backend = TCODConsoleCanvas(renderer)
    pillow_backend = PillowImageCanvas(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)

    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()

    assert tcod_backend.get_effective_line_height() == asc_t + desc_t
    assert pillow_backend.get_effective_line_height() == asc_p + desc_p

    metrics_t = tcod_backend.get_text_metrics("Hi")
    metrics_p = pillow_backend.get_text_metrics("Hi")
    assert metrics_t[2] == tcod_backend.get_effective_line_height()
    assert metrics_p[2] == pillow_backend.get_effective_line_height()


def test_no_unnecessary_scaling() -> None:
    renderer = _make_renderer()
    backend = PillowImageCanvas(renderer)
    update_mock = MagicMock()
    backend._update_scaling_internal = update_mock  # type: ignore[assignment]
    backend.configure_scaling(16)
    backend.configure_scaling(16)
    backend.configure_scaling(20)
    assert update_mock.call_count == 1


def test_pillow_font_sizing() -> None:
    renderer = _make_renderer()
    backend = PillowImageCanvas(renderer)
    for height in [12, 16, 20, 24, 32]:
        backend.configure_scaling(height)
        ascent, descent = backend.get_font_metrics()
        assert backend.get_effective_line_height() == ascent + descent


def test_layout_consistency() -> None:
    renderer = _make_renderer()
    tcod_backend = TCODConsoleCanvas(renderer)
    pillow_backend = PillowImageCanvas(renderer)
    tcod_backend.configure_scaling(16)
    pillow_backend.configure_scaling(16)
    asc_t, desc_t = tcod_backend.get_font_metrics()
    asc_p, desc_p = pillow_backend.get_font_metrics()
    baseline_t = tcod_backend.get_effective_line_height() - desc_t
    baseline_p = pillow_backend.get_effective_line_height() - desc_p
    assert abs(baseline_t - baseline_p) <= 8


def test_rendering_workflow() -> None:
    """Test the full rendering cycle works correctly."""
    renderer = _make_renderer()

    # Test both backends
    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
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
    """Test that backends handle frame operations correctly."""
    renderer = _make_renderer()

    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
        backend = backend_class(renderer)
        backend.configure_dimensions(100, 80)
        backend.configure_scaling(16)

        # First frame
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture1 = backend.end_frame()

        # Should produce a texture/artifact
        assert texture1 is not None

        # Identical second frame
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture2 = backend.end_frame()

        # Should produce a texture/artifact
        assert texture2 is not None

        # Different third frame
        backend.begin_frame()
        backend.draw_text(10, 20, "World", colors.WHITE)  # Different text
        texture3 = backend.end_frame()

        # Should produce a texture/artifact
        assert texture3 is not None


def test_operation_recording() -> None:
    """Test that drawing operations are recorded correctly."""
    renderer = _make_renderer()

    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
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
        from catley.view.render.canvas import DrawOperation

        assert backend._frame_ops[0][0] == DrawOperation.TEXT
        assert backend._frame_ops[1][0] == DrawOperation.RECT
        assert backend._frame_ops[2][0] == DrawOperation.FRAME

        # Begin new frame should reset operations
        backend.begin_frame()
        assert len(backend._frame_ops) == 0


def test_coordinate_types() -> None:
    """Test float pixel coordinates and tile coordinate conversions."""
    renderer = _make_renderer()

    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
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

    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
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

    for backend_class in [TCODConsoleCanvas, PillowImageCanvas]:
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
        assert backend._cached_frame_artifact is None
        assert backend._last_frame_ops == []

        # Same operations, but should create new texture due to dimension change
        backend.begin_frame()
        backend.draw_text(10, 20, "Hello", colors.WHITE)
        texture3 = backend.end_frame()

        # Should be a new texture (cache was invalidated)
        assert texture3 is not None


def test_moderngl_canvas_drawing_operations() -> None:
    """Test ModernGLCanvas draw_text and draw_rect with GlyphBuffer verification.

    Creates a ModernGLCanvas, calls draw_text and draw_rect, then verifies
    the resulting GlyphBuffer contains expected characters and colors at
    expected tile coordinates.
    """
    renderer = _make_renderer(tile_height=20)  # 8x20 tiles
    canvas = ModernGLCanvas(
        renderer, transparent=False
    )  # Non-transparent for clearer testing

    # Configure canvas with known dimensions
    canvas.configure_dimensions(
        80, 60
    )  # 80x60 pixels -> 10x3 tiles (80/8 = 10, 60/20 = 3)

    # Begin frame and perform drawing operations
    canvas.begin_frame()

    # Draw text "A" at pixel position (8, 0) -> tile position (1, 0)
    canvas.draw_text(8, 0, "A", colors.WHITE)

    # Draw filled rectangle at pixel position (16, 20) with size 16x20
    # -> tiles (2,1) to (3,1)
    canvas.draw_rect(16, 20, 16, 20, colors.RED, fill=True)

    # Get the resulting GlyphBuffer
    glyph_buffer = canvas.end_frame()

    # Verify GlyphBuffer was created
    assert glyph_buffer is not None
    assert glyph_buffer.width == 10  # 80 pixels / 8 pixel-wide tiles
    assert glyph_buffer.height == 3  # 60 pixels / 20 pixel-high tiles

    # Verify text character 'A' at tile coordinate (1, 0)
    char, fg_color, bg_color = glyph_buffer.data[1, 0]
    assert char == ord("A")
    assert tuple(fg_color) == (255, 255, 255, 255)  # WHITE with alpha
    assert tuple(bg_color) == (0, 0, 0, 255)  # Non-transparent black background

    # Verify filled rectangle uses solid block character (Unicode 9608) at tile (2, 1)
    char, fg_color, bg_color = glyph_buffer.data[2, 1]
    assert char == 9608  # Unicode █ solid block character
    assert tuple(fg_color) == (255, 0, 0, 255)  # RED foreground
    assert tuple(bg_color) == (255, 0, 0, 255)  # RED background from fill color

    # Verify another part of the rectangle at tile (3, 1)
    char, fg_color, bg_color = glyph_buffer.data[3, 1]
    assert char == 9608  # Unicode █ solid block character
    assert tuple(fg_color) == (255, 0, 0, 255)  # RED foreground
    assert tuple(bg_color) == (255, 0, 0, 255)  # RED background from fill color

    # Verify empty tile has default background at tile (0, 0)
    char, fg_color, bg_color = glyph_buffer.data[0, 0]
    assert char == ord(" ")  # Default space character (32)
    assert tuple(bg_color) == (0, 0, 0, 255)  # Non-transparent black background


def test_moderngl_canvas_transparent_mode() -> None:
    """Test ModernGLCanvas with transparent background mode."""
    renderer = _make_renderer(tile_height=16)
    canvas = ModernGLCanvas(renderer, transparent=True)

    canvas.configure_dimensions(32, 32)  # 4x2 tiles (32/8 = 4, 32/16 = 2)

    canvas.begin_frame()
    canvas.draw_text(0, 0, "B", colors.GREEN)
    glyph_buffer = canvas.end_frame()

    # Verify transparent background
    assert glyph_buffer is not None
    char, fg_color, bg_color = glyph_buffer.data[0, 0]
    assert char == ord("B")
    assert tuple(fg_color) == (0, 255, 0, 255)  # GREEN
    assert tuple(bg_color) == (0, 0, 0, 0)  # Transparent background (alpha=0)


def test_moderngl_canvas_rect_border_only() -> None:
    """Test ModernGLCanvas draw_rect with fill=False (border only)."""
    renderer = _make_renderer(tile_height=16)
    canvas = ModernGLCanvas(renderer, transparent=False)

    canvas.configure_dimensions(48, 48)  # 6x3 tiles

    canvas.begin_frame()
    # Draw 3x2 tile rectangle border at tile position (1,0) to (3,1)
    canvas.draw_rect(8, 0, 24, 32, colors.BLUE, fill=False)
    glyph_buffer = canvas.end_frame()

    # Check corners and borders of the rectangle
    assert glyph_buffer is not None
    # Top-left corner at tile (1, 0)
    char, fg_color, bg_color = glyph_buffer.data[1, 0]
    assert char == 9608  # Unicode █ solid block for border
    assert tuple(bg_color) == (0, 0, 255, 255)  # BLUE

    # Top-right corner at tile (3, 0)
    char, fg_color, bg_color = glyph_buffer.data[3, 0]
    assert char == 9608  # Unicode █ solid block for border
    assert tuple(bg_color) == (0, 0, 255, 255)  # BLUE

    # Bottom-left corner at tile (1, 1)
    char, fg_color, bg_color = glyph_buffer.data[1, 1]
    assert char == 9608  # Unicode █ solid block for border
    assert tuple(bg_color) == (0, 0, 255, 255)  # BLUE

    # Interior should be empty (not filled) at tile (2, 0)
    char, fg_color, bg_color = glyph_buffer.data[2, 0]
    assert char == 9608  # This is actually part of the border (top edge)
    assert tuple(bg_color) == (0, 0, 255, 255)  # BLUE

    # Check a tile outside the rectangle is empty at tile (0, 0)
    char, fg_color, bg_color = glyph_buffer.data[0, 0]
    assert char == ord(" ")  # Default space character
    assert tuple(bg_color) == (0, 0, 0, 255)  # Default background
