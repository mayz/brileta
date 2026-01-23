"""Tests for UI utility functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.ui_utils import draw_keycap, wrap_text_by_words


def _make_test_canvas(width: int = 100, height: int = 100) -> PillowImageCanvas:
    """Create a real PillowImageCanvas for pixel-level testing.

    Uses ACTION_PANEL_FONT_SIZE to match actual game rendering configuration.
    """
    renderer = MagicMock(spec=GraphicsContext)
    renderer.tile_dimensions = (20, 20)  # Match actual tileset dimensions
    canvas = PillowImageCanvas(
        renderer,
        font_path=config.UI_FONT_PATH,
        font_size=config.ACTION_PANEL_FONT_SIZE,  # Use 48, matching game config
    )
    canvas.configure_dimensions(width, height)
    return canvas


def _find_color_bounds(
    pixels: np.ndarray, color: tuple[int, int, int]
) -> tuple[int, int, int, int] | None:
    """Find bounding box of pixels matching a color.

    Returns (x_min, x_max, y_min, y_max) or None if no pixels match.
    """
    matches = np.all(pixels[..., :3] == color, axis=2)
    if not matches.any():
        return None
    y_indices, x_indices = np.where(matches)
    return (
        int(x_indices.min()),
        int(x_indices.max()),
        int(y_indices.min()),
        int(y_indices.max()),
    )


def _find_bright_text_bounds(
    pixels: np.ndarray,
    exclude_color: tuple[int, int, int],
    min_brightness: int = 200,
) -> tuple[int, int, int, int] | None:
    """Find bounding box of bright pixels that aren't the background color.

    Used for finding anti-aliased text which may not be pure white.
    Returns (x_min, x_max, y_min, y_max) or None if no pixels match.
    """
    # Find pixels that are bright (any channel >= min_brightness)
    bright = np.any(pixels[..., :3] >= min_brightness, axis=2)
    # Exclude the background color
    not_bg = ~np.all(pixels[..., :3] == exclude_color, axis=2)
    # Also exclude transparent pixels
    has_alpha = pixels.shape[2] == 4
    not_transparent = (
        pixels[..., 3] > 0 if has_alpha else np.ones(pixels.shape[:2], dtype=bool)
    )

    matches = bright & not_bg & not_transparent
    if not matches.any():
        return None
    y_indices, x_indices = np.where(matches)
    return (
        int(x_indices.min()),
        int(x_indices.max()),
        int(y_indices.min()),
        int(y_indices.max()),
    )


def _find_text_only_bounds(
    pixels: np.ndarray,
    min_brightness: int = 200,
) -> tuple[int, int, int, int] | None:
    """Find bounding box of text pixels only (white/near-white).

    Unlike _find_bright_text_bounds, this excludes BOTH the DARK_GREY background
    AND the GREY border by requiring ALL color channels to be bright.
    Returns (x_min, x_max, y_min, y_max) or None if no pixels match.
    """
    # Find pixels where ALL channels are bright (white text only)
    all_bright = np.all(pixels[..., :3] >= min_brightness, axis=2)

    # Exclude transparent pixels
    has_alpha = pixels.shape[2] == 4
    not_transparent = (
        pixels[..., 3] > 0 if has_alpha else np.ones(pixels.shape[:2], dtype=bool)
    )

    matches = all_bright & not_transparent
    if not matches.any():
        return None
    y_indices, x_indices = np.where(matches)
    return (
        int(x_indices.min()),
        int(x_indices.max()),
        int(y_indices.min()),
        int(y_indices.max()),
    )


class MockCanvasForKeycap:
    """Mock canvas for testing draw_keycap with bbox offset support."""

    def __init__(
        self, bbox_offset: tuple[int, int] = (0, 0), char_width: int = 8
    ) -> None:
        """Initialize mock canvas.

        Args:
            bbox_offset: (x_offset, y_offset) to return from get_text_bbox.
            char_width: Width per character for text metrics.
        """
        self._bbox_offset = bbox_offset
        self._char_width = char_width
        self.drawn_texts: list[dict] = []
        self.drawn_rects: list[dict] = []

    def get_font_metrics(self) -> tuple[int, int]:
        """Return mock font metrics (ascent, descent)."""
        return (12, 4)

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        """Return mock text metrics (width, height, line_height)."""
        width = len(text) * self._char_width
        return width, 12, 16

    def get_text_bbox(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int, int]:
        """Return mock text bbox (x_offset, y_offset, width, height)."""
        width = len(text) * self._char_width
        return (self._bbox_offset[0], self._bbox_offset[1], width, 12)

    def draw_text(
        self,
        pixel_x: int,
        pixel_y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Record a text draw call."""
        self.drawn_texts.append(
            {"x": pixel_x, "y": pixel_y, "text": text, "font_size": font_size}
        )

    def draw_rect(
        self,
        pixel_x: int,
        pixel_y: int,
        width: int,
        height: int,
        color: colors.Color,
        fill: bool = False,
    ) -> None:
        """Record a rect draw call."""
        self.drawn_rects.append(
            {"x": pixel_x, "y": pixel_y, "width": width, "height": height, "fill": fill}
        )


class TestDrawKeycap:
    """Tests for draw_keycap function."""

    def test_letter_centered_with_zero_bbox_offset(self) -> None:
        """Letter should be centered when bbox offset is (0, 0)."""
        canvas = MockCanvasForKeycap(bbox_offset=(0, 0), char_width=8)

        draw_keycap(canvas, pixel_x=100, pixel_y=50, key="C")  # type: ignore

        # Verify text was drawn
        assert len(canvas.drawn_texts) == 1
        text_call = canvas.drawn_texts[0]
        assert text_call["text"] == "C"

        # With zero offset, centering is just (keycap_width - text_width) // 2
        # keycap_size = int(16 * 0.85) = 13
        # text_width = 8 (one char)
        # keycap_width = max(13, 8 + 12) = 20
        # text_x = 100 + (20 - 8) // 2 = 106
        assert text_call["x"] == 106

    def test_letter_adjusted_for_positive_bbox_offset(self) -> None:
        """Letter position should be adjusted when bbox has positive offset."""
        # Simulate a font where "C" has bbox starting at (2, 3)
        canvas = MockCanvasForKeycap(bbox_offset=(2, 3), char_width=8)

        draw_keycap(canvas, pixel_x=100, pixel_y=50, key="C")  # type: ignore

        text_call = canvas.drawn_texts[0]

        # With offset (2, 3), the centering should subtract the offset
        # text_x = 100 + (20 - 8) // 2 - 2 = 104
        # text_y needs to account for keycap vertical position
        assert text_call["x"] == 104

        # Verify y is also adjusted
        # keycap_y = pixel_y - 2 = 48
        # keycap_height = 13 (but visual height is 14 due to PIL inclusive bounds)
        # text_height = 12
        # text_y = 48 + (13 + 1 - 12) // 2 - 3 = 48 + 1 - 3 = 46
        assert text_call["y"] == 46

    def test_keycap_returns_consumed_width(self) -> None:
        """draw_keycap should return the width consumed."""
        canvas = MockCanvasForKeycap()

        width = draw_keycap(canvas, pixel_x=100, pixel_y=50, key="C")  # type: ignore

        # keycap_width (20) + 12px padding after keycap
        assert width == 32


class TestWrapTextByWords:
    """Tests for the wrap_text_by_words function."""

    def test_short_text_no_wrap(self) -> None:
        """Text that fits should not be wrapped."""
        result = wrap_text_by_words("Hello", lambda s: len(s) <= 20)
        assert result == ["Hello"]

    def test_wraps_at_word_boundary(self) -> None:
        """Text should wrap at word boundaries, not mid-word."""
        result = wrap_text_by_words("Hello world", lambda s: len(s) <= 6)
        assert result == ["Hello", "world"]

    def test_preserves_word_integrity(self) -> None:
        """Should NOT break words in the middle (e.g., 'reliable' into 'reli')."""
        text = "Basic but reliable protection"
        result = wrap_text_by_words(text, lambda s: len(s) <= 15)
        # "Basic but" fits (9 chars), "reliable" starts new line
        # "reliable" fits (8 chars), "protection" starts new line
        for line in result:
            # No line should contain a partial word like "reli" without full "reliable"
            assert "reli" not in line or "reliable" in line

    def test_multiple_wraps(self) -> None:
        """Long text should wrap into multiple lines."""
        text = "This is a longer text that needs multiple wraps"
        result = wrap_text_by_words(text, lambda s: len(s) <= 15)
        assert len(result) > 1
        # Each line should fit within the constraint
        for line in result:
            assert len(line) <= 15 or len(line.split()) == 1

    def test_empty_text(self) -> None:
        """Empty text should return a list with an empty string."""
        result = wrap_text_by_words("", lambda s: True)
        assert result == [""]

    def test_whitespace_only(self) -> None:
        """Whitespace-only text should return appropriate result."""
        result = wrap_text_by_words("   ", lambda s: True)
        # split() on whitespace returns empty list, so we return the original
        assert result == ["   "]

    def test_single_long_word(self) -> None:
        """A single word that exceeds the limit should still be returned."""
        result = wrap_text_by_words(
            "Supercalifragilisticexpialidocious", lambda s: len(s) <= 10
        )
        # Word is kept intact even though it exceeds limit
        assert result == ["Supercalifragilisticexpialidocious"]

    def test_pixel_based_width_check(self) -> None:
        """Works with pixel-based width checking (simulated)."""
        # Simulate a font where each char is 8 pixels wide
        char_width = 8
        max_pixels = 80  # 10 chars

        def fits(s: str) -> bool:
            return len(s) * char_width <= max_pixels

        result = wrap_text_by_words("Hello world test", fits)
        # "Hello" = 5 chars, "world" = 5 chars, together = 11 chars > 10
        assert result == ["Hello", "world test"]

    def test_exact_fit(self) -> None:
        """Text that exactly fits should not be wrapped."""
        result = wrap_text_by_words("Hello", lambda s: len(s) <= 5)
        assert result == ["Hello"]

    def test_preserves_single_spaces(self) -> None:
        """Single spaces between words should be preserved in output."""
        text = "one two three"
        result = wrap_text_by_words(text, lambda s: len(s) <= 7)
        # "one two" = 7 chars, fits exactly
        assert result[0] == "one two"


class TestWrapTextByWordsEdgeCases:
    """Edge case tests for wrap_text_by_words."""

    def test_always_fits_returns_single_line(self) -> None:
        """If everything fits, return single line."""
        result = wrap_text_by_words("Any text here", lambda s: True)
        assert result == ["Any text here"]

    def test_nothing_fits_returns_words_on_separate_lines(self) -> None:
        """If nothing fits, each word goes on its own line."""
        result = wrap_text_by_words("a b c", lambda s: len(s) <= 1)
        assert result == ["a", "b", "c"]


class TestDrawKeycapPixelLevel:
    """Pixel-level tests that actually render and inspect output."""

    def test_diagnose_keycap_centering(self) -> None:
        """Diagnostic test - prints actual pixel positions for debugging."""
        # Canvas needs to be large enough for the keycap at ACTION_PANEL_FONT_SIZE
        # With font_size=48, line_height=49, keycap_size=41
        # At pixel_y=50, keycap goes from y=48 to y=48+41=89, needs at least 90px height
        canvas = _make_test_canvas(100, 100)

        # Print the values that draw_keycap will use
        ascent, descent = canvas.get_font_metrics()
        line_height = ascent + descent
        keycap_size = int(line_height * 0.85)
        keycap_font_size = max(8, int(keycap_size * 0.65))
        x_off, y_off, text_width, text_height = canvas.get_text_bbox(
            "C", font_size=keycap_font_size
        )
        print(f"\nFont metrics: ascent={ascent}, descent={descent}")
        print(f"line_height={line_height}")
        print(f"keycap_size={keycap_size}, keycap_font_size={keycap_font_size}")
        print(f"get_text_bbox('C'): x_off={x_off}, y_off={y_off}")
        print(f"  width={text_width}, height={text_height}")

        # Use pixel_y with enough room for keycap (needs ~41px above and below)
        pixel_y = 50
        keycap_y = pixel_y - 2  # From draw_keycap
        # Actual visual keycap height is keycap_size + 1 (PIL inclusive bounds)
        text_y_calc = keycap_y + (keycap_size + 1 - text_height) // 2 - y_off
        print(f"\nkeycap_y = {pixel_y} - 2 = {keycap_y}")
        print(f"text_y = {keycap_y} + ({keycap_size}+1 - {text_height})//2 - {y_off}")
        print(f"       = {keycap_y} + {(keycap_size + 1 - text_height) // 2} - {y_off}")
        print(f"       = {text_y_calc}")
        expected_start = text_y_calc + y_off
        print(f"Text pixels should start at: text_y + y_off = {expected_start}")

        canvas.begin_frame()
        draw_keycap(canvas, pixel_x=20, pixel_y=pixel_y, key="C")
        pixels = canvas.end_frame()

        assert pixels is not None

        # Find keycap bounds (DARK_GREY background)
        keycap_bounds = _find_color_bounds(pixels, colors.DARK_GREY)
        assert keycap_bounds is not None, "Keycap background not found"
        kx_min, kx_max, ky_min, ky_max = keycap_bounds
        keycap_center_x = (kx_min + kx_max) / 2
        keycap_center_y = (ky_min + ky_max) / 2
        print(f"\nActual keycap bounds: x=[{kx_min}, {kx_max}], y=[{ky_min}, {ky_max}]")
        print(f"Actual keycap height: {ky_max - ky_min + 1} pixels")
        print(f"Keycap center: x={keycap_center_x}, y={keycap_center_y}")

        # Find text bounds - use text-only function to exclude GREY border
        text_bounds = _find_text_only_bounds(pixels)
        assert text_bounds is not None, "Text pixels not found"
        tx_min, tx_max, ty_min, ty_max = text_bounds
        text_center_x = (tx_min + tx_max) / 2
        text_center_y = (ty_min + ty_max) / 2
        print(f"\nActual text bounds: x=[{tx_min}, {tx_max}], y=[{ty_min}, {ty_max}]")
        print(f"Actual text height: {ty_max - ty_min + 1} pixels")
        print(f"Text center: x={text_center_x}, y={text_center_y}")

        # Calculate offsets
        left_padding = tx_min - kx_min
        right_padding = kx_max - tx_max
        top_padding = ty_min - ky_min
        bottom_padding = ky_max - ty_max

        print(
            f"\nHorizontal: left_padding={left_padding}, right_padding={right_padding}"
        )
        print(f"Horizontal offset from center: {(left_padding - right_padding) / 2} px")
        print(f"Vertical: top_padding={top_padding}, bottom_padding={bottom_padding}")
        print(f"Vertical offset from center: {(top_padding - bottom_padding) / 2} px")

    def test_letter_centered_horizontally_in_keycap(self) -> None:
        """Letter should be horizontally centered within keycap border.

        For odd margins, prefer left margin <= right margin (letter slightly
        left-aligned rather than right-aligned).
        """
        # Test with multiple characters to catch character-specific issues
        for key in ["C", "?", "A", "H"]:
            canvas = _make_test_canvas(100, 100)
            canvas.begin_frame()
            draw_keycap(canvas, pixel_x=20, pixel_y=50, key=key)
            pixels = canvas.end_frame()
            assert pixels is not None

            # Find border bounds (GREY) - this is what the user sees as keycap edge
            border_bounds = _find_color_bounds(pixels, colors.GREY)
            assert border_bounds is not None, f"Border not found for key '{key}'"
            border_left, border_right, _border_top, _border_bottom = border_bounds

            # Find text bounds - look for bright pixels (anti-aliased text)
            text_bounds = _find_bright_text_bounds(pixels, colors.DARK_GREY)
            assert text_bounds is not None, f"Text pixels not found for key '{key}'"
            text_left, text_right, _ty_min, _ty_max = text_bounds

            # Calculate margins from border to text
            # Border is 1px wide, so inner edge is border_left + 1 and border_right - 1
            left_margin = text_left - (border_left + 1)
            right_margin = (border_right - 1) - text_right

            # For perfect centering, margins should be equal.
            # For odd total margin, prefer left_margin <= right_margin.
            assert left_margin <= right_margin + 1, (
                f"Key '{key}': letter too far right. "
                f"left_margin={left_margin}, right_margin={right_margin}"
            )
            assert right_margin <= left_margin + 1, (
                f"Key '{key}': letter too far left. "
                f"left_margin={left_margin}, right_margin={right_margin}"
            )

    def test_letter_centered_vertically_in_keycap(self) -> None:
        """Letter pixels should be vertically centered within keycap box."""
        # Canvas needs enough room for keycap at ACTION_PANEL_FONT_SIZE
        canvas = _make_test_canvas(100, 100)

        canvas.begin_frame()
        draw_keycap(canvas, pixel_x=20, pixel_y=50, key="C")
        pixels = canvas.end_frame()

        assert pixels is not None

        keycap_bounds = _find_color_bounds(pixels, colors.DARK_GREY)
        assert keycap_bounds is not None, "Keycap background not found"
        _, _, ky_min, ky_max = keycap_bounds
        keycap_center_y = (ky_min + ky_max) / 2

        # Find text bounds - look for bright pixels (anti-aliased text)
        text_bounds = _find_bright_text_bounds(pixels, colors.DARK_GREY)
        assert text_bounds is not None, "Text pixels not found"
        _, _, ty_min, ty_max = text_bounds
        text_center_y = (ty_min + ty_max) / 2

        # Verify vertical centering
        assert abs(text_center_y - keycap_center_y) <= 1, (
            f"Letter not vertically centered: text center={text_center_y}, "
            f"keycap center={keycap_center_y}"
        )
