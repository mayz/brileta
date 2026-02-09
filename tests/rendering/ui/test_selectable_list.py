"""Tests for the SelectableListRenderer component."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from brileta import colors, config
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.view.render.graphics import GraphicsContext
from brileta.view.ui.selectable_list import (
    LayoutMode,
    SelectableListRenderer,
    SelectableRow,
)


def _make_test_canvas(width: int = 300, height: int = 100) -> PillowImageCanvas:
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


class MockCanvas:
    """Mock canvas for testing SelectableListRenderer without actual rendering."""

    def __init__(self) -> None:
        self.drawn_rects: list[dict] = []
        self.drawn_texts: list[dict] = []
        self._font_metrics = (12, 4)  # ascent, descent
        self._char_widths: dict[str, int] = {}  # Cache for text widths

    def get_font_metrics(self) -> tuple[int, int]:
        """Return mock font metrics (ascent, descent)."""
        return self._font_metrics

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        """Return mock text metrics (width, height, baseline).

        Width is calculated as 8 pixels per character.
        """
        width = len(text) * 8
        return width, 12, 10

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
            {
                "x": pixel_x,
                "y": pixel_y,
                "width": width,
                "height": height,
                "color": color,
                "fill": fill,
            }
        )

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
            {
                "x": pixel_x,
                "y": pixel_y,
                "text": text,
                "color": color,
            }
        )


class TestSelectableRow:
    """Tests for SelectableRow dataclass."""

    def test_default_values(self) -> None:
        """Verify default values for SelectableRow."""
        row = SelectableRow(text="Test")

        assert row.text == "Test"
        assert row.key is None
        assert row.enabled is True
        assert row.color == colors.WHITE
        assert row.data is None
        assert row.prefix_segments is None
        assert row.suffix is None
        assert row.suffix_color is None
        assert row.force_color is False

    def test_custom_values(self) -> None:
        """Verify custom values are set correctly."""
        data_obj = {"action": "test"}
        row = SelectableRow(
            text="Custom",
            key="A",
            enabled=False,
            color=colors.YELLOW,
            data=data_obj,
            prefix_segments=[("* ", colors.RED)],
            suffix=" (50%)",
            suffix_color=colors.GREY,
            force_color=True,
        )

        assert row.text == "Custom"
        assert row.key == "A"
        assert row.enabled is False
        assert row.color == colors.YELLOW
        assert row.data is data_obj
        assert row.prefix_segments == [("* ", colors.RED)]
        assert row.suffix == " (50%)"
        assert row.suffix_color == colors.GREY
        assert row.force_color is True


class TestSelectableListRendererBasics:
    """Basic functionality tests for SelectableListRenderer."""

    def test_init_keycap_mode(self) -> None:
        """Verify KEYCAP mode initialization."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        assert renderer.mode == LayoutMode.KEYCAP
        assert renderer.rows == []
        assert renderer.hovered_index is None
        assert renderer._hit_areas == []

    def test_init_inline_mode(self) -> None:
        """Verify INLINE mode initialization."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        assert renderer.mode == LayoutMode.INLINE

    def test_default_mode_is_keycap(self) -> None:
        """Verify default mode is KEYCAP."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore

        assert renderer.mode == LayoutMode.KEYCAP

    def test_reset_clears_all_state(self) -> None:
        """Verify reset() clears all state including hover."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore

        renderer.rows = [SelectableRow(text="Test")]
        renderer.hovered_index = 0
        renderer._hit_areas = [((0, 0, 100, 20), 0)]

        renderer.reset()

        assert renderer.rows == []
        assert renderer.hovered_index is None
        assert renderer._hit_areas == []

    def test_render_empty_list_returns_start_position(self) -> None:
        """Rendering empty list should return the start y position."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore

        y_result = renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        assert y_result == 100


class TestKeycapModeRendering:
    """Tests for KEYCAP mode rendering."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_render_creates_hit_areas(self, mock_draw_keycap: MagicMock) -> None:
        """Rendering should create hit areas for each row."""
        mock_draw_keycap.return_value = 40  # Keycap width

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        assert len(renderer._hit_areas) == 2
        # Each hit area is a tuple of (rect, row_index)
        assert renderer._hit_areas[0][1] == 0
        assert renderer._hit_areas[1][1] == 1

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_hover_draws_background(self, mock_draw_keycap: MagicMock) -> None:
        """Hovering over a row should draw a dark grey background."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Action 1", key="A")]
        renderer.hovered_index = 0

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Should have drawn a hover background rect
        hover_rects = [r for r in canvas.drawn_rects if r["fill"]]
        assert len(hover_rects) == 1
        assert hover_rects[0]["color"] == colors.DARK_GREY

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_disabled_row_no_hover_background(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Disabled rows should not show hover background."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Disabled", key="A", enabled=False)]
        renderer.hovered_index = 0

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Should not have drawn a hover background
        hover_rects = [r for r in canvas.drawn_rects if r["fill"]]
        assert len(hover_rects) == 0

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_render_advances_y_position(self, mock_draw_keycap: MagicMock) -> None:
        """Rendering should advance y position based on row count and gaps."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]

        y_result = renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
            # Default gap is line_height // 3 = 5
        )

        # Each row takes line_height + default_gap = 16 + 5 = 21
        # Two rows = 42 pixels advanced
        assert y_result > 100

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_render_with_zero_row_gap(self, mock_draw_keycap: MagicMock) -> None:
        """row_gap=0 should produce tight spacing with no extra gap."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]

        y_result = renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
            row_gap=0,  # Explicit zero gap
        )

        # Each row takes line_height only = 16
        # Two rows = 32 pixels advanced
        # y_result = 100 + 32 = 132
        assert y_result == 132


class TestInlineModeRendering:
    """Tests for INLINE mode rendering."""

    def test_inline_mode_draws_key_as_text(self) -> None:
        """INLINE mode should render key as '(key) ' text, not keycap."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore
        renderer.rows = [SelectableRow(text="Item", key="A")]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Should find "(A) " in drawn texts
        key_texts = [t for t in canvas.drawn_texts if "(A)" in t["text"]]
        assert len(key_texts) == 1

    def test_inline_mode_draws_prefix_segments(self) -> None:
        """INLINE mode should render prefix segments before the key."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore
        renderer.rows = [
            SelectableRow(
                text="Sword",
                key="1",
                prefix_segments=[("[1] ", colors.YELLOW)],
            )
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Should find "[1] " in drawn texts
        prefix_texts = [t for t in canvas.drawn_texts if "[1]" in t["text"]]
        assert len(prefix_texts) == 1
        assert prefix_texts[0]["color"] == colors.YELLOW

    def test_inline_mode_draws_suffix(self) -> None:
        """INLINE mode should render suffix after main text."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore
        renderer.rows = [
            SelectableRow(
                text="Attack",
                suffix=" (75%)",
                suffix_color=colors.GREY,
            )
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Should find suffix in drawn texts
        suffix_texts = [t for t in canvas.drawn_texts if "(75%)" in t["text"]]
        assert len(suffix_texts) == 1
        assert suffix_texts[0]["color"] == colors.GREY


class TestColorHandling:
    """Tests for color handling and hover brightening."""

    def test_brighten_color_on_hover(self) -> None:
        """Hovering should brighten the color by 40 RGB."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        original = (100, 100, 100)
        brightened = renderer._brighten_if_hovered(original, is_hovered=True)

        assert brightened == (140, 140, 140)

    def test_brighten_color_caps_at_255(self) -> None:
        """Brightening should cap at 255."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        original = (230, 240, 250)
        brightened = renderer._brighten_if_hovered(original, is_hovered=True)

        assert brightened == (255, 255, 255)

    def test_no_brighten_when_not_hovered(self) -> None:
        """Color should not change when not hovered."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        original = (100, 100, 100)
        result = renderer._brighten_if_hovered(original, is_hovered=False)

        assert result == original

    def test_disabled_row_uses_grey(self) -> None:
        """Disabled rows should use grey color."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        row = SelectableRow(text="Disabled", enabled=False, color=colors.WHITE)
        color = renderer._get_effective_color(row, is_hovered=False)

        assert color == colors.GREY

    def test_disabled_row_force_color_keeps_original(self) -> None:
        """Disabled rows with force_color should keep original color."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        row = SelectableRow(
            text="Condition", enabled=False, color=colors.RED, force_color=True
        )
        color = renderer._get_effective_color(row, is_hovered=False)

        assert color == colors.RED


class TestHitAreaDetection:
    """Tests for hit area and mouse interaction."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_update_hover_from_pixel_sets_index(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """update_hover_from_pixel should set hovered_index when over a row."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]

        # Render to create hit areas
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Hover over first row's hit area
        rect = renderer._hit_areas[0][0]
        mid_y = (rect[1] + rect[3]) // 2
        mid_x = (rect[0] + rect[2]) // 2

        changed = renderer.update_hover_from_pixel(mid_x, mid_y)

        assert changed is True
        assert renderer.hovered_index == 0

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_update_hover_returns_false_when_unchanged(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """update_hover_from_pixel should return False when hover doesn't change."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Action", key="A")]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Hover over same row twice
        rect = renderer._hit_areas[0][0]
        mid_y = (rect[1] + rect[3]) // 2
        mid_x = (rect[0] + rect[2]) // 2

        renderer.update_hover_from_pixel(mid_x, mid_y)  # First call
        changed = renderer.update_hover_from_pixel(mid_x, mid_y)  # Second call

        assert changed is False

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_update_hover_disabled_row_not_hovered(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Disabled rows should not be hovered."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Disabled", key="A", enabled=False)]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Try to hover over disabled row
        rect = renderer._hit_areas[0][0]
        mid_y = (rect[1] + rect[3]) // 2
        mid_x = (rect[0] + rect[2]) // 2

        renderer.update_hover_from_pixel(mid_x, mid_y)

        assert renderer.hovered_index is None

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_get_row_at_pixel_returns_row(self, mock_draw_keycap: MagicMock) -> None:
        """get_row_at_pixel should return the row at the given position."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        row1 = SelectableRow(text="Action 1", key="A", data="data1")
        row2 = SelectableRow(text="Action 2", key="B", data="data2")
        renderer.rows = [row1, row2]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Get row at first hit area
        rect = renderer._hit_areas[0][0]
        mid_y = (rect[1] + rect[3]) // 2
        mid_x = (rect[0] + rect[2]) // 2

        result = renderer.get_row_at_pixel(mid_x, mid_y)

        assert result is row1

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_get_row_at_pixel_returns_none_for_disabled(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """get_row_at_pixel should return None for disabled rows."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Disabled", key="A", enabled=False)]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        rect = renderer._hit_areas[0][0]
        mid_y = (rect[1] + rect[3]) // 2
        mid_x = (rect[0] + rect[2]) // 2

        result = renderer.get_row_at_pixel(mid_x, mid_y)

        assert result is None

    def test_get_row_at_pixel_returns_none_outside_areas(self) -> None:
        """get_row_at_pixel should return None when outside all hit areas."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Action", key="A")]

        # Don't render, so no hit areas
        result = renderer.get_row_at_pixel(100, 100)

        assert result is None


class TestHotkeyLookup:
    """Tests for hotkey lookup functionality."""

    def test_get_row_by_hotkey_finds_row(self) -> None:
        """get_row_by_hotkey should find a row by its key."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore
        row = SelectableRow(text="Action", key="A", data="action_data")
        renderer.rows = [row]

        result = renderer.get_row_by_hotkey("A")

        assert result is row

    def test_get_row_by_hotkey_case_insensitive(self) -> None:
        """Hotkey lookup should be case-insensitive."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore
        row = SelectableRow(text="Action", key="A", data="action_data")
        renderer.rows = [row]

        result_lower = renderer.get_row_by_hotkey("a")
        result_upper = renderer.get_row_by_hotkey("A")

        assert result_lower is row
        assert result_upper is row

    def test_get_row_by_hotkey_returns_none_for_disabled(self) -> None:
        """Hotkey lookup should not return disabled rows."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore
        renderer.rows = [SelectableRow(text="Disabled", key="A", enabled=False)]

        result = renderer.get_row_by_hotkey("A")

        assert result is None

    def test_get_row_by_hotkey_returns_none_for_missing(self) -> None:
        """Hotkey lookup should return None for non-existent keys."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore
        renderer.rows = [SelectableRow(text="Action", key="A")]

        result = renderer.get_row_by_hotkey("Z")

        assert result is None

    def test_get_row_by_hotkey_with_no_key_rows(self) -> None:
        """Hotkey lookup should handle rows without keys."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas)  # type: ignore
        renderer.rows = [SelectableRow(text="No Key")]

        result = renderer.get_row_by_hotkey("A")

        assert result is None


class TestHitAreaAlignment:
    """Tests for hit area alignment with visual text.

    These tests verify that hit areas properly align with their visual text rows,
    catching bugs where hit areas overlap or don't cover the correct y-range.
    """

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_hit_areas_are_contiguous_keycap_mode(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Hit areas should be contiguous - each row's start equals the previous end."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Talk", key="T"),
            SelectableRow(text="Attack", key="A"),
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        assert len(renderer._hit_areas) == 2

        # First row's hit area should end exactly where the second begins
        rect0 = renderer._hit_areas[0][0]
        rect1 = renderer._hit_areas[1][0]

        assert rect0[3] == rect1[1], (
            f"Hit areas should be contiguous: row 0 ends at {rect0[3]} "
            f"but row 1 starts at {rect1[1]}"
        )

    def test_hit_areas_are_contiguous_inline_mode(self) -> None:
        """Hit areas should be contiguous in inline mode too."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Talk", key="T"),
            SelectableRow(text="Attack", key="A"),
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        assert len(renderer._hit_areas) == 2

        rect0 = renderer._hit_areas[0][0]
        rect1 = renderer._hit_areas[1][0]

        assert rect0[3] == rect1[1], (
            f"Hit areas should be contiguous: row 0 ends at {rect0[3]} "
            f"but row 1 starts at {rect1[1]}"
        )

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_hit_area_height_equals_line_height_plus_gap(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Each hit area height should equal line_height + gap."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Talk", key="T"),
            SelectableRow(text="Attack", key="A"),
        ]

        line_height = 16
        row_gap = 4

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=line_height,
            ascent=12,
            row_gap=row_gap,
        )

        # Each hit area should have height = line_height + gap
        expected_height = line_height + row_gap
        for rect, idx in renderer._hit_areas:
            actual_height = rect[3] - rect[1]
            assert actual_height == expected_height, (
                f"Row {idx} hit area height is {actual_height}, "
                f"expected {expected_height}"
            )

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_hovering_row_1_only_highlights_row_1(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Hovering over the visual area of row 1 should only highlight row 1.

        This test catches the bug where hit area calculations were off, causing
        hovering row 1's text to actually trigger row 0's hit area.
        """
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(text="Talk", key="T"),
            SelectableRow(text="Attack", key="A"),
        ]

        line_height = 16
        ascent = 12
        y_start = 100

        renderer.render(
            x_start=10,
            y_start=y_start,
            max_width=200,
            line_height=line_height,
            ascent=ascent,
            row_gap=0,  # Zero gap makes the calculation clearer
        )

        # Row 1's text is drawn at y_start + line_height - ascent
        # (since y advances by line_height per row)
        # Visual row 1 text baseline is at y_start + line_height = 116
        # Visual row 1 text top is at y_start + line_height - ascent = 104
        row1_visual_y = y_start + line_height - ascent + 2  # +2 to be inside the row

        x_mid = 100  # Somewhere in the middle horizontally

        # Update hover from this position
        renderer.update_hover_from_pixel(x_mid, row1_visual_y)

        # Should be hovering row 1, not row 0
        assert renderer.hovered_index == 1, (
            f"Hovering at y={row1_visual_y} should highlight row 1, "
            f"but got row {renderer.hovered_index}"
        )


class TestExecuteCallback:
    """Tests for SelectableRow execute callback functionality."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_execute_at_pixel_calls_callback(self, mock_draw_keycap: MagicMock) -> None:
        """execute_at_pixel should call the row's execute callback."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        callback_called = []

        def callback() -> None:
            callback_called.append(True)

        renderer.rows = [
            SelectableRow(text="Action", key="A", execute=callback),
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Execute at pixel within the hit area
        rect = renderer._hit_areas[0][0]
        mid_x = (rect[0] + rect[2]) // 2
        mid_y = (rect[1] + rect[3]) // 2

        result = renderer.execute_at_pixel(mid_x, mid_y)

        assert result is True
        assert len(callback_called) == 1

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_execute_at_pixel_returns_false_without_callback(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """execute_at_pixel should return False if row has no callback."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        renderer.rows = [
            SelectableRow(text="Action", key="A"),  # No execute callback
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        rect = renderer._hit_areas[0][0]
        mid_x = (rect[0] + rect[2]) // 2
        mid_y = (rect[1] + rect[3]) // 2

        result = renderer.execute_at_pixel(mid_x, mid_y)

        assert result is False

    def test_execute_at_pixel_returns_false_outside_hit_areas(self) -> None:
        """execute_at_pixel should return False outside all hit areas."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        renderer.rows = [SelectableRow(text="Action", key="A", execute=lambda: None)]

        # No hit areas rendered
        result = renderer.execute_at_pixel(100, 100)

        assert result is False


class TestKeycapVerticalCentering:
    """Tests for keycap vertical centering within hit areas."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_keycap_centered_in_hit_area_with_gap(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Keycap should be vertically centered within the hit area including gap."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Action", key="A")]

        line_height = 16
        ascent = 12
        row_gap = 8

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=line_height,
            ascent=ascent,
            row_gap=row_gap,
        )

        # Verify draw_keycap was called
        assert mock_draw_keycap.called

        # Get the pixel_y argument passed to draw_keycap
        call_args = mock_draw_keycap.call_args
        keycap_pixel_y = call_args.kwargs["pixel_y"]

        # Expected calculation:
        # hit_y_start = y_start - ascent = 100 - 12 = 88
        # hit_area_height = line_height + gap = 16 + 8 = 24
        # keycap_height = int(line_height * 0.85) = 13
        # centered_keycap_y = hit_y_start + (hit_area_height - keycap_height) // 2
        #                   = 88 + (24 - 13) // 2 = 88 + 5 = 93
        # keycap_pixel_y = centered_keycap_y + 2 = 95 (because draw_keycap does -2)

        hit_y_start = 100 - ascent  # 88
        hit_area_height = line_height + row_gap  # 24
        keycap_height = int(line_height * 0.85)  # 13
        expected_centered_y = hit_y_start + (hit_area_height - keycap_height) // 2
        expected_pixel_y = expected_centered_y + 2  # compensate for draw_keycap's -2

        assert keycap_pixel_y == expected_pixel_y

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_keycap_centered_with_zero_gap(self, mock_draw_keycap: MagicMock) -> None:
        """Keycap should still be properly positioned with zero gap."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Action", key="A")]

        line_height = 16
        ascent = 12
        row_gap = 0

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=line_height,
            ascent=ascent,
            row_gap=row_gap,
        )

        call_args = mock_draw_keycap.call_args
        keycap_pixel_y = call_args.kwargs["pixel_y"]

        # With zero gap, hit_area_height = line_height = 16
        # keycap_height = 13
        # centered = (16 - 13) // 2 = 1
        # centered_y = 88 + 1 = 89
        # pixel_y = 89 + 2 = 91
        expected_pixel_y = (
            (100 - ascent) + (line_height - int(line_height * 0.85)) // 2 + 2
        )

        assert keycap_pixel_y == expected_pixel_y


class TestLabelVerticalCentering:
    """Tests for label text vertical centering within hit areas."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_label_vertically_centered_in_hit_area(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Label text should be vertically centered to match keycap centering."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Attack", key="A")]

        line_height = 16
        ascent = 12
        row_gap = 8

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=line_height,
            ascent=ascent,
            row_gap=row_gap,
        )

        # Find the drawn text for "Attack"
        label_texts = [t for t in canvas.drawn_texts if t["text"] == "Attack"]
        assert len(label_texts) == 1
        label_y = label_texts[0]["y"]

        # Expected calculation (mathematical centering for containment):
        # hit_y_start = y_start - ascent = 100 - 12 = 88
        # hit_area_height = line_height + gap = 16 + 8 = 24
        # label_y = hit_y_start + (hit_area_height - line_height) // 2
        #         = 88 + (24 - 16) // 2 = 88 + 4 = 92
        hit_y_start = 100 - ascent  # 88
        hit_area_height = line_height + row_gap  # 24
        expected_label_y = hit_y_start + (hit_area_height - line_height) // 2

        assert label_y == expected_label_y

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_label_at_hit_area_top_with_zero_gap(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """With zero gap, label should be at hit area top (centered = top)."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [SelectableRow(text="Attack", key="A")]

        line_height = 16
        ascent = 12
        row_gap = 0

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=line_height,
            ascent=ascent,
            row_gap=row_gap,
        )

        label_texts = [t for t in canvas.drawn_texts if t["text"] == "Attack"]
        assert len(label_texts) == 1
        label_y = label_texts[0]["y"]

        # With zero gap, hit_area_height = line_height
        # Mathematical centering: (line_height - line_height) // 2 = 0
        # So label_y = hit_y_start = 88
        hit_y_start = 100 - ascent  # 88
        assert label_y == hit_y_start

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_prefix_suffix_same_y_as_label(self, mock_draw_keycap: MagicMock) -> None:
        """Prefix segments and suffix should be at same y as label text."""
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore
        renderer.rows = [
            SelectableRow(
                text="Attack",
                key="A",
                prefix_segments=[("* ", colors.RED)],
                suffix=" (50%)",
                suffix_color=colors.GREY,
            )
        ]

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
            row_gap=8,
        )

        # All text elements should have the same y coordinate
        y_values = [t["y"] for t in canvas.drawn_texts]
        assert len(set(y_values)) == 1, f"All text should have same y, got {y_values}"


class TestRenderResetsHoverState:
    """Tests verifying that render() resets stale hover state."""

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_render_clears_out_of_bounds_hovered_index(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """render() should clear hovered_index when it's out of bounds for new rows.

        This prevents stale hover state from a previous (longer) render from
        incorrectly affecting a new (shorter) render.
        """
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        # First render with some rows
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Simulate hovering over the second row (index 1)
        rect = renderer._hit_areas[1][0]
        mid_x = (rect[0] + rect[2]) // 2
        mid_y = (rect[1] + rect[3]) // 2
        renderer.update_hover_from_pixel(mid_x, mid_y)
        assert renderer.hovered_index == 1

        # Now render with fewer rows - index 1 is now out of bounds
        renderer.rows = [
            SelectableRow(text="New Action", key="X"),
        ]
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # hovered_index should be cleared because it's out of bounds
        assert renderer.hovered_index is None

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_render_preserves_in_bounds_hovered_index(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """render() should preserve hovered_index when still in bounds.

        When re-rendering the same list, the hover state should persist.
        """
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        # Render with rows
        renderer.rows = [
            SelectableRow(text="Action 1", key="A"),
            SelectableRow(text="Action 2", key="B"),
        ]
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Hover over the first row (index 0)
        rect = renderer._hit_areas[0][0]
        mid_x = (rect[0] + rect[2]) // 2
        mid_y = (rect[1] + rect[3]) // 2
        renderer.update_hover_from_pixel(mid_x, mid_y)
        assert renderer.hovered_index == 0

        # Re-render with same number of rows - index 0 is still valid
        renderer.rows = [
            SelectableRow(text="New Action 1", key="X"),
            SelectableRow(text="New Action 2", key="Y"),
        ]
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # hovered_index should be preserved because it's still in bounds
        assert renderer.hovered_index == 0

    @patch("brileta.view.ui.selectable_list.draw_keycap")
    def test_stale_hover_does_not_affect_new_render(
        self, mock_draw_keycap: MagicMock
    ) -> None:
        """Out-of-bounds hover index should not cause incorrect highlighting.

        When multiple lists are rendered sequentially (like in Action Panel),
        a hovered_index of 2 from the first list should be cleared when
        rendering a shorter second list.
        """
        mock_draw_keycap.return_value = 40

        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)  # type: ignore

        # First render: 3 rows
        renderer.rows = [
            SelectableRow(text="Row A", key="A"),
            SelectableRow(text="Row B", key="B"),
            SelectableRow(text="Row C", key="C"),
        ]
        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # Hover over row index 2 in first render
        rect = renderer._hit_areas[2][0]
        mid_x = (rect[0] + rect[2]) // 2
        mid_y = (rect[1] + rect[3]) // 2
        renderer.update_hover_from_pixel(mid_x, mid_y)
        assert renderer.hovered_index == 2

        # Second render: only 1 row
        renderer.rows = [
            SelectableRow(text="Only Row", key="X"),
        ]
        canvas.drawn_rects.clear()  # Clear previous draw calls

        renderer.render(
            x_start=10,
            y_start=100,
            max_width=200,
            line_height=16,
            ascent=12,
        )

        # hovered_index=2 is out of bounds for 1 row, so it should be cleared
        assert renderer.hovered_index is None

        # No hover background should be drawn
        hover_rects = [r for r in canvas.drawn_rects if r["fill"]]
        assert len(hover_rects) == 0


class TestTextTruncation:
    """Tests for text truncation in INLINE mode."""

    def test_truncate_to_fit_short_text_unchanged(self) -> None:
        """Short text should not be truncated."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        result = renderer._truncate_to_fit("Short", max_width=100)

        assert result == "Short"

    def test_truncate_to_fit_long_text_adds_ellipsis(self) -> None:
        """Long text should be truncated with ellipsis."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        # With 8px per char, "VeryLongTextHere" = 16 chars * 8 = 128px
        # Truncate to fit in 50px
        result = renderer._truncate_to_fit("VeryLongTextHere", max_width=50)

        assert result.endswith("...")
        assert len(result) < len("VeryLongTextHere")

    def test_truncate_to_fit_zero_width_returns_empty(self) -> None:
        """Zero width should return empty string."""
        canvas = MockCanvas()
        renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)  # type: ignore

        result = renderer._truncate_to_fit("Text", max_width=0)

        assert result == ""


class TestLabelAlignmentPixelLevel:
    """Pixel-level tests for label vertical alignment with keycap."""

    def test_diagnose_alignment(self) -> None:
        """Diagnostic: print actual pixel positions for keycap and label."""
        canvas = _make_test_canvas(400, 150)
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)
        renderer.rows = [SelectableRow(text="Push", key="C")]

        ascent, descent = canvas.get_font_metrics()
        line_height = ascent + descent
        print(f"\nFont metrics: ascent={ascent}, descent={descent}")
        print(f"line_height={line_height}")

        canvas.begin_frame()
        renderer.render(
            x_start=10,
            y_start=ascent + 10,
            max_width=380,
            line_height=line_height,
            ascent=ascent,
            row_gap=line_height // 3,
        )
        pixels = canvas.end_frame()

        assert pixels is not None

        # Find keycap (DARK_GREY)
        keycap_bounds = _find_color_bounds(pixels, colors.DARK_GREY)
        assert keycap_bounds is not None
        kx_min, kx_max, ky_min, ky_max = keycap_bounds
        print(f"\nKeycap bounds: x=[{kx_min}, {kx_max}], y=[{ky_min}, {ky_max}]")
        print(f"Keycap center: y={(ky_min + ky_max) / 2}")

        # Find white pixels
        white_matches = np.all(pixels[..., :3] == (255, 255, 255), axis=2)
        y_indices, x_indices = np.where(white_matches)

        # Keycap letter (inside keycap bounds)
        keycap_letter_mask = (
            (x_indices >= kx_min)
            & (x_indices <= kx_max)
            & (y_indices >= ky_min)
            & (y_indices <= ky_max)
        )
        if keycap_letter_mask.any():
            letter_x = x_indices[keycap_letter_mask]
            letter_center_x = (letter_x.min() + letter_x.max()) / 2
            keycap_center_x = (kx_min + kx_max) / 2
            print(f"\nLetter bounds: x=[{letter_x.min()}, {letter_x.max()}]")
            print(f"Letter center x: {letter_center_x}, keycap: {keycap_center_x}")
            print(f"Horizontal offset: {letter_center_x - keycap_center_x} px")

        # Label (to the right of keycap)
        label_mask = x_indices > kx_max + 10
        if label_mask.any():
            label_y_vals = y_indices[label_mask]
            label_center_y = (label_y_vals.min() + label_y_vals.max()) / 2
            keycap_center_y = (ky_min + ky_max) / 2
            print(f"\nLabel y bounds: [{label_y_vals.min()}, {label_y_vals.max()}]")
            print(
                f"Label center y: {label_center_y}, keycap center y: {keycap_center_y}"
            )
            print(f"Vertical offset: {label_center_y - keycap_center_y} px")

    def test_label_contained_within_hit_area(self) -> None:
        """Label text (including descenders) should stay within hit area bounds.

        We use mathematical centering which may cause visible text to appear
        slightly higher than the keycap center, but guarantees containment.
        """
        # Canvas must be large enough for actual font size
        canvas = _make_test_canvas(400, 150)
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)
        # Use "Hype" - has descender (y) to test containment
        renderer.rows = [SelectableRow(text="Hype", key="C")]

        # Use ACTUAL font metrics from the canvas, not hardcoded values
        ascent, descent = canvas.get_font_metrics()
        line_height = ascent + descent
        row_gap = line_height // 3

        y_start = ascent + 10
        canvas.begin_frame()
        renderer.render(
            x_start=10,
            y_start=y_start,
            max_width=380,
            line_height=line_height,
            ascent=ascent,
            row_gap=row_gap,
        )
        pixels = canvas.end_frame()

        assert pixels is not None

        # Calculate expected hit area bounds
        hit_y_start = y_start - ascent
        hit_y_end = hit_y_start + line_height + row_gap

        # Find keycap bounds (DARK_GREY background)
        keycap_bounds = _find_color_bounds(pixels, colors.DARK_GREY)
        assert keycap_bounds is not None, "Keycap background not found"
        keycap_x_max = keycap_bounds[1]

        # Find all white pixels for the label (to the right of keycap)
        white_matches = np.all(pixels[..., :3] == (255, 255, 255), axis=2)
        y_indices, x_indices = np.where(white_matches)

        label_mask = x_indices > keycap_x_max + 10
        assert label_mask.any(), "No label text pixels found to the right of keycap"

        label_y_indices = y_indices[label_mask]
        label_y_min = int(label_y_indices.min())
        label_y_max = int(label_y_indices.max())

        # Label should be contained within hit area (with small tolerance for
        # anti-aliasing)
        assert label_y_min >= hit_y_start - 1, (
            f"Label top ({label_y_min}) above hit area top ({hit_y_start})"
        )
        assert label_y_max <= hit_y_end + 1, (
            f"Label bottom ({label_y_max}) below hit area bottom ({hit_y_end})"
        )
