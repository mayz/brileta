"""Tests for ScrollableTextPanel component."""

from __future__ import annotations

from unittest.mock import MagicMock

from catley import input_events
from catley.view.ui.scrollable_text_panel import ScrollableTextPanel


class TestScrollableTextPanel:
    """Tests for the ScrollableTextPanel class."""

    def test_no_overflow_shows_all_lines(self) -> None:
        """Panel with fewer lines than max should show all lines."""
        panel = ScrollableTextPanel(max_visible_lines=5, width_chars=40)
        panel.set_content(["Line 1", "Line 2", "Line 3"])
        assert not panel.has_overflow()
        assert panel.get_visible_lines() == ["Line 1", "Line 2", "Line 3"]

    def test_overflow_shows_first_n_lines(self) -> None:
        """Panel with more lines than max should show only visible lines."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        assert panel.has_overflow()
        assert panel.get_visible_lines() == ["A", "B"]

    def test_scroll_down(self) -> None:
        """Scrolling down should shift visible lines."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down()
        assert panel.get_visible_lines() == ["B", "C"]

    def test_scroll_up(self) -> None:
        """Scrolling up should shift visible lines back."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down(2)  # Now at C, D
        panel.scroll_up()
        assert panel.get_visible_lines() == ["B", "C"]

    def test_scroll_up_at_top(self) -> None:
        """Scrolling up at top should have no effect."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_up(10)  # Can't go negative
        assert panel.scroll_offset == 0
        assert panel.get_visible_lines() == ["A", "B"]

    def test_scroll_down_at_bottom(self) -> None:
        """Scrolling down at bottom should stop at max offset."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C"])
        panel.scroll_down(10)  # Can't exceed max
        # Max offset = 3 - 2 = 1, so we see B, C
        assert panel.get_visible_lines() == ["B", "C"]

    def test_set_content_resets_scroll(self) -> None:
        """Setting new content should reset scroll to top."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C"])
        panel.scroll_down()
        panel.set_content(["X", "Y", "Z"])
        assert panel.scroll_offset == 0
        assert panel.get_visible_lines() == ["X", "Y"]

    def test_can_scroll_up_indicator(self) -> None:
        """can_scroll_up should return correct state."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        assert not panel.can_scroll_up()  # At top
        panel.scroll_down()
        assert panel.can_scroll_up()  # Not at top anymore

    def test_can_scroll_down_indicator(self) -> None:
        """can_scroll_down should return correct state."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        assert panel.can_scroll_down()  # Content below
        panel.scroll_down(2)  # Move to end
        assert not panel.can_scroll_down()  # At bottom

    def test_scroll_indicators_at_middle(self) -> None:
        """Both scroll indicators should be true in middle of content."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down()  # In middle
        assert panel.can_scroll_up()
        assert panel.can_scroll_down()

    def test_no_overflow_no_scroll(self) -> None:
        """Panel without overflow should not allow scrolling."""
        panel = ScrollableTextPanel(max_visible_lines=5, width_chars=40)
        panel.set_content(["A", "B"])
        assert not panel.has_overflow()
        assert not panel.can_scroll_up()
        assert not panel.can_scroll_down()

    def test_empty_content(self) -> None:
        """Empty content should be handled gracefully."""
        panel = ScrollableTextPanel(max_visible_lines=5, width_chars=40)
        panel.set_content([])
        assert not panel.has_overflow()
        assert panel.get_visible_lines() == []

    def test_scroll_amount(self) -> None:
        """Scrolling by custom amount should work correctly."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D", "E", "F"])
        panel.scroll_down(3)
        assert panel.get_visible_lines() == ["D", "E"]


class TestScrollableTextPanelInput:
    """Tests for ScrollableTextPanel input handling."""

    def _make_keydown(self, sym: input_events.KeySym) -> input_events.KeyDown:
        """Create a KeyDown event for testing."""
        return input_events.KeyDown(
            scancode=0,
            sym=sym,
            mod=input_events.Modifier.NONE,
            repeat=False,
        )

    def test_page_down_scrolls(self) -> None:
        """Page Down should scroll down by max_visible_lines."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D", "E", "F"])

        event = self._make_keydown(input_events.KeySym.PAGEDOWN)
        consumed = panel.handle_input(event)

        assert consumed
        assert panel.scroll_offset == 2  # Scrolled by max_visible_lines

    def test_page_up_scrolls(self) -> None:
        """Page Up should scroll up by max_visible_lines."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D", "E", "F"])
        panel.scroll_down(4)  # Start near bottom

        event = self._make_keydown(input_events.KeySym.PAGEUP)
        consumed = panel.handle_input(event)

        assert consumed
        assert panel.scroll_offset == 2  # Scrolled back by 2

    def test_no_overflow_ignores_input(self) -> None:
        """Input should not be consumed when there's no overflow."""
        panel = ScrollableTextPanel(max_visible_lines=5, width_chars=40)
        panel.set_content(["A", "B"])  # No overflow

        event = self._make_keydown(input_events.KeySym.PAGEDOWN)
        consumed = panel.handle_input(event)

        assert not consumed  # Event not consumed

    def test_other_keys_not_consumed(self) -> None:
        """Non-scroll keys should not be consumed."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])

        event = self._make_keydown(input_events.KeySym.RETURN)
        consumed = panel.handle_input(event)

        assert not consumed

    def test_mouse_events_not_consumed(self) -> None:
        """Mouse events should not be consumed."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])

        event = input_events.MouseMotion(position=input_events.Point(0, 0))
        consumed = panel.handle_input(event)

        assert not consumed


class TestScrollableTextPanelDraw:
    """Tests for ScrollableTextPanel draw functionality."""

    def test_draw_calls_canvas(self) -> None:
        """Draw should call canvas.draw_text for visible lines."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["Line A", "Line B", "Line C"])

        # Create mock canvas with get_text_metrics returning reasonable values
        mock_canvas = MagicMock()
        mock_canvas.get_text_metrics.return_value = (
            50,
            12,
            16,
        )  # width, ascent, height

        panel.draw(
            canvas=mock_canvas,
            x=10,
            y=20,
            line_height=16,
            char_width=8,
        )

        # Should draw visible lines plus ellipsis for overflow indicator
        calls = mock_canvas.draw_text.call_args_list
        assert len(calls) >= 2

        # Check first line is drawn at correct position
        first_call = calls[0]
        assert first_call[0][0] == 10  # x
        assert first_call[0][1] == 20  # y
        assert first_call[0][2] == "Line A"

    def test_draw_scroll_indicators_when_needed(self) -> None:
        """Ellipsis indicators should be drawn when content overflows."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down()  # Now in middle, can scroll both ways

        mock_canvas = MagicMock()
        mock_canvas.get_text_metrics.return_value = (
            20,
            12,
            16,
        )  # width, ascent, height

        panel.draw(
            canvas=mock_canvas,
            x=10,
            y=20,
            line_height=16,
            char_width=8,
        )

        # Should draw ellipsis at start (content above) and end (content below)
        drawn_text = [call[0][2] for call in mock_canvas.draw_text.call_args_list]
        assert "..." in drawn_text  # Ellipsis indicators present

    def test_hover_zone_colors_ellipsis_top(self) -> None:
        """Top ellipsis should be white when hover_zone < 0 and can scroll up."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down()  # Can now scroll up

        mock_canvas = MagicMock()
        mock_canvas.get_text_metrics.return_value = (20, 12, 16)

        # Draw with hover_zone = -1 (top zone hovered)
        panel.draw(
            canvas=mock_canvas,
            x=10,
            y=20,
            line_height=16,
            char_width=8,
            hover_zone=-1,
        )

        # Find the first ellipsis draw call (top indicator)
        ellipsis_calls = [
            call for call in mock_canvas.draw_text.call_args_list if call[0][2] == "..."
        ]
        assert len(ellipsis_calls) >= 1
        # First ellipsis should be white (255, 255, 255) when hovered
        first_ellipsis_color = ellipsis_calls[0][0][3]
        assert first_ellipsis_color == (255, 255, 255)

    def test_hover_zone_colors_ellipsis_bottom(self) -> None:
        """Bottom ellipsis should be white when hover_zone > 0 and can scroll down."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        # At top, can scroll down but not up

        mock_canvas = MagicMock()
        mock_canvas.get_text_metrics.return_value = (20, 12, 16)

        # Draw with hover_zone = 1 (bottom zone hovered)
        panel.draw(
            canvas=mock_canvas,
            x=10,
            y=20,
            line_height=16,
            char_width=8,
            hover_zone=1,
        )

        # Find ellipsis draw calls
        ellipsis_calls = [
            call for call in mock_canvas.draw_text.call_args_list if call[0][2] == "..."
        ]
        assert len(ellipsis_calls) >= 1
        # Bottom ellipsis should be white when hovered
        bottom_ellipsis_color = ellipsis_calls[-1][0][3]
        assert bottom_ellipsis_color == (255, 255, 255)

    def test_hover_zone_zero_keeps_grey(self) -> None:
        """Ellipsis should be grey when hover_zone = 0 (not hovered)."""
        panel = ScrollableTextPanel(max_visible_lines=2, width_chars=40)
        panel.set_content(["A", "B", "C", "D"])
        panel.scroll_down()  # In middle, both indicators visible

        mock_canvas = MagicMock()
        mock_canvas.get_text_metrics.return_value = (20, 12, 16)

        # Draw with hover_zone = 0 (nothing hovered)
        panel.draw(
            canvas=mock_canvas,
            x=10,
            y=20,
            line_height=16,
            char_width=8,
            hover_zone=0,
        )

        # All ellipsis should be grey
        ellipsis_calls = [
            call for call in mock_canvas.draw_text.call_args_list if call[0][2] == "..."
        ]
        for call in ellipsis_calls:
            color = call[0][3]
            assert color == (128, 128, 128)  # Grey
