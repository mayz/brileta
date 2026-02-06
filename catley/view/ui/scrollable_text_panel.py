"""Reusable scrollable text display component for UI overlays.

ScrollableTextPanel encapsulates content, scroll state, rendering, and input
handling for text that may overflow a fixed display area.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors, input_events

if TYPE_CHECKING:
    from catley.backends.pillow.canvas import PillowImageCanvas


class ScrollableTextPanel:
    """Reusable scrollable text display component.

    Fully encapsulates:
    - Content and scroll state
    - Rendering (including scroll indicators)
    - Input handling (Page Up/Down)

    Usage:
        panel = ScrollableTextPanel(max_lines=4, width_chars=36)
        panel.set_content(["Line 1", "Line 2", ...])
        panel.draw(canvas, pixel_x, pixel_y, line_height, char_width)
        consumed = panel.handle_input(event)
    """

    def __init__(self, max_visible_lines: int, width_chars: int) -> None:
        """Initialize the scrollable panel.

        Args:
            max_visible_lines: Maximum number of lines visible at once.
            width_chars: Width in characters (for scroll indicator placement).
        """
        self.lines: list[str] = []
        self.scroll_offset: int = 0
        self.max_visible_lines = max_visible_lines
        self.width_chars = width_chars

    def set_content(self, lines: list[str]) -> None:
        """Set content and reset scroll to top.

        Args:
            lines: List of text lines to display.
        """
        self.lines = lines
        self.scroll_offset = 0

    def has_overflow(self) -> bool:
        """Return True if content exceeds visible area."""
        return len(self.lines) > self.max_visible_lines

    def can_scroll_up(self) -> bool:
        """Return True if there's content above the current view."""
        return self.scroll_offset > 0

    def can_scroll_down(self) -> bool:
        """Return True if there's content below the current view."""
        return self.scroll_offset + self.max_visible_lines < len(self.lines)

    def scroll_up(self, amount: int = 1) -> None:
        """Scroll up by the specified number of lines."""
        self.scroll_offset = max(0, self.scroll_offset - amount)

    def scroll_down(self, amount: int = 1) -> None:
        """Scroll down by the specified number of lines."""
        max_offset = max(0, len(self.lines) - self.max_visible_lines)
        self.scroll_offset = min(max_offset, self.scroll_offset + amount)

    def get_visible_lines(self) -> list[str]:
        """Return the currently visible lines based on scroll offset."""
        end = self.scroll_offset + self.max_visible_lines
        return self.lines[self.scroll_offset : end]

    def draw(
        self,
        canvas: PillowImageCanvas,
        x: int,
        y: int,
        line_height: int,
        char_width: int,
        text_color: colors.Color = colors.WHITE,
        first_line_color: colors.Color | None = None,
        hover_zone: int = 0,
    ) -> None:
        """Draw visible content and scroll indicators.

        When content overflows, ellipsis ("...") is shown:
        - At the start of the first line when scrolled down (content above)
        - At the end of the last line when there's more content below

        The ellipsis turns white when hovered and scrollable, signaling
        interactivity. Click on the ellipsis to scroll.

        Args:
            canvas: Canvas to draw on.
            x: Top-left pixel X position.
            y: Top-left pixel Y position.
            line_height: Pixels per line.
            char_width: Pixels per character.
            text_color: Default text color.
            first_line_color: Optional different color for first visible line.
            hover_zone: Which scroll zone is hovered (-1=top, 0=none, +1=bottom).
        """
        visible = self.get_visible_lines()
        ellipsis = "..."
        ellipsis_default: colors.Color = (128, 128, 128)
        ellipsis_hovered: colors.Color = colors.WHITE

        for i, line in enumerate(visible):
            color = first_line_color if (i == 0 and first_line_color) else text_color
            line_y = y + i * line_height

            # First line: prepend ellipsis if there's content above
            if i == 0 and self.can_scroll_up():
                # White if top zone is hovered and can scroll up
                top_color = ellipsis_hovered if hover_zone < 0 else ellipsis_default
                canvas.draw_text(x, line_y, ellipsis, top_color)
                ellipsis_width, _, _ = canvas.get_text_metrics(ellipsis)
                canvas.draw_text(x + ellipsis_width, line_y, line, color)
            # Last line: append ellipsis if there's content below
            elif i == len(visible) - 1 and self.can_scroll_down():
                canvas.draw_text(x, line_y, line, color)
                line_width, _, _ = canvas.get_text_metrics(line)
                # White if bottom zone is hovered and can scroll down
                bottom_color = ellipsis_hovered if hover_zone > 0 else ellipsis_default
                canvas.draw_text(x + line_width, line_y, ellipsis, bottom_color)
            else:
                canvas.draw_text(x, line_y, line, color)

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle Page Up/Down input for scrolling.

        Args:
            event: The input event to handle.

        Returns:
            True if the event was consumed (scroll action occurred), False otherwise.
        """
        if not self.has_overflow():
            return False

        match event:
            case input_events.KeyDown(sym=input_events.KeySym.PAGEUP):
                self.scroll_up(self.max_visible_lines)
                return True
            case input_events.KeyDown(sym=input_events.KeySym.PAGEDOWN):
                self.scroll_down(self.max_visible_lines)
                return True

        return False
