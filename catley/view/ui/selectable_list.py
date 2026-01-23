"""Unified selectable list rendering for keycap+label lists.

This module provides a unified component for rendering selectable lists with
keycaps and labels. It supports two layout modes:
- KEYCAP: Right-aligned keycaps for action panels (Controls, Actions)
- INLINE: Left-aligned with prefix segments for inventory menus

The SelectableListRenderer handles:
- Hover highlighting with dark grey background
- Hit area tracking for mouse click detection
- Hotkey lookup for keyboard shortcuts
- Color brightening on hover
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from catley import colors
from catley.types import PixelRect
from catley.view.ui.ui_utils import draw_keycap

if TYPE_CHECKING:
    from catley.backends.pillow.canvas import PillowImageCanvas


class LayoutMode(Enum):
    """Layout mode for the selectable list renderer."""

    KEYCAP = auto()  # Right-aligned keycaps + fixed label column
    INLINE = auto()  # prefix_segments + "(key) " + text, left-aligned


@dataclass
class SelectableRow:
    """A row in a selectable list. Supports all current use cases.

    Attributes:
        text: Main label text to display.
        key: Hotkey string (e.g., "A", "Space", "R-Click"). Displayed as keycap
            in KEYCAP mode, or as "(key) " text in INLINE mode.
        enabled: Whether the row is selectable/clickable.
        color: Text color for the main label.
        data: Associated data object (e.g., ActionOption, Item).
        prefix_segments: List of (text, color) tuples for prefix rendering.
            Used for bullets, slot indicators like "[1]", category dots.
        suffix: Optional text to display after the main label.
        suffix_color: Color for the suffix text.
        force_color: If True, keep the specified color even when disabled.
        execute: Optional callback to execute when this row is clicked.
            Called with no arguments. Return value is ignored.
    """

    text: str
    key: str | None = None
    enabled: bool = True
    color: colors.Color = colors.WHITE
    data: Any = None
    prefix_segments: list[tuple[str, colors.Color]] | None = None
    suffix: str | None = None
    suffix_color: colors.Color | None = None
    force_color: bool = False
    execute: Callable[[], None] | None = field(default=None)


class SelectableListRenderer:
    """Renders selectable lists with hover highlighting and hit area tracking.

    Supports two layout modes:
    - KEYCAP: Right-aligned keycaps for action panels
    - INLINE: Left-aligned with prefix segments for inventory menus

    Usage:
        renderer = SelectableListRenderer(canvas, LayoutMode.KEYCAP)
        renderer.rows = [SelectableRow(text="Attack", key="A", data=action)]
        renderer.hovered_index = 0  # Set externally from mouse position
        y_after = renderer.render(x_start, y_start, max_width, line_height, ascent)

        # For mouse interaction:
        if renderer.update_hover_from_pixel(px, py):
            # Hover changed, need redraw
            pass
        row = renderer.get_row_at_pixel(px, py)
        row = renderer.get_row_by_hotkey("a")
    """

    def __init__(
        self, canvas: PillowImageCanvas, mode: LayoutMode = LayoutMode.KEYCAP
    ) -> None:
        """Initialize the renderer.

        Args:
            canvas: The PillowImageCanvas to draw on.
            mode: Layout mode (KEYCAP or INLINE).
        """
        self.canvas = canvas
        self.mode = mode
        self.rows: list[SelectableRow] = []
        self.hovered_index: int | None = None
        self._hit_areas: list[tuple[PixelRect, int]] = []

    def render(
        self,
        x_start: int,
        y_start: int,
        max_width: int,
        line_height: int,
        ascent: int,
        row_gap: int | None = None,
    ) -> int:
        """Render all rows.

        Args:
            x_start: X pixel position for left edge of the list.
            y_start: Y pixel position for baseline of first row.
            max_width: Maximum width in pixels for the entire list area.
            line_height: Height of each line in pixels.
            ascent: Font ascent in pixels.
            row_gap: Extra gap between rows in pixels. None means use default
                (line_height // 3 for KEYCAP mode, 0 for INLINE mode).

        Returns:
            Final y position after rendering all rows (for continuation).
        """
        # Clear stale hovered_index if it's out of bounds for the new row list.
        # This prevents a hovered_index from a previous (longer) list from
        # causing issues when rendering a new (shorter) list.
        if self.hovered_index is not None and self.hovered_index >= len(self.rows):
            self.hovered_index = None
        self._hit_areas.clear()

        if not self.rows:
            return y_start

        if self.mode == LayoutMode.KEYCAP:
            return self._render_keycap_mode(
                x_start, y_start, max_width, line_height, ascent, row_gap
            )
        return self._render_inline_mode(
            x_start, y_start, max_width, line_height, ascent, row_gap
        )

    def _get_keycap_width(self, key: str | None, line_height: int) -> int:
        """Calculate the width of a keycap for a given key.

        Args:
            key: The key text, or None for no keycap.
            line_height: Line height for sizing.

        Returns:
            Width in pixels, or 0 if no key.
        """
        if not key:
            return 0

        keycap_size = int(line_height * 0.85)
        keycap_font_size = max(8, int(keycap_size * 0.65))
        keycap_internal_padding = 12

        text_width, _, _ = self.canvas.get_text_metrics(
            key.upper(), font_size=keycap_font_size
        )
        return max(keycap_size, text_width + keycap_internal_padding)

    def _render_keycap_mode(
        self,
        x_start: int,
        y_start: int,
        max_width: int,
        line_height: int,
        ascent: int,
        row_gap: int | None,
    ) -> int:
        """Render rows with right-aligned keycaps and fixed label column.

        The keycaps are right-aligned to a column edge calculated from the
        widest keycap. Labels start at a fixed column after the keycap area.
        """
        # Calculate keycap column alignment
        keycap_gap = 12  # Gap after keycap (from draw_keycap)
        label_gap = 8  # Extra gap before label

        # Find widest keycap to set the right edge of keycap column
        keycap_widths = [
            self._get_keycap_width(r.key, line_height) for r in self.rows if r.key
        ]
        widest_keycap = max(keycap_widths) if keycap_widths else 0

        # Right edge of keycap column (keycaps right-align to this)
        keycap_right_edge = x_start + widest_keycap
        # Label column starts after keycap + gaps
        label_column_x = keycap_right_edge + keycap_gap + label_gap

        y_pixel = y_start
        # Use provided row_gap if specified, otherwise default to line_height // 3
        gap = row_gap if row_gap is not None else line_height // 3

        for i, row in enumerate(self.rows):
            is_hovered = self.hovered_index == i and row.enabled

            # Calculate hit area for this row
            if self._hit_areas:
                # Continue from where the previous hit area ended
                prev_rect = self._hit_areas[-1][0]
                hit_y_start = prev_rect[3]
            else:
                # First row starts at top of text
                hit_y_start = y_pixel - ascent

            hit_y_end = hit_y_start + line_height + gap
            hit_rect: PixelRect = (
                x_start,
                hit_y_start,
                x_start + max_width,
                hit_y_end,
            )
            self._hit_areas.append((hit_rect, i))

            # Draw hover background
            if is_hovered:
                self.canvas.draw_rect(
                    pixel_x=x_start,
                    pixel_y=hit_y_start,
                    width=max_width,
                    height=hit_y_end - hit_y_start,
                    color=colors.DARK_GREY,
                    fill=True,
                )

            # Draw keycap (right-aligned, vertically centered in hit area)
            if row.key:
                kw = self._get_keycap_width(row.key, line_height)
                keycap_x = keycap_right_edge - kw

                # Calculate keycap height (same formula as draw_keycap)
                keycap_height = int(line_height * 0.85)
                hit_area_height = hit_y_end - hit_y_start

                # Center keycap vertically within the hit area
                # draw_keycap does keycap_y = pixel_y - 2, so we add 2 to compensate
                centered_keycap_y = hit_y_start + (hit_area_height - keycap_height) // 2
                keycap_pixel_y = centered_keycap_y + 2

                draw_keycap(
                    canvas=self.canvas,
                    pixel_x=keycap_x,
                    pixel_y=keycap_pixel_y,
                    key=row.key,
                    bg_color=colors.DARK_GREY,
                    border_color=colors.GREY,
                    text_color=colors.WHITE,
                )

            # Center label text vertically within the hit area using mathematical
            # centering. This ensures descenders (g, p, y, etc.) stay within the
            # hit area bounds. The visible ink may appear slightly high since most
            # text mass is in the ascent, but containment is more important.
            hit_area_height = hit_y_end - hit_y_start
            label_y = hit_y_start + (hit_area_height - line_height) // 2

            # Draw prefix segments if present (before the main text)
            current_x = label_column_x
            if row.prefix_segments:
                for seg_text, seg_color in row.prefix_segments:
                    color = self._brighten_if_hovered(seg_color, is_hovered)
                    self.canvas.draw_text(
                        pixel_x=current_x,
                        pixel_y=label_y,
                        text=seg_text,
                        color=color,
                    )
                    seg_width, _, _ = self.canvas.get_text_metrics(seg_text)
                    current_x += seg_width

            # Draw text at current position (after any prefix)
            text_color = self._get_effective_color(row, is_hovered)
            self.canvas.draw_text(
                pixel_x=current_x,
                pixel_y=label_y,
                text=row.text,
                color=text_color,
            )

            # Draw suffix if present
            if row.suffix:
                text_width, _, _ = self.canvas.get_text_metrics(row.text)
                suffix_x = current_x + text_width
                suffix_color = row.suffix_color or row.color
                suffix_color = self._brighten_if_hovered(suffix_color, is_hovered)
                self.canvas.draw_text(
                    pixel_x=suffix_x,
                    pixel_y=label_y,
                    text=row.suffix,
                    color=suffix_color,
                )

            y_pixel += line_height + gap

        return y_pixel

    def _render_inline_mode(
        self,
        x_start: int,
        y_start: int,
        max_width: int,
        line_height: int,
        ascent: int,
        row_gap: int | None,
    ) -> int:
        """Render rows with left-aligned prefix_segments + "(key) " + text.

        This mode renders:
        1. prefix_segments (if any) - e.g., "[1] " slot indicators, category dots
        2. "(key) " in yellow (not a keycap)
        3. Main text
        4. Suffix (if any)
        """
        y_pixel = y_start
        # Use provided row_gap if specified, otherwise default to 0 for inline mode
        gap = row_gap if row_gap is not None else 0

        for i, row in enumerate(self.rows):
            is_hovered = self.hovered_index == i and row.enabled
            current_x = x_start

            # Calculate hit area for this row
            if self._hit_areas:
                prev_rect = self._hit_areas[-1][0]
                hit_y_start = prev_rect[3]
            else:
                hit_y_start = y_pixel - ascent

            hit_y_end = hit_y_start + line_height + gap
            hit_rect: PixelRect = (
                x_start,
                hit_y_start,
                x_start + max_width,
                hit_y_end,
            )
            self._hit_areas.append((hit_rect, i))

            # Draw hover background
            if is_hovered:
                self.canvas.draw_rect(
                    pixel_x=x_start,
                    pixel_y=hit_y_start,
                    width=max_width,
                    height=hit_y_end - hit_y_start,
                    color=colors.DARK_GREY,
                    fill=True,
                )

            # Draw prefix segments (category dots, slot indicators)
            if row.prefix_segments:
                for seg_text, seg_color in row.prefix_segments:
                    color = self._brighten_if_hovered(seg_color, is_hovered)
                    self.canvas.draw_text(
                        pixel_x=current_x,
                        pixel_y=y_pixel - ascent,
                        text=seg_text,
                        color=color,
                    )
                    seg_width, _, _ = self.canvas.get_text_metrics(seg_text)
                    current_x += seg_width

            # Draw key as "(key) " in yellow (not keycap)
            if row.key:
                key_text = f"({row.key}) "
                key_color = self._brighten_if_hovered(colors.YELLOW, is_hovered)
                self.canvas.draw_text(
                    pixel_x=current_x,
                    pixel_y=y_pixel - ascent,
                    text=key_text,
                    color=key_color,
                )
                key_width, _, _ = self.canvas.get_text_metrics(key_text)
                current_x += key_width

            # Draw main text
            text_color = self._get_effective_color(row, is_hovered)

            # Truncate text if needed
            available_width = x_start + max_width - current_x
            truncated_text = self._truncate_to_fit(row.text, available_width)

            self.canvas.draw_text(
                pixel_x=current_x,
                pixel_y=y_pixel - ascent,
                text=truncated_text,
                color=text_color,
            )
            text_width, _, _ = self.canvas.get_text_metrics(truncated_text)
            current_x += text_width

            # Draw suffix if present
            if row.suffix:
                suffix_color = row.suffix_color or row.color
                suffix_color = self._brighten_if_hovered(suffix_color, is_hovered)
                self.canvas.draw_text(
                    pixel_x=current_x,
                    pixel_y=y_pixel - ascent,
                    text=row.suffix,
                    color=suffix_color,
                )

            y_pixel += line_height + gap

        return y_pixel

    def _truncate_to_fit(self, text: str, max_width: int) -> str:
        """Truncate text to fit within max_width pixels.

        Args:
            text: Text to potentially truncate.
            max_width: Maximum width in pixels.

        Returns:
            The text, truncated with "..." if needed.
        """
        if max_width <= 0:
            return ""

        text_width, _, _ = self.canvas.get_text_metrics(text)
        if text_width <= max_width:
            return text

        # Need to truncate
        ellipsis = "..."

        # Iteratively remove characters until text fits
        while len(text) > 0:
            text = text[:-1]
            text_width, _, _ = self.canvas.get_text_metrics(text + ellipsis)
            if text_width <= max_width:
                return text + ellipsis

        return ellipsis

    def _get_effective_color(
        self, row: SelectableRow, is_hovered: bool
    ) -> colors.Color:
        """Get the effective text color for a row.

        Args:
            row: The row to get color for.
            is_hovered: Whether the row is currently hovered.

        Returns:
            The color to use for the main text.
        """
        base_color = row.color if row.enabled or row.force_color else colors.GREY
        return self._brighten_if_hovered(base_color, is_hovered)

    def _brighten_if_hovered(
        self, color: colors.Color, is_hovered: bool
    ) -> colors.Color:
        """Brighten color by ~40 RGB if hovered, capped at 255.

        Args:
            color: The base color.
            is_hovered: Whether to brighten.

        Returns:
            The original color if not hovered, or brightened color if hovered.
        """
        if not is_hovered:
            return color
        return (
            min(255, color[0] + 40),
            min(255, color[1] + 40),
            min(255, color[2] + 40),
        )

    def update_hover_from_pixel(self, px: int, py: int) -> bool:
        """Update hovered_index based on pixel coordinates.

        Args:
            px: X pixel coordinate relative to the list's coordinate space.
            py: Y pixel coordinate relative to the list's coordinate space.

        Returns:
            True if the hovered index changed (needs redraw).
        """
        old_index = self.hovered_index
        new_index = None

        for rect, row_index in self._hit_areas:
            x1, y1, x2, y2 = rect
            if x1 <= px < x2 and y1 <= py < y2:
                # Check if this row is enabled
                if row_index < len(self.rows) and self.rows[row_index].enabled:
                    new_index = row_index
                break

        self.hovered_index = new_index
        return old_index != new_index

    def get_row_at_pixel(self, px: int, py: int) -> SelectableRow | None:
        """Get the row at the given pixel coordinates.

        Args:
            px: X pixel coordinate relative to the list's coordinate space.
            py: Y pixel coordinate relative to the list's coordinate space.

        Returns:
            The SelectableRow at that position, or None if no row or disabled.
        """
        for rect, row_index in self._hit_areas:
            x1, y1, x2, y2 = rect
            if x1 <= px < x2 and y1 <= py < y2:
                if row_index < len(self.rows) and self.rows[row_index].enabled:
                    return self.rows[row_index]
                return None
        return None

    def execute_at_pixel(self, px: int, py: int) -> bool:
        """Execute the callback for the row at the given pixel coordinates.

        If the row at (px, py) has an execute callback, calls it.

        Args:
            px: X pixel coordinate relative to the list's coordinate space.
            py: Y pixel coordinate relative to the list's coordinate space.

        Returns:
            True if a callback was executed, False otherwise.
        """
        row = self.get_row_at_pixel(px, py)
        if row is not None and row.execute is not None:
            row.execute()
            return True
        return False

    def get_row_by_hotkey(self, key: str) -> SelectableRow | None:
        """Get an enabled row by its hotkey (case-insensitive).

        Args:
            key: The hotkey character to search for.

        Returns:
            The matching SelectableRow, or None if not found or disabled.
        """
        key_lower = key.lower()
        for row in self.rows:
            if row.key and row.key.lower() == key_lower and row.enabled:
                return row
        return None

    def reset(self) -> None:
        """Reset renderer to initial state (clears all state including hover)."""
        self.rows.clear()
        self._hit_areas.clear()
        self.hovered_index = None

    def clear_hit_areas(self) -> None:
        """Clear hit areas without affecting rows or hover state.

        Use this when a section transitions from rendered to not-rendered,
        to prevent stale hit areas from intercepting clicks meant for other
        sections that are now rendered in the same screen region.
        """
        self._hit_areas.clear()
