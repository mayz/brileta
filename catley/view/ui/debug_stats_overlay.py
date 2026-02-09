from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from catley import colors, config, input_events
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.util.live_vars import LiveVariable, live_variable_registry
from catley.view.render.canvas import Canvas
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from catley.controller import Controller


class DebugStatsOverlay(TextOverlay):
    """Overlay that displays watched live variables."""

    _REFRESH_INTERVAL_SECONDS = 0.5  # 2 Hz refresh cadence

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.is_interactive = False
        # Re-render only on the fixed refresh cadence or watch-list changes.
        self.always_dirty = False
        self._display_lines: list[str] = []
        self._watched_names: tuple[str, ...] = ()
        self._next_refresh_time_s = 0.0

    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(
            self.controller.graphics, font_path=config.UI_FONT_PATH, transparent=True
        )

    def _format_live_variable_line(self, var: LiveVariable) -> str:
        """Build the display line for one watched live variable."""
        raw_value = var.get_value()
        display_value = var.formatter(raw_value) if var.formatter else str(raw_value)
        return f"{var.name}: {display_value}"

    def _refresh_display_lines(self, watched: list[LiveVariable] | None = None) -> None:
        """Snapshot watched variable values for this refresh tick.

        Formatted values may contain newlines for multi-line display (e.g.
        score breakdowns). Each ``\\n`` becomes a separate display line.
        """
        active_watched = (
            watched
            if watched is not None
            else live_variable_registry.get_watched_variables()
        )
        self._watched_names = tuple(var.name for var in active_watched)
        lines: list[str] = []
        for var in active_watched:
            formatted = self._format_live_variable_line(var)
            lines.extend(formatted.split("\n"))
        self._display_lines = lines

    def show(self) -> None:
        """Show overlay and force an immediate snapshot refresh."""
        self._next_refresh_time_s = 0.0
        super().show()

    def draw(self) -> None:
        """Refresh overlay snapshot at 2Hz, then draw cached texture if needed."""
        if not self.is_active:
            return

        watched = live_variable_registry.get_watched_variables()
        watched_names = tuple(var.name for var in watched)
        now = perf_counter()
        should_refresh = (
            watched_names != self._watched_names
            or now >= self._next_refresh_time_s
            or not self._display_lines
        )
        if should_refresh:
            self._refresh_display_lines(watched)
            self._next_refresh_time_s = now + self._REFRESH_INTERVAL_SECONDS
            self.invalidate()

        super().draw()

    def _calculate_dimensions(self) -> None:
        watched = live_variable_registry.get_watched_variables()
        watched_names = tuple(var.name for var in watched)
        if watched_names != self._watched_names or not self._display_lines:
            self._refresh_display_lines(watched)
        self.tile_dimensions = self.controller.graphics.tile_dimensions

        if not self._display_lines:
            self.width = 0
            self.height = 0
            self.pixel_width = 0
            self.pixel_height = 0
            self.x_tiles = 0
            self.y_tiles = 0
            return

        max_width = max(len(line) for line in self._display_lines)

        self.width = max_width + 2  # small padding
        self.height = len(self._display_lines)

        # Right-aligned at a fixed column offset from the screen edge so
        # the overlay doesn't jump horizontally when line widths change.
        _FIXED_WIDTH = 60  # reserve enough columns for the widest score display
        console_w = self.controller.graphics.console_width_tiles
        self.x_tiles = max(0, console_w - _FIXED_WIDTH)
        self.y_tiles = 1

        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]

    def draw_content(self) -> None:
        watched = live_variable_registry.get_watched_variables()
        watched_names = tuple(var.name for var in watched)
        if watched_names != self._watched_names or not self._display_lines:
            self._refresh_display_lines(watched)
        assert self.canvas is not None

        _tile_width, tile_height = self.tile_dimensions
        for y_offset, text in enumerate(self._display_lines):
            self.canvas.draw_text(
                0,
                y_offset * tile_height,
                text,
                colors.YELLOW,
            )

    def handle_input(self, event: input_events.InputEvent) -> bool:  # pragma: no cover
        return False
