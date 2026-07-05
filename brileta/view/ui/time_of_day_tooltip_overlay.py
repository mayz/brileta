"""Hover tooltip for the HUD time-of-day dial."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from brileta import colors, config, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.game.clock import format_clock_time
from brileta.types import PixelRect
from brileta.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.view.render.canvas import Canvas
    from brileta.view.views.time_of_day_dial_view import TimeOfDayDialView


class TimeOfDayTooltipOverlay(TextOverlay):
    """Passive tooltip that shows precise clock time while hovering the dial."""

    _FONT_SIZE_PX = config.PLAYER_STATUS_FONT_SIZE
    _GAP_PX = 2
    # Match the right padding the HP/Armor status text keeps from the edge.
    _EDGE_MARGIN_PX = 8
    # Refresh the displayed time at most this often so a fast day/night cycle
    # (small day_length_seconds) does not flicker the label illegibly.
    _UPDATE_INTERVAL_S = 0.5
    _TEXT_COLOR: colors.Color = (255, 222, 95)

    def __init__(self, controller: Controller, dial_view: TimeOfDayDialView) -> None:
        super().__init__(controller)
        self.is_interactive = False
        self.always_dirty = True
        self.dial_view = dial_view
        self._label_text = ""
        self._last_label_update_s = 0.0
        self._text_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            transparent=True,
            font_size=self._FONT_SIZE_PX,
            line_spacing=1.0,
        )

    def _calculate_dimensions(self) -> None:
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        dial_rect = self._dial_screen_rect()
        if dial_rect is None or not self._is_hovering_rect(dial_rect):
            self._clear_dimensions()
            return

        # Throttle the label refresh so a fast clock does not update it every
        # frame. Always refresh on the first frame of a hover (empty label).
        now = perf_counter()
        if not self._label_text or now - self._last_label_update_s >= (
            self._UPDATE_INTERVAL_S
        ):
            self._label_text = format_clock_time(
                self.controller.gw.clock.time_of_day,
                use_24_hour=self.controller.use_24_hour_clock,
            )
            self._last_label_update_s = now

        assert self.canvas is not None
        x_off, y_off, text_w, text_h = self.canvas.get_text_bbox(self._label_text)
        self._text_bbox = (int(x_off), int(y_off), int(text_w), int(text_h))
        line_h = max(1, int(self.canvas.get_effective_line_height()))
        self.pixel_width = text_w
        self.pixel_height = max(text_h, line_h)

        screen_w = (
            self.controller.graphics.console_width_tiles * self.tile_dimensions[0]
        )
        screen_h = (
            self.controller.graphics.console_height_tiles * self.tile_dimensions[1]
        )
        dial_x1, _dial_y1, dial_x2, dial_y2 = dial_rect
        tooltip_x = ((dial_x1 + dial_x2) // 2) - (self.pixel_width // 2)
        tooltip_y = dial_y2 + self._GAP_PX
        max_x = screen_w - self.pixel_width - self._EDGE_MARGIN_PX
        tooltip_x = max(0, min(tooltip_x, max_x))
        tooltip_y = max(0, min(tooltip_y, screen_h - self.pixel_height))

        tile_w, tile_h = self.tile_dimensions
        self.x_tiles = tooltip_x / max(1, tile_w)
        self.y_tiles = tooltip_y / max(1, tile_h)
        self.width = max(1, int((self.pixel_width + tile_w - 1) // max(1, tile_w)))
        self.height = max(1, int((self.pixel_height + tile_h - 1) // max(1, tile_h)))

    def _clear_dimensions(self) -> None:
        self._label_text = ""
        self.pixel_width = 0
        self.pixel_height = 0
        self.x_tiles = 0
        self.y_tiles = 0
        self.width = 0
        self.height = 0

    def _dial_screen_rect(self) -> PixelRect | None:
        view = self.dial_view
        if not view.visible or view.width <= 0 or view.height <= 0:
            return None

        x1, y1 = self.controller.graphics.console_to_screen_coords(view.x, view.y)
        x2, y2 = self.controller.graphics.console_to_screen_coords(
            view.x + view.width,
            view.y + view.height,
        )
        return (int(x1), int(y1), int(x2), int(y2))

    def _is_hovering_rect(self, rect: PixelRect) -> bool:
        x1, y1, x2, y2 = rect
        cursor = self.controller.frame_manager.cursor_manager
        mouse_x = float(cursor.mouse_pixel_x)
        mouse_y = float(cursor.mouse_pixel_y)
        return x1 <= mouse_x < x2 and y1 <= mouse_y < y2

    def draw_content(self) -> None:
        if not self._label_text:
            return

        assert self.canvas is not None
        x_off, y_off, _text_w, _text_h = self._text_bbox
        text_x = -x_off
        text_y = -y_off
        self.canvas.draw_text(text_x, text_y, self._label_text, self._TEXT_COLOR)

    def handle_input(self, event: input_events.InputEvent) -> bool:
        _ = event
        return False
