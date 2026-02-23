"""Viewport-local zoom indicator overlay for the world view.

Displays the current world-view zoom level as a small passive chip inside the
world viewport. The indicator is hidden at the default zoom unless the user
recently changed zoom, in which case it appears briefly and fades out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from brileta import colors, config, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.types import Opacity, PixelRect
from brileta.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.view.render.canvas import Canvas
    from brileta.view.views.world_view import WorldView


class ZoomIndicatorOverlay(TextOverlay):
    """Passive viewport-local chip showing the current world zoom level."""

    _PADDING_X_PX = 8
    _PADDING_Y_PX = 8
    _INNER_PAD_X_PX = 8
    _INNER_PAD_Y_PX = 4
    _SHADOW_OFFSET_PX = 1

    _WAKE_DURATION_S = 1.0
    _WAKE_ALPHA = 0.92
    _PERSISTENT_ALPHA = 0.35
    _RESET_HINT_TEXT = "Press 0 to reset"

    def __init__(self, controller: Controller, world_view: WorldView) -> None:
        super().__init__(controller)
        self.is_interactive = False
        self.world_view = world_view
        self._label_text = ""
        self._reset_hint_text = ""
        self._last_zoom_change_time_s: float | None = None
        self._present_pixel_x = 0
        self._present_pixel_y = 0
        self._text_bbox_x_offset = 0
        self._text_bbox_y_offset = 0
        self._line_height_px = 0
        self._text_bbox_width = 0
        self._text_bbox_height = 0
        self._reset_hint_bbox_x_offset = 0
        self._reset_hint_bbox_y_offset = 0
        self._reset_hint_bbox_width = 0
        self._reset_hint_bbox_height = 0

    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            transparent=True,
            font_size=config.ACTION_PANEL_FONT_SIZE,
            line_spacing=1.0,
        )

    def notify_zoom_changed(self) -> None:
        """Record a zoom-change pulse so the indicator wakes/fades appropriately."""
        self._last_zoom_change_time_s = float(self.controller.clock.last_time)
        self.invalidate()

    def _format_zoom_label(self) -> str:
        zoom = self.world_view.viewport_zoom
        zoom_str = f"{zoom:.2f}".rstrip("0").rstrip(".")
        return f"Zoom {zoom_str}x"

    def _compute_alpha(self) -> float:
        """Return current display alpha for the hybrid zoom-indicator behavior."""
        is_default = self.world_view.is_default_viewport_zoom
        now = float(self.controller.clock.last_time)

        if self._last_zoom_change_time_s is None:
            return 0.0 if is_default else self._PERSISTENT_ALPHA

        elapsed = max(0.0, now - self._last_zoom_change_time_s)
        if elapsed >= self._WAKE_DURATION_S:
            return 0.0 if is_default else self._PERSISTENT_ALPHA

        t = elapsed / self._WAKE_DURATION_S
        target_alpha = 0.0 if is_default else self._PERSISTENT_ALPHA
        return self._WAKE_ALPHA + (target_alpha - self._WAKE_ALPHA) * t

    def _show_transient_reset_hint(self) -> bool:
        """Show reset guidance only during the wake phase at non-default zoom."""
        if self.world_view.is_default_viewport_zoom:
            return False
        if self._last_zoom_change_time_s is None:
            return False
        elapsed = max(
            0.0,
            float(self.controller.clock.last_time) - self._last_zoom_change_time_s,
        )
        return elapsed < self._WAKE_DURATION_S

    def _view_screen_rect(self, view: Any) -> PixelRect | None:
        """Return a view's screen-space pixel rect, or None when not drawable."""
        if not getattr(view, "visible", False):
            return None
        width = int(getattr(view, "width", 0))
        height = int(getattr(view, "height", 0))
        if width <= 0 or height <= 0:
            return None

        left_px, top_px = self.controller.graphics.console_to_screen_coords(
            float(getattr(view, "x", 0)),
            float(getattr(view, "y", 0)),
        )
        right_px, bottom_px = self.controller.graphics.console_to_screen_coords(
            float(getattr(view, "x", 0) + width),
            float(getattr(view, "y", 0) + height),
        )
        return (int(left_px), int(top_px), int(right_px), int(bottom_px))

    @staticmethod
    def _rects_intersect(a: PixelRect, b: PixelRect) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1

    def _collect_avoid_rects(self) -> list[PixelRect]:
        """Collect visible HUD view rects that should not be covered by the chip."""
        frame_manager = getattr(self.controller, "frame_manager", None)
        if frame_manager is None:
            return []

        rects: list[PixelRect] = []
        for view_name in ("player_status_view", "equipment_view"):
            rect = self._view_screen_rect(getattr(frame_manager, view_name, None))
            if rect is not None:
                rects.append(rect)
        return rects

    def _calculate_dimensions(self) -> None:
        """Compute chip size and pixel placement inside the world viewport."""
        if not self.world_view.visible:
            self.width = self.height = 0
            self.pixel_width = self.pixel_height = 0
            return

        if self._compute_alpha() <= 0.0:
            self.width = self.height = 0
            self.pixel_width = self.pixel_height = 0
            return

        self.tile_dimensions = self.controller.graphics.tile_dimensions
        self._label_text = self._format_zoom_label()
        self._reset_hint_text = ""

        assert self.canvas is not None
        self._line_height_px = max(1, int(self.canvas.get_effective_line_height()))
        text_x_offset, text_y_offset, text_w, text_h = self.canvas.get_text_bbox(
            self._label_text
        )
        self._text_bbox_x_offset = int(text_x_offset)
        self._text_bbox_y_offset = int(text_y_offset)
        self._text_bbox_width = int(text_w)
        self._text_bbox_height = int(text_h)

        content_width = self._text_bbox_width
        content_height = self._line_height_px
        if self._show_transient_reset_hint():
            self._reset_hint_text = self._RESET_HINT_TEXT
            hint_x_off, hint_y_off, hint_w, hint_h = self.canvas.get_text_bbox(
                self._reset_hint_text
            )
            self._reset_hint_bbox_x_offset = int(hint_x_off)
            self._reset_hint_bbox_y_offset = int(hint_y_off)
            self._reset_hint_bbox_width = int(hint_w)
            self._reset_hint_bbox_height = int(hint_h)
            content_width = max(content_width, self._reset_hint_bbox_width)
            content_height += self._line_height_px
        else:
            self._reset_hint_bbox_x_offset = 0
            self._reset_hint_bbox_y_offset = 0
            self._reset_hint_bbox_width = 0
            self._reset_hint_bbox_height = 0

        self.pixel_width = (
            content_width + (self._INNER_PAD_X_PX * 2) + self._SHADOW_OFFSET_PX
        )
        self.pixel_height = (
            content_height + (self._INNER_PAD_Y_PX * 2) + self._SHADOW_OFFSET_PX
        )

        # The base TextOverlay pipeline expects tile coords, but we present this
        # overlay directly in pixel space to avoid tile quantization jitter.
        self.x_tiles = 0
        self.y_tiles = 0
        self.width = max(1, self.pixel_width // max(1, self.tile_dimensions[0]))
        self.height = max(1, self.pixel_height // max(1, self.tile_dimensions[1]))

        left_px, top_px = self.controller.graphics.console_to_screen_coords(
            self.world_view.x, self.world_view.y
        )
        right_px, _ = self.controller.graphics.console_to_screen_coords(
            self.world_view.x + self.world_view.width, self.world_view.y
        )
        _left_px2, bottom_px = self.controller.graphics.console_to_screen_coords(
            self.world_view.x, self.world_view.y + self.world_view.height
        )
        world_rect: PixelRect = (
            int(left_px),
            int(top_px),
            int(right_px),
            int(bottom_px),
        )
        chip_w = int(self.pixel_width)
        chip_h = int(self.pixel_height)

        candidates: list[tuple[int, int]] = [
            # Keep the original top-right preference when unobstructed.
            (
                world_rect[2] - chip_w - self._PADDING_X_PX,
                world_rect[1] + self._PADDING_Y_PX,
            ),
            (
                world_rect[0] + self._PADDING_X_PX,
                world_rect[1] + self._PADDING_Y_PX,
            ),
            (
                world_rect[2] - chip_w - self._PADDING_X_PX,
                world_rect[3] - chip_h - self._PADDING_Y_PX,
            ),
            (
                world_rect[0] + self._PADDING_X_PX,
                world_rect[3] - chip_h - self._PADDING_Y_PX,
            ),
        ]
        avoid_rects = self._collect_avoid_rects()

        chosen_x, chosen_y = candidates[0]
        for cand_x, cand_y in candidates:
            chip_rect: PixelRect = (cand_x, cand_y, cand_x + chip_w, cand_y + chip_h)
            if any(self._rects_intersect(chip_rect, avoid) for avoid in avoid_rects):
                continue
            chosen_x, chosen_y = cand_x, cand_y
            break

        self._present_pixel_x = chosen_x
        self._present_pixel_y = chosen_y

    def draw_content(self) -> None:
        """Draw the chip background and text (alpha is applied at present time)."""
        if not self._label_text:
            return

        assert self.canvas is not None
        text_x = self._INNER_PAD_X_PX - self._text_bbox_x_offset
        text_y = self._INNER_PAD_Y_PX - self._text_bbox_y_offset
        self.canvas.draw_rect(
            0,
            0,
            self.pixel_width - 1,
            self.pixel_height - 1,
            colors.BLACK,
            fill=True,
        )
        self.canvas.draw_rect(
            0,
            0,
            self.pixel_width - 1,
            self.pixel_height - 1,
            colors.DARK_GREY,
            fill=False,
        )
        self.canvas.draw_text(
            text_x + self._SHADOW_OFFSET_PX,
            text_y + self._SHADOW_OFFSET_PX,
            self._label_text,
            colors.BLACK,
        )
        self.canvas.draw_text(
            text_x,
            text_y,
            self._label_text,
            colors.LIGHT_GREY,
        )
        if self._reset_hint_text:
            hint_x = self._INNER_PAD_X_PX - self._reset_hint_bbox_x_offset
            hint_y = (
                self._INNER_PAD_Y_PX
                + self._line_height_px
                - self._reset_hint_bbox_y_offset
            )
            self.canvas.draw_text(
                hint_x + self._SHADOW_OFFSET_PX,
                hint_y + self._SHADOW_OFFSET_PX,
                self._reset_hint_text,
                colors.BLACK,
            )
            self.canvas.draw_text(
                hint_x,
                hint_y,
                self._reset_hint_text,
                colors.GREY,
            )

    def present(self) -> None:
        """Present the cached chip texture with a time-varying alpha fade."""
        if not self.is_active or self._cached_texture is None:
            return

        alpha = self._compute_alpha()
        if alpha <= 0.0:
            return

        self.controller.graphics.draw_texture_alpha(
            self._cached_texture,
            self._present_pixel_x,
            self._present_pixel_y,
            Opacity(alpha),
        )

    def handle_input(self, event: input_events.InputEvent) -> bool:
        _ = event
        return False
