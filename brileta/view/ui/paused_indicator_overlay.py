"""Bottom-center "Paused" indicator overlay for the world view.

A passive, non-interactive chip that appears at the bottom center of the world
viewport while the game is paused (``controller.paused``). It renders nothing
when the game is running.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors, config, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.types import Opacity
from brileta.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.view.render.canvas import Canvas
    from brileta.view.views.world_view import WorldView


class PausedIndicatorOverlay(TextOverlay):
    """Passive chip showing "Paused" at the bottom center of the world view."""

    _LABEL_TEXT = "Paused"
    _INNER_PAD_X_PX = 12
    _INNER_PAD_Y_PX = 6
    _BOTTOM_MARGIN_PX = 16
    _SHADOW_OFFSET_PX = 1
    _ALPHA = 0.9

    def __init__(self, controller: Controller, world_view: WorldView) -> None:
        super().__init__(controller)
        self.is_interactive = False
        self.world_view = world_view
        self._present_pixel_x = 0
        self._present_pixel_y = 0
        self._text_bbox_x_offset = 0
        self._text_bbox_y_offset = 0
        self._line_height_px = 0

    def _get_backend(self) -> Canvas:
        return PillowImageCanvas(
            self.controller.graphics,
            font_path=config.UI_FONT_PATH,
            transparent=True,
            font_size=config.ACTION_PANEL_FONT_SIZE,
            line_spacing=1.0,
        )

    def _banner_suppressed(self) -> bool:
        """Whether to hide the chip this frame.

        Shown for a *deliberate* pause, hidden only when the pause is redundant:
        when the world is not actually running, or when an overlay that freezes
        the sim itself (e.g. a conversation) is up - that overlay is its own
        "world frozen" signal. A manual space-pause under a non-freezing overlay
        (dev console, help) still shows the banner, since nothing else conveys it.
        """
        if not self.controller.paused or not self.world_view.visible:
            return True
        overlay_system = getattr(self.controller, "overlay_system", None)
        if overlay_system is None:
            return False
        return any(o.freezes_sim for o in overlay_system.active_overlays)

    def _calculate_dimensions(self) -> None:
        """Compute chip size and pixel placement inside the world viewport."""
        # Hidden entirely unless a deliberate pause is in effect.
        if self._banner_suppressed():
            self.width = self.height = 0
            self.pixel_width = self.pixel_height = 0
            return

        self.tile_dimensions = self.controller.graphics.tile_dimensions

        assert self.canvas is not None
        self._line_height_px = max(1, int(self.canvas.get_effective_line_height()))
        x_off, y_off, text_w, _text_h = self.canvas.get_text_bbox(self._LABEL_TEXT)
        self._text_bbox_x_offset = int(x_off)
        self._text_bbox_y_offset = int(y_off)

        self.pixel_width = (
            int(text_w) + (self._INNER_PAD_X_PX * 2) + self._SHADOW_OFFSET_PX
        )
        self.pixel_height = (
            self._line_height_px + (self._INNER_PAD_Y_PX * 2) + self._SHADOW_OFFSET_PX
        )

        # Present directly in pixel space to avoid tile quantization jitter.
        self.x_tiles = 0
        self.y_tiles = 0
        self.width = max(1, self.pixel_width // max(1, self.tile_dimensions[0]))
        self.height = max(1, self.pixel_height // max(1, self.tile_dimensions[1]))

        # Center horizontally, sit just above the world view's bottom edge.
        left_px, _top_px = self.controller.graphics.console_to_screen_coords(
            self.world_view.x, self.world_view.y
        )
        right_px, bottom_px = self.controller.graphics.console_to_screen_coords(
            self.world_view.x + self.world_view.width,
            self.world_view.y + self.world_view.height,
        )
        self._present_pixel_x = int((left_px + right_px) / 2 - self.pixel_width / 2)
        self._present_pixel_y = int(
            bottom_px - self.pixel_height - self._BOTTOM_MARGIN_PX
        )

    def draw_content(self) -> None:
        """Draw the chip background and label."""
        assert self.canvas is not None
        text_x = self._INNER_PAD_X_PX - self._text_bbox_x_offset
        text_y = self._INNER_PAD_Y_PX - self._text_bbox_y_offset
        self.canvas.draw_rect(
            0, 0, self.pixel_width - 1, self.pixel_height - 1, colors.BLACK, fill=True
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
            self._LABEL_TEXT,
            colors.BLACK,
        )
        self.canvas.draw_text(text_x, text_y, self._LABEL_TEXT, colors.YELLOW)

    def present(self) -> None:
        """Present the cached chip texture at the bottom center of the view."""
        if not self.is_active or self._cached_texture is None:
            return
        if self._banner_suppressed():
            return
        self.controller.graphics.draw_texture_alpha(
            self._cached_texture,
            self._present_pixel_x,
            self._present_pixel_y,
            Opacity(self._ALPHA),
        )

    def handle_input(self, event: input_events.InputEvent) -> bool:
        _ = event
        return False
