from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.util.message_log import MessageLog


class MessageLogView(TextView):
    """View for displaying the message log in the bottom-left corner."""

    def __init__(self, message_log: MessageLog, graphics: GraphicsContext) -> None:
        super().__init__()
        self.message_log = message_log
        self.canvas = PillowImageCanvas(
            graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.MESSAGE_LOG_FONT_SIZE,
            line_spacing=1.0,
        )
        self.renderer = graphics

        # View pixel dimensions will be calculated when resize() is called
        self.view_width_px = 0
        self.view_height_px = 0

        # Override the cache from the base class,
        # as this view requires a more complex, tuple-based key.
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render",
            max_size=1,
            on_evict=lambda tex: self.renderer.release_texture(tex),
        )

    def get_cache_key(self) -> tuple:
        """The key is a combination of data revision and view dimensions."""
        current_tile_dimensions = self.renderer.tile_dimensions
        new_view_width_px = self.width * current_tile_dimensions[0]
        new_view_height_px = self.height * current_tile_dimensions[1]

        return (
            self.message_log.revision,
            current_tile_dimensions,
            new_view_width_px,
            new_view_height_px,
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update pixel dimensions when view is resized."""
        super().set_bounds(x1, y1, x2, y2)
        # Update pixel dimensions based on new size
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]
        self.canvas.configure_scaling(self.tile_dimensions[1])

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        # Update cached tile dimensions and recalculate pixel dimensions
        self.tile_dimensions = graphics.tile_dimensions
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]
        self.canvas.draw_rect(
            0, 0, self.view_width_px, self.view_height_px, colors.BLACK, fill=True
        )

        # Padding from edges
        x_padding = 8
        bottom_padding = 8

        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent
        y_baseline = self.view_height_px - descent - bottom_padding

        # Available width for text (accounting for left and right padding)
        text_width = self.view_width_px - (x_padding * 2)

        for message in reversed(self.message_log.messages):
            if y_baseline < line_height:
                break

            wrapped_lines = self.canvas.wrap_text(message.full_text, text_width)
            for line in reversed(wrapped_lines):
                text_top = y_baseline - ascent
                if text_top < 0:
                    y_baseline = -1
                    break

                self.canvas.draw_text(
                    pixel_x=x_padding, pixel_y=text_top, text=line, color=message.fg
                )
                y_baseline -= line_height

            if y_baseline == -1:
                break
