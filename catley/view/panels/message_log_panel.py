from __future__ import annotations

from typing import TYPE_CHECKING

from tcod.sdl.render import Texture

from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import PillowTextBackend

from .panel import TextPanel

if TYPE_CHECKING:
    from catley.util.message_log import MessageLog


class MessageLogPanel(TextPanel):
    """Panel for displaying the message log in the bottom-left corner."""

    def __init__(self, message_log: MessageLog, renderer: Renderer) -> None:
        super().__init__()
        self.message_log = message_log
        self.text_backend = PillowTextBackend(renderer)

        # Panel pixel dimensions will be calculated when resize() is called
        self.panel_width_px = 0
        self.panel_height_px = 0

        self._cached_texture: Texture | None = None
        # store the MessageLog.revision used to build _cached_texture so we
        # know when a new texture needs to be generated
        self._render_revision = -1
        # Track the texture dimensions to detect when we need to regenerate
        self._cached_texture_width = 0
        self._cached_texture_height = 0

    def needs_redraw(self, renderer: Renderer) -> bool:
        current_tile_dimensions = renderer.tile_dimensions
        new_panel_width_px = self.width * current_tile_dimensions[0]
        new_panel_height_px = self.height * current_tile_dimensions[1]

        return (
            self._render_revision != self.message_log.revision
            or current_tile_dimensions != self.tile_dimensions
            or new_panel_width_px != self._cached_texture_width
            or new_panel_height_px != self._cached_texture_height
        )

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override resize to update pixel dimensions when panel is resized."""
        super().resize(x1, y1, x2, y2)
        # Update pixel dimensions based on new size
        self.panel_width_px = self.width * self.tile_dimensions[0]
        self.panel_height_px = self.height * self.tile_dimensions[1]

    def draw_content(self, renderer: Renderer) -> None:
        # Update cached tile dimensions and recalculate pixel dimensions
        self.tile_dimensions = renderer.tile_dimensions
        self.text_backend.configure_scaling(self.tile_dimensions[1])
        self.panel_width_px = self.width * self.tile_dimensions[0]
        self.panel_height_px = self.height * self.tile_dimensions[1]
        self._cached_texture_width = self.panel_width_px
        self._cached_texture_height = self.panel_height_px
        self.text_backend.configure_renderer(renderer.sdl_renderer)

        ascent, descent = self.text_backend.get_font_metrics()
        line_height = ascent + descent
        y_baseline = self.panel_height_px - descent

        for message in reversed(self.message_log.messages):
            if y_baseline < line_height:
                break

            wrapped_lines = self.text_backend.wrap_text(
                message.full_text, self.panel_width_px
            )
            for line in reversed(wrapped_lines):
                text_top = y_baseline - ascent
                if text_top < 0:
                    y_baseline = -1
                    break

                self.text_backend.draw_text(
                    pixel_x=0, pixel_y=text_top, text=line, color=message.fg
                )
                y_baseline -= line_height

            if y_baseline == -1:
                break

        # Update render revision now that the frame has been drawn.
        self._render_revision = self.message_log.revision
