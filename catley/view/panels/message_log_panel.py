from __future__ import annotations

from typing import TYPE_CHECKING

from tcod.sdl.render import Texture

from catley.view.renderer import Renderer
from catley.view.text_backend import TextBackend

from .panel import Panel

if TYPE_CHECKING:
    from catley.util.message_log import MessageLog


class MessageLogPanel(Panel):
    """Panel for displaying the message log in the bottom-left corner."""

    def __init__(
        self,
        message_log: MessageLog,
        *,
        tile_dimensions: tuple[int, int],
        text_backend: TextBackend | None = None,
    ) -> None:
        super().__init__(text_backend)
        self.message_log = message_log
        self.tile_dimensions = tile_dimensions

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

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override resize to update pixel dimensions when panel is resized."""
        super().resize(x1, y1, x2, y2)
        # Update pixel dimensions based on new size
        self.panel_width_px = self.width * self.tile_dimensions[0]
        self.panel_height_px = self.height * self.tile_dimensions[1]

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        if not self.text_backend:
            return

        # Check if tile dimensions have changed (window resize) or content changed
        current_tile_dimensions = renderer.tile_dimensions
        new_panel_width_px = self.width * current_tile_dimensions[0]
        new_panel_height_px = self.height * current_tile_dimensions[1]

        # Check if we need to regenerate texture
        needs_regeneration = (
            self._render_revision != self.message_log.revision
            or current_tile_dimensions != self.tile_dimensions
            or new_panel_width_px != self._cached_texture_width
            or new_panel_height_px != self._cached_texture_height
        )

        if not needs_regeneration:
            return

        # Update cached tile dimensions and recalculate pixel dimensions
        self.tile_dimensions = current_tile_dimensions
        self.text_backend.configure_scaling(self.tile_dimensions[1])
        self.panel_width_px = new_panel_width_px
        self.panel_height_px = new_panel_height_px
        self._cached_texture_width = new_panel_width_px
        self._cached_texture_height = new_panel_height_px
        self.text_backend.configure_dimensions(
            self.panel_width_px, self.panel_height_px
        )
        self.text_backend.configure_renderer(renderer.sdl_renderer)
        self.text_backend.begin_frame()

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

                self.text_backend.draw_text(0, y_baseline, line, message.fg)
                y_baseline -= line_height

            if y_baseline == -1:
                break

        texture = self.text_backend.end_frame()
        if texture is not None:
            self._cached_texture = texture
            self._render_revision = self.message_log.revision

    def present(self, renderer: Renderer) -> None:
        if not self.visible or self._cached_texture is None:
            return

        # Use current tile dimensions from renderer instead of cached ones
        current_tile_dimensions = renderer.tile_dimensions

        # Calculate destination rectangle
        dest_x = self.x * current_tile_dimensions[0]
        dest_y = self.y * current_tile_dimensions[1]
        dest_width = self.width * current_tile_dimensions[0]
        dest_height = self.height * current_tile_dimensions[1]

        dest_rect = (dest_x, dest_y, dest_width, dest_height)
        renderer.sdl_renderer.copy(self._cached_texture, dest=dest_rect)
