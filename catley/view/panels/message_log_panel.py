from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tcod.sdl.render import BlendMode, Texture

from catley import config
from catley.view.render.renderer import Renderer

from .panel import Panel

if TYPE_CHECKING:
    from catley.util.message_log import MessageLog


class MessageLogPanel(Panel):
    """Panel for displaying the message log in the bottom-left corner."""

    def __init__(
        self,
        message_log: MessageLog,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        tile_dimensions: tuple[int, int],
    ) -> None:
        super().__init__(x, y, width, height)
        self.message_log = message_log
        self.tile_dimensions = tile_dimensions

        self.base_tile_height = tile_dimensions[1]
        self._current_font_size = 0
        self._update_font_for_tile_height(tile_dimensions[1])

        self.panel_width_px = width * self.tile_dimensions[0]
        self.panel_height_px = height * self.tile_dimensions[1]

        self._cached_texture: Texture | None = None
        # store the MessageLog.revision used to build _cached_texture so we
        # know when a new texture needs to be generated
        self._render_revision = -1
        # Track the texture dimensions to detect when we need to regenerate
        self._cached_texture_width = 0
        self._cached_texture_height = 0

    def _update_font_for_tile_height(self, tile_height: int) -> None:
        """Set font and line metrics for the given tile height."""
        scale = tile_height / self.base_tile_height
        font_size = max(8, round(config.MESSAGE_LOG_FONT_SIZE * scale))
        if font_size == self._current_font_size:
            return

        self.font = ImageFont.truetype(str(config.MESSAGE_LOG_FONT_PATH), font_size)
        ascent, descent = self.font.getmetrics()
        self.line_height_px = ascent + descent
        self._current_font_size = font_size

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
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
        self._update_font_for_tile_height(self.tile_dimensions[1])
        self.panel_width_px = new_panel_width_px
        self.panel_height_px = new_panel_height_px
        self._cached_texture_width = new_panel_width_px
        self._cached_texture_height = new_panel_height_px

        image = self._render_messages_to_image()
        pixels_rgba = np.array(image, dtype=np.uint8)
        pixels_rgba = np.ascontiguousarray(pixels_rgba)

        texture = renderer.sdl_renderer.upload_texture(pixels_rgba)
        texture.blend_mode = BlendMode.BLEND
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

    def _render_messages_to_image(self) -> PILImage.Image:
        img = PILImage.new(
            "RGBA", (self.panel_width_px, self.panel_height_px), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(img)

        # Start with the baseline positioned correctly
        # PIL uses top-down coordinates (Y=0 at top)
        # We want messages to appear from bottom up, so start at bottom and work up
        ascent, descent = self.font.getmetrics()
        # Start at the bottom with enough room for the text
        y_baseline = self.panel_height_px - descent
        lines_drawn = 0

        for message in reversed(self.message_log.messages):
            # If the next message would start off the top of the panel, stop.
            if y_baseline < self.line_height_px:
                break

            wrapped_lines = self._wrap_text(message.full_text)

            for line in reversed(wrapped_lines):
                # We need to check if drawing this line would go off-screen.
                # Check if the top of the text would be above the panel
                text_top = y_baseline - ascent
                if text_top < 0:
                    y_baseline = -1  # Signal to break outer loop
                    break

                # Draw the text using the baseline as the anchor.
                # "ls" means the (x, y) coordinate is the Left side of the baSeline.
                draw.text(
                    (0, y_baseline),
                    line,
                    font=self.font,
                    fill=(*message.fg, 255),
                    anchor="ls",
                )
                lines_drawn += 1

                # Move the baseline up for the next line of text.
                y_baseline -= self.line_height_px

            if y_baseline == -1:
                break

        return img

    def _wrap_text(self, text: str) -> list[str]:
        # A simple estimate for character width. '8' is a reasonable guess
        # for a 16pt font. A more accurate way would be to measure an average character
        # with the font object.
        avg_char_width = 8

        # Calculate how many characters can fit in the panel's pixel width.
        wrap_at_character_count = self.panel_width_px // avg_char_width

        return textwrap.wrap(text, wrap_at_character_count)
