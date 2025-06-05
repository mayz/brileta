from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tcod.sdl.render import BlendMode, Texture

from catley import config
from catley.render.renderer import Renderer

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

        self.font = ImageFont.truetype(
            str(config.MESSAGE_LOG_FONT_PATH), config.MESSAGE_LOG_FONT_SIZE
        )
        ascent, descent = self.font.getmetrics()
        self.line_height_px = ascent + descent

        self.panel_width_px = width * self.tile_dimensions[0]
        self.panel_height_px = height * self.tile_dimensions[1]

        self._cached_texture: Texture | None = None
        # store the MessageLog.revision used to build _cached_texture so we
        # know when a new texture needs to be generated
        self._render_revision = -1

    def draw(self, renderer: Renderer) -> None:
        if not self.visible:
            return

        if self._render_revision == self.message_log.revision:
            return

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

        dest_rect = (
            self.x * self.tile_dimensions[0],
            self.y * self.tile_dimensions[1],
            self.panel_width_px,
            self.panel_height_px,
        )

        renderer.sdl_renderer.copy(self._cached_texture, dest=dest_rect)

    def _render_messages_to_image(self) -> PILImage.Image:
        img = PILImage.new(
            "RGBA", (self.panel_width_px, self.panel_height_px), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(img)

        # Start with the baseline for drawing at the very bottom of the panel.
        y_baseline = self.panel_height_px

        for message in reversed(self.message_log.messages):
            # If the next message would start off the top of the panel, stop.
            if y_baseline < 0:
                break

            wrapped_lines = self._wrap_text(message.full_text)
            for line in reversed(wrapped_lines):
                # We need to check if drawing this line would go off-screen.
                # If the baseline for the *top* of this line is less than 0, break.
                if y_baseline - self.line_height_px < -self.line_height_px / 2:
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
