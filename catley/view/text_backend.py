"""Abstractions for text rendering backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import tcod.sdl.render
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from tcod.console import Console
from tcod.sdl.render import BlendMode, Texture

from catley import colors
from catley.constants.view import ViewConstants as View


class TextBackend(ABC):
    """Abstract interface for text rendering backends."""

    def __init__(self) -> None:
        self.drawing_offset_x = 0
        self.drawing_offset_y = 0

    @abstractmethod
    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Draw text at ``(x, y)`` using ``color``."""

    @abstractmethod
    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        """Return ``(width, height, line_height)`` for ``text``."""

    @abstractmethod
    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        """Wrap ``text`` into lines that fit within ``max_width`` pixels."""

    @abstractmethod
    def begin_frame(self) -> None:
        """Prepare backend for a new frame."""

    @abstractmethod
    def end_frame(self) -> Texture | None:
        """Finalize the frame and return a texture if one was produced."""

    # ------------------------------------------------------------------
    # Optional configuration helpers
    # ------------------------------------------------------------------
    def configure_dimensions(self, width: int, height: int) -> None:
        """Configure canvas dimensions. Default implementation does nothing."""
        _ = width
        _ = height

    def configure_scaling(self, tile_height: int) -> None:
        """Configure scaling based on tile height."""
        # Default implementation does nothing.
        _ = tile_height

    def configure_renderer(self, sdl_renderer) -> None:
        """Configure SDL renderer. Default implementation does nothing."""
        _ = sdl_renderer

    def configure_drawing_offset(self, offset_x: int, offset_y: int) -> None:
        """Configure coordinate translation for backends."""
        self.drawing_offset_x = offset_x
        self.drawing_offset_y = offset_y

    def get_font_metrics(self) -> tuple[int, int]:
        """Return ``(ascent, descent)``. Default implementation returns ``(0, 0)``."""
        return (0, 0)


class TCODTextBackend(TextBackend):
    """Text backend that draws directly to a tcod :class:`Console`."""

    def __init__(self, console: Console, tile_dimensions: tuple[int, int]) -> None:
        super().__init__()
        self.console = console
        self.tile_width, self.tile_height = tile_dimensions

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        _ = font_size
        absolute_x = x + self.drawing_offset_x
        absolute_y = y + self.drawing_offset_y
        self.console.print(x=absolute_x, y=absolute_y, text=text, fg=color)

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        _ = font_size
        width = len(text) * self.tile_width
        height = self.tile_height
        return width, height, self.tile_height

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        _ = font_size
        chars_per_line = max_width // self.tile_width
        if chars_per_line <= 0:
            return [text]
        return (
            text.splitlines()
            if len(text) <= chars_per_line
            else [
                text[i : i + chars_per_line]
                for i in range(0, len(text), chars_per_line)
            ]
        )

    def configure_dimensions(self, width: int, height: int) -> None:
        """No-op for TCOD backend."""
        _ = width
        _ = height

    def begin_frame(self) -> None:  # pragma: no cover - no work needed
        pass

    def end_frame(self) -> Texture | None:  # pragma: no cover - no work needed
        return None


class PillowTextBackend(TextBackend):
    """Text backend that renders using PIL and uploads to an SDL texture."""

    def __init__(
        self,
        font_path: Path,
        tile_height: int,
        sdl_renderer,
    ) -> None:
        super().__init__()
        self.font_path = Path(font_path)
        self.base_tile_height = View.MESSAGE_LOG_MIN_FONT_SIZE
        self.tile_height = self.base_tile_height
        self._current_font_size = 0
        self.font = ImageFont.truetype(str(self.font_path), self.base_tile_height)
        ascent, descent = self.font.getmetrics()
        self.line_height_px = ascent + descent
        self.image: PILImage.Image | None = None
        self._drawer: ImageDraw.ImageDraw | None = None
        self.sdl_renderer: tcod.sdl.render.Renderer | None = None
        self.width = 0
        self.height = 0

        self.configure_scaling(tile_height)
        self.configure_renderer(sdl_renderer)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def configure_dimensions(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def configure_scaling(self, tile_height: int) -> None:
        self.tile_height = tile_height
        self._update_font_for_tile_height(tile_height)

    def configure_renderer(self, sdl_renderer) -> None:
        self.sdl_renderer = sdl_renderer

    def get_font_metrics(self) -> tuple[int, int]:
        return self.font.getmetrics()

    def _update_font_for_tile_height(self, tile_height: int) -> None:
        scale = tile_height / self.base_tile_height
        font_size = max(
            View.MESSAGE_LOG_MIN_FONT_SIZE,
            round(View.MESSAGE_LOG_MIN_FONT_SIZE * scale),
        )
        if font_size == self._current_font_size:
            return
        self.font = ImageFont.truetype(str(self.font_path), font_size)
        ascent, descent = self.font.getmetrics()
        self.line_height_px = ascent + descent
        self._current_font_size = font_size

    def begin_frame(self) -> None:
        self.image = PILImage.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._drawer = ImageDraw.Draw(self.image)

    def draw_text(
        self,
        x: int,
        y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Draw text using relative coordinates with reasonable defaults for UI text."""
        if self._drawer is None:
            return

        # If no font size specified, use a reasonable default based on tile height
        if font_size is None:
            font_size = max(12, self.tile_height)  # Reasonable UI font size

        font = ImageFont.truetype(str(self.font_path), font_size)

        # Adjust Y position to account for baseline positioning
        # Add font size to Y to prevent clipping at top
        adjusted_y = y + font_size if y < font_size else y

        self._drawer.text(
            (x, adjusted_y), text, font=font, fill=(*color, 255), anchor="ls"
        )

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        font = (
            self.font
            if font_size is None
            else ImageFont.truetype(str(self.font_path), font_size)
        )
        bbox = font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        ascent, descent = font.getmetrics()
        return int(width), int(height), ascent + descent

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        font = (
            self.font
            if font_size is None
            else ImageFont.truetype(str(self.font_path), font_size)
        )
        words = text.split()
        lines: list[str] = []
        current_line = ""
        for word in words:
            test = word if not current_line else f"{current_line} {word}"
            if font.getbbox(test)[2] - font.getbbox(test)[0] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def end_frame(self) -> Texture | None:
        if self.image is None or self.sdl_renderer is None:
            return None
        pixels = np.array(self.image, dtype=np.uint8)
        pixels = np.ascontiguousarray(pixels)
        texture = self.sdl_renderer.upload_texture(pixels)
        texture.blend_mode = BlendMode.BLEND
        self.image = None
        self._drawer = None
        return texture
