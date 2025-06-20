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

from catley import colors, config

from .renderer import Renderer


class TextBackend(ABC):
    """Abstract interface for text rendering backends."""

    def __init__(self) -> None:
        self.drawing_offset_x = 0
        self.drawing_offset_y = 0
        self._last_tile_height: int = 0

    @abstractmethod
    def draw_text(
        self,
        pixel_x: int,
        pixel_y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Draw a line of text whose top-left corner is at ``(pixel_x, pixel_y)``
        using ``color``."""

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

    @abstractmethod
    def draw_frame(
        self,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Draw a framed box, using tile coordinates."""

    @abstractmethod
    def draw_rect(
        self,
        pixel_x: int,
        pixel_y: int,
        width: int,
        height: int,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Draw a rectangle at the given pixel coordinates."""

    # ------------------------------------------------------------------
    # Optional configuration helpers
    # ------------------------------------------------------------------
    def configure_dimensions(self, width: int, height: int) -> None:
        if self.width != width or self.height != height:
            # Dimensions changed - clear cache
            self._cached_frame_texture = None  # Let GC handle cleanup
            self._last_frame_ops = []

        self.width = width
        self.height = height

    def configure_scaling(self, tile_height: int) -> None:
        """Configure scaling based on tile height."""
        if self._last_tile_height != tile_height:
            self._last_tile_height = tile_height
            self._update_scaling_internal(tile_height)

    @abstractmethod
    def _update_scaling_internal(self, tile_height: int) -> None:
        """Backend-specific scaling logic."""

    @abstractmethod
    def get_effective_line_height(self) -> int:
        """Return the line height currently used for text rendering."""

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

    def __init__(self, renderer: Renderer) -> None:
        super().__init__()
        self.renderer = renderer
        self.private_console: Console | None = None
        self.width = 0
        self.height = 0

        # Caching logic
        self._frame_ops: list = []  # Defer drawing operations
        self._last_frame_ops: list = []
        self._cached_frame_texture: Texture | None = None

        self.configure_scaling(renderer.tile_dimensions[1])

    def draw_text(
        self,
        pixel_x: int,
        pixel_y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Record a text drawing operation to be rendered later."""
        _ = font_size
        self._frame_ops.append(("text", pixel_x, pixel_y, text, color))

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        _ = font_size
        tile_width, _ = self.renderer.tile_dimensions
        width = len(text) * tile_width
        line_height = self.get_effective_line_height()
        return width, line_height, line_height

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        _ = font_size
        tile_width, _ = self.renderer.tile_dimensions
        if tile_width == 0:
            return [text]  # Avoid division by zero
        chars_per_line = max_width // tile_width
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

    def draw_frame(
        self,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Record a frame drawing operation."""
        self._frame_ops.append(("frame", tile_x, tile_y, width, height, fg, bg))

    def draw_rect(
        self,
        pixel_x: int,
        pixel_y: int,
        width: int,
        height: int,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Record a rectangle drawing operation."""
        self._frame_ops.append(("rect", pixel_x, pixel_y, width, height, color, fill))

    def _update_scaling_internal(
        self, tile_height: int
    ) -> None:  # pragma: no cover - nothing to do
        _ = tile_height

    def get_effective_line_height(self) -> int:
        return self._last_tile_height

    def get_font_metrics(self) -> tuple[int, int]:
        line_height = self.get_effective_line_height()
        ascent = int(line_height * 0.8)
        descent = line_height - ascent
        return ascent, descent

    def configure_dimensions(self, width: int, height: int) -> None:
        """Create a private console based on pixel dimensions."""
        # Call parent first to set width/height attributes
        super().configure_dimensions(width, height)

        tile_w, tile_h = self.renderer.tile_dimensions
        if not tile_w or not tile_h:
            return

        width_tiles = width // tile_w
        height_tiles = height // tile_h

        if (
            not self.private_console
            or self.private_console.width != width_tiles
            or self.private_console.height != height_tiles
        ):
            self.private_console = Console(width_tiles, height_tiles, order="F")
            # New console means cache is invalid
            self._cached_frame_texture = None
            self._last_frame_ops = []

    def begin_frame(self) -> None:
        self._frame_ops = []

    def end_frame(self) -> Texture | None:
        # If content hasn't changed, return the cached texture
        if self._frame_ops == self._last_frame_ops and self._cached_frame_texture:
            return self._cached_frame_texture

        if not self.private_console:
            return None

        # Content has changed, so re-render
        self.private_console.clear()

        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return None

        for op in self._frame_ops:
            op_type = op[0]
            if op_type == "text":
                _, pixel_x, pixel_y, text, color = op
                tile_x = pixel_x // tile_width
                tile_y = pixel_y // tile_height
                self.private_console.print(x=tile_x, y=tile_y, text=text, fg=color)
            elif op_type == "frame":
                _, tile_x, tile_y, width, height, fg, bg = op
                self.private_console.draw_frame(
                    x=tile_x,
                    y=tile_y,
                    width=width,
                    height=height,
                    title="",
                    clear=False,
                    fg=fg,
                    bg=bg,
                )
            elif op_type == "rect":
                _, pixel_x, pixel_y, width, height, color, fill = op
                start_tx = pixel_x // tile_width
                start_ty = pixel_y // tile_height
                end_tx = (pixel_x + width) // tile_width
                end_ty = (pixel_y + height) // tile_height
                for tx in range(start_tx, end_tx):
                    for ty in range(start_ty, end_ty):
                        if fill:
                            self.private_console.bg[tx, ty] = color
                        else:
                            if tx in (start_tx, end_tx - 1) or ty in (
                                start_ty,
                                end_ty - 1,
                            ):
                                self.private_console.bg[tx, ty] = color

        texture = self.renderer.console_render.render(self.private_console)
        texture.blend_mode = BlendMode.ADD

        # Cache the new texture and the operations that created it
        self._cached_frame_texture = texture
        self._last_frame_ops = self._frame_ops.copy()

        return texture


class PillowTextBackend(TextBackend):
    """Text backend that renders using PIL and uploads to an SDL texture."""

    def __init__(
        self,
        renderer: Renderer,
    ) -> None:
        super().__init__()
        self.font_path = Path(config.MESSAGE_LOG_FONT_PATH)
        self._current_font_size = 0
        # Temporary default font before scaling is configured.
        self.font = ImageFont.truetype(str(self.font_path), 12)
        ascent, descent = self.font.getmetrics()
        self._actual_ascent = ascent
        self._actual_descent = descent
        self._effective_line_height = ascent + descent
        self._target_tile_height: int = 0
        self._font_needs_update: bool = True
        self.image: PILImage.Image | None = None
        self._drawer: ImageDraw.ImageDraw | None = None
        self.sdl_renderer: tcod.sdl.render.Renderer | None = None
        self.width = 0
        self.height = 0

        self._frame_ops = []  # [(x, y, text, color, font_size), ...]
        self._cached_frame_texture: Texture | None = None
        self._last_frame_ops: list = []

        tile_height = renderer.tile_dimensions[1]

        self.configure_scaling(tile_height)
        self.configure_renderer(renderer.sdl_renderer)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def configure_dimensions(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def configure_renderer(self, sdl_renderer) -> None:
        self.sdl_renderer = sdl_renderer

    def get_font_metrics(self) -> tuple[int, int]:
        ascent = self._actual_ascent
        descent = self._actual_descent
        total = ascent + descent
        if total == 0:
            return (0, 0)
        if total == self._effective_line_height:
            return ascent, descent
        scale = self._effective_line_height / total
        scaled_ascent = round(ascent * scale)
        scaled_descent = self._effective_line_height - scaled_ascent
        return scaled_ascent, scaled_descent

    def _update_scaling_internal(self, tile_height: int) -> None:
        self._target_tile_height = tile_height
        self._update_font_for_tile_height(tile_height)
        self._font_needs_update = False

    def get_effective_line_height(self) -> int:
        return self._effective_line_height

    def _update_font_for_tile_height(self, tile_height: int) -> None:
        # Use simple proportional scaling - works reliably with any font
        # Font size should be slightly larger than tile height for good
        # visual appearance
        target_font_size = max(8, round(tile_height * 1.1))

        try:
            self.font = ImageFont.truetype(str(self.font_path), target_font_size)
            self._current_font_size = target_font_size
        except OSError:
            # Fallback to minimum size if font loading fails
            self.font = ImageFont.truetype(str(self.font_path), 8)
            self._current_font_size = 8

        ascent, descent = self.font.getmetrics()
        self._actual_ascent = ascent
        self._actual_descent = descent
        self._effective_line_height = tile_height  # For layout consistency

    def begin_frame(self) -> None:
        self._frame_ops = []  # Reset frame text list

    def draw_text(
        self,
        pixel_x: int,
        pixel_y: int,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        # Just record what text to draw - don't render yet
        self._frame_ops.append(("text", pixel_x, pixel_y, text, color, font_size))

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
        return int(width), int(height), self._effective_line_height

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

    def draw_rect(
        self,
        pixel_x: int,
        pixel_y: int,
        width: int,
        height: int,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Record a rectangle drawing operation."""
        self._frame_ops.append(("rect", pixel_x, pixel_y, width, height, color, fill))

    def draw_frame(
        self,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Draw a frame using Pillow drawing methods. Coordinates are in tiles."""
        px_x = tile_x * self.font.getbbox(" ")[2]
        px_y = tile_y * self._effective_line_height
        px_width = width * self.font.getbbox(" ")[2]
        px_height = height * self._effective_line_height
        self._frame_ops.append(("frame", px_x, px_y, px_width, px_height, fg, bg))

    def end_frame(self) -> Texture | None:
        if self.sdl_renderer is None:
            return None

        # Check if frame content changed
        if self._frame_ops == self._last_frame_ops and self._cached_frame_texture:
            return self._cached_frame_texture

        # Content changed - need to re-render
        self._cached_frame_texture = None  # Let GC handle cleanup

        # Create new image and render all text
        self.image = PILImage.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self._drawer = ImageDraw.Draw(self.image)

        for op in self._frame_ops:
            op_type = op[0]
            if op_type == "text":
                _, pixel_x, pixel_y, text, color, font_size = op
                self._render_single_text(pixel_x, pixel_y, text, color, font_size)
            elif op_type == "rect":
                _, pixel_x, pixel_y, width, height, color, fill = op
                fill_color = color if fill else None
                outline_color = color if not fill else None
                self._drawer.rectangle(
                    [pixel_x, pixel_y, pixel_x + width, pixel_y + height],
                    fill=fill_color,
                    outline=outline_color,
                )
            elif op_type == "frame":
                _, px_x, px_y, px_width, px_height, fg, bg = op
                self._drawer.rectangle(
                    [px_x, px_y, px_x + px_width, px_y + px_height],
                    fill=bg,
                    outline=fg,
                )

        # Convert to texture
        pixels = np.array(self.image, dtype=np.uint8)
        pixels = np.ascontiguousarray(pixels)
        texture = self.sdl_renderer.upload_texture(pixels)
        texture.blend_mode = BlendMode.BLEND

        # Cache the result
        self._cached_frame_texture = texture
        self._last_frame_ops = self._frame_ops.copy()

        # Cleanup
        self.image = None
        self._drawer = None

        return texture

    def _render_single_text(self, pixel_x, pixel_y, text, color, font_size):
        """Render a single piece of text (extracted from old draw_text)."""
        if self._drawer is None:
            return

        font_to_use = self.font
        if font_size is not None:
            font_to_use = ImageFont.truetype(str(self.font_path), font_size)

        ascent, descent = font_to_use.getmetrics()
        top_padding = (self._effective_line_height - (ascent + descent)) // 2
        baseline_y = pixel_y + top_padding + ascent

        self._drawer.text(
            (pixel_x, baseline_y),
            text,
            font=font_to_use,
            fill=(*color, 255),
            anchor="ls",
        )
