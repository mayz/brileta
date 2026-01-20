from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from catley import colors
from catley.util.coordinates import PixelCoord, TileCoord
from catley.view.render.canvas import Canvas

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class PillowImageCanvas(Canvas):
    """Canvas that renders using PIL and uploads to an SDL texture."""

    def __init__(
        self,
        renderer: GraphicsContext,
        font_path: Path,
        transparent: bool = True,
        font_size: int | None = None,
        line_spacing: float = 1.0,
    ) -> None:
        """Initialize a Pillow-based canvas for text and shape rendering.

        Args:
            renderer: Graphics context providing tile dimensions.
            font_path: Path to TTF font file.
            transparent: Whether the canvas background is transparent.
            font_size: Explicit font size in pixels. If None, scales with tile height.
            line_spacing: Multiplier for line height (1.0 = tight, 1.3 = comfortable).
        """
        super().__init__(transparent)
        self.font_path = font_path
        self._explicit_font_size = font_size
        self._line_spacing = line_spacing
        self._current_font_size = 0
        # Temporary default font before scaling is configured.
        self.font = ImageFont.truetype(str(self.font_path), 12)
        ascent, descent = self.font.getmetrics()
        self._actual_ascent = ascent
        self._actual_descent = descent
        # Measure actual glyph height for box-drawing characters.
        bbox = self.font.getbbox("║")
        glyph_height = bbox[3] - bbox[1]
        self._effective_line_height = int(glyph_height * self._line_spacing)
        self._target_tile_height: int = 0
        self._font_needs_update: bool = True
        self.image: PILImage.Image | None = None
        self._drawer: ImageDraw.ImageDraw | None = None

        tile_height = renderer.tile_dimensions[1]

        self.configure_scaling(tile_height)

    @property
    def artifact_type(self) -> str:
        return "numpy"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Creates a backend-specific texture from this canvas's artifact."""
        return renderer.texture_from_numpy(artifact, self.transparent)

    def configure_dimensions(self, width: PixelCoord, height: PixelCoord) -> None:
        super().configure_dimensions(width, height)

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
        # Use explicit font size if provided, otherwise scale from tile height
        if self._explicit_font_size is not None:
            target_font_size = self._explicit_font_size
        else:
            # Default scaling - font size slightly larger than tile height
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

        # Measure actual height of box-drawing character for seamless connections.
        # font.getmetrics() returns declared metrics for text layout, which include
        # extra space that box-drawing characters don't use. Using getbbox() on ║
        # gives us the actual rendered glyph height.
        bbox = self.font.getbbox("║")
        glyph_height = bbox[3] - bbox[1]
        self._effective_line_height = int(glyph_height * self._line_spacing)

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

    def _prepare_for_rendering(self) -> bool:
        """Prepare PIL image for rendering operations."""
        # Create new image and drawer
        self.image = PILImage.new(
            "RGBA", (int(self.width), int(self.height)), (0, 0, 0, 0)
        )
        self._drawer = ImageDraw.Draw(self.image)
        return True

    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Render a single text operation to the PIL image."""
        if self._drawer is None:
            return

        font_to_use = self.font
        if font_size is not None:
            font_to_use = ImageFont.truetype(str(self.font_path), font_size)

        self._drawer.text(
            (int(pixel_x), int(pixel_y)),
            text,
            font=font_to_use,
            fill=(*color, 255),
            anchor="la",
        )

    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Render a single rectangle operation to the PIL image."""
        if self._drawer is None:
            return

        fill_color = color if fill else None
        outline_color = color if not fill else None
        self._drawer.rectangle(
            [int(pixel_x), int(pixel_y), int(pixel_x + width), int(pixel_y + height)],
            fill=fill_color,
            outline=outline_color,
        )

    def _render_frame_op(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Render a single frame operation to the PIL image."""
        if self._drawer is None:
            return

        # Convert tile coordinates to pixel coordinates
        px_x = int(tile_x * self.font.getbbox(" ")[2])
        px_y = int(tile_y * self._effective_line_height)
        px_width = int(width * self.font.getbbox(" ")[2])
        px_height = int(height * self._effective_line_height)

        self._drawer.rectangle(
            [px_x, px_y, px_x + px_width, px_y + px_height],
            fill=bg,
            outline=fg,
        )

    def _create_artifact_from_rendered_content(self) -> np.ndarray | None:
        """Return the numpy array as the intermediate artifact."""
        if self.image is None:
            return None

        # Convert to numpy array
        pixels = np.array(self.image, dtype=np.uint8)
        pixels = np.ascontiguousarray(pixels)

        # Cleanup
        self.image = None
        self._drawer = None

        return pixels
