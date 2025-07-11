"""WGPUCanvas - WGPU canvas implementation extending Canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors
from catley.util.coordinates import PixelCoord, TileCoord
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.canvas import Canvas

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class WGPUCanvas(Canvas):
    """WGPU canvas implementation extending base Canvas."""

    def __init__(self, renderer: GraphicsContext, transparent: bool = True) -> None:
        """Initialize WGPU canvas.

        Args:
            renderer: Graphics context (WGPUGraphicsContext)
            transparent: Whether the canvas should support transparency
        """
        super().__init__(transparent)
        self.renderer = renderer
        self.private_glyph_buffer: GlyphBuffer | None = None

        # Configure scaling based on renderer tile dimensions
        self.configure_scaling(renderer.tile_dimensions[1])

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        """Return text metrics (width, height, line_height)."""
        # For console-based rendering, each character is one tile
        tile_width, tile_height = self.renderer.tile_dimensions

        width = len(text) * tile_width
        height = tile_height
        line_height = tile_height

        return width, height, line_height

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        """Wrap text to fit within max_width pixels."""
        tile_width, _ = self.renderer.tile_dimensions
        max_chars_per_line = max_width // tile_width

        if max_chars_per_line <= 0:
            return [text]  # Can't wrap, return as is

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, break it
                    while len(word) > max_chars_per_line:
                        lines.append(word[:max_chars_per_line])
                        word = word[max_chars_per_line:]
                    current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [""]

    @property
    def artifact_type(self) -> str:
        """Return type of artifact this canvas produces."""
        return "glyph_buffer"

    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Create a WGPU texture from this canvas's artifact."""
        if not isinstance(artifact, GlyphBuffer):
            return None

        # Use the renderer's render_glyph_buffer_to_texture method
        return renderer.render_glyph_buffer_to_texture(artifact)

    def _update_scaling_internal(self, tile_height: int) -> None:
        """Backend-specific scaling logic."""
        # For console rendering, scaling is handled by the tile dimensions
        # No additional scaling needed in the canvas itself
        pass

    def get_effective_line_height(self) -> int:
        """Return the line height currently used for text rendering."""
        return self.renderer.tile_dimensions[1]

    def _prepare_for_rendering(self) -> bool:
        """Prepare the backend for rendering operations."""
        if self.width <= 0 or self.height <= 0:
            return False

        # Calculate buffer dimensions in tiles - match ModernGL exactly
        tile_width, tile_height = self.renderer.tile_dimensions
        buffer_width = int(self.width // tile_width)
        buffer_height = int(self.height // tile_height)

        # Create or resize the glyph buffer
        if (
            self.private_glyph_buffer is None
            or self.private_glyph_buffer.width != buffer_width
            or self.private_glyph_buffer.height != buffer_height
        ):
            self.private_glyph_buffer = GlyphBuffer(buffer_width, buffer_height)

        # Clear the buffer
        if self.transparent:
            self.private_glyph_buffer.clear(
                ch=ord(" "), fg=(0, 0, 0, 0), bg=(0, 0, 0, 0)
            )
        else:
            self.private_glyph_buffer.clear(
                ch=ord(" "), fg=(255, 255, 255, 255), bg=(0, 0, 0, 255)
            )

        return True

    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Render a single text operation."""
        if not self.private_glyph_buffer:
            return

        # Convert pixel coordinates to tile coordinates
        tile_width, tile_height = self.renderer.tile_dimensions
        tile_x = int(pixel_x // tile_width)
        tile_y = int(pixel_y // tile_height)

        # Convert color to RGBA
        fg_color = (color[0], color[1], color[2], 255)
        bg_color = (0, 0, 0, 0) if self.transparent else (0, 0, 0, 255)

        # Draw the text to the glyph buffer
        self.private_glyph_buffer.print(tile_x, tile_y, text, fg_color, bg_color)

    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Render a single rectangle operation."""
        if not self.private_glyph_buffer:
            return

        # Convert pixel coordinates to tile coordinates
        tile_width, tile_height = self.renderer.tile_dimensions
        tile_x = int((pixel_x + self.drawing_offset_x) // tile_width)
        tile_y = int((pixel_y + self.drawing_offset_y) // tile_height)
        tile_w = max(1, int(width // tile_width))
        tile_h = max(1, int(height // tile_height))

        # Convert color to RGBA
        fg_color = (color[0], color[1], color[2], 255)
        bg_color = fg_color if fill else (0, 0, 0, 0)

        # Draw rectangle using block characters
        char = ord("█") if fill else ord("□")

        for y in range(tile_h):
            for x in range(tile_w):
                if fill or x == 0 or x == tile_w - 1 or y == 0 or y == tile_h - 1:
                    self.private_glyph_buffer.put_char(
                        tile_x + x, tile_y + y, char, fg_color, bg_color
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
        """Render a single frame operation."""
        if not self.private_glyph_buffer:
            return

        # Convert colors to RGBA
        fg_color = (fg[0], fg[1], fg[2], 255)
        bg_color = (bg[0], bg[1], bg[2], 255)

        # Draw the frame using the glyph buffer's draw_frame method
        self.private_glyph_buffer.draw_frame(
            tile_x, tile_y, width, height, fg_color, bg_color
        )

    def _create_artifact_from_rendered_content(self) -> Any:
        """Create and return intermediate artifact from the rendered content."""
        return self.private_glyph_buffer
