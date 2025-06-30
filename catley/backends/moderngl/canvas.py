from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from catley import colors
from catley.util.coordinates import PixelCoord, TileCoord
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.canvas import Canvas

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class ModernGLCanvas(Canvas):
    """Canvas that draws to a GlyphBuffer for ModernGL rendering."""

    def __init__(self, renderer: GraphicsContext, transparent: bool = True) -> None:
        super().__init__(transparent)
        self.renderer = renderer
        self.private_glyph_buffer: GlyphBuffer | None = None

        self.configure_scaling(renderer.tile_dimensions[1])

    @property
    def artifact_type(self) -> str:
        return "glyph_buffer"

    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Creates a backend-specific texture from this canvas's artifact."""
        # This import is here to avoid a circular dependency.
        from catley.backends.moderngl.graphics import ModernGLGraphicsContext

        # We need to cast the renderer to access its ModernGL-specific method.
        # This is acceptable because this canvas is ModernGL-specific.
        moderngl_renderer = cast(ModernGLGraphicsContext, renderer)
        return moderngl_renderer.render_glyph_buffer_to_texture(artifact)

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

    def configure_dimensions(self, width: PixelCoord, height: PixelCoord) -> None:
        """Create a private glyph buffer based on pixel dimensions."""
        # Call parent first to set width/height attributes
        super().configure_dimensions(width, height)

        tile_w, tile_h = self.renderer.tile_dimensions
        if not tile_w or not tile_h:
            return

        width_tiles = int(width // tile_w)
        height_tiles = int(height // tile_h)

        if (
            not self.private_glyph_buffer
            or self.private_glyph_buffer.width != width_tiles
            or self.private_glyph_buffer.height != height_tiles
        ):
            self.private_glyph_buffer = GlyphBuffer(width_tiles, height_tiles)
            # New buffer means cache is invalid
            self._cached_frame_artifact = None
            self._last_frame_ops = []

    def _prepare_for_rendering(self) -> bool:
        """Prepare glyph buffer for rendering operations."""
        if not self.private_glyph_buffer:
            return False

        # Clear buffer for new frame
        bg_color = (0, 0, 0, 0) if self.transparent else (0, 0, 0, 255)
        self.private_glyph_buffer.clear(bg=bg_color)
        return True

    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Render a single text operation to the glyph buffer."""
        if not self.private_glyph_buffer:
            return

        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return

        tile_x = int(pixel_x // tile_width)
        tile_y = int(pixel_y // tile_height)

        # Convert color to RGBA if needed
        rgba_color = (*color, 255) if len(color) == 3 else color

        # Draw each character
        for i, char in enumerate(text):
            x = tile_x + i
            if x >= self.private_glyph_buffer.width:
                break
            if (
                0 <= x < self.private_glyph_buffer.width
                and 0 <= tile_y < self.private_glyph_buffer.height
            ):
                bg_color = (0, 0, 0, 0) if self.transparent else (0, 0, 0, 255)
                self.private_glyph_buffer.put_char(
                    x, tile_y, ord(char), rgba_color, bg_color
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
        """Render a single rectangle operation to the glyph buffer."""
        if not self.private_glyph_buffer:
            return

        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return

        # Convert color to RGBA if needed
        rgba_color = (*color, 255) if len(color) == 3 else color

        start_tx = int(pixel_x // tile_width)
        start_ty = int(pixel_y // tile_height)
        end_tx = int((pixel_x + width) // tile_width)
        end_ty = int((pixel_y + height) // tile_height)

        # Clamp to buffer bounds
        start_tx = max(0, start_tx)
        start_ty = max(0, start_ty)
        end_tx = min(self.private_glyph_buffer.width, end_tx)
        end_ty = min(self.private_glyph_buffer.height, end_ty)

        for tx in range(start_tx, end_tx):
            for ty in range(start_ty, end_ty):
                if fill:
                    # Fill with solid block character (CP437 219)
                    self.private_glyph_buffer.put_char(
                        tx, ty, 219, rgba_color, rgba_color
                    )
                else:
                    # Border only
                    if tx in (start_tx, end_tx - 1) or ty in (start_ty, end_ty - 1):
                        self.private_glyph_buffer.put_char(
                            tx, ty, 219, rgba_color, rgba_color
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
        """Render a single frame operation to the glyph buffer."""
        if not self.private_glyph_buffer:
            return

        # Convert colors to RGBA if needed
        fg_rgba = (*fg, 255) if len(fg) == 3 else fg
        bg_rgba = (*bg, 255) if len(bg) == 3 else bg

        # Draw frame using CP437 box drawing characters
        # Top and bottom borders
        for x in range(tile_x, tile_x + width):
            if 0 <= x < self.private_glyph_buffer.width:
                # Top border
                if 0 <= tile_y < self.private_glyph_buffer.height:
                    # CP437: horizontal=196, top-left=218, top-right=191
                    char_code = (
                        196  # horizontal line
                        if x != tile_x and x != tile_x + width - 1
                        else (218 if x == tile_x else 191)  # top-left : top-right
                    )
                    self.private_glyph_buffer.put_char(
                        x, tile_y, char_code, fg_rgba, bg_rgba
                    )
                # Bottom border
                if 0 <= tile_y + height - 1 < self.private_glyph_buffer.height:
                    # CP437: horizontal=196, bottom-left=192, bottom-right=217
                    char_code = (
                        196  # horizontal line
                        if x != tile_x and x != tile_x + width - 1
                        else (192 if x == tile_x else 217)  # bottom-left : bottom-right
                    )
                    self.private_glyph_buffer.put_char(
                        x, tile_y + height - 1, char_code, fg_rgba, bg_rgba
                    )

        # Left and right borders
        for y in range(tile_y + 1, tile_y + height - 1):
            if 0 <= y < self.private_glyph_buffer.height:
                # Left border (CP437 vertical line = 179)
                if 0 <= tile_x < self.private_glyph_buffer.width:
                    self.private_glyph_buffer.put_char(tile_x, y, 179, fg_rgba, bg_rgba)
                # Right border (CP437 vertical line = 179)
                if 0 <= tile_x + width - 1 < self.private_glyph_buffer.width:
                    self.private_glyph_buffer.put_char(
                        tile_x + width - 1, y, 179, fg_rgba, bg_rgba
                    )

    def _create_artifact_from_rendered_content(self) -> GlyphBuffer | None:
        """Return the glyph buffer as the intermediate artifact."""
        return self.private_glyph_buffer
