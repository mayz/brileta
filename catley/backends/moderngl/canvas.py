from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import moderngl

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
        self.renderer = renderer  # This is ModernGLGraphicsContext
        self.private_glyph_buffer: GlyphBuffer | None = None

        # Canvas owns its GPU resources for isolation
        self.vbo: moderngl.Buffer | None = None
        self.vao: moderngl.VertexArray | None = None

        # Get a reference to the shader program from the TextureRenderer
        from catley.backends.moderngl.graphics import ModernGLGraphicsContext

        moderngl_renderer = cast(ModernGLGraphicsContext, renderer)

        # Handle both real and mock renderers
        if hasattr(moderngl_renderer, "texture_renderer") and hasattr(
            moderngl_renderer, "mgl_context"
        ):
            self.texture_program = moderngl_renderer.texture_renderer.texture_program
            self.mgl_context = moderngl_renderer.mgl_context
        else:
            # Mock renderer for tests - create dummy values
            self.texture_program = None
            self.mgl_context = None

        self.configure_scaling(renderer.tile_dimensions[1])

    def release(self) -> None:
        """Release GPU resources when canvas is no longer needed."""
        if self.vbo:
            self.vbo.release()
            self.vbo = None
        if self.vao:
            self.vao.release()
            self.vao = None

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

        # Pass THIS canvas's VBO and VAO to the renderer for isolation
        return moderngl_renderer.render_glyph_buffer_to_texture(
            artifact,
            vbo_override=self.vbo,
            vao_override=self.vao,
        )

    def create_texture_with_cache_key(
        self, renderer: GraphicsContext, artifact: Any, cache_key: str
    ) -> Any:
        """Creates a backend-specific texture with cache key for unique caching."""
        # This import is here to avoid a circular dependency.
        from catley.backends.moderngl.graphics import ModernGLGraphicsContext

        # We need to cast the renderer to access its ModernGL-specific method.
        moderngl_renderer = cast(ModernGLGraphicsContext, renderer)

        # Pass cache key for unique caching per overlay
        return moderngl_renderer.render_glyph_buffer_to_texture(
            artifact,
            vbo_override=self.vbo,
            vao_override=self.vao,
            cache_key_suffix=cache_key,
        )

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

            # Recreate VBO and VAO for this canvas with the new dimensions
            # Only create GPU resources if we have a real renderer (not a mock)
            if self.mgl_context is not None and self.texture_program is not None:
                max_vertices = width_tiles * height_tiles * 12
                if self.vbo:
                    self.vbo.release()
                if self.vao:
                    self.vao.release()

                # Create VBO sized for this canvas, not the global max
                # We need to import the VERTEX_DTYPE from texture_renderer
                from catley.backends.moderngl.texture_renderer import VERTEX_DTYPE

                vbo_size = max_vertices * VERTEX_DTYPE.itemsize
                initial_data = b"\x00" * vbo_size  # Initialize with zeros
                self.vbo = self.mgl_context.buffer(initial_data, dynamic=True)
                self.vao = self.mgl_context.vertex_array(
                    self.texture_program,
                    [
                        (
                            self.vbo,
                            "2f 2f 4f 4f",
                            "in_vert",
                            "in_uv",
                            "in_fg_color",
                            "in_bg_color",
                        )
                    ],
                )

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
                # Preserve existing background color, similar to TCOD behavior
                current_bg = self.private_glyph_buffer.data[x, tile_y]["bg"]
                self.private_glyph_buffer.put_char(
                    x, tile_y, ord(char), rgba_color, tuple(current_bg)
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
                    # For filled rectangles, use solid blocks but also set bg color
                    # This provides the solid block for general rectangle drawing while
                    # ensuring background color is available for text overlay
                    self.private_glyph_buffer.put_char(
                        tx, ty, 9608, rgba_color, rgba_color
                    )
                else:
                    # Border only - draw outline using solid block characters
                    if tx in (start_tx, end_tx - 1) or ty in (start_ty, end_ty - 1):
                        self.private_glyph_buffer.put_char(
                            tx, ty, 9608, rgba_color, rgba_color
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

        # Draw frame using Unicode box drawing characters
        # Top and bottom borders
        for x in range(tile_x, tile_x + width):
            if 0 <= x < self.private_glyph_buffer.width:
                # Top border
                if 0 <= tile_y < self.private_glyph_buffer.height:
                    # Unicode: horizontal=─, top-left=┌, top-right=┐
                    char_code = (
                        9472  # ─ horizontal line
                        if x != tile_x and x != tile_x + width - 1
                        else (9484 if x == tile_x else 9488)  # ┌ top-left : ┐ top-right
                    )
                    self.private_glyph_buffer.put_char(
                        x, tile_y, char_code, fg_rgba, bg_rgba
                    )
                # Bottom border
                if 0 <= tile_y + height - 1 < self.private_glyph_buffer.height:
                    # Unicode: horizontal=─, bottom-left=└, bottom-right=┘
                    char_code = (
                        9472  # ─ horizontal line
                        if x != tile_x and x != tile_x + width - 1
                        else (
                            9492 if x == tile_x else 9496
                        )  # └ bottom-left : ┘ bottom-right
                    )
                    self.private_glyph_buffer.put_char(
                        x, tile_y + height - 1, char_code, fg_rgba, bg_rgba
                    )

        # Left and right borders
        for y in range(tile_y + 1, tile_y + height - 1):
            if 0 <= y < self.private_glyph_buffer.height:
                # Left border (Unicode vertical line │)
                if 0 <= tile_x < self.private_glyph_buffer.width:
                    self.private_glyph_buffer.put_char(
                        tile_x, y, 9474, fg_rgba, bg_rgba
                    )
                # Right border (Unicode vertical line │)
                if 0 <= tile_x + width - 1 < self.private_glyph_buffer.width:
                    self.private_glyph_buffer.put_char(
                        tile_x + width - 1, y, 9474, fg_rgba, bg_rgba
                    )

    def _create_artifact_from_rendered_content(self) -> GlyphBuffer | None:
        """Return the glyph buffer as the intermediate artifact."""
        return self.private_glyph_buffer
