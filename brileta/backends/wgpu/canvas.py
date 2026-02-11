"""WGPUCanvas - WGPU canvas implementation extending Canvas."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu

from brileta import colors
from brileta.util.coordinates import PixelCoord, TileCoord
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.view.render.canvas import Canvas

if TYPE_CHECKING:
    from brileta.view.render.graphics import GraphicsContext

    from .resource_manager import WGPUResourceManager


class WGPUCanvas(Canvas):
    """WGPU canvas implementation extending base Canvas."""

    def __init__(
        self,
        renderer: GraphicsContext,
        resource_manager: WGPUResourceManager,
        transparent: bool = True,
    ) -> None:
        """Initialize WGPU canvas.

        Args:
            renderer: Graphics context (WGPUGraphicsContext)
            resource_manager: The WGPU resource manager for creating buffers
            transparent: Whether the canvas should support transparency
        """
        super().__init__(transparent)
        self.renderer = renderer
        self.resource_manager = resource_manager
        self.private_glyph_buffer: GlyphBuffer | None = None
        self.vertex_buffer: wgpu.GPUBuffer | None = None
        self.cpu_vertex_buffer: np.ndarray | None = None

        # Unique cache key for this canvas instance to prevent texture sharing
        self._cache_key = str(id(self))

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
        """Wrap text at word boundaries to fit within max_width pixels."""
        from brileta.view.ui.ui_utils import wrap_text_by_words

        _ = font_size
        tile_width, _ = self.renderer.tile_dimensions
        if tile_width == 0:
            return [text]  # Avoid division by zero

        def fits(s: str) -> bool:
            return len(s) * tile_width <= max_width

        return wrap_text_by_words(text, fits)

    @property
    def artifact_type(self) -> str:
        """Return type of artifact this canvas produces."""
        return "glyph_buffer"

    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Create a WGPU texture from this canvas's artifact."""
        if not isinstance(artifact, GlyphBuffer):
            return None

        # Use the renderer's method, but pass this canvas's OWN vertex buffers
        return renderer.render_glyph_buffer_to_texture(
            artifact,
            buffer_override=self.vertex_buffer,
            secondary_override=self.cpu_vertex_buffer,
        )

    def create_texture_with_cache_key(
        self, renderer: GraphicsContext, artifact: Any, cache_key: str
    ) -> Any:
        """Creates a backend-specific texture with cache key for unique caching."""
        if not isinstance(artifact, GlyphBuffer):
            return None

        # Pass cache key for unique caching per overlay
        return renderer.render_glyph_buffer_to_texture(
            artifact,
            buffer_override=self.vertex_buffer,
            secondary_override=self.cpu_vertex_buffer,
            cache_key_suffix=cache_key,
        )

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

    def configure_dimensions(self, width: PixelCoord, height: PixelCoord) -> None:
        """Create or resize the canvas's private glyph buffer and GPU vertex buffer."""
        from brileta.backends.wgpu.glyph_renderer import TEXTURE_VERTEX_DTYPE

        super().configure_dimensions(width, height)

        # Recreate the GPU vertex buffer if dimensions change
        tile_w, tile_h = self.renderer.tile_dimensions
        if not tile_w or not tile_h:
            return

        width_tiles = int(width // tile_w)
        height_tiles = int(height // tile_h)
        max_vertices = width_tiles * height_tiles * 6

        # Create or resize GPU vertex buffer
        if (
            self.vertex_buffer is None
            or self.vertex_buffer.size < max_vertices * TEXTURE_VERTEX_DTYPE.itemsize
        ):
            if self.vertex_buffer:
                self.vertex_buffer.destroy()

            buffer_size = max_vertices * TEXTURE_VERTEX_DTYPE.itemsize
            if buffer_size > 0:
                self.vertex_buffer = self.resource_manager.device.create_buffer(
                    size=buffer_size,
                    usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
                    label=f"canvas_vbo_{id(self)}",
                )
            else:
                self.vertex_buffer = None

        # Create or resize CPU vertex buffer to match GPU buffer
        if self.cpu_vertex_buffer is None or len(self.cpu_vertex_buffer) < max_vertices:
            if max_vertices > 0:
                self.cpu_vertex_buffer = np.zeros(
                    max_vertices, dtype=TEXTURE_VERTEX_DTYPE
                )
            else:
                self.cpu_vertex_buffer = None

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

        # Draw each character, preserving background like ModernGL
        for i, char in enumerate(text):
            x = tile_x + i
            if (
                0 <= x < self.private_glyph_buffer.width
                and 0 <= tile_y < self.private_glyph_buffer.height
            ):
                # Preserve existing background color when printing over a tile
                current_bg = self.private_glyph_buffer.data[x, tile_y]["bg"]
                self.private_glyph_buffer.put_char(
                    x, tile_y, ord(char), fg_color, tuple(current_bg)
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
        """Render a single rectangle operation."""
        if not self.private_glyph_buffer:
            return

        # Convert pixel coordinates to tile coordinates
        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return

        start_tx = int(pixel_x // tile_width)
        start_ty = int(pixel_y // tile_height)
        end_tx = int((pixel_x + width) // tile_width)
        end_ty = int((pixel_y + height) // tile_height)

        # Convert color to RGBA if needed
        rgba_color = (*color, 255) if len(color) == 3 else color

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
