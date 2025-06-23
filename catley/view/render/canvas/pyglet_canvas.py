from __future__ import annotations

from typing import Any

from pyglet.shapes import BorderedRectangle, Rectangle
from pyglet.text import Label

from catley import config
from catley.util.coordinates import PixelCoord, TileCoord

from .base import Canvas


class PygletCanvas(Canvas):
    """Canvas that renders directly to the renderer's UI batch."""

    def __init__(self, renderer, transparent: bool = True):
        super().__init__(transparent)
        self.renderer = renderer
        self.batch = renderer.ui_batch  # Get a direct reference to the UI batch
        self.window = renderer.window
        self.configure_scaling(renderer.tile_dimensions[1])

        # Keep references to UI objects to prevent garbage collection
        self._ui_objects = []

    # --- Frame Lifecycle: We don't need to manage objects anymore ---

    def begin_frame(self) -> None:
        """Clear previous UI objects and prepare for new frame."""
        self._ui_objects.clear()

    def end_frame(self):
        """Return the batch as the artifact."""
        return self._create_artifact_from_rendered_content()

    # Override the base Canvas draw methods to do immediate execution
    def draw_text(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color,
        font_size: int | None = None,
    ) -> None:
        """Draw text immediately to the batch."""
        self._render_text_op(pixel_x, pixel_y, text, color, font_size)

    def draw_rect(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color,
        fill: bool,
    ) -> None:
        """Draw rectangle immediately to the batch."""
        self._render_rect_op(pixel_x, pixel_y, width, height, color, fill)

    def draw_frame(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg,
        bg,
    ) -> None:
        """Draw frame immediately to the batch."""
        self._render_frame_op(tile_x, tile_y, width, height, fg, bg)

    # --- Abstract Method Implementations ---

    @property
    def artifact_type(self) -> str:
        return "pyglet_batch"

    def create_texture(self, renderer, artifact):
        # The artifact is the batch itself. No texture is created.
        return artifact

    def _prepare_for_rendering(self) -> bool:
        # No preparation needed.
        return True

    def _create_artifact_from_rendered_content(self) -> Any:
        """The 'artifact' is the batch we've populated."""
        return self.batch

    # --- Backend-Specific Rendering Operations ---

    def _render_text_op(self, pixel_x, pixel_y, text: str, color, font_size=None):
        label = Label(
            text,
            font_name=str(config.MESSAGE_LOG_FONT_PATH),
            font_size=font_size or self.renderer.tile_dimensions[1],
            x=pixel_x,
            y=self.window.height - pixel_y,  # Flip Y-coord for top-left origin
            anchor_y="top",
            color=(*color, 255),
            batch=self.batch,
        )
        # Keep reference to prevent garbage collection
        self._ui_objects.append(label)

    def _render_rect_op(self, pixel_x, pixel_y, width, height, color, fill):
        flipped_y = self.window.height - pixel_y - height
        color_with_alpha = (*color, 255) if len(color) == 3 else color

        rect = Rectangle(
            x=pixel_x,
            y=flipped_y,
            width=width,
            height=height,
            color=color_with_alpha,
            batch=self.batch,
        )
        if not fill:
            rect.opacity = 128

        # Keep reference to prevent garbage collection
        self._ui_objects.append(rect)

    def _render_frame_op(self, tile_x, tile_y, width, height, fg, bg):
        tile_w, tile_h = self.renderer.tile_dimensions
        pixel_x = tile_x * tile_w
        pixel_y = tile_y * tile_h
        pixel_width = width * tile_w
        pixel_height = height * tile_h
        flipped_y = self.window.height - pixel_y - pixel_height

        frame = BorderedRectangle(
            x=pixel_x,
            y=flipped_y,
            width=pixel_width,
            height=pixel_height,
            border=1,
            color=bg,
            border_color=fg,
            batch=self.batch,
        )
        # Keep reference to prevent garbage collection
        self._ui_objects.append(frame)

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        tile_width, tile_height = self.renderer.tile_dimensions
        return len(text) * tile_width, tile_height, tile_height

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        tile_width, _ = self.renderer.tile_dimensions
        if tile_width == 0:
            return [text]
        chars_per_line = max_width // tile_width
        if chars_per_line <= 0:
            return [text]
        return [
            text[i : i + chars_per_line] for i in range(0, len(text), chars_per_line)
        ]

    def _update_scaling_internal(self, tile_height: int) -> None:
        pass

    def get_effective_line_height(self) -> int:
        return self.renderer.tile_dimensions[1]
