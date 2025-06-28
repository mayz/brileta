from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from tcod.console import Console

from catley import colors
from catley.util.coordinates import PixelCoord, TileCoord
from catley.view.render.canvas import Canvas

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class TCODConsoleCanvas(Canvas):
    """Canvas that draws directly to a tcod :class:`Console`."""

    def __init__(self, renderer: GraphicsContext, transparent: bool = True) -> None:
        super().__init__(transparent)
        self.renderer = renderer
        self.private_console: Console | None = None

        self.configure_scaling(renderer.tile_dimensions[1])

    @property
    def artifact_type(self) -> str:
        return "console"

    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Creates a backend-specific texture from this canvas's artifact."""
        # This import is here to avoid a circular dependency.
        from catley.backends.tcod.graphics import TCODGraphicsContext

        # We need to cast the renderer to access its TCOD-specific method.
        # This is acceptable because this canvas is TCOD-specific.
        tcod_renderer = cast(TCODGraphicsContext, renderer)
        return tcod_renderer.texture_from_console(artifact, self.transparent)

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
        """Create a private console based on pixel dimensions."""
        # Call parent first to set width/height attributes
        super().configure_dimensions(width, height)

        tile_w, tile_h = self.renderer.tile_dimensions
        if not tile_w or not tile_h:
            return

        width_tiles = int(width // tile_w)
        height_tiles = int(height // tile_h)

        if (
            not self.private_console
            or self.private_console.width != width_tiles
            or self.private_console.height != height_tiles
        ):
            self.private_console = Console(width_tiles, height_tiles, order="F")
            # New console means cache is invalid
            self._cached_frame_artifact = None
            self._last_frame_ops = []

    def _prepare_for_rendering(self) -> bool:
        """Prepare console for rendering operations."""
        if not self.private_console:
            return False

        # Clear console for new frame
        self.private_console.clear()
        return True

    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Render a single text operation to the console."""
        if not self.private_console:
            return

        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return

        tile_x = int(pixel_x // tile_width)
        tile_y = int(pixel_y // tile_height)
        self.private_console.print(x=tile_x, y=tile_y, text=text, fg=color)

    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Render a single rectangle operation to the console."""
        if not self.private_console:
            return

        tile_width, tile_height = self.renderer.tile_dimensions
        if tile_width == 0 or tile_height == 0:
            return

        start_tx = int(pixel_x // tile_width)
        start_ty = int(pixel_y // tile_height)
        end_tx = int((pixel_x + width) // tile_width)
        end_ty = int((pixel_y + height) // tile_height)

        for tx in range(start_tx, end_tx):
            for ty in range(start_ty, end_ty):
                if fill:
                    self.private_console.bg[tx, ty] = color
                else:
                    if tx in (start_tx, end_tx - 1) or ty in (start_ty, end_ty - 1):
                        self.private_console.bg[tx, ty] = color

    def _render_frame_op(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Render a single frame operation to the console."""
        if not self.private_console:
            return

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

    def _create_artifact_from_rendered_content(self) -> Console | None:
        """Return the console as the intermediate artifact."""
        return self.private_console
