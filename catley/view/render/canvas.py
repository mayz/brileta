"""
Defines the Canvas, an immediate-mode drawing API for Views and Overlays.

A Canvas's role is to provide a standardized, implementation-agnostic interface
for drawing primitives like text and rectangles. It acts as an intermediary between
the high-level View components and the low-level Renderer.

Key Principles:
- A Canvas is a "drawing command recorder." It's used by a View to describe its
  visual content for a single frame.
- It is stateless between frames. A new frame begins with a `begin_frame()` call.
- The `end_frame()` method processes the recorded commands and produces an
  intermediate "artifact" (e.g., a `tcod.Console` object or a NumPy array of
  pixels).
- This artifact is then passed to the `Renderer` to be converted into a final
  texture for display.

This separation allows Views to be ignorant of the underlying rendering technology
(e.g., TCOD glyphs vs. Pillow TrueType fonts).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from catley import colors
from catley.util.coordinates import PixelCoord, TileCoord

if TYPE_CHECKING:
    from catley.view.render.graphics import GraphicsContext


class DrawOperation(Enum):
    """Types of drawing operations that can be recorded and rendered."""

    TEXT = auto()
    RECT = auto()
    FRAME = auto()


class Canvas(ABC):
    """Abstract interface for drawing canvases."""

    def __init__(self, transparent: bool = True) -> None:
        self.drawing_offset_x = 0
        self.drawing_offset_y = 0
        self._last_tile_height: int = 0
        self.width = 0
        self.height = 0
        self.transparent = transparent

        # Shared caching logic
        self._frame_ops: list = []
        self._last_frame_ops: list = []
        self._cached_frame_artifact = None

    def draw_text(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Draw a line of text whose top-left corner is at ``(pixel_x, pixel_y)``
        using ``color``."""
        self._record_text_op(pixel_x, pixel_y, text, color, font_size)

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

    @property
    @abstractmethod
    def artifact_type(self) -> str:
        """Return type of artifact this canvas produces (e.g., 'console', 'numpy')."""
        pass

    def begin_frame(self) -> None:
        """Prepare backend for a new frame."""
        self._frame_ops = []

    def end_frame(self):
        """Finalize the frame and return intermediate artifact."""
        # Check if frame content changed
        if self._should_rerender():
            return self._cached_frame_artifact

        # Content changed - need to re-render
        self._cached_frame_artifact = None  # Let GC handle cleanup

        if not self._prepare_for_rendering():
            return None

        # Render operations
        for op in self._frame_ops:
            match op[0]:
                case DrawOperation.TEXT:
                    _, pixel_x, pixel_y, text, color, font_size = op
                    self._render_text_op(pixel_x, pixel_y, text, color, font_size)
                case DrawOperation.RECT:
                    _, pixel_x, pixel_y, width, height, color, fill = op
                    self._render_rect_op(pixel_x, pixel_y, width, height, color, fill)
                case DrawOperation.FRAME:
                    _, tile_x, tile_y, width, height, fg, bg = op
                    self._render_frame_op(tile_x, tile_y, width, height, fg, bg)

        # Create intermediate artifact from rendered content
        artifact = self._create_artifact_from_rendered_content()

        if artifact is not None:
            # Cache the result
            self._cached_frame_artifact = artifact
            self._last_frame_ops = self._frame_ops.copy()

        return artifact

    @abstractmethod
    def create_texture(self, renderer: GraphicsContext, artifact: Any) -> Any:
        """Creates a backend-specific texture from this canvas's artifact."""
        pass

    def draw_frame(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Draw a framed box, using tile coordinates."""
        self._record_frame_op(tile_x, tile_y, width, height, fg, bg)

    def draw_rect(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Draw a rectangle at the given pixel coordinates."""
        self._record_rect_op(pixel_x, pixel_y, width, height, color, fill)

    # ------------------------------------------------------------------
    # Optional configuration helpers
    # ------------------------------------------------------------------
    def configure_dimensions(self, width: PixelCoord, height: PixelCoord) -> None:
        if self.width != width or self.height != height:
            # Dimensions changed - clear cache
            self._cached_frame_artifact = None  # Let GC handle cleanup
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

    # ------------------------------------------------------------------
    # Operation recording helpers
    # ------------------------------------------------------------------
    def _record_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> None:
        """Record a text drawing operation."""
        self._frame_ops.append(
            (DrawOperation.TEXT, pixel_x, pixel_y, text, color, font_size)
        )

    def _record_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> None:
        """Record a rectangle drawing operation."""
        self._frame_ops.append(
            (DrawOperation.RECT, pixel_x, pixel_y, width, height, color, fill)
        )

    def _record_frame_op(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg: colors.Color,
        bg: colors.Color,
    ) -> None:
        """Record a frame drawing operation."""
        self._frame_ops.append(
            (DrawOperation.FRAME, tile_x, tile_y, width, height, fg, bg)
        )

    def _should_rerender(self) -> bool:
        """Check if we should reuse cached artifact instead of re-rendering."""
        return (
            self._frame_ops == self._last_frame_ops
            and self._cached_frame_artifact is not None
        )

    # ------------------------------------------------------------------
    # Abstract rendering methods
    # ------------------------------------------------------------------
    @abstractmethod
    def _prepare_for_rendering(self) -> bool:
        """Prepare the backend for rendering operations.

        Return False if rendering should be skipped.
        """

    @abstractmethod
    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: colors.Color,
        font_size: int | None = None,
    ) -> Any:
        """Render a single text operation."""

    @abstractmethod
    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: colors.Color,
        fill: bool,
    ) -> Any:
        """Render a single rectangle operation."""

    @abstractmethod
    def _render_frame_op(
        self,
        tile_x: TileCoord,
        tile_y: TileCoord,
        width: TileCoord,
        height: TileCoord,
        fg: colors.Color,
        bg: colors.Color,
    ) -> Any:
        """Render a single frame operation."""

    @abstractmethod
    def _create_artifact_from_rendered_content(self) -> Any:
        """Create and return intermediate artifact from the rendered content."""
