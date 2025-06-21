"""
Defines the View class, the base component for all persistent UI regions.

A View is a logical, hierarchical component responsible for a specific part of the
UI's layout and content. It's the fundamental building block for the main game
screen (world map, message log, status bars, etc.).

Key Principles:
- A View describes *what* to draw, but not *how*.
- It's responsible for its own content and sub-layout, but its overall position
  and size are managed by the top-level `FrameManager`.
- To render, a View uses a `Canvas` object to record drawing commands, which
  are then turned into a final texture by the `Renderer`.

This architecture decouples layout and content logic from the low-level details
of rendering, making the system more modular and maintainable.
"""

import abc
from typing import Any

from catley.view.render.canvas import Canvas
from catley.view.render.renderer import Renderer


class View(abc.ABC):
    """Base class for persistent UI views.

    Views are always visible and show continuously updated game info.
    Positioned by FrameManager, rendered every frame.

    Two phases: draw() for console rendering, present() for SDL effects.
    """

    def __init__(self) -> None:
        """Initialize view. Call resize() before drawing."""
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.visible = True
        self.tile_dimensions: tuple[int, int] = (0, 0)
        # Cache the most recently rendered texture if the backend provides one.
        self._cached_texture: Any | None = None

    @abc.abstractmethod
    def draw(self, renderer: Renderer) -> None:
        """Draw view content. Subclasses implement this."""
        pass

    def present(self, renderer: Renderer) -> None:
        """Handle texture presentation for backends that produce textures."""
        if not self.visible or self._cached_texture is None:
            return

        renderer.present_texture(
            self._cached_texture, self.x, self.y, self.width, self.height
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set view screen boundaries. Called by an external layout manager."""
        self.x = x1
        self.y = y1
        self.width = x2 - x1
        self.height = y2 - y1

    def show(self) -> None:
        """Make view visible in rendering pipeline."""
        self.visible = True

    def hide(self) -> None:
        """Skip view during rendering."""
        self.visible = False


class TextView(View):
    """An abstract view that is guaranteed to have a Canvas."""

    canvas: Canvas

    def __init__(self) -> None:
        super().__init__()
        self._last_pixel_dims: tuple[int, int] = (0, 0)

    @abc.abstractmethod
    def needs_redraw(self, renderer: Renderer) -> bool:
        """Return ``True`` if the view needs to regenerate its texture."""
        pass

    @abc.abstractmethod
    def draw_content(self, renderer: Renderer) -> None:
        """Subclasses implement the actual drawing commands."""
        pass

    def draw(self, renderer: Renderer) -> None:
        """Orchestrate rendering with caching support."""
        if not self.visible:
            return

        if self.needs_redraw(renderer):
            self.canvas.begin_frame()
            self.draw_content(renderer)
            artifact = self.canvas.end_frame()
            if artifact is not None:
                # Convert artifact to texture using renderer
                if self.canvas.artifact_type == "console":
                    texture = renderer.texture_from_console(
                        artifact, self.canvas.transparent
                    )
                elif self.canvas.artifact_type == "numpy":
                    texture = renderer.texture_from_numpy(
                        artifact, self.canvas.transparent
                    )
                else:
                    # Handle unknown artifact types gracefully
                    texture = None

                self._cached_texture = texture

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set bounds and configure the text backend."""
        super().set_bounds(x1, y1, x2, y2)

        if self.tile_dimensions != (0, 0):
            tile_width, tile_height = self.tile_dimensions
            pixel_width = self.width * tile_width
            pixel_height = self.height * tile_height
            if (pixel_width, pixel_height) != self._last_pixel_dims:
                self._last_pixel_dims = (pixel_width, pixel_height)
                self.canvas.configure_dimensions(pixel_width, pixel_height)
            self.canvas.configure_scaling(tile_height)
            self.canvas.configure_drawing_offset(self.x, self.y)
