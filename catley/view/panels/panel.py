"""Panel System - Persistent sub-regions of the game window that
know how to draw themselves.

Panels are always-visible UI that show game state:
- The main, central game world view
- The message log
- Health, ammo, FPS displays
- Fixed-position interface components

Use panels for persistent UI. Use overlays for temporary UI.

Two-phase rendering: draw() for console ops, present() for SDL ops.
"""

import abc
from typing import Any

from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import TextBackend


class Panel(abc.ABC):
    """Base class for persistent UI panels.

    Panels are always visible and show continuously updated game info.
    Positioned by FrameManager, rendered every frame.

    Two phases: draw() for console rendering, present() for SDL effects.
    """

    def __init__(self) -> None:
        """Initialize panel. Call resize() before drawing."""
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
        """Draw panel content. Subclasses implement this."""
        pass

    def present(self, renderer: Renderer) -> None:
        """Handle texture presentation for backends that produce textures."""
        if not self.visible or not self._cached_texture:
            return

        if self.tile_dimensions != (0, 0):
            tile_width, tile_height = self.tile_dimensions
            dest_x = self.x * tile_width
            dest_y = self.y * tile_height
            dest_width = self.width * tile_width
            dest_height = self.height * tile_height

            dest_rect = (dest_x, dest_y, dest_width, dest_height)
            renderer.sdl_renderer.copy(self._cached_texture, dest=dest_rect)

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set panel screen boundaries."""
        self.x = x1
        self.y = y1
        self.width = x2 - x1
        self.height = y2 - y1

    def show(self) -> None:
        """Make panel visible in rendering pipeline."""
        self.visible = True

    def hide(self) -> None:
        """Skip panel during rendering."""
        self.visible = False


class TextPanel(Panel):
    """An abstract panel that is guaranteed to have a TextBackend."""

    text_backend: TextBackend

    @abc.abstractmethod
    def needs_redraw(self, renderer: Renderer) -> bool:
        """Return ``True`` if the panel needs to regenerate its texture."""
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
            self.text_backend.begin_frame()
            self.draw_content(renderer)
            texture = self.text_backend.end_frame()
            if texture is not None:
                self._cached_texture = texture

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Resize and configure the text backend."""
        super().resize(x1, y1, x2, y2)

        if self.tile_dimensions != (0, 0):
            tile_width, tile_height = self.tile_dimensions
            pixel_width = self.width * tile_width
            pixel_height = self.height * tile_height
            self.text_backend.configure_dimensions(pixel_width, pixel_height)
            self.text_backend.configure_drawing_offset(self.x, self.y)
