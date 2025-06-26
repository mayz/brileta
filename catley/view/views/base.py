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

import tcod.sdl.render

from catley.util.caching import ResourceCache
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
    def draw(self, renderer: Renderer, alpha: float) -> None:
        """Draw view content. Subclasses implement this.

        Args:
            renderer: The rendering backend to use
            alpha: Interpolation factor between game logic steps (0.0-1.0)
                  Used for smooth animations independent of visual framerate
        """
        pass

    def present(self, renderer: Renderer, alpha: float) -> None:
        """Handle texture presentation for backends that produce textures.

        Args:
            renderer: The rendering backend to use
            alpha: Interpolation factor (unused in base implementation)
        """
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
        # Each view gets its own tiny cache. max_size=1 is perfect
        # for views that only ever need to cache their most recent state.
        self._texture_cache = ResourceCache[Any, tcod.sdl.render.Texture](
            name=self.__class__.__name__, max_size=1
        )

    @abc.abstractmethod
    def get_cache_key(self) -> Any:
        """
        Return a hashable key representing the current state of the view's data.
        If this key is unchanged, the view does not need to be redrawn.
        Examples: a revision number, a tuple of values (hp, ap), etc.
        """
        pass

    @abc.abstractmethod
    def draw_content(self, renderer: Renderer, alpha: float) -> None:
        """Subclasses implement the actual drawing commands.

        Args:
            renderer: The rendering backend to use
            alpha: Interpolation factor for smooth animations (0.0-1.0)
                  Most UI views don't use this, but it's available
        """
        pass

    def draw(self, renderer: Renderer, alpha: float) -> None:
        """Orchestrate rendering with caching support.

        Args:
            renderer: The rendering backend to use
            alpha: Interpolation factor passed to draw_content
        """
        if not self.visible:
            return

        cache_key = self.get_cache_key()
        cached_texture = self._texture_cache.get(cache_key)

        if cached_texture is None:
            # CACHE MISS: The state has changed. We need to re-render.
            self.canvas.begin_frame()
            self.draw_content(renderer, alpha)
            artifact = self.canvas.end_frame()

            if artifact is not None:
                texture = self.canvas.create_texture(renderer, artifact)
                if texture:
                    # Store the newly rendered texture in the cache
                    self._texture_cache.store(cache_key, texture)
                    self._cached_texture = texture
        else:
            # CACHE HIT: Use the existing texture
            self._cached_texture = cached_texture

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set bounds and configure the text backend."""
        super().set_bounds(x1, y1, x2, y2)

        if self.tile_dimensions != (0, 0):
            tile_width, tile_height = self.tile_dimensions
            pixel_width = self.width * tile_width
            pixel_height = self.height * tile_height
            self.canvas.configure_dimensions(pixel_width, pixel_height)
            self.canvas.configure_scaling(tile_height)
            self.canvas.configure_drawing_offset(self.x, self.y)
