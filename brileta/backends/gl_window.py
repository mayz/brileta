from typing import Protocol

from brileta.types import PixelPos


class GLWindow(Protocol):
    """
    Defines the minimal interface a windowing library must provide
    to be a host for an OpenGL-based GraphicsContext.

    The existence of this window *implies* that a valid, current OpenGL context exists
    to attach to.
    """

    def get_size(self) -> PixelPos:
        """Returns the current pixel dimensions (width, height)
        of the window's drawable area."""
        ...

    def get_framebuffer_size(self) -> PixelPos:
        """Returns the framebuffer dimensions (can be different on high DPI/Retina)."""
        ...

    def flip(self) -> None:
        """Swaps the back and front buffers, displaying the rendered frame."""
        ...
