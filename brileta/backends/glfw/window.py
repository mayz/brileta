"""GLFW window wrapper implementing the GLWindow protocol."""

import glfw

from brileta.types import PixelPos


class GlfwWindow:
    """
    Wraps a GLFW window to implement the GLWindow protocol.

    This allows GLFW windows to be used with the ModernGLGraphicsContext
    without modification to the graphics context.
    """

    def __init__(self, glfw_window):
        """
        Initialize the wrapper with a GLFW window handle.

        Args:
            glfw_window: The GLFW window handle from glfw.create_window()
        """
        self.glfw_window = glfw_window

    def get_size(self) -> PixelPos:
        """
        Returns the current pixel dimensions (width, height) of the
        window's drawable area.

        Returns:
            Tuple of (width, height) in pixels
        """
        return glfw.get_window_size(self.glfw_window)

    def get_framebuffer_size(self) -> PixelPos:
        """
        Returns the framebuffer dimensions (can be different on high DPI/Retina).

        Returns:
            Tuple of (framebuffer_width, framebuffer_height) in pixels
        """
        return glfw.get_framebuffer_size(self.glfw_window)

    def flip(self) -> None:
        """Swaps the back and front buffers, displaying the rendered frame."""
        glfw.swap_buffers(self.glfw_window)
