"""WGPU Window Wrapper - Bridges Catley's GLFW window with WGPU's canvas system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import wgpu.gui.base
import wgpu.gui.glfw

from catley.types import PixelPos

if TYPE_CHECKING:
    from catley.backends.glfw.window import GlfwWindow


class WGPUWindowWrapper(wgpu.gui.base.WgpuCanvasBase):
    """
    A WGPU window wrapper that integrates Catley's existing GLFW window
    with WGPU's canvas/context system.

    This class implements WGPU's canvas interface while wrapping our
    existing GLFW window instead of creating a new one. This avoids
    the naming conflict between WGPU's "canvas" (window wrapper) and
    Catley's "canvas" (render target).
    """

    def __init__(self, glfw_window: GlfwWindow) -> None:
        """Initialize the WGPU window wrapper.

        Args:
            glfw_window: Catley's GLFW window wrapper
        """
        # Disable WGPU's internal frame limiting (max_fps=999999 effectively uncaps it)
        super().__init__(max_fps=1000_000, vsync=False)
        self.glfw_window = glfw_window

        # Cache window properties
        self._physical_size = self.glfw_window.get_framebuffer_size()
        self._logical_size = self.glfw_window.get_size()
        self._pixel_ratio = (
            self._physical_size[0] / self._logical_size[0]
            if self._logical_size[0] > 0
            else 1.0,
            self._physical_size[1] / self._logical_size[1]
            if self._logical_size[1] > 0
            else 1.0,
        )

    def get_present_methods(self) -> dict:
        """Get present methods for this GLFW window.

        Returns:
            Dictionary with platform-specific window information for WGPU surface
        """
        # Use WGPU's built-in GLFW present method detection
        return wgpu.gui.glfw.get_glfw_present_methods(self.glfw_window.glfw_window)

    def get_physical_size(self) -> PixelPos:
        """Get the physical size of the window in pixels.

        Returns:
            Tuple of (width, height) in physical pixels
        """
        self._physical_size = self.glfw_window.get_framebuffer_size()
        return self._physical_size

    def get_logical_size(self) -> PixelPos:
        """Get the logical size of the window.

        Returns:
            Tuple of (width, height) in logical pixels
        """
        self._logical_size = self.glfw_window.get_size()
        return self._logical_size

    def get_pixel_ratio(self) -> tuple[float, float]:
        """Get the pixel ratio (physical pixels per logical pixel).

        Returns:
            Tuple of (x_ratio, y_ratio)
        """
        physical = self.get_physical_size()
        logical = self.get_logical_size()

        self._pixel_ratio = (
            physical[0] / logical[0] if logical[0] > 0 else 1.0,
            physical[1] / logical[1] if logical[1] > 0 else 1.0,
        )
        return self._pixel_ratio

    # Required abstract methods from WgpuCanvasBase that we don't need for our use case
    def set_logical_size(self, width: int, height: int) -> None:
        """Set the logical size of the window.

        Note: We don't implement this since Catley manages window sizing elsewhere.
        """
        pass

    def set_title(self, title: str) -> None:
        """Set the window title.

        Note: We don't implement this since Catley manages window title elsewhere.
        """
        pass

    def close(self) -> None:
        """Close the window.

        Note: We don't implement this since Catley manages window lifecycle elsewhere.
        """
        pass

    def is_closed(self) -> bool:
        """Check if the window is closed.

        Returns:
            False - Catley manages window lifecycle, so we assume window is open
        """
        return False

    def _request_draw(self) -> None:
        """Request a draw operation.

        Note: We don't implement this since Catley manages the render loop elsewhere.
        """
        pass
