"""WGPU Window Wrapper - Bridges Brileta's GLFW window with WGPU's canvas system.

This module integrates Brileta's existing GLFW window with the rendercanvas/wgpu
canvas system, allowing WGPU to create a surface for the existing window rather
than creating its own window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rendercanvas import BaseRenderCanvas
from rendercanvas.glfw import get_glfw_present_methods

if TYPE_CHECKING:
    from brileta.backends.glfw.window import GlfwWindow


class WGPUWindowWrapper(BaseRenderCanvas):
    """
    A WGPU window wrapper that integrates Brileta's existing GLFW window
    with rendercanvas/wgpu's canvas system.

    This class implements rendercanvas's canvas interface while wrapping our
    existing GLFW window instead of creating a new one. Brileta manages the
    window lifecycle and event loop - this wrapper just provides the surface
    for WGPU rendering.
    """

    # No canvas group - Brileta manages the event loop
    _rc_canvas_group = None

    def __init__(self, glfw_window: GlfwWindow, vsync: bool = True) -> None:
        """Initialize the WGPU window wrapper.

        Args:
            glfw_window: Brileta's GLFW window wrapper
            vsync: Whether to enable vsync (controls WGPU present mode)
        """
        # Disable rendercanvas's internal frame limiting - Brileta manages the loop
        super().__init__(
            max_fps=1_000_000,
            vsync=vsync,
            present_method="screen",
        )
        self.glfw_window = glfw_window

        # Set initial size in the rendercanvas size tracking system
        physical_size = self.glfw_window.get_framebuffer_size()
        logical_size = self.glfw_window.get_size()
        pixel_ratio = physical_size[0] / logical_size[0] if logical_size[0] > 0 else 1.0
        self._size_info.set_physical_size(
            int(physical_size[0]), int(physical_size[1]), pixel_ratio
        )

        # Finalize canvas initialization (sets size and title on the canvas)
        self._final_canvas_init()

    def _rc_get_present_methods(self) -> dict:
        """Get present methods for this GLFW window.

        Returns a dict describing how to create a WGPU surface for this window.
        Uses rendercanvas's built-in GLFW present method detection.
        """
        return get_glfw_present_methods(self.glfw_window.glfw_window)

    # The following methods are required by BaseRenderCanvas but are
    # no-ops because Brileta manages the window and event loop.

    def _rc_gui_poll(self) -> None:
        """Process GUI events. No-op - Brileta polls GLFW events."""
        pass

    def _rc_request_draw(self) -> None:
        """Request a draw. No-op - Brileta manages the render loop."""
        pass

    def _rc_force_draw(self) -> None:
        """Force a synchronous draw. No-op - Brileta manages rendering."""
        pass

    def _rc_present_bitmap(
        self,
        *,
        data: memoryview,
        format: str,  # noqa: A002
        **kwargs: object,
    ) -> None:
        """Present a bitmap. Not used - we render to screen."""
        pass

    def _rc_set_logical_size(self, width: float, height: float) -> None:
        """Set logical size. No-op - Brileta manages window sizing."""
        pass

    def _rc_set_title(self, title: str) -> None:
        """Set window title. No-op - Brileta manages the window title."""
        pass

    def _rc_set_cursor(self, cursor: str) -> None:
        """Set cursor shape. No-op - Brileta manages cursors."""
        pass

    def _rc_close(self) -> None:
        """Close the window. No-op - Brileta manages window lifecycle."""
        pass

    def _rc_get_closed(self) -> bool:
        """Check if window is closed. Returns False - Brileta manages lifecycle."""
        return False

    def update_size(self) -> None:
        """Update the cached size info from the current GLFW window size.

        Call this when the window is resized to keep the rendercanvas
        size tracking in sync.
        """
        physical_size = self.glfw_window.get_framebuffer_size()
        logical_size = self.glfw_window.get_size()
        pixel_ratio = physical_size[0] / logical_size[0] if logical_size[0] > 0 else 1.0
        self._size_info.set_physical_size(
            int(physical_size[0]), int(physical_size[1]), pixel_ratio
        )
