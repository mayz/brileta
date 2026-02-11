"""WGPU Surface Creation - Creates a WGPU surface from Brileta's GLFW window.

Provides a direct bridge between GLFW's native window handle and wgpu-py's
surface creation, without going through rendercanvas's managed canvas system.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import glfw as glfw_lib
import wgpu

if TYPE_CHECKING:
    from brileta.backends.glfw.window import GlfwWindow


def _glfw_native(name: str, *args: Any) -> int:
    """Call a platform-specific glfw function by name, returning an int handle.

    The glfw module exposes get_cocoa_window, get_x11_window, etc. only on
    their respective platforms, so we use getattr to avoid type-checker
    complaints about missing attributes on other platforms.
    """
    fn = getattr(glfw_lib, name)
    return int(fn(*args))


def _get_present_info(glfw_window: int) -> dict:
    """Build the present-info dict wgpu needs to create a surface.

    Extracts the platform-specific native window handle from a GLFW window
    and returns it in the format expected by wgpu.gpu.get_canvas_context().

    Args:
        glfw_window: The raw GLFW window handle (from glfw.create_window).
    """
    if sys.platform == "darwin":
        return {
            "method": "screen",
            "platform": "cocoa",
            "window": _glfw_native("get_cocoa_window", glfw_window),
        }
    if sys.platform == "win32":
        return {
            "method": "screen",
            "platform": "windows",
            "window": _glfw_native("get_win32_window", glfw_window),
        }
    if sys.platform.startswith("linux"):
        # Detect Wayland vs X11 by checking if the Wayland display is available
        try:
            wayland_display = _glfw_native("get_wayland_display")
        except Exception:
            wayland_display = None

        if wayland_display:
            return {
                "method": "screen",
                "platform": "wayland",
                "window": _glfw_native("get_wayland_window", glfw_window),
                "display": wayland_display,
            }
        return {
            "method": "screen",
            "platform": "x11",
            "window": _glfw_native("get_x11_window", glfw_window),
            "display": _glfw_native("get_x11_display"),
        }
    raise RuntimeError(
        f"Unsupported platform for WGPU surface creation: {sys.platform}"
    )


def create_wgpu_context(
    window: GlfwWindow, *, vsync: bool = True
) -> wgpu.GPUCanvasContext:
    """Create a WGPU canvas context (surface) for an existing GLFW window.

    Args:
        window: Brileta's GLFW window wrapper.
        vsync: Whether to use vsync. Controls the wgpu present mode
               (fifo for vsync, immediate for no vsync).

    Returns:
        A GPUCanvasContext ready to be configured and rendered to.
    """
    present_info = _get_present_info(window.glfw_window)
    present_info["vsync"] = vsync
    return wgpu.gpu.get_canvas_context(present_info)
