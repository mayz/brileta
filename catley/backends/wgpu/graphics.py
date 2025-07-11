"""WGPUGraphicsContext - Main entry point for WGPU rendering."""

from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu
from wgpu.gui.glfw import GlfwWgpuCanvas

from catley import colors
from catley.game.enums import BlendMode
from catley.types import (
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    TileDimensions,
)
from catley.util.coordinates import CoordinateConverter, Rect
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from catley.view.render.graphics import GraphicsContext

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class WGPUGraphicsContext(GraphicsContext):
    """WGPU graphics context implementation."""

    def __init__(
        self, window_size: tuple[int, int] = (800, 600), title: str = "Catley"
    ) -> None:
        """Initialize WGPU graphics context.

        Args:
            window_size: Window dimensions (width, height)
            title: Window title
        """
        super().__init__()
        # Let GlfwWgpuCanvas create its own window - don't pass existing GLFW window
        self.canvas = GlfwWgpuCanvas(size=window_size, title=title)
        self.window = self.canvas._window  # Access GLFW window after creation
        self.device: wgpu.GPUDevice | None = None
        self.queue: wgpu.GPUQueue | None = None
        self.context: Any = None

        # Initialize coordinate converter placeholder (set up after initialize())
        self._coordinate_converter: CoordinateConverter | None = None
        self._tile_dimensions = (20, 20)  # Default tile size

    def initialize(self) -> None:
        """Initialize WGPU device and queue."""
        # Correct WGPU API usage patterns:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=wgpu.PowerPreference.high_performance  # type: ignore
        )
        self.device = adapter.request_device_sync()
        self.queue = self.device.queue

        # Create and configure canvas context
        self.context = self.canvas.get_context("wgpu")
        render_texture_format = self.context.get_preferred_format(adapter)
        self.context.configure(device=self.device, format=render_texture_format)

    @property
    def tile_dimensions(self) -> TileDimensions:
        """Get tile dimensions."""
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        """Get console width in tiles."""
        # TODO: Get from config or calculate based on window size
        return 80  # Placeholder

    @property
    def console_height_tiles(self) -> int:
        """Get console height in tiles."""
        # TODO: Get from config or calculate based on window size
        return 50  # Placeholder

    def render_glyph_buffer_to_texture(self, glyph_buffer: GlyphBuffer) -> Any:
        """Render GlyphBuffer to a WGPU texture."""
        # TODO: Implement WGPU glyph buffer rendering
        raise NotImplementedError("WGPU glyph buffer rendering not yet implemented")

    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha | None = None,
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates."""
        if interpolation_alpha is None:
            interpolation_alpha = InterpolationAlpha(1.0)
        # TODO: Implement WGPU actor drawing
        raise NotImplementedError("WGPU actor drawing not yet implemented")

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get display scale factor for high-DPI displays."""
        # TODO: Implement proper DPI scaling
        return (1.0, 1.0)

    def draw_mouse_cursor(self, cursor_manager: "CursorManager") -> None:
        """Draw the mouse cursor."""
        # TODO: Implement WGPU cursor drawing
        raise NotImplementedError("WGPU cursor drawing not yet implemented")

    def draw_tile_highlight(
        self,
        root_x: int,
        root_y: int,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draw a tile highlight."""
        # TODO: Implement WGPU tile highlighting
        raise NotImplementedError("WGPU tile highlighting not yet implemented")

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: RootConsoleTilePos,
    ) -> None:
        """Render particles."""
        # TODO: Implement WGPU particle rendering
        raise NotImplementedError("WGPU particle rendering not yet implemented")

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        """Apply environmental effect."""
        # TODO: Implement WGPU environmental effects
        raise NotImplementedError("WGPU environmental effects not yet implemented")

    def update_dimensions(self) -> None:
        """Update dimensions when window size changes."""
        # TODO: Update coordinate converter with new dimensions
        pass

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """Convert console coordinates to screen pixel coordinates."""
        if self._coordinate_converter is None:
            # Fallback calculation if coordinate converter not initialized
            return PixelPos(
                (
                    int(console_x * self._tile_dimensions[0]),
                    int(console_y * self._tile_dimensions[1]),
                )
            )
        return self._coordinate_converter.console_to_screen_coords(console_x, console_y)

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Convert screen pixel coordinates to console tile coordinates."""
        if self._coordinate_converter is None:
            # Fallback calculation if coordinate converter not initialized
            return RootConsoleTilePos(
                (
                    int(pixel_x // self._tile_dimensions[0]),
                    int(pixel_y // self._tile_dimensions[1]),
                )
            )
        return self._coordinate_converter.pixel_to_tile(pixel_x, pixel_y)

    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Create a WGPU texture from numpy array."""
        # TODO: Implement WGPU texture creation from numpy
        raise NotImplementedError("WGPU texture creation not yet implemented")

    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Present a texture to the screen."""
        # TODO: Implement WGPU texture presentation
        raise NotImplementedError("WGPU texture presentation not yet implemented")

    def draw_background(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Draw background texture."""
        # TODO: Implement WGPU background drawing
        raise NotImplementedError("WGPU background drawing not yet implemented")

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draw debug rectangle."""
        # TODO: Implement WGPU debug rectangle drawing
        raise NotImplementedError("WGPU debug rectangle drawing not yet implemented")

    @property
    def coordinate_converter(self) -> CoordinateConverter:
        """Get the coordinate converter."""
        if self._coordinate_converter is None:
            # Create a basic coordinate converter if not initialized
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=800,  # Default window width
                renderer_scaled_height=600,  # Default window height
            )
        return self._coordinate_converter

    def create_canvas(self, transparent: bool = True) -> Any:
        """Create a WGPU canvas."""
        # TODO: Implement WGPU canvas creation
        raise NotImplementedError("WGPU canvas creation not yet implemented")
