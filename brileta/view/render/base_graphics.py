"""Base implementation for graphics contexts with common functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from brileta import colors, config
from brileta.types import (
    Opacity,
    PixelCoord,
    PixelPos,
    RootConsoleTilePos,
    TileDimensions,
    ViewOffset,
)
from brileta.util.coordinates import (
    CoordinateConverter,
    Rect,
    convert_particle_to_screen_coords,
)
from brileta.view.render.effects.decals import DecalSystem
from brileta.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from brileta.view.render.graphics import GraphicsContext
from brileta.view.render.viewport import ViewportSystem

if TYPE_CHECKING:
    pass


@runtime_checkable
class ScreenRenderer(Protocol):
    """Protocol for the vertex-buffer renderer used by BaseGraphicsContext.

    Any object that implements ``add_quad`` satisfies this protocol.
    The WGPU backend's ``WGPUScreenRenderer`` is the primary implementation.
    """

    def add_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        color_rgba: tuple[float, float, float, float],
    ) -> None: ...


class BaseGraphicsContext(GraphicsContext):
    """Base implementation of GraphicsContext with common functionality.

    This class provides shared implementations for methods that are identical
    or nearly identical across different graphics backends (WGPU, ModernGL, etc).
    Backend-specific classes should inherit from this and override only the
    methods that require backend-specific implementations.
    """

    def __init__(self) -> None:
        """Initialize common properties."""
        super().__init__()
        # Common constants
        self.SOLID_BLOCK_CHAR = 219

        # Common state that will be set by subclasses
        self._tile_dimensions: TileDimensions = (20, 20)  # Default
        self._coordinate_converter: CoordinateConverter | None = None
        self.letterbox_geometry: tuple[int, int, int, int] | None = None
        self.uv_map: np.ndarray | None = None
        self.screen_renderer: ScreenRenderer | None = None

    @property
    def console_width_tiles(self) -> int:
        """Get console width in tiles from config."""
        return config.SCREEN_WIDTH

    @property
    def console_height_tiles(self) -> int:
        """Get console height in tiles from config."""
        return config.SCREEN_HEIGHT

    @property
    def tile_dimensions(self) -> TileDimensions:
        """Get current tile dimensions."""
        return self._tile_dimensions

    def _precalculate_uv_map(self) -> np.ndarray:
        """Pre-calculates UV coordinates for all 256 possible characters.

        This method is identical across all backends.
        """
        uv_map = np.zeros((256, 4), dtype="f4")
        for i in range(256):
            col = i % config.TILESET_COLUMNS
            row = i // config.TILESET_COLUMNS
            u1 = col / config.TILESET_COLUMNS
            v1 = row / config.TILESET_ROWS
            u2 = (col + 1) / config.TILESET_COLUMNS
            v2 = (row + 1) / config.TILESET_ROWS
            uv_map[i] = [u1, v1, u2, v2]
        return uv_map

    def draw_rect_outline(
        self,
        px_x: int,
        px_y: int,
        px_w: int,
        px_h: int,
        color: colors.Color,
        alpha: float,
    ) -> None:
        """Draw a rectangle outline using the screen renderer."""
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        color_rgba = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            max(0.0, min(1.0, alpha)),
        )
        uv_coords = self.uv_map[self.SOLID_BLOCK_CHAR]

        # Top edge
        self.screen_renderer.add_quad(
            px_x, px_y, px_w, 1, uv_coords, color_rgba
        )  # Bottom edge
        self.screen_renderer.add_quad(
            px_x, px_y + px_h - 1, px_w, 1, uv_coords, color_rgba
        )
        # Left edge
        self.screen_renderer.add_quad(
            px_x, px_y, 1, px_h, uv_coords, color_rgba
        )  # Right edge
        self.screen_renderer.add_quad(
            px_x + px_w - 1, px_y, 1, px_h, uv_coords, color_rgba
        )

    def draw_tile_highlight(
        self, root_x: float, root_y: float, color: colors.Color, alpha: Opacity
    ) -> None:
        """Draw a tile highlight.

        This implementation is nearly identical across backends.
        """
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        px_x, px_y = self.console_to_screen_coords(root_x, root_y)
        w, h = self.tile_dimensions
        uv = self.uv_map[self.SOLID_BLOCK_CHAR]
        color_rgba = (color[0] / 255, color[1] / 255, color[2] / 255, alpha)
        self.screen_renderer.add_quad(px_x, px_y, w, h, uv, color_rgba)

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem | None = None,
    ) -> None:
        """Render particles by adding each one to the vertex buffer.

        This high-level logic is identical across backends.
        """
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        # Loop through active particles
        for i in range(particle_system.active_count):
            if particle_system.layers[i] != layer.value:
                continue

            coords = convert_particle_to_screen_coords(
                particle_system, i, viewport_bounds, view_offset, self, viewport_system
            )
            if coords is None:
                continue

            screen_x, screen_y = coords

            # Calculate alpha based on lifetime or flash intensity
            if not np.isnan(particle_system.flash_intensity[i]):
                alpha = particle_system.flash_intensity[i]
            else:
                alpha = particle_system.lifetimes[i] / particle_system.max_lifetimes[i]

            # Render this particle
            self._draw_particle_to_buffer(
                particle_system.chars[i],
                tuple(particle_system.colors[i]),
                screen_x,
                screen_y,
                alpha,
            )

    def render_decals(
        self,
        decal_system: DecalSystem,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem,
        game_time: float,
    ) -> None:
        """Render visible decals from the decal system with sub-tile precision.

        Decals have sub-tile float coordinates and are rendered at precise
        pixel positions, similar to particles.
        """
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        # Iterate through all tiles with decals
        for (tile_x, tile_y), decals in decal_system.decals.items():
            # Convert tile position to screen position for culling
            screen_tile_x, screen_tile_y = viewport_system.world_to_screen(
                tile_x, tile_y
            )

            # Cull tiles outside the viewport (with 1-tile margin for sub-tile overflow)
            if not (
                -1 <= screen_tile_x < viewport_bounds.width + 1
                and -1 <= screen_tile_y < viewport_bounds.height + 1
            ):
                continue

            # Draw each decal at its sub-tile position
            for decal in decals:
                alpha = decal_system.get_alpha(decal, game_time)
                if alpha <= 0:
                    continue

                # Convert world sub-tile position to screen position (float coords)
                world_screen_x, world_screen_y = viewport_system.world_to_screen_float(
                    decal.x, decal.y
                )

                # Convert to final screen pixel coordinates
                console_x = view_offset[0] + world_screen_x
                console_y = view_offset[1] + world_screen_y
                pixel_x, pixel_y = self.console_to_screen_coords(console_x, console_y)

                self._draw_particle_to_buffer(
                    decal.char,
                    decal.color,
                    pixel_x,
                    pixel_y,
                    alpha,
                )

    def _draw_particle_to_buffer(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        alpha: float,
    ) -> None:
        """Draw a single particle by adding its quad to the vertex buffer.

        This implementation is identical across backends.
        """
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        # Convert color to 0-1 range and apply alpha
        final_color = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            max(0.0, min(1.0, alpha)),  # Clamp alpha to valid range
        )

        # Get UV coordinates for the particle character
        char_code = ord(char) if char else ord(" ")
        uv_coords = self.uv_map[char_code]
        w, h = self.tile_dimensions

        # Add the particle quad to the vertex buffer
        self.screen_renderer.add_quad(screen_x, screen_y, w, h, uv_coords, final_color)

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """Convert console coordinates to screen pixel coordinates.

        The core logic is the same across backends.
        """
        if self.letterbox_geometry is None:
            # Fallback calculation if letterbox not initialized
            return (
                int(console_x * self._tile_dimensions[0]),
                int(console_y * self._tile_dimensions[1]),
            )

        # Use letterbox geometry for accurate conversion
        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry
        screen_x = offset_x + console_x * (scaled_w / self.console_width_tiles)
        screen_y = offset_y + console_y * (scaled_h / self.console_height_tiles)
        return (screen_x, screen_y)

    def pixel_to_tile(
        self, pixel_x: PixelCoord, pixel_y: PixelCoord
    ) -> RootConsoleTilePos:
        """Convert screen pixel coordinates to console tile coordinates.

        This logic is nearly identical across backends.
        """
        if self._coordinate_converter is None:
            # Fallback calculation if coordinate converter not initialized
            return (
                int(pixel_x // self._tile_dimensions[0]),
                int(pixel_y // self._tile_dimensions[1]),
            )

        # Adjust for letterboxing offsets before passing to the converter
        if self.letterbox_geometry is not None:
            offset_x, offset_y, _, _ = self.letterbox_geometry
            adjusted_pixel_x = pixel_x - offset_x
            adjusted_pixel_y = pixel_y - offset_y
        else:
            adjusted_pixel_x = pixel_x
            adjusted_pixel_y = pixel_y

        return self._coordinate_converter.pixel_to_tile(
            adjusted_pixel_x, adjusted_pixel_y
        )

    def _calculate_letterbox_geometry(
        self, window_width: int, window_height: int
    ) -> tuple[int, int, int, int]:
        """Calculate letterbox geometry for the current window size.

        This calculation is identical across backends.
        """
        console_width, console_height = (
            self.console_width_tiles,
            self.console_height_tiles,
        )
        console_aspect = console_width / console_height
        window_aspect = window_width / window_height

        if console_aspect > window_aspect:
            scaled_w = window_width
            scaled_h = int(window_width / console_aspect)
            offset_x = 0
            offset_y = (window_height - scaled_h) // 2
        else:
            scaled_h = window_height
            scaled_w = int(window_height * console_aspect)
            offset_x = (window_width - scaled_w) // 2
            offset_y = 0

        return (offset_x, offset_y, scaled_w, scaled_h)

    def _setup_coordinate_converter(self) -> None:
        """Set up the coordinate converter with letterbox scaled dimensions.

        This setup is nearly identical across backends.
        """
        if self.letterbox_geometry is not None:
            # Use letterbox scaled dimensions
            _, _, scaled_w, scaled_h = self.letterbox_geometry
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=scaled_w,
                renderer_scaled_height=scaled_h,
            )
        else:
            # Fallback to a default converter - subclasses should provide window dims
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=800,  # Default
                renderer_scaled_height=600,  # Default
            )

    @property
    def coordinate_converter(self) -> CoordinateConverter:
        """Get the coordinate converter."""
        if self._coordinate_converter is None:
            self._setup_coordinate_converter()
        converter = self._coordinate_converter
        assert converter is not None
        return converter

    def _preprocess_pixels_for_texture(
        self, pixels: np.ndarray, transparent: bool = True
    ) -> np.ndarray:
        """Preprocess pixels for texture creation.

        This helper method handles the common preprocessing that's identical
        across backends. Subclasses should call this then create the texture.
        """
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("Expected RGBA or RGB image array")

        height, width = pixels.shape[:2]
        components = pixels.shape[2]

        # Ensure we have RGBA data
        if components == 3:
            rgba_pixels = np.zeros((height, width, 4), dtype=pixels.dtype)
            rgba_pixels[:, :, :3] = pixels
            rgba_pixels[:, :, 3] = 255 if not transparent else 0
            pixels = rgba_pixels

        return pixels
