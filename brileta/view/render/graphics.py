"""Shared graphics context interface and common rendering implementations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from brileta import colors, config
from brileta.game.enums import BlendMode
from brileta.types import (
    ColorRGBAf,
    ColorRGBf,
    InterpolationAlpha,
    Opacity,
    PixelCoord,
    PixelPos,
    PixelRect,
    RootConsoleTilePos,
    SpriteUV,
    TileDimensions,
    UVRect,
    ViewOffset,
    WorldTilePos,
    saturate,
)
from brileta.util.coordinates import (
    CoordinateConverter,
    Rect,
    convert_particle_to_screen_coords,
)
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.util.tilesets import unicode_to_cp437
from brileta.view.render.effects.decals import DecalSystem
from brileta.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from brileta.view.render.viewport import ViewportSystem

if TYPE_CHECKING:
    from brileta.backends.wgpu.resource_manager import WGPUResourceManager
    from brileta.view.ui.cursor_manager import CursorManager


@runtime_checkable
class ScreenRenderer(Protocol):
    """Protocol for the screen-quad renderer used by GraphicsContext."""

    def add_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: UVRect,
        color_rgba: ColorRGBAf,
    ) -> None: ...


class GraphicsContext:
    """Concrete base class for all graphics backends.

    This class provides shared coordinate conversion and render helper logic.
    Backend-specific contexts should inherit and override backend-dependent
    methods.
    """

    # CP437 index for the solid-block glyph (█), used for filled quads.
    SOLID_BLOCK_CHAR: int = 219

    # Backends without explicit resource managers inherit the default None value.
    resource_manager: WGPUResourceManager | None = None

    def __init__(self) -> None:
        """Initialize common state shared by graphics backends."""

        self._tile_dimensions: TileDimensions = (20, 20)
        self._native_tile_size: TileDimensions = (20, 20)
        self._scale_factor: float = 1.0
        self._console_width_tiles: int = 80
        self._console_height_tiles: int = 50
        self._coordinate_converter: CoordinateConverter | None = None

        self.letterbox_geometry: PixelRect | None = None
        self.uv_map: np.ndarray | None = None
        self.screen_renderer: ScreenRenderer | None = None

    @property
    def tile_dimensions(self) -> TileDimensions:
        """Current tile dimensions used for display rendering."""
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        """Current console width in tiles."""
        return self._console_width_tiles

    @property
    def console_height_tiles(self) -> int:
        """Current console height in tiles."""
        return self._console_height_tiles

    @property
    def native_tile_size(self) -> TileDimensions:
        """Native tile dimensions loaded from the tileset atlas."""
        return self._native_tile_size

    @property
    def scale_factor(self) -> float:
        """Effective content scale factor (DPI scale * zoom)."""
        return self._scale_factor

    @property
    def gpu_max_texture_dimension_2d(self) -> int:
        """GPU max 2D texture dimension.

        Backends can override with device-specific limits.
        """
        return 8192

    @property
    def locked_content_scale(self) -> int | None:
        """Locked content scale for DPI stabilization, if supported."""
        return None

    @property
    def coordinate_converter(self) -> CoordinateConverter:
        """Converter for screen pixel <-> tile coordinate mapping."""
        if self._coordinate_converter is None:
            self._setup_coordinate_converter()
        converter = self._coordinate_converter
        assert converter is not None
        return converter

    def _precalculate_uv_map(self) -> np.ndarray:
        """Pre-calculate UV coordinates for all CP437 atlas characters."""
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

    @staticmethod
    def _cp437_index_for_char(char: str) -> int:
        """Map a Python character to a CP437 atlas index."""
        codepoint = ord(char) if char else ord(" ")
        if 0 <= codepoint <= 255:
            return codepoint
        return unicode_to_cp437(codepoint)

    @staticmethod
    def _color_to_rgba(color: colors.Color, alpha: Opacity) -> ColorRGBAf:
        """Normalize a 0-255 RGB color and opacity to a 0.0-1.0 RGBA tuple."""
        return (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            saturate(alpha),
        )

    def render_glyph_buffer_to_texture(
        self,
        glyph_buffer: GlyphBuffer,
        buffer_override: Any = None,
        secondary_override: Any = None,
        cache_key_suffix: str = "",
    ) -> Any:
        """Render a GlyphBuffer to a backend-specific texture."""
        raise NotImplementedError

    def draw_actor(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: ColorRGBf = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        world_pos: WorldTilePos | None = None,
        tile_bg: colors.Color | None = None,
    ) -> None:
        """Draw an actor character at sub-pixel screen coordinates."""
        raise NotImplementedError

    def draw_actor_shadow(
        self,
        char: str,
        screen_x: float,
        screen_y: float,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_pixels: float,
        shadow_alpha: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        fade_tip: bool = True,
    ) -> None:
        """Draw a projected glyph shadow as a stretched parallelogram quad."""
        raise NotImplementedError

    def draw_sprite_shadow(
        self,
        sprite_uv: SpriteUV,
        screen_x: float,
        screen_y: float,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_pixels: float,
        shadow_alpha: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        ground_anchor_y: float = 1.0,
        fade_tip: bool = True,
    ) -> None:
        """Draw a projected shadow using a sprite atlas silhouette if supported."""
        return

    def draw_sprite_smooth(
        self,
        sprite_uv: SpriteUV,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: ColorRGBf = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        ground_anchor_y: float = 1.0,
        world_pos: WorldTilePos | None = None,
        tile_bg: colors.Color | None = None,
    ) -> None:
        """Draw a sprite from the dynamic sprite atlas at sub-pixel coordinates."""
        return

    def draw_sprite_smooth_batch(
        self,
        *,
        sprite_uvs: np.ndarray,
        actor_colors: np.ndarray,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        light_intensity: np.ndarray,
        scale_x: np.ndarray,
        scale_y: np.ndarray,
        ground_anchor_y: np.ndarray,
        world_pos: np.ndarray,
        tile_bg: np.ndarray,
    ) -> None:
        """Batch-draw multiple sprites from the sprite atlas."""
        return

    def draw_sprite_shadow_batch(
        self,
        *,
        sprite_uvs: np.ndarray,
        screen_x: np.ndarray,
        screen_y: np.ndarray,
        shadow_dir_x: float,
        shadow_dir_y: float,
        shadow_length_pixels: np.ndarray,
        shadow_alpha: np.ndarray | float,
        scale_x: np.ndarray,
        scale_y: np.ndarray,
        ground_anchor_y: np.ndarray,
        fade_tip: bool = True,
    ) -> None:
        """Batch-draw many sprite-silhouette shadows."""
        return

    def create_sprite_atlas(self, width: int, height: int) -> Any | None:
        """Create a dynamic sprite atlas if supported by the backend."""
        return None

    def draw_sprite_outline(
        self,
        sprite_uv: SpriteUV,
        screen_x: float,
        screen_y: float,
        color: colors.Color,
        alpha: Opacity,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        ground_anchor_y: float = 1.0,
    ) -> None:
        """Draw a 1px outline extracted from a sprite's alpha silhouette."""
        return

    def set_sprite_atlas_texture(self, texture: Any | None) -> None:
        """Bind the sprite atlas texture for rendering, if supported."""
        return

    def set_sun_direction(self, sun_dx: float, sun_dy: float) -> None:
        """Set per-frame sun direction for sprite shading, if supported."""
        return

    def draw_actor_outline(
        self,
        char: str,
        screen_x: float,
        screen_y: float,
        color: colors.Color,
        alpha: Opacity,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw an outlined version of a glyph for targeting."""
        raise NotImplementedError

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Return display scale factor for high-DPI displays."""
        raise NotImplementedError

    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draw the active cursor."""
        raise NotImplementedError

    def draw_tile_highlight(
        self,
        root_x: float,
        root_y: float,
        color: colors.Color,
        alpha: Opacity,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw a filled highlight quad over a tile."""
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        px_x, px_y = self.console_to_screen_coords(root_x, root_y)
        w, h = self.tile_dimensions
        scaled_w = w * scale_x
        scaled_h = h * scale_y
        uv = self.uv_map[self.SOLID_BLOCK_CHAR]
        self.screen_renderer.add_quad(
            px_x, px_y, scaled_w, scaled_h, uv, self._color_to_rgba(color, alpha)
        )

    def render_particles(
        self,
        particle_system: SubTileParticleSystem,
        layer: ParticleLayer,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem | None = None,
    ) -> None:
        """Render particles by adding each active particle to the quad buffer."""
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        particle_scale = (1.0, 1.0)
        if viewport_system is not None:
            particle_scale = viewport_system.get_display_scale_factors()

        for i in range(particle_system.active_count):
            if particle_system.layers[i] != layer.value:
                continue

            coords = convert_particle_to_screen_coords(
                particle_system,
                i,
                viewport_bounds,
                view_offset,
                self,
                viewport_system,
            )
            if coords is None:
                continue

            screen_x, screen_y = coords
            if not np.isnan(particle_system.flash_intensity[i]):
                alpha = particle_system.flash_intensity[i]
            else:
                alpha = particle_system.lifetimes[i] / particle_system.max_lifetimes[i]

            self._draw_particle_to_buffer(
                particle_system.chars[i],
                tuple(particle_system.colors[i]),
                screen_x,
                screen_y,
                Opacity(alpha),
                scale_x=particle_scale[0],
                scale_y=particle_scale[1],
            )

    def render_decals(
        self,
        decal_system: DecalSystem,
        viewport_bounds: Rect,
        view_offset: ViewOffset,
        viewport_system: ViewportSystem,
        game_time: float,
    ) -> None:
        """Render decals from world-space sub-tile coordinates."""
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        decal_scale_x, decal_scale_y = viewport_system.get_display_scale_factors()
        for (tile_x, tile_y), decals in decal_system.decals.items():
            screen_tile_x, screen_tile_y = viewport_system.world_to_screen(
                tile_x, tile_y
            )

            if not (
                -1 <= screen_tile_x < viewport_bounds.width + 1
                and -1 <= screen_tile_y < viewport_bounds.height + 1
            ):
                continue

            for decal in decals:
                alpha = decal_system.get_alpha(decal, game_time)
                if alpha <= 0:
                    continue

                world_screen_x, world_screen_y = viewport_system.world_to_screen_float(
                    decal.x,
                    decal.y,
                )
                console_x = view_offset[0] + world_screen_x
                console_y = view_offset[1] + world_screen_y
                pixel_x, pixel_y = self.console_to_screen_coords(console_x, console_y)

                self._draw_particle_to_buffer(
                    decal.char,
                    decal.color,
                    pixel_x,
                    pixel_y,
                    Opacity(alpha),
                    scale_x=decal_scale_x,
                    scale_y=decal_scale_y,
                )

    def _draw_particle_to_buffer(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        alpha: Opacity,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw a single particle by adding its quad to the screen buffer.

        Callers must guard against ``screen_renderer`` / ``uv_map`` being None
        before entering their render loops.
        """
        assert self.screen_renderer is not None
        assert self.uv_map is not None

        uv_coords = self.uv_map[self._cp437_index_for_char(char)]
        w, h = self.tile_dimensions
        self.screen_renderer.add_quad(
            screen_x,
            screen_y,
            w * scale_x,
            h * scale_y,
            uv_coords,
            self._color_to_rgba(color, alpha),
        )

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
        *,
        radius_scale_x: float = 1.0,
        radius_scale_y: float = 1.0,
    ) -> None:
        """Apply a circular environmental effect using cached textures."""
        raise NotImplementedError

    def update_dimensions(self, zoom_only: bool = False) -> None:
        """Update dimension-dependent coordinate state when window changes."""
        raise NotImplementedError

    def console_to_screen_coords(self, console_x: float, console_y: float) -> PixelPos:
        """Convert root-console coordinates to final screen pixel coordinates."""
        if self.letterbox_geometry is None:
            return (
                int(console_x * self._tile_dimensions[0]),
                int(console_y * self._tile_dimensions[1]),
            )

        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry
        screen_x = offset_x + console_x * (scaled_w / self.console_width_tiles)
        screen_y = offset_y + console_y * (scaled_h / self.console_height_tiles)
        return (screen_x, screen_y)

    def pixel_to_tile(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
    ) -> RootConsoleTilePos:
        """Convert screen pixel coordinates to root-console tile coordinates."""
        if self._coordinate_converter is None:
            return (
                int(pixel_x // self._tile_dimensions[0]),
                int(pixel_y // self._tile_dimensions[1]),
            )

        if self.letterbox_geometry is not None:
            offset_x, offset_y, _, _ = self.letterbox_geometry
            adjusted_pixel_x = pixel_x - offset_x
            adjusted_pixel_y = pixel_y - offset_y
        else:
            adjusted_pixel_x = pixel_x
            adjusted_pixel_y = pixel_y

        return self._coordinate_converter.pixel_to_tile(
            adjusted_pixel_x,
            adjusted_pixel_y,
        )

    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Create a backend texture from RGBA or RGB numpy pixels."""
        raise NotImplementedError

    def present_texture(
        self,
        texture: Any,
        x_tile: float,
        y_tile: float,
        width_tiles: float,
        height_tiles: float,
    ) -> None:
        """Present a texture in a tile-coordinate destination region."""
        raise NotImplementedError

    def draw_texture_alpha(
        self,
        texture: Any,
        screen_x: PixelCoord,
        screen_y: PixelCoord,
        alpha: Opacity,
        *,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draw a texture at pixel coordinates with alpha modulation."""
        raise NotImplementedError

    def draw_background(
        self,
        texture: Any,
        x_tile: float,
        y_tile: float,
        width_tiles: float,
        height_tiles: float,
        offset_x_pixels: float = 0.0,
        offset_y_pixels: float = 0.0,
    ) -> None:
        """Draw the main world background texture."""
        raise NotImplementedError

    def draw_rect_outline(
        self,
        px_x: int,
        px_y: int,
        px_w: int,
        px_h: int,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draw an unfilled rectangle outline using the screen renderer."""
        if self.screen_renderer is None:
            return
        if self.uv_map is None:
            return

        color_rgba = self._color_to_rgba(color, alpha)
        uv_coords = self.uv_map[self.SOLID_BLOCK_CHAR]

        self.screen_renderer.add_quad(px_x, px_y, px_w, 1, uv_coords, color_rgba)
        self.screen_renderer.add_quad(
            px_x,
            px_y + px_h - 1,
            px_w,
            1,
            uv_coords,
            color_rgba,
        )
        self.screen_renderer.add_quad(px_x, px_y, 1, px_h, uv_coords, color_rgba)
        self.screen_renderer.add_quad(
            px_x + px_w - 1,
            px_y,
            1,
            px_h,
            uv_coords,
            color_rgba,
        )

    def create_canvas(self, transparent: bool = True) -> Any:
        """Create a backend canvas for drawing operations."""
        raise NotImplementedError

    def release_texture(self, texture: Any) -> None:
        """Release a backend texture and any associated GPU resources."""
        raise NotImplementedError

    def draw_debug_tile_grid(
        self,
        view_origin: WorldTilePos,
        view_size: tuple[int, int],
        offset_pixels: ViewOffset,
    ) -> None:
        """Render a debug tile grid overlay for the current world view."""
        raise NotImplementedError

    def compose_light_overlay_gpu(
        self,
        dark_texture: Any,
        light_texture: Any,
        lightmap_texture: Any,
        visible_mask_buffer: np.ndarray,
        viewport_bounds: Rect,
        viewport_offset: WorldTilePos,
        pad_tiles: int,
    ) -> Any:
        """Compose light overlay directly on GPU, when backend supports it."""
        raise NotImplementedError

    def set_atmospheric_layer(
        self,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: Any | None,
        explored_texture: Any | None,
        visible_texture: Any | None,
        roof_surface_mask_buffer: np.ndarray | None,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        strength: float,
        tint_color: colors.Color,
        drift_offset: ViewOffset,
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        blend_mode: str,
        pixel_bounds: PixelRect,
        *,
        affects_foreground: bool = True,
    ) -> None:
        """Queue or ignore atmospheric layer data."""
        return

    def set_rain_effect(
        self,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        tile_dimensions: TileDimensions,
        intensity: float,
        angle: float,
        drop_length: float,
        drop_speed: float,
        drop_spacing: float,
        stream_spacing: float,
        rain_color: colors.Color,
        time: float,
        rain_exclusion_mask_buffer: np.ndarray | None,
        pixel_bounds: PixelRect,
    ) -> None:
        """Queue or ignore rain overlay data."""
        return

    def set_noise_seed(self, seed: int) -> None:
        """Set sub-tile noise seed, if the backend supports it."""
        return

    def set_noise_tile_offset(self, offset_x: int, offset_y: int) -> None:
        """Set world-space tile offset for stable noise hashing."""
        return

    def cleanup(self) -> None:
        """Release backend resources before shutdown."""
        return

    @contextmanager
    def shadow_pass(self) -> Iterator[None]:
        """Bracket shadow geometry generation, if supported by backend."""
        yield

    def _calculate_letterbox_geometry(
        self,
        framebuffer_width: int,
        framebuffer_height: int,
        display_tile_width: int,
        display_tile_height: int,
    ) -> PixelRect:
        """Calculate integer-scaled letterbox geometry for the framebuffer."""
        tile_width = max(1, display_tile_width)
        tile_height = max(1, display_tile_height)
        console_width, console_height = self._calculate_console_tile_dimensions(
            framebuffer_width,
            framebuffer_height,
            tile_width,
            tile_height,
        )

        rendered_width = console_width * tile_width
        rendered_height = console_height * tile_height
        offset_x = (framebuffer_width - rendered_width) // 2
        offset_y = (framebuffer_height - rendered_height) // 2

        return (offset_x, offset_y, rendered_width, rendered_height)

    @staticmethod
    def _calculate_console_tile_dimensions(
        framebuffer_width: int,
        framebuffer_height: int,
        display_tile_width: int,
        display_tile_height: int,
    ) -> tuple[int, int]:
        """Calculate console dimensions from framebuffer and display tile size."""
        tile_width = max(1, display_tile_width)
        tile_height = max(1, display_tile_height)
        console_width = max(1, framebuffer_width // tile_width)
        console_height = max(1, framebuffer_height // tile_height)
        return (console_width, console_height)

    def _setup_coordinate_converter(self) -> None:
        """Set up coordinate conversion using letterbox geometry when available."""
        if self.letterbox_geometry is not None:
            _, _, scaled_w, scaled_h = self.letterbox_geometry
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=scaled_w,
                renderer_scaled_height=scaled_h,
            )
        else:
            fallback_w, fallback_h = self._coordinate_converter_fallback_scaled_size()
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=fallback_w,
                renderer_scaled_height=fallback_h,
            )

    def _coordinate_converter_fallback_scaled_size(self) -> tuple[int, int]:
        """Return fallback scaled dimensions for converter setup."""
        return (800, 600)

    def _preprocess_pixels_for_texture(
        self,
        pixels: np.ndarray,
        transparent: bool = True,
    ) -> np.ndarray:
        """Normalize RGB/RGBA input into an RGBA pixel array."""
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError("Expected RGBA or RGB image array")

        height, width = pixels.shape[:2]
        components = pixels.shape[2]

        if components == 3:
            rgba_pixels = np.zeros((height, width, 4), dtype=pixels.dtype)
            rgba_pixels[:, :, :3] = pixels
            rgba_pixels[:, :, 3] = 255 if not transparent else 0
            pixels = rgba_pixels

        return pixels
