"""WGPUGraphicsContext - Main entry point for WGPU rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu
from PIL import Image as PILImage

from catley import colors, config
from catley.backends.glfw.window import GlfwWindow
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

from .resource_manager import WGPUResourceManager
from .screen_renderer import WGPUScreenRenderer
from .shader_manager import WGPUShaderManager

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class WGPUGraphicsContext(GraphicsContext):
    """WGPU graphics context implementation."""

    def __init__(self, window: GlfwWindow) -> None:
        """Initialize WGPU graphics context.

        Args:
            window: The GLFW window to render to
        """
        super().__init__()
        self.window = window

        self.device: wgpu.GPUDevice | None = None
        self.queue: wgpu.GPUQueue | None = None
        self.surface: wgpu.GPUSurface | None = None
        self.swap_chain: Any = None

        # Resource managers (initialized after device)
        self.resource_manager: WGPUResourceManager | None = None
        self.shader_manager: WGPUShaderManager | None = None

        # Atlas texture and UV mapping
        self.atlas_texture: wgpu.GPUTexture | None = None
        self.uv_map: np.ndarray | None = None

        # Screen renderer
        self.screen_renderer: WGPUScreenRenderer | None = None

        # Letterbox/viewport state
        self.letterbox_geometry: tuple[int, int, int, int] | None = None

        # Initialize coordinate converter placeholder (set up after initialize())
        self._coordinate_converter: CoordinateConverter | None = None
        self._tile_dimensions = (20, 20)  # Default tile size

    def initialize(self) -> None:
        """Initialize WGPU device and surface."""

        # Create WGPU adapter and device
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=wgpu.PowerPreference.high_performance  # type: ignore
        )
        self.device = adapter.request_device_sync()
        self.queue = self.device.queue

        # Create WGPU surface from the GLFW window
        self.surface = wgpu.create_surface_from_window(self.window.glfw_window)

        if self.surface is None:
            raise RuntimeError("Failed to create WGPU surface")

        # Configure surface
        width, height = self.window.get_framebuffer_size()
        surface_format = self.surface.get_preferred_format(adapter)
        self.surface.configure(
            device=self.device,
            format=surface_format,
            width=width,
            height=height,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        # Initialize resource managers
        self.resource_manager = WGPUResourceManager(self.device, self.queue)
        self.shader_manager = WGPUShaderManager(self.device)

        # Load atlas texture and prepare UV mapping
        self.atlas_texture = self._load_atlas_texture()
        self.uv_map = self._precalculate_uv_map()

        # Initialize screen renderer
        self.screen_renderer = WGPUScreenRenderer(
            self.resource_manager, self.shader_manager, self.atlas_texture
        )

        # Set up coordinate converter with proper dimensions
        self._setup_coordinate_converter()

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

    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
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

        # Reconfigure surface for new size
        if self.surface and self.device:
            width, height = self.window.get_framebuffer_size()
            adapter = wgpu.gpu.request_adapter_sync()
            surface_format = self.surface.get_preferred_format(adapter)
            self.surface.configure(
                device=self.device,
                format=surface_format,
                width=width,
                height=height,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )

        # Update coordinate converter
        self._setup_coordinate_converter()

        # Update letterbox geometry
        self._calculate_letterbox_geometry()

    def prepare_to_present(self) -> None:
        """Prepare for presenting the frame."""
        if self.screen_renderer is None:
            return
        self.screen_renderer.begin_frame()

    def finalize_present(self) -> None:
        """Finalize frame presentation by rendering to screen."""

        if (
            self.screen_renderer is None
            or self.surface is None
            or self.resource_manager is None
        ):
            return

        # Get the current surface texture
        surface_texture = self.surface.get_current_texture()

        # Create command encoder
        command_encoder = self.resource_manager.device.create_command_encoder()

        # Create render pass
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": surface_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),  # Black background
                    "load_op": "clear",
                    "store_op": "store",
                }
            ]
        )

        # Render screen content
        window_size = self.window.get_framebuffer_size()
        self.screen_renderer.render_to_screen(
            render_pass, window_size, self.letterbox_geometry
        )

        # End render pass and submit commands
        render_pass.end()
        command_buffer = command_encoder.finish()
        self.resource_manager.queue.submit([command_buffer])

        # Present the frame
        surface_texture.present()

    def add_tile_to_screen(
        self,
        char_code: int,
        x_tile: int,
        y_tile: int,
        color: colors.Color,
        bg_color: colors.Color | None = None,
    ) -> None:
        """Add a tile to the screen renderer for batched rendering."""
        if self.screen_renderer is None or self.uv_map is None:
            return

        # Convert tile position to pixel coordinates
        px_x, px_y = self.console_to_screen_coords(x_tile, y_tile)
        tile_w, tile_h = self._tile_dimensions

        # Get UV coordinates for the character
        uv_coords = tuple(self.uv_map[char_code % 256])

        # Convert color to RGBA tuple
        color_rgba = (
            color.r / 255.0,
            color.g / 255.0,
            color.b / 255.0,
            1.0,  # Full opacity
        )

        # Add quad to screen renderer
        self.screen_renderer.add_quad(
            x=px_x,
            y=px_y,
            w=tile_w,
            h=tile_h,
            uv_coords=uv_coords,
            color_rgba=color_rgba,
        )

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

    def texture_from_numpy(
        self, pixels: np.ndarray, transparent: bool = True
    ) -> wgpu.GPUTexture:
        """Create a WGPU texture from numpy array."""
        # TODO: Implement WGPU texture creation from numpy
        raise NotImplementedError("WGPU texture creation not yet implemented")

    def present_texture(
        self,
        texture: wgpu.GPUTexture,
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

    # --- Atlas Texture Loading Methods ---

    def _load_atlas_texture(self) -> wgpu.GPUTexture:
        """Loads the tileset PNG and creates a WGPU texture."""
        if self.resource_manager is None:
            raise RuntimeError("Resource manager not initialized")

        img = PILImage.open(str(config.TILESET_PATH)).convert("RGBA")
        pixels = np.array(img, dtype="u1")

        # Handle magenta transparency (same as ModernGL version)
        magenta_mask = (
            (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 255)
        )
        pixels[magenta_mask, 3] = 0

        # Create WGPU texture using resource manager
        return self.resource_manager.create_atlas_texture(
            width=img.size[0],
            height=img.size[1],
            data=pixels.tobytes(),
            format="rgba8unorm",
        )

    def _precalculate_uv_map(self) -> np.ndarray:
        """Pre-calculates UV coordinates for all 256 possible characters."""
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

    def _setup_coordinate_converter(self) -> None:
        """Set up the coordinate converter with current window dimensions."""

        width, height = self.window.get_framebuffer_size()
        self._coordinate_converter = CoordinateConverter(
            console_width_in_tiles=self.console_width_tiles,
            console_height_in_tiles=self.console_height_tiles,
            renderer_scaled_width=width,
            renderer_scaled_height=height,
        )

    def _calculate_letterbox_geometry(self) -> None:
        """Calculate letterbox geometry for the current window size."""

        if self._coordinate_converter is None:
            return

        # Use full window size for now
        width, height = self.window.get_framebuffer_size()
        self.letterbox_geometry = (0, 0, width, height)
