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
from catley.util.coordinates import (
    CoordinateConverter,
    Rect,
    convert_particle_to_screen_coords,
)
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.effects.particles import ParticleLayer, SubTileParticleSystem
from catley.view.render.graphics import GraphicsContext
from catley.view.render.viewport import ViewportSystem

from .resource_manager import WGPUResourceManager
from .screen_renderer import WGPUScreenRenderer
from .shader_manager import WGPUShaderManager
from .texture_renderer import WGPUTextureRenderer
from .ui_texture_renderer import WGPUUITextureRenderer

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class WGPUGraphicsContext(GraphicsContext):
    """WGPU graphics context implementation."""

    def __init__(self, window: GlfwWindow, *, _defer_init: bool = False) -> None:
        """Initialize WGPU graphics context.

        Args:
            window: The GLFW window to render to
            _defer_init: Internal parameter to defer initialization for testing
        """
        super().__init__()
        self.window = window
        self.vsync = config.VSYNC

        self.device: wgpu.GPUDevice | None = None
        self.queue: wgpu.GPUQueue | None = None
        self.window_wrapper: Any = None  # WGPUWindowWrapper
        self.wgpu_context: Any = None  # GPUCanvasContext
        self.swap_chain: Any = None

        # Resource managers (initialized after device)
        self.resource_manager: WGPUResourceManager | None = None
        self.shader_manager: WGPUShaderManager | None = None

        # Atlas texture and UV mapping
        self.atlas_texture: wgpu.GPUTexture | None = None
        self.uv_map: np.ndarray | None = None

        # Screen renderer
        self.screen_renderer: WGPUScreenRenderer | None = None

        # Texture renderer for off-screen rendering
        self.texture_renderer: WGPUTextureRenderer | None = None

        # UI texture renderer for arbitrary textures
        self.ui_texture_renderer: WGPUUITextureRenderer | None = None
        # Background renderer for immediate texture rendering
        self.background_renderer = None
        # Environmental effect renderer for batched effect rendering
        self.environmental_effect_renderer = None

        # Letterbox/viewport state
        self.letterbox_geometry: tuple[int, int, int, int] | None = None

        # Initialize coordinate converter placeholder (set up after initialize())
        self._coordinate_converter: CoordinateConverter | None = None
        self._tile_dimensions = (20, 20)  # Default tile size

        # Character code for solid block used in tile highlighting
        self.SOLID_BLOCK_CHAR = 219

        # Reusable CPU buffer for WorldView (to avoid per-frame allocations)
        self._world_view_cpu_buffer: np.ndarray | None = None

        # Initialize immediately unless deferred for testing
        if not _defer_init:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize WGPU device and context."""
        # Create WGPU adapter and device first (required before getting context)
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=wgpu.PowerPreference.high_performance  # type: ignore
        )
        self.device = adapter.request_device_sync(
            required_features=[],
            required_limits={},
            label="catley_main_device",
        )
        self.queue = self.device.queue

        # Create the initial rendering surface and context.
        if not self._recreate_surface_and_context():
            # This can happen if the window starts minimized.
            # The first call to update_dimensions will fix it.
            print("Warning: Window started with zero size. Deferring surface creation.")

        # Initialize resource managers (these only need to be created once)
        self.resource_manager = WGPUResourceManager(self.device, self.queue)
        self.shader_manager = WGPUShaderManager(self.device)

        # Define surface format for renderer initialization
        surface_format = wgpu.TextureFormat.bgra8unorm

        # Load atlas texture and prepare UV mapping
        self.atlas_texture = self._load_atlas_texture()
        self.uv_map = self._precalculate_uv_map()

        # Initialize screen renderer with the correct surface format
        self.screen_renderer = WGPUScreenRenderer(
            self.resource_manager,
            self.shader_manager,
            self.atlas_texture,
            surface_format,
        )

        # Initialize texture renderer for off-screen rendering
        self.texture_renderer = WGPUTextureRenderer(
            self.resource_manager,
            self.shader_manager,
            self.atlas_texture,
            self._tile_dimensions,
            self.uv_map,
        )

        # Initialize UI texture renderer
        self.ui_texture_renderer = WGPUUITextureRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        # Initialize background renderer for immediate rendering
        from .background_renderer import WGPUBackgroundRenderer

        self.background_renderer = WGPUBackgroundRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        # Initialize environmental effect renderer for batched effect rendering
        from .environmental_effect_renderer import WGPUEnvironmentalEffectRenderer

        self.environmental_effect_renderer = WGPUEnvironmentalEffectRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        # Set up dimensions and coordinate converter
        self.update_dimensions()

        # Pre-compile shaders to avoid runtime compilation stutters
        self._precompile_shaders()

        # Apply vsync setting after WGPU initialization
        import glfw

        glfw.swap_interval(1 if self.vsync else 0)

    def _precompile_shaders(self) -> None:
        """Pre-compile commonly used shaders to avoid runtime compilation stutters.

        This ensures that shaders are compiled during initialization rather than
        on first use, preventing frame drops when particles/effects first appear.
        """
        if self.shader_manager is None:
            return

        # Pre-load commonly used shaders
        shaders_to_precompile = [
            "wgsl/screen/main.wgsl",  # Main screen rendering (particles, actors)
            "wgsl/glyph/render.wgsl",  # Glyph buffer rendering
            "wgsl/ui/texture.wgsl",  # UI texture rendering
        ]

        for shader_path in shaders_to_precompile:
            try:
                # Loading the shader source triggers caching
                self.shader_manager.load_shader_source(shader_path)
                # Create shader module to trigger compilation
                self.shader_manager.create_shader_module(shader_path, shader_path)
            except Exception as e:
                # Log but don't fail initialization if a shader can't be pre-compiled
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to pre-compile shader {shader_path}: {e}"
                )

    def _recreate_surface_and_context(self) -> bool:
        """
        Creates (or recreates) the WGPU surface and rendering context.
        This is the core fix for the resize bug, ensuring a fresh surface
        is used instead of trying to reconfigure a stale one.

        Returns:
            bool: True if the surface was created successfully, False otherwise.
        """
        from .window_wrapper import WGPUWindowWrapper

        # CRITICAL: Prevent configuring a zero-size surface, which is an error.
        # This happens when the window is minimized.
        width, height = self.window.get_framebuffer_size()
        if width == 0 or height == 0:
            return False  # Skip creation/recreation until we have a valid size

        # Create WGPU window wrapper from our existing GLFW window
        self.window_wrapper = WGPUWindowWrapper(self.window)

        # Get WGPU context (GPUSurface) from the window wrapper
        self.wgpu_context = self.window_wrapper.get_context()

        # Configure the WGPU context
        surface_format = wgpu.TextureFormat.bgra8unorm
        self.wgpu_context.configure(
            device=self.device,
            format=surface_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,  # type: ignore
        )
        return True

    @property
    def tile_dimensions(self) -> TileDimensions:
        """Get tile dimensions."""
        return self._tile_dimensions

    @property
    def console_width_tiles(self) -> int:
        """Get console width in tiles."""
        return config.SCREEN_WIDTH

    @property
    def console_height_tiles(self) -> int:
        """Get console height in tiles."""
        return config.SCREEN_HEIGHT

    def render_glyph_buffer_to_texture(
        self,
        glyph_buffer: GlyphBuffer,
        canvas_vbo: wgpu.GPUBuffer | None = None,
        cpu_buffer_override: np.ndarray | None = None,
        cache_key_suffix: str = "",
    ) -> wgpu.GPUTexture:
        """Render GlyphBuffer to a WGPU texture."""
        if self.texture_renderer is None:
            raise RuntimeError("Texture renderer not initialized")

        # Handle case where no CPU buffer is provided (e.g., WorldView)
        if cpu_buffer_override is None:
            # This is the WorldView case. Reuse our internal buffer.
            from .texture_renderer import TEXTURE_VERTEX_DTYPE

            max_vertices = glyph_buffer.width * glyph_buffer.height * 6

            # Check if our cached buffer is big enough, if not, recreate it.
            if (
                self._world_view_cpu_buffer is None
                or len(self._world_view_cpu_buffer) < max_vertices
            ):
                self._world_view_cpu_buffer = np.zeros(
                    max_vertices, dtype=TEXTURE_VERTEX_DTYPE
                )

            cpu_buffer_to_use = self._world_view_cpu_buffer
        else:
            # This is a UI Canvas case. Use the buffer it provided.
            cpu_buffer_to_use = cpu_buffer_override

        # Pass the chosen buffer and cache key to the renderer
        return self.texture_renderer.render(
            glyph_buffer,
            cpu_buffer_to_use,
            buffer_override=canvas_vbo,
            cache_key_suffix=cache_key_suffix,
        )

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
        if self.screen_renderer is None or self.uv_map is None:
            return

        # Convert color to 0-1 range
        base_color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        # Apply colored lighting using a blend that preserves actor color and light tint
        # This approach ensures actors are visibly tinted by colored lights
        # We use a mix of multiplication (for shading) and addition (for color tint)
        final_color = (
            min(
                1.0, base_color[0] * light_intensity[0] * 0.7 + light_intensity[0] * 0.3
            ),
            min(
                1.0, base_color[1] * light_intensity[1] * 0.7 + light_intensity[1] * 0.3
            ),
            min(
                1.0, base_color[2] * light_intensity[2] * 0.7 + light_intensity[2] * 0.3
            ),
            1.0,  # Always fully opaque
        )
        uv_coords = self.uv_map[ord(char)]

        # Use integer tile dimensions like TCOD backend for consistent positioning
        tile_w, tile_h = self.tile_dimensions

        # Screen renderer should now work correctly!
        self.screen_renderer.add_quad(
            screen_x, screen_y, tile_w, tile_h, uv_coords, final_color
        )

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get display scale factor for high-DPI displays."""
        try:
            # Get the actual content scale from GLFW for high-DPI displays
            import glfw

            scale_x, scale_y = glfw.get_window_content_scale(self.window.glfw_window)
            return (scale_x, scale_y)
        except (AttributeError, TypeError, Exception):
            # Fallback for tests or when GLFW window is not available
            return (1.0, 1.0)

    def draw_mouse_cursor(self, cursor_manager: CursorManager) -> None:
        """Draw the mouse cursor."""
        if self.ui_texture_renderer is None or self.resource_manager is None:
            return

        cursor_data = cursor_manager.cursors.get(cursor_manager.active_cursor_type)
        if not cursor_data:
            return

        # Create texture if not already cached
        if cursor_data.texture is None:
            # Convert numpy array to WGPU texture
            pixels = cursor_data.pixels
            height, width = pixels.shape[:2]
            cursor_data.texture = self.resource_manager.create_atlas_texture(
                width, height, pixels.tobytes(), "rgba8unorm"
            )

        texture = cursor_data.texture
        hotspot_x, hotspot_y = cursor_data.hotspot

        # Calculate position adjusted for hotspot with proper scaling
        scale_factor = 2  # Make cursor 2x tile size for visibility
        dest_x = cursor_manager.mouse_pixel_x - hotspot_x * scale_factor
        dest_y = cursor_manager.mouse_pixel_y - hotspot_y * scale_factor
        dest_w = texture.width * scale_factor
        dest_h = texture.height * scale_factor

        # Create vertices for the cursor quad
        vertices = np.zeros(6, dtype=self.ui_texture_renderer.VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Standard UV coordinates
        color_rgba = (1.0, 1.0, 1.0, 1.0)  # White color, no tint

        # Calculate quad corners
        px_x1, px_y1 = dest_x, dest_y
        px_x2, px_y2 = dest_x + dest_w, dest_y + dest_h

        # Create triangle vertices (2 triangles = 6 vertices)
        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        # Queue the cursor for batched rendering
        self.ui_texture_renderer.add_textured_quad(texture, vertices)

    def draw_tile_highlight(
        self,
        root_x: int,
        root_y: int,
        color: colors.Color,
        alpha: Opacity,
    ) -> None:
        """Draw a tile highlight."""
        if self.screen_renderer is None or self.uv_map is None:
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
        view_offset: RootConsoleTilePos,
        viewport_system: ViewportSystem | None = None,
    ) -> None:
        """Render particles by adding each one to the vertex buffer."""
        if self.screen_renderer is None or self.uv_map is None:
            return

        # Loop through active particles, same logic as ModernGL implementation
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

            # Render this particle using the same approach as draw_actor_smooth
            self._draw_particle_to_buffer(
                particle_system.chars[i],
                tuple(particle_system.colors[i]),
                screen_x,
                screen_y,
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
        """Draw a single particle by adding its quad to the vertex buffer."""
        if self.screen_renderer is None or self.uv_map is None:
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

    def apply_environmental_effect(
        self,
        position: tuple[float, float],
        radius: float,
        tint_color: colors.Color,
        intensity: float,
        blend_mode: BlendMode,
    ) -> None:
        """Apply a circular environmental effect like lighting or fog."""
        if radius <= 0 or intensity <= 0 or self.environmental_effect_renderer is None:
            return

        # Convert tile coordinates to screen coordinates
        screen_x, screen_y = self.console_to_screen_coords(*position)

        # Calculate pixel radius based on tile dimensions
        px_radius = max(1, int(radius * self.tile_dimensions[0]))
        size = px_radius * 2

        # Calculate final color with intensity
        final_color = (
            tint_color[0] / 255.0 * intensity,
            tint_color[1] / 255.0 * intensity,
            tint_color[2] / 255.0 * intensity,
            intensity,
        )

        # Position the effect centered on the given position
        dest_x = screen_x - px_radius
        dest_y = screen_y - px_radius

        # Add to the batched environmental effect renderer
        self.environmental_effect_renderer.add_effect_quad(
            dest_x, dest_y, size, size, final_color
        )

    def update_dimensions(self) -> None:
        """Update dimensions when window size changes."""

        # Recreate the WGPU surface and context to match the new window size.
        if self.device:
            self._recreate_surface_and_context()

        # Update coordinate converter
        self._setup_coordinate_converter()

        # Update letterbox geometry
        self._calculate_letterbox_geometry()

    def prepare_to_present(self) -> None:
        """Prepare for presenting the frame."""
        if (
            self.screen_renderer is None
            or self.ui_texture_renderer is None
            or self.background_renderer is None
            or self.environmental_effect_renderer is None
        ):
            return
        self.screen_renderer.begin_frame()
        self.ui_texture_renderer.begin_frame()
        self.background_renderer.begin_frame()
        self.environmental_effect_renderer.begin_frame()

    def finalize_present(self) -> None:
        """Finalize frame presentation by rendering to screen."""

        if (
            self.screen_renderer is None
            or self.wgpu_context is None
            or self.resource_manager is None
        ):
            return

        try:
            # Get the current surface texture from WGPU context
            surface_texture = self.wgpu_context.get_current_texture()

            # Create command encoder
            command_encoder = self.resource_manager.device.create_command_encoder()

            # Get window size and create surface view
            window_size = self.window.get_framebuffer_size()
            surface_view = surface_texture.create_view()

            # Create render pass that clears the surface for background rendering
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": surface_view,
                        "resolve_target": None,
                        "clear_value": (
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ),  # Black background
                        "load_op": "clear",  # Clear for fresh background
                        "store_op": "store",
                    }
                ]
            )

            # First render background textures (lowest layer)
            if self.background_renderer is not None:
                self.background_renderer.render(
                    render_pass, window_size, self.letterbox_geometry
                )

            # Then render screen content (actors, particles, tiles)
            self.screen_renderer.render_to_screen(
                render_pass, window_size, self.letterbox_geometry
            )

            # Then render environmental effects (atmospheric overlays)
            if (
                self.environmental_effect_renderer is not None
                and self.environmental_effect_renderer.vertex_count > 0
            ):
                self.environmental_effect_renderer.render(
                    render_pass, window_size, self.letterbox_geometry
                )

            # Finally render UI content on top (menus, dialogs, etc.)
            if (
                self.ui_texture_renderer is not None
                and self.ui_texture_renderer.vertex_count > 0
            ):
                self.ui_texture_renderer.render(
                    render_pass, window_size, self.letterbox_geometry
                )

            render_pass.end()

            command_buffer = command_encoder.finish()
            self.resource_manager.queue.submit([command_buffer])

        finally:
            # Always present the frame, even if rendering failed
            self.wgpu_context.present()

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
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
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
        if self.letterbox_geometry is None:
            # Fallback calculation if letterbox not initialized
            return PixelPos(
                (
                    int(console_x * self._tile_dimensions[0]),
                    int(console_y * self._tile_dimensions[1]),
                )
            )

        # Use the same calculation as ModernGL backend
        offset_x, offset_y, scaled_w, scaled_h = self.letterbox_geometry
        screen_x = offset_x + console_x * (scaled_w / self.console_width_tiles)
        screen_y = offset_y + console_y * (scaled_h / self.console_height_tiles)
        return (screen_x, screen_y)

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

    def texture_from_numpy(
        self, pixels: np.ndarray, transparent: bool = True
    ) -> wgpu.GPUTexture:
        """Create a WGPU texture from numpy array."""
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

        # Use the resource manager to create the texture
        assert self.resource_manager is not None, "Resource manager not initialized"
        return self.resource_manager.create_atlas_texture(
            width, height, pixels.tobytes(), "rgba8unorm"
        )

    def present_texture(
        self,
        texture: wgpu.GPUTexture,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Present a texture to the screen."""
        if not isinstance(texture, wgpu.GPUTexture) or self.ui_texture_renderer is None:
            return

        # Convert tile coordinates to pixel coordinates
        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Create vertices for this texture quad
        vertices = np.zeros(6, dtype=self.ui_texture_renderer.VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Use non-flipped UVs
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        # Queue the texture and vertices for batched rendering
        self.ui_texture_renderer.add_textured_quad(texture, vertices)

    def draw_background(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Queue background texture for batched rendering."""
        if not isinstance(texture, wgpu.GPUTexture) or self.background_renderer is None:
            return

        # Convert tile coordinates to pixel coordinates
        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Create vertices for background quad using background renderer's VERTEX_DTYPE
        from .background_renderer import VERTEX_DTYPE

        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Use non-flipped UVs
        color_rgba = (1.0, 1.0, 1.0, 1.0)  # White, no tint

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        # Queue the background texture for batched rendering
        self.background_renderer.add_background_quad(texture, vertices)

    def draw_debug_rect(
        self, px_x: int, px_y: int, px_w: int, px_h: int, color: colors.Color
    ) -> None:
        """Draw debug rectangle."""
        if self.screen_renderer is None or self.uv_map is None:
            return

        color_rgba = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, 1.0)
        uv_coords = self.uv_map[self.SOLID_BLOCK_CHAR]

        # Top edge
        self.screen_renderer.add_quad(px_x, px_y, px_w, 1, uv_coords, color_rgba)
        # Bottom edge
        self.screen_renderer.add_quad(
            px_x, px_y + px_h - 1, px_w, 1, uv_coords, color_rgba
        )
        # Left edge
        self.screen_renderer.add_quad(px_x, px_y, 1, px_h, uv_coords, color_rgba)
        # Right edge
        self.screen_renderer.add_quad(
            px_x + px_w - 1, px_y, 1, px_h, uv_coords, color_rgba
        )

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
        from catley.backends.wgpu.canvas import WGPUCanvas

        # Pass the resource_manager to the canvas constructor
        assert self.resource_manager is not None
        return WGPUCanvas(self, self.resource_manager, transparent)

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
            texture_format="rgba8unorm",
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
        """Set up the coordinate converter with letterbox scaled dimensions."""
        if self.letterbox_geometry is not None:
            # Use letterbox scaled dimensions like ModernGL
            _, _, scaled_w, scaled_h = self.letterbox_geometry
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=scaled_w,
                renderer_scaled_height=scaled_h,
            )
        else:
            # Fallback to window dimensions
            width, height = self.window.get_framebuffer_size()
            self._coordinate_converter = CoordinateConverter(
                console_width_in_tiles=self.console_width_tiles,
                console_height_in_tiles=self.console_height_tiles,
                renderer_scaled_width=width,
                renderer_scaled_height=height,
            )

    def _calculate_letterbox_geometry(self) -> None:
        """Calculate letterbox geometry for the current window size."""
        window_width, window_height = map(int, self.window.get_framebuffer_size())
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

        self.letterbox_geometry = (offset_x, offset_y, scaled_w, scaled_h)

        # Calculate dynamic tile dimensions like ModernGL - use full window dimensions
        tile_width = max(1, window_width // self.console_width_tiles)
        tile_height = max(1, window_height // self.console_height_tiles)
        self._tile_dimensions = (tile_width, tile_height)

        # Update coordinate converter
        self._setup_coordinate_converter()

    def cleanup(self) -> None:
        """Clean up WGPU resources before shutdown."""
        if self.resource_manager is not None:
            self.resource_manager.release_all()

        # Clear references to help with cleanup order
        self.screen_renderer = None
        self.texture_renderer = None
        self.ui_texture_renderer = None
        self.background_renderer = None
        self.environmental_effect_renderer = None
        self.atlas_texture = None
        self.shader_manager = None
        self.resource_manager = None
        self._world_view_cpu_buffer = None

        # Let WGPU context cleanup happen naturally
        self.wgpu_context = None
        self.window_wrapper = None
