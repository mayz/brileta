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
)
from catley.util.coordinates import (
    CoordinateConverter,
)
from catley.util.glyph_buffer import GlyphBuffer
from catley.util.tilesets import derive_outlined_atlas
from catley.view.render.base_graphics import BaseGraphicsContext

from .atmospheric_renderer import WGPUAtmosphericRenderer
from .resource_manager import WGPUResourceManager
from .screen_renderer import WGPUScreenRenderer
from .shader_manager import WGPUShaderManager
from .texture_renderer import WGPUTextureRenderer
from .ui_texture_renderer import WGPUUITextureRenderer

if TYPE_CHECKING:
    from catley.view.ui.cursor_manager import CursorManager


class WGPUGraphicsContext(BaseGraphicsContext):
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
        self.outlined_atlas_texture: wgpu.GPUTexture | None = None
        self.uv_map: np.ndarray | None = None
        self._atlas_pixels: np.ndarray | None = None

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
        # Atmospheric renderer for cloud shadows and mist
        self.atmospheric_renderer: WGPUAtmosphericRenderer | None = None
        # Queued atmospheric layers for this frame
        self._atmospheric_layers: list[tuple] = []

        # Letterbox/viewport state
        self.letterbox_geometry: tuple[int, int, int, int] | None = None

        # Reusable CPU buffer for WorldView (to avoid per-frame allocations)
        self._world_view_cpu_buffer: np.ndarray | None = None

        # Initialize immediately unless deferred for testing
        if not _defer_init:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize WGPU device and context."""
        # Create WGPU adapter and device first (required before getting context)
        # Note: This takes ~290ms due to GPU enumeration - unavoidable with WGPU.
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=wgpu.PowerPreference.high_performance
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
        self.outlined_atlas_texture = self._load_outlined_atlas_texture()
        self.uv_map = self._precalculate_uv_map()

        # Initialize all renderers
        # Note: First pipeline creation pays ~165ms for Metal shader compiler init.
        # This is unavoidable with current wgpu-py (no Metal Binary Archive support).
        self.screen_renderer = WGPUScreenRenderer(
            self.resource_manager,
            self.shader_manager,
            self.atlas_texture,
            surface_format,
        )

        self.texture_renderer = WGPUTextureRenderer(
            self.resource_manager,
            self.shader_manager,
            self.atlas_texture,
            self._tile_dimensions,
            self.uv_map,
        )

        self.ui_texture_renderer = WGPUUITextureRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        from .background_renderer import WGPUBackgroundRenderer

        self.background_renderer = WGPUBackgroundRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        from .environmental_effect_renderer import WGPUEnvironmentalEffectRenderer

        self.environmental_effect_renderer = WGPUEnvironmentalEffectRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        self.atmospheric_renderer = WGPUAtmosphericRenderer(
            self.resource_manager,
            self.shader_manager,
            surface_format,
        )

        # Set up dimensions and coordinate converter
        self.update_dimensions()

        # Note: Shader precompilation is not needed here because all renderers
        # compile their shaders during pipeline creation above. The shader manager
        # caches compiled modules, so calling _precompile_shaders() would just
        # do redundant cache lookups.

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
        # Pass vsync setting so WGPU uses the correct present mode (fifo vs immediate)
        self.window_wrapper = WGPUWindowWrapper(self.window, vsync=self.vsync)

        # Get WGPU context (GPUSurface) from the window wrapper
        self.wgpu_context = self.window_wrapper.get_context("wgpu")

        # Configure the WGPU context
        surface_format = wgpu.TextureFormat.bgra8unorm
        self.wgpu_context.configure(  # type: ignore[unresolved-attribute]
            device=self.device,
            format=surface_format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        return True

    def render_glyph_buffer_to_texture(
        self,
        glyph_buffer: GlyphBuffer,
        buffer_override: wgpu.GPUBuffer | None = None,
        secondary_override: np.ndarray | None = None,
        cache_key_suffix: str = "",
    ) -> wgpu.GPUTexture:
        """Render GlyphBuffer to a WGPU texture."""
        if self.texture_renderer is None or self.resource_manager is None:
            raise RuntimeError("Texture renderer not initialized")

        # Handle case where no CPU buffer is provided (e.g., WorldView)
        if secondary_override is None:
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
            cpu_buffer_to_use = secondary_override

        # Pass the chosen buffer and cache key to the renderer
        return self.texture_renderer.render(
            glyph_buffer,
            cpu_buffer_to_use,
            buffer_override=buffer_override,
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
        scale_x: float = 1.0,
        scale_y: float = 1.0,
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

        # Use integer tile dimensions for consistent positioning
        tile_w, tile_h = self.tile_dimensions

        # Apply non-uniform scaling: scale dimensions and center on tile
        scaled_w = tile_w * scale_x
        scaled_h = tile_h * scale_y
        offset_x = (tile_w - scaled_w) / 2
        offset_y = (tile_h - scaled_h) / 2

        self.screen_renderer.add_quad(
            screen_x + offset_x,
            screen_y + offset_y,
            scaled_w,
            scaled_h,
            uv_coords,
            final_color,
        )

    def draw_actor_outline(
        self,
        char: str,
        screen_x: float,
        screen_y: float,
        color: colors.Color,
        alpha: float,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draws an outlined glyph using the pre-generated outlined atlas.

        The outlined atlas contains white outlines which are tinted by the color
        parameter and rendered with the specified alpha for shimmer effects.
        """
        if (
            self.ui_texture_renderer is None
            or self.uv_map is None
            or self.outlined_atlas_texture is None
        ):
            return

        # Convert color to 0-1 range with alpha
        final_color = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            max(0.0, min(1.0, alpha)),
        )

        uv_coords = self.uv_map[ord(char)]
        u1, v1, u2, v2 = uv_coords
        tile_w, tile_h = self.tile_dimensions

        # Apply scaling and center on tile
        scaled_w = tile_w * scale_x
        scaled_h = tile_h * scale_y
        offset_x = (tile_w - scaled_w) / 2
        offset_y = (tile_h - scaled_h) / 2

        # Calculate final position
        x = screen_x + offset_x
        y = screen_y + offset_y

        # Create vertices for the outlined quad
        vertices = np.zeros(6, dtype=self.ui_texture_renderer.VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), final_color)
        vertices[1] = ((x + scaled_w, y), (u2, v1), final_color)
        vertices[2] = ((x, y + scaled_h), (u1, v2), final_color)
        vertices[3] = ((x + scaled_w, y), (u2, v1), final_color)
        vertices[4] = ((x, y + scaled_h), (u1, v2), final_color)
        vertices[5] = ((x + scaled_w, y + scaled_h), (u2, v2), final_color)

        # Queue the outlined texture for batched rendering
        self.ui_texture_renderer.add_textured_quad(
            self.outlined_atlas_texture, vertices
        )

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get display scale factor for high-DPI displays."""
        try:
            # Get the actual content scale from GLFW for high-DPI displays
            import glfw

            scale_x, scale_y = glfw.get_window_content_scale(self.window.glfw_window)  # type: ignore[possibly-missing-attribute]
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
        self._calculate_letterbox_geometry_and_tiles()

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
        if self.atmospheric_renderer is not None:
            self.atmospheric_renderer.begin_frame()
        self._atmospheric_layers = []

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

            # Then render atmospheric layers (cloud shadows, ground mist)
            if self._atmospheric_layers and self.atmospheric_renderer is not None:
                letterbox = self.letterbox_geometry or (
                    0,
                    0,
                    int(window_size[0]),
                    int(window_size[1]),
                )
                for layer_state in self._atmospheric_layers:
                    (
                        viewport_offset,
                        viewport_size,
                        map_size,
                        sky_exposure_threshold,
                        sky_exposure_texture,
                        explored_texture,
                        visible_texture,
                        noise_scale,
                        noise_threshold_low,
                        noise_threshold_high,
                        strength,
                        tint_color,
                        drift_offset,
                        turbulence_offset,
                        turbulence_strength,
                        turbulence_scale,
                        blend_mode,
                        pixel_bounds,
                    ) = layer_state
                    self.atmospheric_renderer.render(
                        render_pass,
                        window_size,
                        letterbox,
                        viewport_offset,
                        viewport_size,
                        map_size,
                        sky_exposure_threshold,
                        sky_exposure_texture,
                        explored_texture,
                        visible_texture,
                        noise_scale,
                        noise_threshold_low,
                        noise_threshold_high,
                        strength,
                        tint_color,
                        drift_offset,
                        turbulence_offset,
                        turbulence_strength,
                        turbulence_scale,
                        blend_mode,
                        pixel_bounds,
                    )

            # Then render environmental effects (discrete effects like smoke puffs)
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
            # Always present the frame, even if rendering failed.
            # rendercanvas wraps wgpu's GPUCanvasContext - access the underlying
            # context to call present() since we manage our own render loop.
            self.wgpu_context._wgpu_context.present()

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

    def texture_from_numpy(
        self, pixels: np.ndarray, transparent: bool = True
    ) -> wgpu.GPUTexture:
        """Create a WGPU texture from numpy array."""
        # Use base class helper to preprocess pixels
        pixels = self._preprocess_pixels_for_texture(pixels, transparent)

        height, width = pixels.shape[:2]

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

    def draw_texture_alpha(
        self,
        texture: wgpu.GPUTexture,
        screen_x: PixelCoord,
        screen_y: PixelCoord,
        alpha: Opacity,
    ) -> None:
        """Draw a texture at pixel coordinates with alpha modulation."""
        if alpha <= 0.0 or not isinstance(texture, wgpu.GPUTexture):
            return
        if self.ui_texture_renderer is None:
            return

        px_x1 = float(screen_x)
        px_y1 = float(screen_y)
        px_x2 = px_x1 + texture.width
        px_y2 = px_y1 + texture.height

        # Create vertices with alpha in the color
        vertices = np.zeros(6, dtype=self.ui_texture_renderer.VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        color_rgba = (1.0, 1.0, 1.0, float(alpha))

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        self.ui_texture_renderer.add_textured_quad(texture, vertices)

    def draw_background(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
        offset_x_pixels: float = 0.0,
        offset_y_pixels: float = 0.0,
    ) -> None:
        """Queue background texture for batched rendering."""
        if not isinstance(texture, wgpu.GPUTexture) or self.background_renderer is None:
            return

        # Convert tile coordinates to pixel coordinates
        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Apply screen shake pixel offset
        px_x1 += offset_x_pixels
        px_x2 += offset_x_pixels
        px_y1 += offset_y_pixels
        px_y2 += offset_y_pixels

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

        # Store pixels for outlined atlas generation
        self._atlas_pixels = pixels

        # Create WGPU texture using resource manager
        return self.resource_manager.create_atlas_texture(
            width=img.size[0],
            height=img.size[1],
            data=pixels.tobytes(),
            texture_format="rgba8unorm",
        )

    def _load_outlined_atlas_texture(self) -> wgpu.GPUTexture | None:
        """Creates an outlined version of the tileset atlas.

        Uses the derive_outlined_atlas utility to create a tileset where each
        glyph is replaced with just its 1-pixel outline. The outline is white
        so it can be tinted to any color at render time.

        Returns None if required resources are not available (e.g., in tests).
        """
        if self.resource_manager is None or self._atlas_pixels is None:
            return None

        # Calculate tile dimensions from the atlas
        height, width = self._atlas_pixels.shape[:2]
        tile_width = width // config.TILESET_COLUMNS
        tile_height = height // config.TILESET_ROWS

        # Generate the outlined atlas in white so it can be tinted to any color
        outlined_pixels = derive_outlined_atlas(
            self._atlas_pixels,
            tile_width,
            tile_height,
            config.TILESET_COLUMNS,
            config.TILESET_ROWS,
            color=(255, 255, 255, 255),
        )

        # Create WGPU texture using resource manager
        return self.resource_manager.create_atlas_texture(
            width=width,
            height=height,
            data=outlined_pixels.tobytes(),
            texture_format="rgba8unorm",
        )

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

    def _calculate_letterbox_geometry_and_tiles(self) -> None:
        """Calculate letterbox geometry for the current window size."""
        window_width, window_height = map(int, self.window.get_framebuffer_size())
        self.letterbox_geometry = super()._calculate_letterbox_geometry(
            window_width, window_height
        )

        # Calculate dynamic tile dimensions like ModernGL - use full window dimensions
        tile_width = max(1, window_width // self.console_width_tiles)
        tile_height = max(1, window_height // self.console_height_tiles)
        self._tile_dimensions = (tile_width, tile_height)

        # Update coordinate converter
        self._setup_coordinate_converter()

    def set_atmospheric_layer(
        self,
        viewport_offset: tuple[int, int],
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: wgpu.GPUTexture | None,
        explored_texture: wgpu.GPUTexture | None,
        visible_texture: wgpu.GPUTexture | None,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        strength: float,
        tint_color: tuple[int, int, int],
        drift_offset: tuple[float, float],
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        blend_mode: str,
        pixel_bounds: tuple[int, int, int, int],
    ) -> None:
        """Store atmospheric layer data for rendering this frame."""
        self._atmospheric_layers.append(
            (
                viewport_offset,
                viewport_size,
                map_size,
                sky_exposure_threshold,
                sky_exposure_texture,
                explored_texture,
                visible_texture,
                noise_scale,
                noise_threshold_low,
                noise_threshold_high,
                strength,
                tint_color,
                drift_offset,
                turbulence_offset,
                turbulence_strength,
                turbulence_scale,
                blend_mode,
                pixel_bounds,
            )
        )

    def draw_debug_tile_grid(
        self,
        view_origin: tuple[int, int],
        view_size: tuple[int, int],
        offset_pixels: tuple[float, float],
    ) -> None:
        """Render a 1-pixel cyan tile grid overlay for the given view bounds."""
        if self.screen_renderer is None or self.uv_map is None:
            return

        origin_x, origin_y = view_origin
        width_tiles, height_tiles = view_size
        offset_x, offset_y = offset_pixels

        px_left, px_top = self.console_to_screen_coords(origin_x, origin_y)
        px_right, px_bottom = self.console_to_screen_coords(
            origin_x + width_tiles, origin_y + height_tiles
        )

        left = round(px_left + offset_x)
        top = round(px_top + offset_y)
        width_px = round(px_right - px_left)
        height_px = round(px_bottom - px_top)

        if width_px <= 0 or height_px <= 0:
            return

        uv = self.uv_map[self.SOLID_BLOCK_CHAR]
        color_rgba = (
            colors.CYAN[0] / 255.0,
            colors.CYAN[1] / 255.0,
            colors.CYAN[2] / 255.0,
            1.0,
        )

        # Vertical grid lines
        for i in range(width_tiles + 1):
            px_x, _ = self.console_to_screen_coords(origin_x + i, origin_y)
            x = round(px_x + offset_x)
            self.screen_renderer.add_quad(x, top, 1, height_px, uv, color_rgba)

        # Horizontal grid lines
        for j in range(height_tiles + 1):
            _, px_y = self.console_to_screen_coords(origin_x, origin_y + j)
            y = round(px_y + offset_y)
            self.screen_renderer.add_quad(left, y, width_px, 1, uv, color_rgba)

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
        self.atmospheric_renderer = None
        self.atlas_texture = None
        self.outlined_atlas_texture = None
        self.shader_manager = None
        self.resource_manager = None
        self._world_view_cpu_buffer = None
        self._atlas_pixels = None

        # Let WGPU context cleanup happen naturally
        self.wgpu_context = None
        self.window_wrapper = None

    def release_texture(self, texture: Any) -> None:
        """Release WGPU texture resources."""
        if texture is not None and hasattr(texture, "destroy"):
            texture.destroy()
