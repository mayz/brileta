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
)
from catley.util.coordinates import (
    CoordinateConverter,
)
from catley.util.glyph_buffer import GlyphBuffer
from catley.view.render.base_graphics import BaseGraphicsContext

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
