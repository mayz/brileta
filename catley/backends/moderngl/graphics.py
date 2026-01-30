from typing import Any

import moderngl
import numpy as np
from PIL import Image as PILImage

from catley import colors, config
from catley.backends.gl_window import GLWindow
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

from .atmospheric_renderer import ModernGLAtmosphericRenderer
from .resource_manager import ModernGLResourceManager
from .screen_renderer import VERTEX_DTYPE, ScreenRenderer
from .shader_manager import ShaderManager
from .texture_renderer import TextureRenderer


class UITextureRenderer:
    """
    A batched renderer for textured quads where each quad can have a
    different source texture. This version is optimized to reuse GPU resources.
    """

    # Define a reasonable max number of UI quads per frame.
    # This is for reserving buffer space.
    MAX_UI_QUADS = 500

    def __init__(self, mgl_context: moderngl.Context) -> None:
        self.mgl_context = mgl_context
        self.shader_manager = ShaderManager(mgl_context)
        self.render_queue: list[tuple[moderngl.Texture, int]] = []

        # Create shader program for UI textures
        self.program = self.shader_manager.create_program(
            "glsl/ui/texture.vert", "glsl/ui/texture.frag", "ui_texture_renderer"
        )
        self.program["u_texture"].value = 0  # type: ignore[invalid-assignment]

        # Create CPU-side buffer to aggregate all vertex data
        self.cpu_vertex_buffer = np.zeros(self.MAX_UI_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0

        # Create dynamic VBO large enough for all UI quads
        self.vbo = self.mgl_context.buffer(
            reserve=self.MAX_UI_QUADS * 6 * VERTEX_DTYPE.itemsize, dynamic=True
        )
        self.vao = self.mgl_context.vertex_array(
            self.program,
            [(self.vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )

    def begin_frame(self) -> None:
        """Clear the internal queue of textures to be rendered for the new frame."""
        self.render_queue.clear()
        self.vertex_count = 0

    def add_textured_quad(
        self, texture: moderngl.Texture, vertices: np.ndarray
    ) -> None:
        """Add a texture and its corresponding vertex data to the internal
        render queue."""
        # Copy vertices into the CPU buffer
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

        # Queue only the texture and vertex count for this quad
        self.render_queue.append((texture, 6))

    def render(self, letterbox_geometry: tuple[int, int, int, int]) -> None:
        """Render all queued textured quads."""
        if not self.render_queue or self.vertex_count == 0:
            return

        # Set letterbox uniform once for all UI elements
        self.program["u_letterbox"].value = letterbox_geometry  # type: ignore[invalid-assignment]

        # Upload all vertex data to GPU in a single operation
        self.vbo.write(self.cpu_vertex_buffer[: self.vertex_count].tobytes())

        # Render each quad with the correct texture binding
        vertex_offset = 0
        for texture, vertex_count in self.render_queue:
            # Bind the specific texture for this draw call
            texture.use(location=0)

            # Render this quad using the correct slice of the VBO
            self.vao.render(
                moderngl.TRIANGLES, vertices=vertex_count, first=vertex_offset
            )

            vertex_offset += vertex_count

    def release(self) -> None:
        """Clean up GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.program.release()


class ModernGLGraphicsContext(BaseGraphicsContext):
    """
    A high-performance GraphicsContext that uses ModernGL. It uses a two-shader
    architecture to correctly handle different rendering contexts.
    """

    def __init__(self, window: GLWindow) -> None:
        super().__init__()
        self.window = window
        self.mgl_context = moderngl.create_context()
        self.mgl_context.enable(moderngl.BLEND)
        self.mgl_context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.mgl_context.disable(moderngl.CULL_FACE)

        # --- Shader and Texture Setup ---
        self.atlas_texture = self._load_atlas_texture()
        self.outlined_atlas_texture = self._load_outlined_atlas_texture()
        self.uv_map = self._precalculate_uv_map()

        # --- Shared Resource Manager for GPU Resources ---
        self.resource_manager = ModernGLResourceManager(self.mgl_context)

        # --- Reusable Radial Gradient Texture for Environmental Effects ---
        self.radial_gradient_texture = self._create_radial_gradient_texture(256)

        # --- ScreenRenderer for main screen rendering ---
        self.screen_renderer = ScreenRenderer(self.mgl_context, self.atlas_texture)
        self.atmospheric_renderer = ModernGLAtmosphericRenderer(self.mgl_context)
        self._atmospheric_layers: list[
            tuple[
                tuple[int, int],
                tuple[int, int],
                tuple[int, int],
                float,
                moderngl.Texture | None,
                moderngl.Texture | None,
                moderngl.Texture | None,
                float,
                float,
                float,
                float,
                tuple[int, int, int],
                tuple[float, float],
                float,
                float,
                float,
                str,
                tuple[int, int, int, int],
            ]
        ] = []

        # Set initial viewport and update dimensions (this will set _tile_dimensions)
        window_width, window_height = map(int, self.window.get_framebuffer_size())
        self.mgl_context.viewport = (0, 0, window_width, window_height)
        self.update_dimensions()

        # --- TextureRenderer for UI rendering (created after dimensions are set) ---
        self.texture_renderer = TextureRenderer(
            self.mgl_context, self.atlas_texture, self._tile_dimensions, self.uv_map
        )

        # --- UITextureRenderer for batched texture rendering ---
        self.ui_texture_renderer = UITextureRenderer(self.mgl_context)

        # --- Persistent VBO/VAO for Immediate Rendering (after renderers created) ---
        self._create_immediate_render_resources()

    def _draw_single_texture_immediately(
        self, texture: moderngl.Texture, vertices: np.ndarray
    ) -> None:
        """Helper to render a single quad with a custom texture immediately."""
        # Use persistent resources instead of creating temp VBO/VAO
        program = self.ui_texture_renderer.program
        program["u_letterbox"].value = self.letterbox_geometry  # type: ignore[invalid-assignment]

        texture.use(location=0)

        # Upload vertices to persistent VBO and render with persistent VAO
        self.immediate_vbo.write(vertices.tobytes())
        self.immediate_vao_ui.render(moderngl.TRIANGLES, vertices=6)

        # IMPORTANT: Restore the main atlas texture for the ScreenRenderer
        self.atlas_texture.use(location=0)

    def draw_background(
        self,
        texture: moderngl.Texture,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
        offset_x_pixels: float = 0.0,
        offset_y_pixels: float = 0.0,
    ) -> None:
        """Draws the main world background texture. This is an immediate draw call
        that should happen before other rendering."""
        if not isinstance(texture, moderngl.Texture):
            return

        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Apply screen shake pixel offset
        px_x1 += offset_x_pixels
        px_x2 += offset_x_pixels
        px_y1 += offset_y_pixels
        px_y2 += offset_y_pixels

        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0  # Use non-flipped UVs
        color_rgba = (1.0, 1.0, 1.0, 1.0)  # White, no tint

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        self._draw_single_texture_immediately(texture, vertices)

    # --- Initialization Helpers ---

    def _load_atlas_texture(self) -> moderngl.Texture:
        """Loads the tileset PNG and creates a ModernGL texture."""
        img = PILImage.open(str(config.TILESET_PATH)).convert("RGBA")
        pixels = np.array(img, dtype="u1")
        magenta_mask = (
            (pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 255)
        )
        pixels[magenta_mask, 3] = 0
        # Store pixels for outlined atlas generation
        self._atlas_pixels = pixels
        texture = self.mgl_context.texture(img.size, 4, pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return texture

    def _load_outlined_atlas_texture(self) -> moderngl.Texture | None:
        """Creates an outlined version of the tileset atlas.

        Uses the derive_outlined_atlas utility to create a tileset where each
        glyph is replaced with just its 1-pixel outline. The outline is white
        so it can be tinted to any color at render time.

        Returns None if the atlas pixels are not available (e.g., in tests).
        """
        # Handle case where atlas pixels are not available (mocked in tests)
        if not hasattr(self, "_atlas_pixels") or self._atlas_pixels is None:
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

        # Create the texture
        texture = self.mgl_context.texture(
            (width, height), 4, outlined_pixels.tobytes()
        )
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return texture

    def _get_or_create_render_target(
        self, width: int, height: int, cache_key_suffix: str = ""
    ) -> tuple[moderngl.Framebuffer, moderngl.Texture]:
        """
        Get or create a cached FBO and texture for the given dimensions.
        This prevents per-frame GPU resource creation/destruction.
        """
        return self.resource_manager.get_or_create_simple_fbo(
            width, height, cache_key_suffix
        )

    def _create_radial_gradient_texture(self, resolution: int) -> moderngl.Texture:
        """
        Create a reusable white radial gradient texture for environmental effects.
        This prevents per-frame texture creation in apply_environmental_effect.
        """
        center = resolution // 2

        # Create gradient using numpy
        y_coords, x_coords = np.ogrid[:resolution, :resolution]
        distances = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)

        # Create alpha gradient that fades from center to edge
        alpha = np.clip(1.0 - distances / center, 0.0, 1.0)

        # Create RGBA texture data (white with alpha gradient)
        effect_data = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        effect_data[:, :, :3] = 255  # White base color
        effect_data[:, :, 3] = (alpha * 255).astype(np.uint8)

        # Create the texture
        texture = self.mgl_context.texture(
            (resolution, resolution), 4, effect_data.tobytes()
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        return texture

    def _create_immediate_render_resources(self) -> None:
        """
        Create persistent VBO/VAO resources for immediate rendering operations.
        This prevents per-frame GPU resource creation in immediate drawing methods.
        """
        # Create a persistent VBO for single quad immediate rendering (6 vertices)
        self.immediate_vbo = self.mgl_context.buffer(
            reserve=6 * VERTEX_DTYPE.itemsize, dynamic=True
        )

        # Create persistent VAOs for different shader programs
        self.immediate_vao_screen = self.mgl_context.vertex_array(
            self.screen_renderer.screen_program,
            [(self.immediate_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
        )
        self.immediate_vao_ui = self.mgl_context.vertex_array(
            self.ui_texture_renderer.program,
            [(self.immediate_vbo, "2f 2f 4f", "in_vert", "in_uv", "in_color")],
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

    # --- Frame Lifecycle & Main Rendering ---

    def prepare_to_present(self) -> None:
        """Clears the screen and resets the vertex buffer for a new frame."""
        self.mgl_context.clear(0.0, 0.0, 0.0)
        self.screen_renderer.begin_frame()
        self.ui_texture_renderer.begin_frame()
        self._atmospheric_layers = []

    def finalize_present(self) -> None:
        """Uploads all vertex data and draws the scene in a single batch."""
        window_size = self.window.get_framebuffer_size()
        self.screen_renderer.render_to_screen(
            window_size,
            letterbox_geometry=self.letterbox_geometry,
        )
        if self._atmospheric_layers:
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
                    letterbox,
                )
        if self.letterbox_geometry is not None:
            self.ui_texture_renderer.render(self.letterbox_geometry)

    def set_atmospheric_layer(
        self,
        viewport_offset: tuple[int, int],
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: moderngl.Texture | None,
        explored_texture: moderngl.Texture | None,
        visible_texture: moderngl.Texture | None,
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

    # --- GraphicsContext ABC Implementation ---

    @property
    def program(self) -> moderngl.Program:
        """Backward compatibility: returns the screen program."""
        return self.screen_renderer.screen_program

    def render_glyph_buffer_to_texture(
        self,
        glyph_buffer: GlyphBuffer,
        buffer_override: moderngl.Buffer | None = None,
        secondary_override: moderngl.VertexArray | None = None,
        cache_key_suffix: str = "",
    ) -> Any:
        """Renders a GlyphBuffer to a cached, correctly oriented texture."""
        # Calculate the pixel dimensions of the output texture
        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        if width_px == 0 or height_px == 0:
            # Return a minimal 1x1 transparent texture for empty buffers
            empty_data = np.zeros((1, 1, 4), dtype=np.uint8)
            return self.mgl_context.texture((1, 1), 4, empty_data.tobytes())

        # Get or create cached render target
        target_fbo, target_texture = self._get_or_create_render_target(
            width_px, height_px, cache_key_suffix
        )

        # Render into the cached FBO
        # Pass along the optional overrides. If they are None, the
        # TextureRenderer will use its internal defaults.
        self.texture_renderer.render(
            glyph_buffer,
            target_fbo,
            vbo_override=buffer_override,
            vao_override=secondary_override,
        )

        return target_texture

    def present_texture(
        self,
        texture: Any,
        x_tile: int,
        y_tile: int,
        width_tiles: int,
        height_tiles: int,
    ) -> None:
        """Queues a pre-rendered UI texture for batched rendering."""
        if not isinstance(texture, moderngl.Texture):
            return

        px_x1, px_y1 = self.console_to_screen_coords(x_tile, y_tile)
        px_x2, px_y2 = self.console_to_screen_coords(
            x_tile + width_tiles, y_tile + height_tiles
        )

        # Create vertices for this texture quad
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
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
        texture: moderngl.Texture,
        screen_x: PixelCoord,
        screen_y: PixelCoord,
        alpha: Opacity,
    ) -> None:
        """Draw a texture at pixel coordinates with alpha modulation."""
        if alpha <= 0.0 or not isinstance(texture, moderngl.Texture):
            return

        px_x1 = float(screen_x)
        px_y1 = float(screen_y)
        px_x2 = px_x1 + texture.width
        px_y2 = px_y1 + texture.height

        # Create vertices with alpha in the color
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        color_rgba = (1.0, 1.0, 1.0, float(alpha))

        vertices[0] = ((px_x1, px_y1), (u1, v1), color_rgba)
        vertices[1] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[2] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[3] = ((px_x2, px_y1), (u2, v1), color_rgba)
        vertices[4] = ((px_x1, px_y2), (u1, v2), color_rgba)
        vertices[5] = ((px_x2, px_y2), (u2, v2), color_rgba)

        self.ui_texture_renderer.add_textured_quad(texture, vertices)

    def draw_actor_smooth(
        self,
        char: str,
        color: colors.Color,
        screen_x: float,
        screen_y: float,
        light_intensity: tuple[float, float, float] = (1.0, 1.0, 1.0),
        interpolation_alpha: InterpolationAlpha = InterpolationAlpha(1.0),  # noqa: B008
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ) -> None:
        """Draws a single character by adding its quad to the vertex buffer.

        Note: interpolation_alpha is for position interpolation only, NOT for color
        transparency. The color alpha is always 1.0 (fully opaque).
        """
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
            1.0,  # Always fully opaque - interpolation_alpha is NOT used for color
        )
        assert self.uv_map is not None
        uv_coords = self.uv_map[ord(char)]

        # Use integer tile dimensions like TCOD backend for consistent positioning
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
        if self.outlined_atlas_texture is None or self.uv_map is None:
            return

        # Convert color to 0-1 range with alpha
        final_color = (
            color[0] / 255.0,
            color[1] / 255.0,
            color[2] / 255.0,
            max(0.0, min(1.0, alpha)),
        )

        uv_coords = self.uv_map[ord(char)]
        tile_w, tile_h = self.tile_dimensions

        # Apply scaling and center on tile
        scaled_w = tile_w * scale_x
        scaled_h = tile_h * scale_y
        offset_x = (tile_w - scaled_w) / 2
        offset_y = (tile_h - scaled_h) / 2

        # Render immediately using the outlined atlas texture
        self._draw_outlined_quad_immediately(
            screen_x + offset_x,
            screen_y + offset_y,
            scaled_w,
            scaled_h,
            uv_coords,
            final_color,
        )

    def _draw_outlined_quad_immediately(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: np.ndarray,
        color_rgba: tuple[float, float, float, float],
    ) -> None:
        """Helper to render a quad with the outlined atlas texture immediately."""
        # Guard against None (caller should check, but type checker needs this)
        if self.outlined_atlas_texture is None:
            return

        u1, v1, u2, v2 = uv_coords

        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y), (u2, v1), color_rgba)
        vertices[2] = ((x, y + h), (u1, v2), color_rgba)
        vertices[3] = ((x + w, y), (u2, v1), color_rgba)
        vertices[4] = ((x, y + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y + h), (u2, v2), color_rgba)

        # Set up shader uniforms
        program = self.screen_renderer.screen_program
        program["u_letterbox"].value = self.letterbox_geometry  # type: ignore[invalid-assignment]

        # Bind the outlined atlas texture
        self.outlined_atlas_texture.use(location=0)

        # Render using persistent VBO/VAO
        self.immediate_vbo.write(vertices.tobytes())
        self.immediate_vao_screen.render(moderngl.TRIANGLES, vertices=6)

        # Restore the main atlas texture
        self.atlas_texture.use(location=0)

    # --- Coordinate and Dimension Management ---

    def update_dimensions(self) -> None:
        """Recalculates letterboxing and updates the coordinate converter."""
        window_width, window_height = map(int, self.window.get_framebuffer_size())

        # Use base class method to calculate letterbox geometry
        self.letterbox_geometry = self._calculate_letterbox_geometry(
            window_width, window_height
        )

        # Calculate dynamic tile dimensions like TCOD backend for consistent positioning
        tile_width = max(1, window_width // self.console_width_tiles)
        tile_height = max(1, window_height // self.console_height_tiles)
        self._tile_dimensions = (tile_width, tile_height)

        self._coordinate_converter = self._create_coordinate_converter()

        # Set OpenGL viewport to full framebuffer size
        self.mgl_context.viewport = (0, 0, window_width, window_height)

    def _create_coordinate_converter(self) -> CoordinateConverter:
        # Call base class method to set up converter
        self._setup_coordinate_converter()
        assert self._coordinate_converter is not None
        return self._coordinate_converter

    # --- Internal Helpers ---

    def draw_mouse_cursor(self, cursor_manager: Any) -> None:
        """Queues the active cursor for batched rendering."""
        cursor_data = cursor_manager.cursors.get(cursor_manager.active_cursor_type)
        if not cursor_data:
            return

        # Create texture if not already cached
        if cursor_data.texture is None:
            # Convert numpy array to ModernGL texture
            pixels = cursor_data.pixels
            height, width = pixels.shape[:2]
            texture = self.mgl_context.texture((width, height), 4, pixels.tobytes())
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            cursor_data.texture = texture

        texture = cursor_data.texture
        hotspot_x, hotspot_y = cursor_data.hotspot

        # Calculate position adjusted for hotspot with proper scaling
        scale_factor = 2  # Make cursor 2x tile size for visibility
        dest_x = cursor_manager.mouse_pixel_x - hotspot_x * scale_factor
        dest_y = cursor_manager.mouse_pixel_y - hotspot_y * scale_factor
        dest_w = texture.width * scale_factor
        dest_h = texture.height * scale_factor

        # Create vertices for the cursor quad
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
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
        if radius <= 0 or intensity <= 0:
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
        uv_coords = (0.0, 0.0, 1.0, 1.0)  # Use full gradient texture

        # Use immediate rendering approach to avoid batching complications
        # This approach is more direct and avoids needing to manage batch state
        self._draw_environmental_effect_immediately(
            dest_x, dest_y, size, size, uv_coords, final_color
        )

    def _draw_environmental_effect_immediately(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        color_rgba: tuple[float, float, float, float],
    ) -> None:
        """Helper to render a single environmental effect quad immediately."""
        # Create vertices for this effect quad
        u1, v1, u2, v2 = uv_coords
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y), (u2, v1), color_rgba)
        vertices[2] = ((x, y + h), (u1, v2), color_rgba)
        vertices[3] = ((x + w, y), (u2, v1), color_rgba)
        vertices[4] = ((x, y + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y + h), (u2, v2), color_rgba)

        # Set up shader uniforms like ScreenRenderer does
        program = self.screen_renderer.screen_program
        program["u_letterbox"].value = self.letterbox_geometry  # type: ignore[invalid-assignment]

        # Use the reusable gradient texture
        self.radial_gradient_texture.use(location=0)

        # Use persistent VBO/VAO instead of creating temporary resources
        self.immediate_vbo.write(vertices.tobytes())
        self.immediate_vao_screen.render(moderngl.TRIANGLES, vertices=6)

        # Restore atlas texture for normal rendering
        self.atlas_texture.use(location=0)

    def get_display_scale_factor(self) -> tuple[float, float]:
        """Get the (x_scale, y_scale) factor for high-DPI displays."""
        # Get logical window size (what the OS reports)
        logical_w, logical_h = self.window.get_size()
        # Get actual framebuffer size (high-DPI aware)
        physical_w, physical_h = self.window.get_framebuffer_size()

        scale_x = physical_w / logical_w if logical_w > 0 else 1.0
        scale_y = physical_h / logical_h if logical_h > 0 else 1.0

        return (scale_x, scale_y)

    def texture_from_numpy(self, pixels: np.ndarray, transparent: bool = True) -> Any:
        """Creates a backend-specific texture from a raw NumPy RGBA pixel array."""
        # Use base class helper to preprocess pixels
        pixels = self._preprocess_pixels_for_texture(pixels, transparent)

        height, width = pixels.shape[:2]

        texture = self.mgl_context.texture((width, height), 4, pixels.tobytes())
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return texture

    def create_canvas(self, transparent: bool = True) -> Any:
        """Creates a ModernGL canvas that uses GlyphBuffer."""
        from catley.backends.moderngl.canvas import ModernGLCanvas

        return ModernGLCanvas(self, transparent)

    def release_texture(self, texture: Any) -> None:
        """Release ModernGL texture resources."""
        if texture is not None and hasattr(texture, "release"):
            texture.release()
