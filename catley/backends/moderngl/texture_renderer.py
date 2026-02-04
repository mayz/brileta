# catley/backends/moderngl/texture_renderer.py

import moderngl
import numpy as np

from catley.util.caching import ResourceCache
from catley.util.glyph_buffer import GlyphBuffer
from catley.util.live_vars import record_time_live_variable
from catley.util.tilesets import unicode_to_cp437

from . import get_uniform
from .shader_manager import ShaderManager

# Vertex structure for rendering
VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("fg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
        ("bg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class TextureRenderer:
    """
    A specialized class for rendering GlyphBuffer objects into moderngl.Texture objects.
    This isolates the logic for UI rendering from the main game world rendering.
    """

    # Maximum dimensions we expect for glyph buffers - can be adjusted if needed
    MAX_BUFFER_WIDTH = 200
    MAX_BUFFER_HEIGHT = 100

    def __init__(
        self,
        mgl_context: moderngl.Context,
        atlas_texture: moderngl.Texture,
        tile_dimensions: tuple[int, int],
        uv_map: np.ndarray,
    ) -> None:
        """
        Initialize the TextureRenderer.

        Args:
            mgl_context: The ModernGL context
            atlas_texture: The main atlas texture for glyphs
            tile_dimensions: (width, height) of each tile in pixels
            uv_map: Pre-calculated UV coordinates for all 256 characters
        """
        self.mgl_context = mgl_context
        self.atlas_texture = atlas_texture
        self.tile_dimensions = tile_dimensions
        self.uv_map = uv_map
        self.shader_manager = ShaderManager(mgl_context)
        self.SOLID_BLOCK_CHAR = 9608  # Unicode █ character

        # Pre-compute unicode_to_cp437 mapping for common characters (0-255)
        # This eliminates thousands of Python function calls per frame
        self.unicode_to_cp437_map = np.array(
            [unicode_to_cp437(i) for i in range(256)], dtype=np.uint8
        )

        # Pre-compute the background character CP437 value (solid block)
        self.bg_char_cp437 = unicode_to_cp437(self.SOLID_BLOCK_CHAR)

        # Create dedicated shader program for texture rendering
        self.texture_program = self._create_texture_shader_program()

        # Pre-allocate vertex buffer for maximum expected size
        max_cells = self.MAX_BUFFER_WIDTH * self.MAX_BUFFER_HEIGHT
        max_vertices = max_cells * 6  # 1 quad * 6 vertices per quad
        self.cpu_vertex_buffer = np.zeros(max_vertices, dtype=VERTEX_DTYPE)

        # Create persistent VBO with dynamic updates
        # Initialize with zeros to ensure clean state
        initial_data = np.zeros(max_vertices * VERTEX_DTYPE.itemsize, dtype=np.uint8)
        self.vbo = self.mgl_context.buffer(initial_data.tobytes(), dynamic=True)
        self.vao = self.mgl_context.vertex_array(
            self.texture_program,
            [
                (
                    self.vbo,
                    "2f 2f 4f 4f",
                    "in_vert",
                    "in_uv",
                    "in_fg_color",
                    "in_bg_color",
                )
            ],
        )

        # Cache for change detection (TCOD-style optimization)
        # Uses ResourceCache for proper LRU eviction
        self.buffer_cache: ResourceCache[int, GlyphBuffer] = ResourceCache(
            name="texture_renderer_buffers", max_size=100
        )

    def _create_texture_shader_program(self) -> moderngl.Program:
        """
        Creates the GLSL program for rendering to an off-screen texture (FBO).
        This shader normalizes coordinates to the texture's own size and does NOT
        flip the Y-axis.
        """
        program = self.shader_manager.create_program(
            "glsl/glyph/render.vert", "glsl/glyph/render.frag", "texture_renderer"
        )
        get_uniform(program, "u_atlas").value = 0
        return program

    def _encode_glyph_buffer_to_vertices(
        self, glyph_buffer: GlyphBuffer
    ) -> tuple[np.ndarray, int]:
        """
        Translates a GlyphBuffer into the pre-allocated vertex buffer.
        Fully vectorized - no Python loops over cells.

        Args:
            glyph_buffer: The GlyphBuffer to encode

        Returns:
            Tuple of (vertex_data slice, vertex_count) for compatibility
        """
        w, h = glyph_buffer.data.shape
        num_cells = w * h
        if num_cells == 0:
            return np.array([], dtype=VERTEX_DTYPE), 0

        num_vertices = num_cells * 6
        if num_vertices > len(self.cpu_vertex_buffer):
            raise ValueError(
                f"GlyphBuffer too large: needs {num_vertices} vertices, "
                f"buffer has {len(self.cpu_vertex_buffer)}"
            )

        tile_w, tile_h = self.tile_dimensions

        # Build coordinate grids matching original loop order (y outer, x inner).
        # meshgrid with indexing="xy" gives (h, w) arrays where:
        #   x_grid[y, x] = x, y_grid[y, x] = y
        x_indices = np.arange(w, dtype=np.float32)
        y_indices = np.arange(h, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_indices, y_indices, indexing="xy")

        # Screen coordinates for each cell corner - shape (h, w)
        x1 = x_grid * tile_w
        y1 = y_grid * tile_h
        x2 = x1 + tile_w
        y2 = y1 + tile_h

        # Swap axes from (w, h, ...) to (h, w, ...) to match grid order
        fg_colors = (
            np.swapaxes(glyph_buffer.data["fg"], 0, 1).astype(np.float32) / 255.0
        )  # (h, w, 4)
        bg_colors = (
            np.swapaxes(glyph_buffer.data["bg"], 0, 1).astype(np.float32) / 255.0
        )

        # Convert characters to CP437 indices for UV lookup
        chars = glyph_buffer.data["ch"].T  # (w, h) -> (h, w)
        # Clamp to 0-255 for lookup table (rare chars > 255 become 0)
        cp437_indices = np.where(
            chars < 256, self.unicode_to_cp437_map[np.clip(chars, 0, 255)], 0
        )

        # Look up UV coordinates: uv_map is (256, 4) -> (h, w, 4)
        uvs = self.uv_map[cp437_indices]
        u1 = uvs[..., 0]
        v1 = uvs[..., 1]
        u2 = uvs[..., 2]
        v2 = uvs[..., 3]

        # Reshape vertex buffer to (h, w, 6) - matches row-major iteration order
        verts = self.cpu_vertex_buffer[:num_vertices].reshape(h, w, 6)

        # Vertex 0: bottom-left
        verts["position"][..., 0, 0] = x1
        verts["position"][..., 0, 1] = y1
        verts["uv"][..., 0, 0] = u1
        verts["uv"][..., 0, 1] = v1

        # Vertex 1: bottom-right
        verts["position"][..., 1, 0] = x2
        verts["position"][..., 1, 1] = y1
        verts["uv"][..., 1, 0] = u2
        verts["uv"][..., 1, 1] = v1

        # Vertex 2: top-left
        verts["position"][..., 2, 0] = x1
        verts["position"][..., 2, 1] = y2
        verts["uv"][..., 2, 0] = u1
        verts["uv"][..., 2, 1] = v2

        # Vertex 3: bottom-right (same as 1)
        verts["position"][..., 3, 0] = x2
        verts["position"][..., 3, 1] = y1
        verts["uv"][..., 3, 0] = u2
        verts["uv"][..., 3, 1] = v1

        # Vertex 4: top-left (same as 2)
        verts["position"][..., 4, 0] = x1
        verts["position"][..., 4, 1] = y2
        verts["uv"][..., 4, 0] = u1
        verts["uv"][..., 4, 1] = v2

        # Vertex 5: top-right
        verts["position"][..., 5, 0] = x2
        verts["position"][..., 5, 1] = y2
        verts["uv"][..., 5, 0] = u2
        verts["uv"][..., 5, 1] = v2

        # Colors are the same for all 6 vertices of each cell
        # Broadcast (h, w, 4) to (h, w, 6, 4)
        for i in range(6):
            verts["fg_color"][..., i, :] = fg_colors
            verts["bg_color"][..., i, :] = bg_colors

        return self.cpu_vertex_buffer[:num_vertices], num_vertices

    def render(
        self,
        glyph_buffer: GlyphBuffer,
        target_fbo: moderngl.Framebuffer,
        vbo_override: moderngl.Buffer | None = None,
        vao_override: moderngl.VertexArray | None = None,
    ) -> None:
        """
        Renders a GlyphBuffer into the provided target framebuffer.

        Uses vectorized vertex encoding for fast updates. Skips rendering
        if the buffer hasn't changed since last frame.

        Args:
            glyph_buffer: The GlyphBuffer to render
            target_fbo: The pre-allocated framebuffer to render into
            vbo_override: Optional VBO to use instead of the default one
            vao_override: Optional VAO to use instead of the default one
        """
        vbo_to_use = vbo_override if vbo_override is not None else self.vbo
        vao_to_use = vao_override if vao_override is not None else self.vao

        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        if width_px == 0 or height_px == 0:
            return

        buffer_id = id(glyph_buffer)
        cache_buffer = self.buffer_cache.get(buffer_id)

        # Fast path: skip if buffer hasn't changed at all
        if cache_buffer is not None and not self._has_changes(
            glyph_buffer, cache_buffer
        ):
            return

        # --- Something changed - do a full re-render ---
        with record_time_live_variable("cpu.texture_renderer.fbo_bind_clear_ms"):
            target_fbo.use()
            # Clear to transparent black (0, 0, 0, 0) so alpha blending works correctly
            # for overlay textures. Default clear color is opaque black which prevents
            # transparent cells from being truly transparent.
            target_fbo.clear(0.0, 0.0, 0.0, 0.0)

        with record_time_live_variable("cpu.texture_renderer.vbo_update_ms"):
            self._update_complete_vbo(glyph_buffer, vbo_to_use)

        # Now, render the entire buffer. The VBO is up-to-date.
        # Unchanged tiles use their old vertex data, dirty tiles have new data.
        with record_time_live_variable("cpu.texture_renderer.render_ms"):
            self._render_complete_buffer(glyph_buffer, target_fbo, vao_to_use)

        # Update cache with current buffer state for the next frame's comparison.
        self._update_cache(glyph_buffer, buffer_id)

        # Restore default framebuffer
        if self.mgl_context.screen:
            self.mgl_context.screen.use()

    def _has_changes(
        self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer
    ) -> bool:
        """
        Check if the buffer has any changes since last frame.

        Args:
            glyph_buffer: Current frame's buffer
            cache_buffer: Previous frame's buffer

        Returns:
            True if any tiles changed, False if identical
        """
        # Dimension mismatch means changes
        if (
            cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        ):
            return True

        # Vectorized comparison - check if ANY cell differs
        has_char_diff = np.any(glyph_buffer.data["ch"] != cache_buffer.data["ch"])
        if has_char_diff:
            return True

        has_fg_diff = np.any(glyph_buffer.data["fg"] != cache_buffer.data["fg"])
        if has_fg_diff:
            return True

        return bool(np.any(glyph_buffer.data["bg"] != cache_buffer.data["bg"]))

    def _update_complete_vbo(
        self, glyph_buffer: GlyphBuffer, vbo: moderngl.Buffer
    ) -> None:
        """
        Update the entire VBO with data from the complete buffer.
        Used for first render or when buffer dimensions change.

        Args:
            glyph_buffer: The GlyphBuffer to encode completely
            vbo: The VBO to update
        """
        vertex_data, vertex_count = self._encode_glyph_buffer_to_vertices(glyph_buffer)

        if vertex_count > 0:
            vbo.orphan()
            vbo.write(vertex_data.tobytes())

    def _render_complete_buffer(
        self,
        glyph_buffer: GlyphBuffer,
        target_fbo: moderngl.Framebuffer,
        vao: moderngl.VertexArray,
    ) -> None:
        """
        Render only the vertices that correspond to the current buffer size.

        Args:
            glyph_buffer: The GlyphBuffer being rendered
            target_fbo: Target framebuffer for rendering
            vao: The VAO to use for rendering
        """
        # Calculate pixel dimensions for shader uniforms
        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        # Only render vertices for the current buffer size, not the entire VBO
        total_vertices = glyph_buffer.width * glyph_buffer.height * 6

        # Set uniforms and render only the vertices for this buffer
        get_uniform(self.texture_program, "u_texture_size").value = (
            width_px,
            height_px,
        )
        self.atlas_texture.use(location=0)

        # Critical: only render the vertices that belong to this buffer
        vao.render(moderngl.TRIANGLES, vertices=total_vertices)

    def _update_cache(self, glyph_buffer: GlyphBuffer, buffer_id: int) -> None:
        """
        Update the cache with the current buffer state.

        Args:
            glyph_buffer: Current buffer to cache
            buffer_id: ID of the buffer for cache key
        """
        # Get or create cache buffer
        cache_buffer = self.buffer_cache.get(buffer_id)

        if cache_buffer is None:
            # First time caching this buffer
            cache_buffer = GlyphBuffer(glyph_buffer.width, glyph_buffer.height)
        elif (
            cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        ):
            # Buffer dimensions changed - recreate
            cache_buffer = GlyphBuffer(glyph_buffer.width, glyph_buffer.height)

        # Deep copy the data to preserve for next frame comparison
        np.copyto(cache_buffer.data, glyph_buffer.data)

        # Store in cache (ResourceCache handles LRU eviction)
        self.buffer_cache.store(buffer_id, cache_buffer)

    def release(self) -> None:
        """Clean up GPU resources."""
        self.vao.release()
        self.vbo.release()
        self.texture_program.release()
