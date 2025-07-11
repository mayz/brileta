# catley/backends/moderngl/texture_renderer.py

import moderngl
import numpy as np

from catley.util.caching import ResourceCache
from catley.util.glyph_buffer import GlyphBuffer
from catley.util.tilesets import unicode_to_cp437

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
        program["u_atlas"].value = 0
        return program

    def _encode_glyph_buffer_to_vertices(
        self, glyph_buffer: GlyphBuffer
    ) -> tuple[np.ndarray, int]:
        """
        Translates a GlyphBuffer into the pre-allocated vertex buffer.

        Args:
            glyph_buffer: The GlyphBuffer to encode

        Returns:
            Tuple of (vertex_data slice, vertex_count) for compatibility
        """
        w, h = glyph_buffer.data.shape
        num_cells = h * w
        if num_cells == 0:
            return np.array([], dtype=VERTEX_DTYPE), 0

        # Verify buffer size is sufficient
        max_vertices_needed = num_cells * 6  # 1 quad * 6 vertices per quad
        if max_vertices_needed > len(self.cpu_vertex_buffer):
            raise ValueError(
                f"GlyphBuffer too large: needs {max_vertices_needed} vertices, "
                f"buffer has {len(self.cpu_vertex_buffer)}"
            )

        tile_w, tile_h = self.tile_dimensions
        vertex_idx = 0

        # Pre-normalize colors for all cells using vectorized operations
        fg_colors_raw = glyph_buffer.data["fg"]
        bg_colors_raw = glyph_buffer.data["bg"]
        fg_colors_norm = fg_colors_raw.astype(np.float32) / 255.0
        bg_colors_norm = bg_colors_raw.astype(np.float32) / 255.0

        # Pre-calculate screen coordinates to avoid per-cell multiplication
        screen_x_coords = np.arange(w) * tile_w
        screen_y_coords = np.arange(h) * tile_h
        screen_x2_coords = screen_x_coords + tile_w
        screen_y2_coords = screen_y_coords + tile_h

        for y_console in range(h):
            # Cache screen Y coordinates for this row
            screen_y = screen_y_coords[y_console]
            screen_y2 = screen_y2_coords[y_console]

            for x_console in range(w):
                cell = glyph_buffer.data[x_console, y_console]
                char = cell["ch"]

                # Use pre-calculated screen coordinates
                screen_x = screen_x_coords[x_console]
                screen_x2 = screen_x2_coords[x_console]

                # Get pre-normalized colors
                fg_color_norm = fg_colors_norm[x_console, y_console]
                bg_color_norm = bg_colors_norm[x_console, y_console]

                # Convert Unicode character to CP437 and get UV coordinates
                # Fast array lookup for common characters, fallback for others
                if char < 256:
                    fg_char = self.unicode_to_cp437_map[char]
                else:
                    fg_char = unicode_to_cp437(char)  # Rare edge case

                fg_uv = self.uv_map[fg_char]
                u1, v1, u2, v2 = fg_uv

                # Write ONE quad (6 vertices) with both colors
                self.cpu_vertex_buffer[vertex_idx : vertex_idx + 6] = [
                    ((screen_x, screen_y), (u1, v1), fg_color_norm, bg_color_norm),
                    ((screen_x2, screen_y), (u2, v1), fg_color_norm, bg_color_norm),
                    ((screen_x, screen_y2), (u1, v2), fg_color_norm, bg_color_norm),
                    ((screen_x2, screen_y), (u2, v1), fg_color_norm, bg_color_norm),
                    ((screen_x, screen_y2), (u1, v2), fg_color_norm, bg_color_norm),
                    ((screen_x2, screen_y2), (u2, v2), fg_color_norm, bg_color_norm),
                ]
                vertex_idx += 6

        return self.cpu_vertex_buffer[:vertex_idx], vertex_idx

    def render(
        self,
        glyph_buffer: GlyphBuffer,
        target_fbo: moderngl.Framebuffer,
        # NEW: Optional arguments for UI isolation
        vbo_override: moderngl.Buffer | None = None,
        vao_override: moderngl.VertexArray | None = None,
    ) -> None:
        """
        Renders a GlyphBuffer into the provided target framebuffer.
        Uses TCOD-style change detection to only process dirty tiles.

        Args:
            glyph_buffer: The GlyphBuffer to render
            target_fbo: The pre-allocated framebuffer to render into
            vbo_override: Optional VBO to use instead of the default one
            vao_override: Optional VAO to use instead of the default one
        """
        # --- Determine which VBO/VAO to use ---
        vbo_to_use = vbo_override if vbo_override is not None else self.vbo
        vao_to_use = vao_override if vao_override is not None else self.vao
        # Calculate the pixel dimensions for the shader uniforms
        width_px = glyph_buffer.width * self.tile_dimensions[0]
        height_px = glyph_buffer.height * self.tile_dimensions[1]

        if width_px == 0 or height_px == 0:
            return  # Nothing to render

        buffer_id = id(glyph_buffer)
        cache_buffer = self.buffer_cache.get(buffer_id)
        needs_full_update = (
            cache_buffer is None
            or cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        )

        if not needs_full_update:
            dirty_tiles = self._find_dirty_tiles(glyph_buffer, cache_buffer)
            if not dirty_tiles:
                return  # Fast path for totally static buffers
        else:
            # If it's a full update, all tiles are dirty by definition.
            dirty_tiles = [
                (x, y)
                for x in range(glyph_buffer.width)
                for y in range(glyph_buffer.height)
            ]

        # --- If we are here, something has changed and we need to draw ---
        target_fbo.use()
        target_fbo.clear()  # ALWAYS clear the destination texture to prevent artifacts.

        if needs_full_update:
            # A new buffer is being rendered, or dimensions changed.
            # Do a full, single-write VBO update to overwrite any stale data.
            self._update_complete_vbo(glyph_buffer, vbo_to_use)
        else:
            # An existing buffer changed. Do a partial VBO update for performance.
            self._update_dirty_tiles_in_vbo(glyph_buffer, dirty_tiles, vbo_to_use)

        # Now, render the entire buffer. The VBO is up-to-date.
        # Unchanged tiles use their old vertex data, dirty tiles have new data.
        self._render_complete_buffer(glyph_buffer, target_fbo, vao_to_use)

        # Update cache with current buffer state for the next frame's comparison.
        self._update_cache(glyph_buffer, buffer_id)

        # Restore default framebuffer
        if self.mgl_context.screen:
            self.mgl_context.screen.use()

    def _find_dirty_tiles(
        self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer | None
    ) -> list[tuple[int, int]]:
        """
        Find tiles that have changed since the last frame using vectorized comparison.

        Args:
            glyph_buffer: Current frame's buffer
            cache_buffer: Previous frame's buffer (None if first time)

        Returns:
            List of (x, y) coordinates of dirty tiles
        """
        if cache_buffer is None:
            # First time rendering this buffer - all tiles are dirty
            return [
                (x, y)
                for x in range(glyph_buffer.width)
                for y in range(glyph_buffer.height)
            ]

        # Check for dimension mismatch - if dimensions changed, all tiles are dirty
        if (
            cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        ):
            return [
                (x, y)
                for x in range(glyph_buffer.width)
                for y in range(glyph_buffer.height)
            ]

        # Vectorized comparison using NumPy - this is the key TCOD optimization
        # Compare character, foreground, and background data
        dirty_mask = (
            (glyph_buffer.data["ch"] != cache_buffer.data["ch"])
            | np.any(glyph_buffer.data["fg"] != cache_buffer.data["fg"], axis=-1)
            | np.any(glyph_buffer.data["bg"] != cache_buffer.data["bg"], axis=-1)
        )

        # Get coordinates of dirty tiles
        dirty_coords = np.argwhere(dirty_mask)
        return [(int(coord[0]), int(coord[1])) for coord in dirty_coords.tolist()]

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
        # Use the original complete encoding method
        vertex_data, vertex_count = self._encode_glyph_buffer_to_vertices(glyph_buffer)

        if vertex_count > 0:
            # Update the entire VBO with new vertex data
            vbo.write(vertex_data.tobytes())

    def _update_dirty_tiles_in_vbo(
        self,
        glyph_buffer: GlyphBuffer,
        dirty_tiles: list[tuple[int, int]],
        vbo: moderngl.Buffer,
    ) -> None:
        """
        Update only the dirty tiles in the VBO - the real TCOD optimization.
        This generates vertices for only the changed tiles and updates their
        specific sections of the VBO.

        Args:
            glyph_buffer: The current GlyphBuffer
            dirty_tiles: List of (x, y) coordinates of tiles that changed
            vbo: The VBO to update
        """
        for x, y in dirty_tiles:
            # Generate vertices for just this one tile
            tile_vertices = self._generate_vertices_for_single_tile(
                x, y, glyph_buffer.data[x, y]
            )

            # Calculate VBO offset for this specific tile
            # Each tile uses 6 vertices (1 quad * 6 vertices each)
            tile_index = (y * glyph_buffer.width) + x
            vbo_offset = tile_index * 6 * VERTEX_DTYPE.itemsize

            # Update only this tile's section of the VBO
            vbo.write(tile_vertices.tobytes(), offset=vbo_offset)

    def _generate_vertices_for_single_tile(
        self, x: int, y: int, tile_data: np.ndarray
    ) -> np.ndarray:
        """
        Generate the 6 vertices (1 quad) for a single tile.
        This is extracted from the main vertex generation logic.

        Args:
            x: Tile X coordinate
            y: Tile Y coordinate
            tile_data: Single tile's data from GlyphBuffer.data[x, y]

        Returns:
            Array of 6 vertices for this tile
        """
        # Extract tile data
        char = int(tile_data["ch"])
        fg_color = tile_data["fg"].astype(np.float32) / 255.0
        bg_color = tile_data["bg"].astype(np.float32) / 255.0

        # Convert character to CP437 for atlas lookup
        if char < 256:
            fg_char = self.unicode_to_cp437_map[char]
        else:
            fg_char = unicode_to_cp437(char)

        # Get UV coordinates for foreground character
        fg_uv = self.uv_map[fg_char]
        u1, v1, u2, v2 = fg_uv

        # Calculate screen position for this tile
        screen_x = x * self.tile_dimensions[0]
        screen_y = y * self.tile_dimensions[1]
        tile_width, tile_height = self.tile_dimensions

        # Create the 6 vertices for this tile (single quad with both colors)
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)

        # Single quad (6 vertices) - structured assignment
        vertices[0]["position"] = (screen_x, screen_y)
        vertices[0]["uv"] = (u1, v1)
        vertices[0]["fg_color"] = fg_color
        vertices[0]["bg_color"] = bg_color

        vertices[1]["position"] = (screen_x + tile_width, screen_y)
        vertices[1]["uv"] = (u2, v1)
        vertices[1]["fg_color"] = fg_color
        vertices[1]["bg_color"] = bg_color

        vertices[2]["position"] = (screen_x, screen_y + tile_height)
        vertices[2]["uv"] = (u1, v2)
        vertices[2]["fg_color"] = fg_color
        vertices[2]["bg_color"] = bg_color

        vertices[3]["position"] = (screen_x + tile_width, screen_y)
        vertices[3]["uv"] = (u2, v1)
        vertices[3]["fg_color"] = fg_color
        vertices[3]["bg_color"] = bg_color

        vertices[4]["position"] = (screen_x, screen_y + tile_height)
        vertices[4]["uv"] = (u1, v2)
        vertices[4]["fg_color"] = fg_color
        vertices[4]["bg_color"] = bg_color

        vertices[5]["position"] = (screen_x + tile_width, screen_y + tile_height)
        vertices[5]["uv"] = (u2, v2)
        vertices[5]["fg_color"] = fg_color
        vertices[5]["bg_color"] = bg_color

        return vertices

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
        self.texture_program["u_texture_size"].value = (width_px, height_px)
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
