"""WGPU texture rendering - port of ModernGL version for off-screen rendering."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from catley.types import ViewportTilePos
from catley.util.caching import ResourceCache
from catley.util.glyph_buffer import GlyphBuffer
from catley.util.tilesets import unicode_to_cp437

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

# Maximum number of quads for texture rendering
MAX_TEXTURE_QUADS = 5000

# Vertex structure for glyph rendering (foreground + background colors)
TEXTURE_VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("fg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
        ("bg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class WGPUTextureRenderer:
    """WGPU texture renderer for rendering GlyphBuffer objects to off-screen textures.

    This is the WGPU equivalent of ModernGL's TextureRenderer class.
    It handles render-to-texture functionality for creating textures from glyph buffers.
    """

    # Maximum dimensions we expect for glyph buffers - can be adjusted if needed
    MAX_BUFFER_WIDTH = 200
    MAX_BUFFER_HEIGHT = 100

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        atlas_texture: wgpu.GPUTexture,
        tile_dimensions: tuple[int, int],
        uv_map: np.ndarray,
    ) -> None:
        """Initialize WGPU texture renderer.

        Args:
            resource_manager: WGPU resource manager for buffer/texture caching
            shader_manager: WGPU shader manager for pipeline creation
            atlas_texture: The main tileset atlas texture
            tile_dimensions: (width, height) of each tile in pixels
            uv_map: Pre-calculated UV coordinates for all 256 characters
        """
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.atlas_texture = atlas_texture
        self.tile_dimensions = tile_dimensions
        self.uv_map = uv_map

        # Pre-compute unicode_to_cp437 mapping for common characters (0-255)
        # This eliminates thousands of Python function calls per frame
        self.unicode_to_cp437_map = np.array(
            [unicode_to_cp437(i) for i in range(256)], dtype=np.uint8
        )

        # Note: WGPUTextureRenderer is now stateless.
        # CPU vertex buffers are provided by each canvas, and vertex counts
        # are calculated locally during each render call.

        # Cache for change detection (TCOD-style optimization)
        # Uses ResourceCache for proper LRU eviction
        self.buffer_cache: ResourceCache[int, GlyphBuffer] = ResourceCache(
            name="wgpu_texture_renderer_buffers", max_size=100
        )

        # Cache for rendered textures to avoid re-rendering static buffers
        self.texture_cache: dict[int, wgpu.GPUTexture] = {}

        # Create vertex buffer
        buffer_size = MAX_TEXTURE_QUADS * 6 * TEXTURE_VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="texture_renderer_vertex_buffer",
        )

        # Create uniform buffer for texture size
        self.uniform_buffer = self.resource_manager.get_or_create_buffer(
            size=8,  # vec2<f32> = 8 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="texture_renderer_uniform_buffer",
        )

        # Create pipeline and bind groups
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for texture rendering."""
        # Create bind group layout
        bind_group_layout = self.shader_manager.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ],
            label="texture_renderer_bind_group_layout",
        )

        # Define vertex buffer layout
        vertex_layout = [
            {
                "array_stride": TEXTURE_VERTEX_DTYPE.itemsize,
                "step_mode": "vertex",
                "attributes": [
                    {
                        "format": "float32x2",
                        "offset": 0,  # position offset
                        "shader_location": 0,
                    },
                    {
                        "format": "float32x2",
                        "offset": 8,  # uv offset (2 * 4 bytes)
                        "shader_location": 1,
                    },
                    {
                        "format": "float32x4",
                        "offset": 16,  # fg_color offset (4 * 4 bytes)
                        "shader_location": 2,
                    },
                    {
                        "format": "float32x4",
                        "offset": 32,  # bg_color offset (8 * 4 bytes)
                        "shader_location": 3,
                    },
                ],
            }
        ]

        # Create render pipeline for RGBA8 unorm format (off-screen rendering)
        targets = [
            {
                "format": "rgba8unorm",
                "blend": {
                    "color": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.src_alpha,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    },
                    "alpha": {
                        "operation": wgpu.BlendOperation.add,
                        "src_factor": wgpu.BlendFactor.one,
                        "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                    },
                },
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ]

        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/glyph/render.wgsl",
            fragment_shader_path="wgsl/glyph/render.wgsl",  # Same file
            vertex_layout=vertex_layout,
            bind_group_layouts=[bind_group_layout],
            targets=targets,
            cache_key="texture_renderer_pipeline",
        )

        # Create sampler
        sampler = self.resource_manager.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,  # type: ignore
            min_filter=wgpu.FilterMode.nearest,  # type: ignore
            mipmap_filter=wgpu.MipmapFilterMode.nearest,  # type: ignore
            address_mode_u=wgpu.AddressMode.clamp_to_edge,  # type: ignore
            address_mode_v=wgpu.AddressMode.clamp_to_edge,  # type: ignore
        )

        # Create bind group
        self.bind_group = self.shader_manager.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.uniform_buffer},
                },
                {
                    "binding": 1,
                    "resource": self.resource_manager.get_texture_view(
                        self.atlas_texture
                    ),
                },
                {
                    "binding": 2,
                    "resource": sampler,
                },
            ],
            label="texture_renderer_bind_group",
        )

    def render(
        self,
        glyph_buffer: GlyphBuffer,
        cpu_buffer_override: np.ndarray,
        buffer_override: wgpu.GPUBuffer | None = None,
        cache_key_suffix: str = "",
    ) -> wgpu.GPUTexture:
        """Render a GlyphBuffer to a WGPU texture.

        Uses TCOD-style change detection to only process dirty tiles.

        Args:
            glyph_buffer: The glyph buffer to render
            buffer_override: An isolated vertex buffer provided by the canvas
            cpu_buffer_override: CPU vertex buffer to use for calculations (required)

        Returns:
            WGPU texture containing the rendered glyph buffer
        """
        # Calculate texture dimensions based on glyph buffer size
        texture_width = glyph_buffer.width * self.tile_dimensions[0]
        texture_height = glyph_buffer.height * self.tile_dimensions[1]

        if texture_width == 0 or texture_height == 0:
            # Return empty texture for zero-sized buffers
            return self.resource_manager.get_or_create_render_texture(
                width=1,
                height=1,
                texture_format="rgba8unorm",
                cache_key_suffix=cache_key_suffix,
            )

        buffer_id = id(glyph_buffer)
        cache_buffer = self.buffer_cache.get(buffer_id)
        needs_full_update = (
            cache_buffer is None
            or cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        )

        # Check if we have a cached texture for this buffer
        if buffer_id in self.texture_cache and not needs_full_update:
            # Find dirty tiles for partial update
            dirty_tiles = self._find_dirty_tiles(glyph_buffer, cache_buffer)
            if not dirty_tiles:
                # Fast path: buffer hasn't changed, return cached texture
                return self.texture_cache[buffer_id]
            # We have dirty tiles, will do partial update below
        else:
            # Need full update
            dirty_tiles = None

        # Get or create render target texture
        render_texture = self.resource_manager.get_or_create_render_texture(
            width=texture_width,
            height=texture_height,
            texture_format="rgba8unorm",
            cache_key_suffix=cache_key_suffix,
        )

        # Determine which VBO to use
        vbo_to_use = (
            buffer_override if buffer_override is not None else self.vertex_buffer
        )
        if vbo_to_use is None:
            raise ValueError("No vertex buffer available for rendering.")

        # Update uniform buffer with texture size
        texture_size_data = struct.pack(
            "2f", float(texture_width), float(texture_height)
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(texture_size_data)
        )

        # Calculate vertex count from glyph buffer dimensions
        vertex_count = glyph_buffer.width * glyph_buffer.height * 6

        if needs_full_update:
            # Full update: rebuild entire vertex buffer
            vertex_count = self._update_complete_vbo(glyph_buffer, cpu_buffer_override)

            # Upload entire vertex buffer to GPU
            if vertex_count > 0:
                self.resource_manager.queue.write_buffer(
                    vbo_to_use,
                    0,
                    memoryview(cpu_buffer_override[:vertex_count].tobytes()),
                )
        else:
            # Partial update: only update dirty tiles
            assert dirty_tiles is not None
            self._update_dirty_tiles_in_vbo(
                glyph_buffer, dirty_tiles, vbo_to_use, cpu_buffer_override
            )
            # vertex_count already calculated above

        if vertex_count == 0:
            # Return empty texture if no vertices to render
            return render_texture

        # Create command encoder
        command_encoder = self.resource_manager.device.create_command_encoder()

        # Create render pass for off-screen rendering
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": render_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),  # Transparent background
                    "load_op": "clear",
                    "store_op": "store",
                }
            ]
        )

        # Set up render pass
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_vertex_buffer(0, vbo_to_use)

        # Draw vertices as triangles
        render_pass.draw(vertex_count)

        # End render pass and submit commands
        render_pass.end()
        command_buffer = command_encoder.finish()
        self.resource_manager.queue.submit([command_buffer])

        # Update cache with current buffer state for next frame's comparison
        self._update_cache(glyph_buffer, buffer_id)

        # Cache the rendered texture
        self.texture_cache[buffer_id] = render_texture

        return render_texture

    def _build_glyph_vertices(
        self, glyph_buffer: GlyphBuffer, cpu_buffer_override: np.ndarray
    ) -> int:
        """Build vertex data from a glyph buffer.

        Returns:
            Number of vertices generated
        """
        tile_w, tile_h = self.tile_dimensions
        vertex_count = 0

        # Iterate through the glyph buffer and create quads
        for y in range(glyph_buffer.height):
            for x in range(glyph_buffer.width):
                cell = glyph_buffer.data[x, y]
                char_code = cell["ch"]
                fg_rgba = cell["fg"]
                bg_rgba = cell["bg"]

                # Note: We do NOT skip transparent cells here.
                # This matches ModernGL behavior and ensures the VBO is fully updated,
                # preventing stale vertex data from previous frames.

                # Convert to CP437 using pre-computed mapping for common chars
                if char_code < 256:
                    char_code = int(self.unicode_to_cp437_map[char_code])
                else:
                    char_code = unicode_to_cp437(char_code)
                char_code = max(0, min(255, char_code))

                # Get UV coordinates for the character
                uv_coords = tuple(self.uv_map[char_code])

                # Convert colors to 0-1 range
                fg_color = (
                    fg_rgba[0] / 255.0,
                    fg_rgba[1] / 255.0,
                    fg_rgba[2] / 255.0,
                    fg_rgba[3] / 255.0,
                )
                bg_color = (
                    bg_rgba[0] / 255.0,
                    bg_rgba[1] / 255.0,
                    bg_rgba[2] / 255.0,
                    bg_rgba[3] / 255.0,
                )

                # Calculate pixel position within the texture
                px_x = float(x * tile_w)
                px_y = float(y * tile_h)

                # Add quad for this glyph
                vertex_count = self._add_glyph_quad(
                    px_x,
                    px_y,
                    tile_w,
                    tile_h,
                    uv_coords,
                    fg_color,
                    bg_color,
                    cpu_buffer_override,
                    vertex_count,
                )

        return vertex_count

    def _add_glyph_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        fg_color: tuple[float, float, float, float],
        bg_color: tuple[float, float, float, float],
        cpu_buffer: np.ndarray,
        vertex_count: int,
    ) -> int:
        """Add a glyph quad to the vertex buffer.

        Returns:
            Updated vertex count
        """
        if vertex_count + 6 > len(cpu_buffer):
            return vertex_count

        u1, v1, u2, v2 = uv_coords
        vertices = np.zeros(6, dtype=TEXTURE_VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), fg_color, bg_color)
        vertices[1] = ((x + w, y), (u2, v1), fg_color, bg_color)
        vertices[2] = ((x, y + h), (u1, v2), fg_color, bg_color)
        vertices[3] = ((x + w, y), (u2, v1), fg_color, bg_color)
        vertices[4] = ((x, y + h), (u1, v2), fg_color, bg_color)
        vertices[5] = ((x + w, y + h), (u2, v2), fg_color, bg_color)
        cpu_buffer[vertex_count : vertex_count + 6] = vertices
        return vertex_count + 6

    def _find_dirty_tiles(
        self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer | None
    ) -> list[ViewportTilePos]:
        """Find tiles that have changed since the last frame.

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

    def _generate_vertices_for_single_tile(
        self, x: int, y: int, tile_data: np.ndarray
    ) -> np.ndarray:
        """Generate the 6 vertices (1 quad) for a single tile.

        This is extracted from the main vertex generation logic for use
        in partial buffer updates.

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
            fg_char = int(self.unicode_to_cp437_map[char])
        else:
            fg_char = unicode_to_cp437(char)

        # Get UV coordinates for foreground character
        fg_uv = self.uv_map[fg_char]
        u1, v1, u2, v2 = fg_uv

        # Calculate screen position for this tile
        tile_w, tile_h = self.tile_dimensions
        screen_x = float(x * tile_w)
        screen_y = float(y * tile_h)

        # Create the 6 vertices for this tile (single quad with both colors)
        vertices = np.zeros(6, dtype=TEXTURE_VERTEX_DTYPE)

        # Single quad (6 vertices) - structured assignment
        vertices[0]["position"] = (screen_x, screen_y)
        vertices[0]["uv"] = (u1, v1)
        vertices[0]["fg_color"] = fg_color
        vertices[0]["bg_color"] = bg_color

        vertices[1]["position"] = (screen_x + tile_w, screen_y)
        vertices[1]["uv"] = (u2, v1)
        vertices[1]["fg_color"] = fg_color
        vertices[1]["bg_color"] = bg_color

        vertices[2]["position"] = (screen_x, screen_y + tile_h)
        vertices[2]["uv"] = (u1, v2)
        vertices[2]["fg_color"] = fg_color
        vertices[2]["bg_color"] = bg_color

        vertices[3]["position"] = (screen_x + tile_w, screen_y)
        vertices[3]["uv"] = (u2, v1)
        vertices[3]["fg_color"] = fg_color
        vertices[3]["bg_color"] = bg_color

        vertices[4]["position"] = (screen_x, screen_y + tile_h)
        vertices[4]["uv"] = (u1, v2)
        vertices[4]["fg_color"] = fg_color
        vertices[4]["bg_color"] = bg_color

        vertices[5]["position"] = (screen_x + tile_w, screen_y + tile_h)
        vertices[5]["uv"] = (u2, v2)
        vertices[5]["fg_color"] = fg_color
        vertices[5]["bg_color"] = bg_color

        return vertices

    def _update_complete_vbo(
        self, glyph_buffer: GlyphBuffer, cpu_buffer_override: np.ndarray
    ) -> int:
        """Update the entire VBO with data from the complete buffer.

        Used for first render or when buffer dimensions change.

        Args:
            glyph_buffer: The GlyphBuffer to encode completely
            cpu_buffer_override: CPU vertex buffer to use for vertex calculations

        Returns:
            Number of vertices generated
        """
        # Build all vertices and return the count
        return self._build_glyph_vertices(glyph_buffer, cpu_buffer_override)

    def _update_dirty_tiles_in_vbo(
        self,
        glyph_buffer: GlyphBuffer,
        dirty_tiles: list[ViewportTilePos],
        vbo: wgpu.GPUBuffer,
        cpu_buffer_override: np.ndarray,
    ) -> None:
        """Update only the dirty tiles in the VBO - the real TCOD optimization.

        This generates vertices for only the changed tiles and updates their
        specific sections of the VBO.

        Args:
            glyph_buffer: The current GlyphBuffer
            dirty_tiles: List of (x, y) coordinates of tiles that changed
            vbo: The VBO to update
            cpu_buffer_override: CPU buffer to update alongside the VBO
        """
        for x, y in dirty_tiles:
            # Generate vertices for just this one tile
            tile_vertices = self._generate_vertices_for_single_tile(
                x, y, glyph_buffer.data[x, y]
            )

            # Calculate VBO offset for this specific tile
            # Each tile uses 6 vertices
            tile_index = (y * glyph_buffer.width) + x
            vbo_offset = tile_index * 6 * TEXTURE_VERTEX_DTYPE.itemsize
            vertex_offset = tile_index * 6

            # Update CPU buffer
            cpu_buffer_override[vertex_offset : vertex_offset + 6] = tile_vertices

            # Update only this tile's section of the VBO
            self.resource_manager.queue.write_buffer(
                vbo,
                vbo_offset,
                memoryview(tile_vertices.tobytes()),
            )

    def _update_cache(self, glyph_buffer: GlyphBuffer, buffer_id: int) -> None:
        """Update the cache with the current buffer state.

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
