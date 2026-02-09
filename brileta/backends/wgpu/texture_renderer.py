"""WGPU texture rendering - port of ModernGL version for off-screen rendering."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from brileta.util.caching import ResourceCache
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.util.live_vars import record_time_live_variable
from brileta.util.tilesets import unicode_to_cp437

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

# Maximum number of quads for texture rendering
MAX_TEXTURE_QUADS = 5000

# Vertex structure for glyph rendering (foreground + background colors + noise)
TEXTURE_VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("fg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
        ("bg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
        ("noise_amplitude", "f4"),  # Sub-tile brightness noise amplitude
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

        # Noise seed passed to the fragment shader for deterministic sub-tile
        # brightness variation via PCG hash. Set per-frame from game_map.decoration_seed.
        self.noise_seed: int = 0
        # World-space tile offset so the fragment shader can convert buffer-space
        # tile indices to world coordinates for stable noise hashing across scrolls.
        self.noise_tile_offset: tuple[int, int] = (0, 0)

        # Cache for change detection - skip re-render if buffer unchanged
        self.buffer_cache: ResourceCache[int, GlyphBuffer] = ResourceCache(
            name="wgpu_texture_renderer_buffers", max_size=100
        )

        # Create vertex buffer
        buffer_size = MAX_TEXTURE_QUADS * 6 * TEXTURE_VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="texture_renderer_vertex_buffer",
        )

        # Create uniform buffer matching WGSL Uniforms struct layout:
        #   vec2<f32> texture_size + vec2<f32> tile_size + vec2<i32> tile_offset
        #   + u32 noise_seed + 4 bytes implicit padding = 32 bytes
        self.uniform_buffer = self.resource_manager.get_or_create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="texture_renderer_uniform_buffer",
        )

        # Create pipeline and bind groups
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for texture rendering."""
        # Use shared bind group layout from resource manager
        bind_group_layout = self.resource_manager.standard_bind_group_layout

        # Define vertex buffer layout (texture renderer has extra bg_color attribute)
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
                    {
                        "format": "float32",
                        "offset": 48,  # noise_amplitude offset
                        "shader_location": 4,
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

        # Create bind group using shared sampler
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
                    "resource": self.resource_manager.nearest_sampler,
                },
            ],
            label="texture_renderer_bind_group",
        )

    def set_noise_seed(self, seed: int) -> None:
        """Set the noise seed for sub-tile brightness variation.

        Called once per frame from game_map.decoration_seed so each map
        gets a unique but stable noise pattern.
        """
        self.noise_seed = seed

    def set_noise_tile_offset(self, offset_x: int, offset_y: int) -> None:
        """Set the world-space tile offset for stable noise hashing.

        The glyph buffer uses buffer-space tile indices (0, 1, 2, ...) that
        shift when the camera scrolls. This offset converts them to world
        coordinates so the noise pattern stays anchored to world tiles.
        """
        self.noise_tile_offset = (offset_x, offset_y)

    def set_tile_dimensions(self, tile_dimensions: tuple[int, int]) -> None:
        """Update tile size used for off-screen glyph rendering.

        Tile dimensions can change when the window size changes. When that
        happens, we must invalidate the glyph change-detection cache because
        cached textures rendered at the old tile size are no longer valid.
        """
        if tile_dimensions == self.tile_dimensions:
            return
        self.tile_dimensions = tile_dimensions
        self.buffer_cache.clear()

    def render(
        self,
        glyph_buffer: GlyphBuffer,
        cpu_buffer_override: np.ndarray,
        buffer_override: wgpu.GPUBuffer | None = None,
        cache_key_suffix: str = "",
    ) -> wgpu.GPUTexture:
        """Render a GlyphBuffer to a WGPU texture.

        Args:
            glyph_buffer: The glyph buffer to render
            buffer_override: An isolated vertex buffer provided by the canvas
            cpu_buffer_override: CPU vertex buffer to use for calculations (required)
            cache_key_suffix: Unique identifier for texture caching

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

        # Get render target texture
        render_texture = self.resource_manager.get_or_create_render_texture(
            width=texture_width,
            height=texture_height,
            texture_format="rgba8unorm",
            cache_key_suffix=cache_key_suffix,
        )

        # Change detection - skip if buffer hasn't changed
        buffer_id = id(glyph_buffer)
        cache_buffer = self.buffer_cache.get(buffer_id)
        if cache_buffer is not None and not self._has_changes(
            glyph_buffer, cache_buffer
        ):
            return render_texture  # Return cached texture, skip re-render

        # Determine which VBO to use
        vbo_to_use = (
            buffer_override if buffer_override is not None else self.vertex_buffer
        )
        if vbo_to_use is None:
            raise ValueError("No vertex buffer available for rendering.")

        # Update uniform buffer matching WGSL Uniforms struct layout:
        # texture_size (2f) + tile_size (2f) + tile_offset (2i) + noise_seed (I) + pad (4x)
        tile_w = float(self.tile_dimensions[0])
        tile_h = float(self.tile_dimensions[1])
        offset_x, offset_y = self.noise_tile_offset
        uniform_data = struct.pack(
            "4f2iI4x",
            float(texture_width),
            float(texture_height),
            tile_w,
            tile_h,
            offset_x,
            offset_y,
            self.noise_seed & 0xFFFFFFFF,
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(uniform_data)
        )

        with record_time_live_variable("time.render.texture.vbo_update_ms"):
            vertex_count = self._build_glyph_vertices(glyph_buffer, cpu_buffer_override)

            # Upload vertex buffer to GPU
            if vertex_count > 0:
                self.resource_manager.queue.write_buffer(
                    vbo_to_use,
                    0,
                    memoryview(cpu_buffer_override[:vertex_count].tobytes()),
                )

        if vertex_count == 0:
            # Return texture even if no vertices (already cleared)
            return render_texture

        with record_time_live_variable("time.render.texture.render_ms"):
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

        # Update cache for next frame's change detection
        self._update_cache(glyph_buffer, buffer_id)

        return render_texture

    def _build_glyph_vertices(
        self, glyph_buffer: GlyphBuffer, cpu_buffer: np.ndarray
    ) -> int:
        """Build vertex data from a glyph buffer.

        Fully vectorized - no Python loops over cells.

        Args:
            glyph_buffer: The GlyphBuffer to encode
            cpu_buffer: Pre-allocated CPU vertex buffer to fill

        Returns:
            Number of vertices generated
        """
        w, h = glyph_buffer.data.shape
        num_cells = w * h
        if num_cells == 0:
            return 0

        num_vertices = num_cells * 6
        if num_vertices > len(cpu_buffer):
            raise ValueError(
                f"GlyphBuffer too large: needs {num_vertices} vertices, "
                f"buffer has {len(cpu_buffer)}"
            )

        tile_w, tile_h = self.tile_dimensions

        # Build coordinate grids matching row-major iteration order (y outer, x inner).
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
        verts = cpu_buffer[:num_vertices].reshape(h, w, 6)

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

        # Broadcast noise amplitude from glyph buffer to all 6 vertices per cell.
        # Swap from (w, h) to (h, w) to match grid order.
        noise_values = glyph_buffer.data["noise"].T  # (h, w)
        for i in range(6):
            verts["noise_amplitude"][..., i] = noise_values

        return num_vertices

    def _has_changes(
        self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer
    ) -> bool:
        """Check if the buffer has any changes since last frame.

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

        has_bg_diff = np.any(glyph_buffer.data["bg"] != cache_buffer.data["bg"])
        if has_bg_diff:
            return True

        return bool(np.any(glyph_buffer.data["noise"] != cache_buffer.data["noise"]))

    def _update_cache(self, glyph_buffer: GlyphBuffer, buffer_id: int) -> None:
        """Update the cache with the current buffer state.

        Args:
            glyph_buffer: Current buffer to cache
            buffer_id: ID of the buffer for cache key
        """
        cache_buffer = self.buffer_cache.get(buffer_id)

        if cache_buffer is None or (
            cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        ):
            cache_buffer = GlyphBuffer(glyph_buffer.width, glyph_buffer.height)

        # Deep copy the data to preserve for next frame comparison
        np.copyto(cache_buffer.data, glyph_buffer.data)

        self.buffer_cache.store(buffer_id, cache_buffer)
