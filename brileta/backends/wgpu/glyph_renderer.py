"""WGPU glyph rendering - port of ModernGL version for off-screen rendering."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, cast

import numpy as np
import wgpu

from brileta.types import TileDimensions, WorldTilePos
from brileta.util._native import build_glyph_vertices as _native_build_glyph_vertices
from brileta.util.caching import ResourceCache
from brileta.util.glyph_buffer import GlyphBuffer
from brileta.util.live_vars import record_time_live_variable
from brileta.util.tilesets import CHARMAP_CP437, unicode_to_cp437

from .resource_manager import ALPHA_BLEND_STATE

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

# Maximum number of quads for texture rendering
MAX_TEXTURE_QUADS = 5000
EDGE_NEIGHBOR_COUNT = 4

# Vertex structure for glyph rendering (foreground + background colors + noise).
# Cast to np.dtype[np.void] because ty infers np.dtype([...]) as dtype[float64];
# structured dtypes are void-typed, and the cast lets field indexing type-check.
TEXTURE_VERTEX_DTYPE = cast(
    "np.dtype[np.void]",
    np.dtype(
        [
            ("position", "2f4"),  # (x, y)
            ("uv", "2f4"),  # (u, v)
            ("fg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
            ("bg_color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
            ("noise_amplitude", "f4"),  # Sub-tile brightness noise amplitude
            (
                "noise_pattern",
                "u4",
            ),  # Sub-tile brightness pattern ID (u32 for WGSL input)
            ("edge_neighbor_mask", "u4"),  # Cardinal tile-type boundary mask (W/N/S/E)
            ("edge_blend", "f4"),  # Organic edge feathering amplitude (0.0-1.0)
        ]
        + [(f"edge_neighbor_bg_{i}", "3f4") for i in range(EDGE_NEIGHBOR_COUNT)]
        + [
            # Sub-tile split fields for perspective offset boundary tiles.
            ("split_y", "f4"),  # Y threshold [0,1]: 0=no split, >0=split point
            ("split_bg_color", "4f4"),  # Background RGBA for below-split portion
            ("split_fg_color", "4f4"),  # Foreground RGBA for below-split portion
            ("split_noise_amplitude", "f4"),  # Noise amplitude for below-split portion
            ("split_noise_pattern", "u4"),  # Noise pattern for below-split portion
            # Packed roof wear data for per-pixel shader effects.
            ("wear_pack", "u4"),
        ]
    ),
)


class WGPUGlyphRenderer:
    """WGPU glyph renderer for rendering GlyphBuffer objects to off-screen textures.

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
        tile_dimensions: TileDimensions,
        uv_map: np.ndarray,
    ) -> None:
        """Initialize WGPU glyph renderer.

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

        # Pre-compute unicode_to_cp437 lookup table covering all CP437 codepoints.
        # Many classic roguelike glyphs (♣ ♠ ♥ ♦ ☺ etc.) have Unicode codepoints
        # well above 255 (e.g., ♣ = U+2663 = 9827), so the table must extend to
        # the highest codepoint in the CP437 mapping. ~10KB for the full range.
        max_codepoint = max(CHARMAP_CP437) + 1
        self.unicode_to_cp437_map = np.full(max_codepoint, ord("?"), dtype=np.uint8)
        for i in range(max_codepoint):
            self.unicode_to_cp437_map[i] = unicode_to_cp437(i)

        # Noise seed passed to the fragment shader for deterministic sub-tile
        # brightness variation via PCG hash. Set per-frame from game_map.decoration_seed.
        self.noise_seed: int = 0
        # World-space tile offset so the fragment shader can convert buffer-space
        # tile indices to world coordinates for stable noise hashing across scrolls.
        self.noise_tile_offset: WorldTilePos = (0, 0)

        # Cache for change detection - skip re-render if buffer unchanged
        self.buffer_cache: ResourceCache[int, GlyphBuffer] = ResourceCache(
            name="wgpu_glyph_renderer_buffers", max_size=100
        )

        # Create vertex buffer
        self._vertex_buffer_capacity_vertices = MAX_TEXTURE_QUADS * 6
        buffer_size = (
            self._vertex_buffer_capacity_vertices * TEXTURE_VERTEX_DTYPE.itemsize
        )
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="glyph_renderer_vertex_buffer",
        )

        # Create uniform buffer matching WGSL Uniforms struct layout:
        #   vec2<f32> texture_size + vec2<f32> tile_size + vec2<i32> tile_offset
        #   + u32 noise_seed + 4 bytes implicit padding = 32 bytes
        self.uniform_buffer = self.resource_manager.get_or_create_buffer(
            size=32,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="glyph_renderer_uniform_buffer",
        )

        # Cache last-written uniform data to skip redundant write_buffer calls.
        # Within a single frame the glyph renderer is called multiple times
        # (dark world, light world) with identical uniform parameters.
        self._last_uniform_data: bytes = b""

        # Create pipeline and bind groups
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for glyph rendering."""
        # Use shared bind group layout from resource manager
        bind_group_layout = self.resource_manager.standard_bind_group_layout

        # Define vertex buffer layout (texture renderer has extra bg_color attribute)
        dtype_fields = TEXTURE_VERTEX_DTYPE.fields
        if dtype_fields is None:
            raise RuntimeError("TEXTURE_VERTEX_DTYPE has no named fields.")
        offsets = {
            "position": dtype_fields["position"][1],
            "uv": dtype_fields["uv"][1],
            "fg_color": dtype_fields["fg_color"][1],
            "bg_color": dtype_fields["bg_color"][1],
            "noise_amplitude": dtype_fields["noise_amplitude"][1],
            "noise_pattern": dtype_fields["noise_pattern"][1],
            "edge_neighbor_mask": dtype_fields["edge_neighbor_mask"][1],
            "edge_blend": dtype_fields["edge_blend"][1],
            "edge_neighbor_bg_0": dtype_fields["edge_neighbor_bg_0"][1],
            "edge_neighbor_bg_1": dtype_fields["edge_neighbor_bg_1"][1],
            "edge_neighbor_bg_2": dtype_fields["edge_neighbor_bg_2"][1],
            "edge_neighbor_bg_3": dtype_fields["edge_neighbor_bg_3"][1],
            "split_y": dtype_fields["split_y"][1],
            "split_bg_color": dtype_fields["split_bg_color"][1],
            "split_fg_color": dtype_fields["split_fg_color"][1],
            "split_noise_amplitude": dtype_fields["split_noise_amplitude"][1],
            "split_noise_pattern": dtype_fields["split_noise_pattern"][1],
            "wear_pack": dtype_fields["wear_pack"][1],
        }
        vertex_layout = [
            {
                "array_stride": TEXTURE_VERTEX_DTYPE.itemsize,
                "step_mode": "vertex",
                "attributes": [
                    {
                        "format": "float32x2",
                        "offset": offsets["position"],
                        "shader_location": 0,
                    },
                    {
                        "format": "float32x2",
                        "offset": offsets["uv"],
                        "shader_location": 1,
                    },
                    {
                        "format": "float32x4",
                        "offset": offsets["fg_color"],
                        "shader_location": 2,
                    },
                    {
                        "format": "float32x4",
                        "offset": offsets["bg_color"],
                        "shader_location": 3,
                    },
                    {
                        "format": "float32",
                        "offset": offsets["noise_amplitude"],
                        "shader_location": 4,
                    },
                    {
                        "format": "uint32",
                        "offset": offsets["noise_pattern"],
                        "shader_location": 5,
                    },
                    {
                        "format": "uint32",
                        "offset": offsets["edge_neighbor_mask"],
                        "shader_location": 6,
                    },
                    {
                        "format": "float32",
                        "offset": offsets["edge_blend"],
                        "shader_location": 7,
                    },
                    *[
                        {
                            "format": "float32x3",
                            "offset": offsets[f"edge_neighbor_bg_{i}"],
                            "shader_location": 8 + i,
                        }
                        for i in range(EDGE_NEIGHBOR_COUNT)
                    ],
                    # Sub-tile split attributes for perspective offset
                    {
                        "format": "float32",
                        "offset": offsets["split_y"],
                        "shader_location": 12,
                    },
                    {
                        "format": "float32x4",
                        "offset": offsets["split_bg_color"],
                        "shader_location": 13,
                    },
                    {
                        "format": "float32x4",
                        "offset": offsets["split_fg_color"],
                        "shader_location": 14,
                    },
                    {
                        "format": "float32",
                        "offset": offsets["split_noise_amplitude"],
                        "shader_location": 15,
                    },
                    {
                        "format": "uint32",
                        "offset": offsets["split_noise_pattern"],
                        "shader_location": 16,
                    },
                    {
                        "format": "uint32",
                        "offset": offsets["wear_pack"],
                        "shader_location": 17,
                    },
                ],
            }
        ]

        # Create render pipeline for RGBA8 unorm format (off-screen rendering)
        targets = [
            {
                "format": "rgba8unorm",
                "blend": ALPHA_BLEND_STATE,
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ]

        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/glyph/main.wgsl",
            fragment_shader_path="wgsl/glyph/main.wgsl",  # Same file
            vertex_layout=vertex_layout,
            bind_group_layouts=[bind_group_layout],
            targets=targets,
            cache_key="glyph_renderer_pipeline",
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
            label="glyph_renderer_bind_group",
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

    def set_tile_dimensions(self, tile_dimensions: TileDimensions) -> None:
        """Update tile size used for off-screen glyph rendering.

        Tile dimensions can change when the window size changes. When that
        happens, we must invalidate the glyph change-detection cache because
        cached textures rendered at the old tile size are no longer valid.
        """
        if tile_dimensions == self.tile_dimensions:
            return
        self.tile_dimensions = tile_dimensions
        self.buffer_cache.clear()

    def _ensure_vertex_buffer_capacity(self, required_vertices: int) -> None:
        """Grow the shared vertex buffer when the visible glyph buffer expands.

        Zooming out can significantly increase the number of tiles rendered into
        the world-view off-screen texture. The original fixed 5000-quad buffer is
        sufficient for the default viewport, but not for larger zoomed-out views.
        """
        if required_vertices <= self._vertex_buffer_capacity_vertices:
            return

        self._vertex_buffer_capacity_vertices = required_vertices
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=required_vertices * TEXTURE_VERTEX_DTYPE.itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="glyph_renderer_vertex_buffer",
        )

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
        # Skip redundant uniform write when called multiple times per frame
        # with the same parameters (e.g. dark world + light world renders).
        if uniform_data != self._last_uniform_data:
            self.resource_manager.queue.write_buffer(
                self.uniform_buffer, 0, memoryview(uniform_data)
            )
            self._last_uniform_data = uniform_data

        with record_time_live_variable("time.render.texture.vbo_update_ms"):
            vertex_count = self._build_glyph_vertices(glyph_buffer, cpu_buffer_override)

            # Upload vertex buffer to GPU
            if vertex_count > 0:
                if buffer_override is None:
                    self._ensure_vertex_buffer_capacity(vertex_count)
                    vbo_to_use = self.vertex_buffer
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
                        "view": self.resource_manager.get_texture_view(render_texture),
                        "resolve_target": None,
                        "clear_value": (0.0, 0.0, 0.0, 0.0),  # Transparent background
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
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
        """Build vertex data from a glyph buffer using the native C encoder.

        Encodes each glyph cell into 6 interleaved triangle vertices (two
        triangles per quad) in a single pass through C, avoiding the overhead
        of dozens of numpy structured-array field writes from Python.

        Args:
            glyph_buffer: The GlyphBuffer to encode
            cpu_buffer: Pre-allocated CPU vertex buffer to fill

        Returns:
            Number of vertices generated
        """
        w, h = glyph_buffer.data.shape
        if w * h == 0:
            return 0

        tile_w = float(self.tile_dimensions[0])
        tile_h = float(self.tile_dimensions[1])

        return _native_build_glyph_vertices(
            glyph_buffer.data,
            cpu_buffer,
            self.uv_map,
            self.unicode_to_cp437_map,
            tile_w,
            tile_h,
        )

    def _has_changes(
        self, glyph_buffer: GlyphBuffer, cache_buffer: GlyphBuffer
    ) -> bool:
        """Check if the buffer has any changes since last frame.

        Uses a byte-level memoryview comparison of the entire structured array,
        which is ~57x faster than field-by-field np.any() checks for the common
        "no changes" case. Safe because the GlyphBuffer dtype is tightly packed
        with no padding bytes.
        """
        if (
            cache_buffer.width != glyph_buffer.width
            or cache_buffer.height != glyph_buffer.height
        ):
            return True

        return glyph_buffer.data.data != cache_buffer.data.data

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
