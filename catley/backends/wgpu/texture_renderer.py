"""WGPU texture rendering - port of ModernGL version for off-screen rendering."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

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

        # Vertex buffer for texture rendering
        self.cpu_vertex_buffer = np.zeros(
            MAX_TEXTURE_QUADS * 6, dtype=TEXTURE_VERTEX_DTYPE
        )
        self.vertex_count = 0

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

    def render(self, glyph_buffer: GlyphBuffer) -> wgpu.GPUTexture:
        """Render a GlyphBuffer to a WGPU texture.

        Args:
            glyph_buffer: The glyph buffer to render

        Returns:
            WGPU texture containing the rendered glyph buffer
        """
        # Calculate texture dimensions based on glyph buffer size
        texture_width = glyph_buffer.width * self.tile_dimensions[0]
        texture_height = glyph_buffer.height * self.tile_dimensions[1]

        # Get or create render target texture
        render_texture = self.resource_manager.get_or_create_render_texture(
            width=texture_width,
            height=texture_height,
            format="rgba8unorm",
        )

        # Reset vertex count for this render
        self.vertex_count = 0

        # Build vertex data from glyph buffer
        self._build_glyph_vertices(glyph_buffer)

        if self.vertex_count == 0:
            # Return empty texture if no vertices to render
            return render_texture

        # Update uniform buffer with texture size
        texture_size_data = struct.pack(
            "2f", float(texture_width), float(texture_height)
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(texture_size_data)
        )

        # Upload vertex data to GPU
        self.resource_manager.queue.write_buffer(
            self.vertex_buffer,
            0,
            memoryview(self.cpu_vertex_buffer[: self.vertex_count].tobytes()),
        )

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
        render_pass.set_vertex_buffer(0, self.vertex_buffer)

        # Draw vertices as triangles
        render_pass.draw(self.vertex_count)

        # End render pass and submit commands
        render_pass.end()
        command_buffer = command_encoder.finish()
        self.resource_manager.queue.submit([command_buffer])

        return render_texture

    def _build_glyph_vertices(self, glyph_buffer: GlyphBuffer) -> None:
        """Build vertex data from a glyph buffer."""
        tile_w, tile_h = self.tile_dimensions

        # Iterate through the glyph buffer and create quads
        for y in range(glyph_buffer.height):
            for x in range(glyph_buffer.width):
                cell = glyph_buffer.data[x, y]
                char_code = cell["ch"]
                fg_rgba = cell["fg"]
                bg_rgba = cell["bg"]

                # Skip fully transparent cells
                if fg_rgba[3] == 0 and bg_rgba[3] == 0:
                    continue

                # Convert to CP437 and clamp to valid range
                if char_code > 255:
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
                self._add_glyph_quad(
                    px_x, px_y, tile_w, tile_h, uv_coords, fg_color, bg_color
                )

    def _add_glyph_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        fg_color: tuple[float, float, float, float],
        bg_color: tuple[float, float, float, float],
    ) -> None:
        """Add a glyph quad to the vertex buffer."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return

        u1, v1, u2, v2 = uv_coords
        vertices = np.zeros(6, dtype=TEXTURE_VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), fg_color, bg_color)
        vertices[1] = ((x + w, y), (u2, v1), fg_color, bg_color)
        vertices[2] = ((x, y + h), (u1, v2), fg_color, bg_color)
        vertices[3] = ((x + w, y), (u2, v1), fg_color, bg_color)
        vertices[4] = ((x, y + h), (u1, v2), fg_color, bg_color)
        vertices[5] = ((x + w, y + h), (u2, v2), fg_color, bg_color)
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6
