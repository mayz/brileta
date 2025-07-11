"""WGPU screen rendering - port of ModernGL version."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

# Maximum number of quads (2 triangles per quad) to draw per frame.
MAX_QUADS = 10000

VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class WGPUScreenRenderer:
    """WGPU screen renderer - handles background tiles and viewport transformations.

    Specialized class for batch-rendering all game world objects that use the
    main tileset atlas to the screen.
    """

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        atlas_texture: wgpu.GPUTexture,
    ) -> None:
        """Initialize WGPU screen renderer.

        Args:
            resource_manager: WGPU resource manager for buffer/texture caching
            shader_manager: WGPU shader manager for pipeline creation
            atlas_texture: The main tileset atlas texture
        """
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.atlas_texture = atlas_texture

        # Main Vertex Buffer for Screen Rendering
        self.cpu_vertex_buffer = np.zeros(MAX_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0

        # Create vertex buffer
        buffer_size = MAX_QUADS * 6 * VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="screen_renderer_vertex_buffer",
        )

        # Create uniform buffer for letterbox parameters
        self.uniform_buffer = self.resource_manager.get_or_create_buffer(
            size=16,  # vec4<f32> = 16 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="screen_renderer_uniform_buffer",
        )

        # Create pipeline and bind groups
        self._create_pipeline()

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for screen rendering."""
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
            label="screen_renderer_bind_group_layout",
        )

        # Define vertex buffer layout
        vertex_layout = [
            {
                "array_stride": VERTEX_DTYPE.itemsize,
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
                        "offset": 16,  # color offset (4 * 4 bytes)
                        "shader_location": 2,
                    },
                ],
            }
        ]

        # Create render pipeline
        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/screen/main.wgsl",
            fragment_shader_path="wgsl/screen/main.wgsl",  # Same file
            vertex_layout=vertex_layout,
            bind_group_layouts=[bind_group_layout],
            cache_key="screen_renderer_pipeline",
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
            label="screen_renderer_bind_group",
        )

    def begin_frame(self) -> None:
        """Reset the internal vertex count to zero at the start of a frame."""
        self.vertex_count = 0

    def add_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        uv_coords: tuple[float, float, float, float],
        color_rgba: tuple[float, float, float, float],
    ) -> None:
        """Add a quad to the vertex buffer."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return

        u1, v1, u2, v2 = uv_coords
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y), (u2, v1), color_rgba)
        vertices[2] = ((x, y + h), (u1, v2), color_rgba)
        vertices[3] = ((x + w, y), (u2, v1), color_rgba)
        vertices[4] = ((x, y + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y + h), (u2, v2), color_rgba)
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

    def render_to_screen(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: tuple[int, int],
        letterbox_geometry: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Main drawing method that renders all batched vertex data to the screen.

        Args:
            render_pass: WGPU render pass encoder
            window_size: Full window dimensions
            letterbox_geometry: (offset_x, offset_y, scaled_w, scaled_h) for letterbox
        """
        if self.vertex_count == 0:
            return

        # Determine letterbox parameters
        if letterbox_geometry is not None:
            offset_x, offset_y, scaled_w, scaled_h = letterbox_geometry
        else:
            # No letterboxing - use full screen
            offset_x, offset_y = 0, 0
            scaled_w, scaled_h = window_size

        # Update uniform buffer with letterbox parameters
        letterbox_data = struct.pack(
            "4f", float(offset_x), float(offset_y), float(scaled_w), float(scaled_h)
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(letterbox_data)
        )  # type: ignore

        # Upload vertex data to GPU
        self.resource_manager.queue.write_buffer(
            self.vertex_buffer,
            0,
            memoryview(self.cpu_vertex_buffer[: self.vertex_count].tobytes()),  # type: ignore
        )

        # Set up render pass
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)

        # Draw vertices as triangles
        render_pass.draw(self.vertex_count)
