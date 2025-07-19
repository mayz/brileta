"""WGPU background renderer for immediate texture rendering."""

from __future__ import annotations

import struct
import weakref
from typing import TYPE_CHECKING

import numpy as np
import wgpu

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y)
        ("uv", "2f4"),  # (u, v)
        ("color", "4f4"),  # (r, g, b, a) as floats 0.0-1.0
    ]
)


class WGPUBackgroundRenderer:
    """WGPU background renderer for batched texture rendering.

    Queues background textures for rendering in the main render pass
    instead of immediate rendering to improve performance.
    """

    # Maximum number of background quads per frame
    MAX_BACKGROUND_QUADS = 100

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        surface_format: str,
    ) -> None:
        """Initialize WGPU background renderer."""
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.surface_format = surface_format

        # Batched rendering queue and vertex buffer
        self.render_queue: list[tuple[wgpu.GPUTexture, int]] = []
        self.cpu_vertex_buffer = np.zeros(
            self.MAX_BACKGROUND_QUADS * 6, dtype=VERTEX_DTYPE
        )
        self.vertex_count = 0

        # Create vertex buffer for batched rendering
        buffer_size = self.MAX_BACKGROUND_QUADS * 6 * VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # Create uniform buffer for letterbox parameters
        self.uniform_buffer = self.resource_manager.device.create_buffer(
            size=4 * 4,  # 4 floats: offset_x, offset_y, scaled_w, scaled_h
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # Cache bind groups using weak references for automatic cleanup
        self._bind_group_cache: weakref.WeakKeyDictionary[
            wgpu.GPUTexture, wgpu.GPUBindGroup
        ] = weakref.WeakKeyDictionary()

        # Create bind group layout
        self.bind_group_layout = self.resource_manager.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ]
        )

        # Create render pipeline
        self._create_pipeline()

        # Create sampler
        self.sampler = self.resource_manager.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,  # type: ignore
            min_filter=wgpu.FilterMode.nearest,  # type: ignore
            mipmap_filter=wgpu.MipmapFilterMode.nearest,  # type: ignore
            address_mode_u=wgpu.AddressMode.clamp_to_edge,  # type: ignore
            address_mode_v=wgpu.AddressMode.clamp_to_edge,  # type: ignore
        )

    def _create_pipeline(self) -> None:
        """Create the render pipeline for background rendering."""
        # Vertex layout
        vertex_layout = [
            {
                "array_stride": VERTEX_DTYPE.itemsize,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 0,
                        "shader_location": 0,
                    },
                    {
                        "format": wgpu.VertexFormat.float32x2,
                        "offset": 8,
                        "shader_location": 1,
                    },
                    {
                        "format": wgpu.VertexFormat.float32x4,
                        "offset": 16,
                        "shader_location": 2,
                    },
                ],
            }
        ]

        # Render targets with alpha blending
        targets = [
            {
                "format": self.surface_format,
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
            }
        ]

        # Create pipeline (reuse UI shaders since they're similar)
        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/ui/texture.wgsl",
            fragment_shader_path="wgsl/ui/texture.wgsl",
            vertex_layout=vertex_layout,
            bind_group_layouts=[self.bind_group_layout],
            targets=targets,
            cache_key="background_renderer_pipeline",
        )

    def begin_frame(self) -> None:
        """Clear the render queue at the start of a new frame."""
        self.render_queue.clear()
        self.vertex_count = 0

    def add_background_quad(
        self,
        texture: wgpu.GPUTexture,
        vertices: np.ndarray,
    ) -> None:
        """Add a background texture quad to the render queue."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return  # Skip if buffer is full

        # Copy vertices into the CPU buffer
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

        # Queue the texture and vertex count for this quad
        self.render_queue.append((texture, 6))

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        letterbox_geometry: tuple[int, int, int, int] | None,
    ) -> None:
        """Render all queued background textures in the current render pass."""
        if not self.render_queue or self.vertex_count == 0:
            return

        # Update uniform buffer with letterbox parameters
        if letterbox_geometry is not None:
            offset_x, offset_y, scaled_w, scaled_h = letterbox_geometry
        else:
            offset_x, offset_y, scaled_w, scaled_h = 0, 0, 800, 600  # Fallback

        letterbox_data = struct.pack(
            "4f", float(offset_x), float(offset_y), float(scaled_w), float(scaled_h)
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(letterbox_data)
        )

        # Upload all vertex data to GPU in a single operation
        self.resource_manager.queue.write_buffer(
            self.vertex_buffer,
            0,
            memoryview(self.cpu_vertex_buffer[: self.vertex_count].tobytes()),
        )

        # Render each quad with the correct texture binding
        vertex_offset = 0
        for texture, vertex_count in self.render_queue:
            # Get or create cached bind group for this texture
            if texture not in self._bind_group_cache:
                self._bind_group_cache[texture] = self._create_bind_group(texture)
            bind_group = self._bind_group_cache[texture]

            # Set up render pass and draw this quad
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, bind_group)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(vertex_count, 1, vertex_offset)

            vertex_offset += vertex_count

    def _create_bind_group(self, texture: wgpu.GPUTexture) -> wgpu.GPUBindGroup:
        """Create a bind group for the given texture."""
        return self.shader_manager.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self.uniform_buffer},
                },
                {
                    "binding": 1,
                    "resource": texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": self.sampler,
                },
            ],
        )

    # No explicit cleanup needed - WeakKeyDictionary handles it automatically
