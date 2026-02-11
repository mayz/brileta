"""WGPU textured quad renderer for batched texture rendering."""

from __future__ import annotations

import struct
import weakref
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from brileta.types import PixelPos

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


class WGPUTexturedQuadRenderer:
    """Batched textured-quad renderer with configurable capacity and sampling."""

    VERTEX_DTYPE = VERTEX_DTYPE

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        surface_format: str,
        *,
        max_quads: int = 500,
        sampler: wgpu.GPUSampler | None = None,
        label: str = "textured_quad",
    ) -> None:
        """Initialize a configurable textured quad renderer.

        Args:
            resource_manager: WGPU resource manager for buffer/texture caching
            shader_manager: WGPU shader manager for pipeline creation
            surface_format: The surface format for render targets
            max_quads: Maximum number of quads to batch each frame
            sampler: Optional sampler override (defaults to nearest sampler)
            label: Debug label prefix for GPU resources
        """
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.surface_format = surface_format
        self.max_quads = max_quads
        self.label = label
        self.sampler = sampler or self.resource_manager.nearest_sampler
        self.render_queue: list[tuple[wgpu.GPUTexture, int]] = []

        # Create CPU-side buffer to aggregate all vertex data
        self.cpu_vertex_buffer = np.zeros(self.max_quads * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0

        # Each renderer instance owns its vertex buffer; sharing would corrupt
        # per-pass batch data if capacities ever collide.
        buffer_size = self.max_quads * 6 * VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label=f"{self.label}_vertex_buffer",
        )

        # Create dedicated uniform buffer for letterbox parameters
        self.uniform_buffer = self.resource_manager.device.create_buffer(
            size=16,  # vec4<f32> = 16 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label=f"{self.label}_uniform_buffer",
        )

        # Create pipeline and bind groups
        self._create_pipeline()

        # Cache for texture bind groups using weak references for automatic cleanup
        self.texture_bind_groups: weakref.WeakKeyDictionary[
            wgpu.GPUTexture, wgpu.GPUBindGroup
        ] = weakref.WeakKeyDictionary()

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for textured quad rendering."""
        # Use shared bind group layout from resource manager
        self.bind_group_layout = self.resource_manager.standard_bind_group_layout

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

        # Create render pipeline with correct surface format
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
                "write_mask": wgpu.ColorWrite.ALL,
            }
        ]

        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/ui/texture.wgsl",
            fragment_shader_path="wgsl/ui/texture.wgsl",  # Same file
            vertex_layout=vertex_layout,
            bind_group_layouts=[self.bind_group_layout],
            targets=targets,
            cache_key="textured_quad_renderer_pipeline",
        )

    def begin_frame(self) -> None:
        """Clear the internal queue of textures to be rendered for the new frame."""
        self.render_queue.clear()
        self.vertex_count = 0

    def add_textured_quad(self, texture: wgpu.GPUTexture, vertices: np.ndarray) -> None:
        """Add texture and vertex data to the internal render queue."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return

        # Copy vertices into the CPU buffer
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

        # Queue only the texture and vertex count for this quad
        self.render_queue.append((texture, 6))

    def _get_or_create_bind_group(self, texture: wgpu.GPUTexture) -> wgpu.GPUBindGroup:
        """Get or create a bind group for the given texture."""
        if texture not in self.texture_bind_groups:
            self.texture_bind_groups[texture] = self.shader_manager.create_bind_group(
                layout=self.bind_group_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {"buffer": self.uniform_buffer},
                    },
                    {
                        "binding": 1,
                        "resource": self.resource_manager.get_texture_view(texture),
                    },
                    {
                        "binding": 2,
                        "resource": self.sampler,
                    },
                ],
                label=f"{self.label}_bind_group_{id(texture)}",
            )
        return self.texture_bind_groups[texture]

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: PixelPos,
        letterbox_geometry: tuple[int, int, int, int] | None,
    ) -> None:
        """Render all queued textured quads."""
        if not self.render_queue or self.vertex_count == 0:
            return

        # Set the viewport to ensure UI renders correctly after a resize.
        render_pass.set_viewport(
            0.0, 0.0, float(window_size[0]), float(window_size[1]), 0.0, 1.0
        )

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

        # Set up render pass
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)

        # Render each quad with the correct texture binding
        vertex_offset = 0
        for texture, vertex_count in self.render_queue:
            # Get or create bind group for this texture
            bind_group = self._get_or_create_bind_group(texture)
            render_pass.set_bind_group(0, bind_group)

            # Render this quad using the correct slice of vertices
            render_pass.draw(vertex_count, 1, vertex_offset)

            vertex_offset += vertex_count

    # No explicit cleanup needed - WeakKeyDictionary handles it automatically
