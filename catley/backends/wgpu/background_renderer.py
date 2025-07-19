"""WGPU background renderer for immediate texture rendering."""

from __future__ import annotations

import struct
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
    """WGPU background renderer for immediate texture rendering.

    Similar to UI texture renderer but renders immediately to screen
    instead of batching for later rendering.
    """

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

        # Create vertex buffer for a single quad
        self.vertex_buffer = self.resource_manager.device.create_buffer(
            size=6 * VERTEX_DTYPE.itemsize,  # 6 vertices per quad
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

        # Create uniform buffer for letterbox parameters
        self.uniform_buffer = self.resource_manager.device.create_buffer(
            size=4 * 4,  # 4 floats: offset_x, offset_y, scaled_w, scaled_h
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,  # type: ignore
        )

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

    def render_immediately(
        self,
        surface_texture: wgpu.GPUTexture,
        texture: wgpu.GPUTexture,
        vertices: np.ndarray,
        letterbox_geometry: tuple[int, int, int, int] | None,
    ) -> None:
        """Render a background texture immediately to the surface."""
        # Create command encoder
        command_encoder = self.resource_manager.device.create_command_encoder()

        # Create surface view
        surface_view = surface_texture.create_view()

        # Create render pass that clears the surface
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": surface_view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),  # Black background
                    "load_op": "clear",
                    "store_op": "store",
                }
            ]
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

        # Upload vertex data
        self.resource_manager.queue.write_buffer(
            self.vertex_buffer, 0, memoryview(vertices.tobytes())
        )

        # Create bind group for this texture
        bind_group = self.shader_manager.create_bind_group(
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

        # Set up render pass and draw
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)
        render_pass.draw(6, 1, 0)  # 6 vertices, 1 instance, starting at vertex 0

        render_pass.end()

        # Submit commands
        command_buffer = command_encoder.finish()
        self.resource_manager.queue.submit([command_buffer])
