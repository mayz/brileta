"""WGPU environmental effect renderer for batched environmental effect rendering."""

from __future__ import annotations

import struct
import weakref
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from catley.types import PixelPos

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


class WGPUEnvironmentalEffectRenderer:
    """WGPU environmental effect renderer for batched effect rendering.

    Queues environmental effects for rendering in the main render pass
    instead of immediate rendering to improve performance and consistency.
    """

    # Maximum number of environmental effect quads per frame
    MAX_EFFECT_QUADS = 200

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        surface_format: str,
    ) -> None:
        """Initialize WGPU environmental effect renderer."""
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.surface_format = surface_format

        # Batched rendering queue and vertex buffer
        self.render_queue: list[tuple[wgpu.GPUTexture, int]] = []
        self.cpu_vertex_buffer = np.zeros(self.MAX_EFFECT_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0

        # Create vertex buffer for batched rendering
        buffer_size = self.MAX_EFFECT_QUADS * 6 * VERTEX_DTYPE.itemsize
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

        # Use shared bind group layout from resource manager
        self.bind_group_layout = self.resource_manager.standard_bind_group_layout

        # Create render pipeline
        self._create_pipeline()

        # Use shared linear sampler from resource manager (for smooth gradients)
        self.sampler = self.resource_manager.linear_sampler

        # Create reusable radial gradient texture for environmental effects
        self.radial_gradient_texture = self._create_radial_gradient_texture(256)

    def _create_pipeline(self) -> None:
        """Create the render pipeline for environmental effect rendering."""
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

        # Create pipeline (reuse screen shaders since they're similar)
        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/screen/main.wgsl",
            fragment_shader_path="wgsl/screen/main.wgsl",
            vertex_layout=vertex_layout,
            bind_group_layouts=[self.bind_group_layout],
            targets=targets,
            cache_key="environmental_effect_renderer_pipeline",
        )

    def _create_radial_gradient_texture(self, resolution: int) -> wgpu.GPUTexture:
        """
        Create a reusable white radial gradient texture for environmental effects.
        This prevents per-frame texture creation in apply_environmental_effect.
        """
        center = resolution // 2

        # Create gradient using numpy
        y_coords, x_coords = np.ogrid[:resolution, :resolution]
        distances = np.sqrt((x_coords - center) ** 2 + (y_coords - center) ** 2)

        # Create alpha gradient that fades from center to edge
        alpha = np.clip(1.0 - distances / center, 0.0, 1.0)

        # Create RGBA texture data (white with alpha gradient)
        effect_data = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        effect_data[:, :, :3] = 255  # White base color
        effect_data[:, :, 3] = (alpha * 255).astype(np.uint8)

        # Create the texture using resource manager
        return self.resource_manager.create_atlas_texture(
            width=resolution,
            height=resolution,
            data=effect_data.tobytes(),
            texture_format="rgba8unorm",
        )

    def begin_frame(self) -> None:
        """Clear the render queue at the start of a new frame."""
        self.render_queue.clear()
        self.vertex_count = 0

    def add_effect_quad(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        color_rgba: tuple[float, float, float, float],
        texture: wgpu.GPUTexture | None = None,
    ) -> None:
        """Add an environmental effect quad to the render queue."""
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return  # Skip if buffer is full

        # Use provided texture or default radial gradient
        effect_texture = (
            texture if texture is not None else self.radial_gradient_texture
        )

        # Create vertices for this effect quad (use full texture UVs)
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = ((x, y), (u1, v1), color_rgba)
        vertices[1] = ((x + w, y), (u2, v1), color_rgba)
        vertices[2] = ((x, y + h), (u1, v2), color_rgba)
        vertices[3] = ((x + w, y), (u2, v1), color_rgba)
        vertices[4] = ((x, y + h), (u1, v2), color_rgba)
        vertices[5] = ((x + w, y + h), (u2, v2), color_rgba)

        # Copy vertices into the CPU buffer
        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

        # Queue the texture and vertex count for this quad
        self.render_queue.append((effect_texture, 6))

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: PixelPos,
        letterbox_geometry: tuple[int, int, int, int] | None,
    ) -> None:
        """Render all queued environmental effect textures in the render pass."""
        if not self.render_queue or self.vertex_count == 0:
            return

        # Set the viewport to ensure effects render correctly after a resize.
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
