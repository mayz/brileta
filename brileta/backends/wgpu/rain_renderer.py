"""WGPU renderer for the stochastic rain overlay effect."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from brileta import colors, config
from brileta.types import (
    PixelPos,
    PixelRect,
    TileDimensions,
    TileDisplacement,
    WorldTilePos,
)

from .resource_manager import ALPHA_BLEND_STATE
from .textured_quad_renderer import VERTEX_DTYPE

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager


class WGPURainRenderer:
    """Renders procedural rain line segments as a full-screen overlay."""

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        surface_format: str,
    ) -> None:
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.surface_format = surface_format

        self._cpu_vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        self._vertex_buffer = self.resource_manager.device.create_buffer(
            size=6 * VERTEX_DTYPE.itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        self._uniform_buffer = self.resource_manager.device.create_buffer(
            size=256,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._bind_group_layout = self._create_bind_group_layout()
        self._pipeline = self._create_pipeline()

        self._default_exclusion_texture: wgpu.GPUTexture | None = None
        self._rain_exclusion_texture: wgpu.GPUTexture | None = None
        self._rain_exclusion_size: tuple[int, int] | None = None
        # Cached bind group, invalidated when the exclusion texture changes.
        self._cached_bind_group: wgpu.GPUBindGroup | None = None
        self._cached_exclusion_texture: wgpu.GPUTexture | None = None
        self._sampler = self.resource_manager.device.create_sampler(
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            mipmap_filter=wgpu.MipmapFilterMode.nearest,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            label="rain_overlay_sampler",
        )

    def _create_bind_group_layout(self) -> wgpu.GPUBindGroupLayout:
        """Create bind group layout for rain shader resources."""
        return self.shader_manager.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": "2d",
                    },
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ],
            label="rain_bind_group_layout",
        )

    def _create_pipeline(self) -> wgpu.GPURenderPipeline:
        """Create the rain render pipeline with alpha blending."""
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

        targets = [
            {
                "format": self.surface_format,
                "blend": ALPHA_BLEND_STATE,
            }
        ]

        return self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/effects/rain.wgsl",
            fragment_shader_path="wgsl/effects/rain.wgsl",
            vertex_layout=vertex_layout,
            bind_group_layouts=[self._bind_group_layout],
            targets=targets,
            cache_key="rain_renderer_pipeline",
        )

    def _get_default_exclusion_texture(self) -> wgpu.GPUTexture | None:
        """Return a 1x1 black texture meaning no rain exclusion."""
        if self._default_exclusion_texture is not None:
            return self._default_exclusion_texture
        if config.IS_TEST_ENVIRONMENT:
            return None

        self._default_exclusion_texture = self.resource_manager.create_default_texture(
            texture_format="r8unorm",
            data=bytes([0]),
            bytes_per_pixel=1,
            label="rain_default_exclusion",
        )
        return self._default_exclusion_texture

    def _upload_exclusion_mask_texture(
        self,
        rain_exclusion_mask_buffer: np.ndarray,
    ) -> wgpu.GPUTexture | None:
        """Upload a viewport-space mask that excludes rain where value > 0."""
        if rain_exclusion_mask_buffer.ndim != 2:
            raise ValueError("rain_exclusion_mask_buffer must be a 2D array")
        if config.IS_TEST_ENVIRONMENT:
            return None

        width = int(rain_exclusion_mask_buffer.shape[0])
        height = int(rain_exclusion_mask_buffer.shape[1])
        if width <= 0 or height <= 0:
            return self._get_default_exclusion_texture()

        self._rain_exclusion_texture, self._rain_exclusion_size = (
            self.resource_manager.upload_mask_texture(
                rain_exclusion_mask_buffer,
                self._rain_exclusion_texture,
                self._rain_exclusion_size,
                label="rain_exclusion_mask_texture",
            )
        )
        return self._rain_exclusion_texture

    def begin_frame(self) -> None:
        """Reset per-frame state."""
        return

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: PixelPos,
        letterbox_geometry: PixelRect | None,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        tile_dimensions: TileDimensions,
        intensity: float,
        angle: float,
        drop_length: float,
        advection: TileDisplacement,
        drop_spacing: float,
        stream_spacing: float,
        rain_color: colors.Color,
        rain_exclusion_mask_buffer: np.ndarray | None,
        pixel_bounds: PixelRect,
    ) -> None:
        """Render rain overlay with current frame parameters."""
        if intensity <= 0.0:
            return

        rain_exclusion_texture = self._get_default_exclusion_texture()
        if rain_exclusion_texture is None:
            return
        if rain_exclusion_mask_buffer is not None:
            uploaded_mask = self._upload_exclusion_mask_texture(
                rain_exclusion_mask_buffer
            )
            if uploaded_mask is not None:
                rain_exclusion_texture = uploaded_mask

        left, top, right, bottom = pixel_bounds
        width_px = right - left
        height_px = bottom - top
        if width_px <= 0 or height_px <= 0:
            return

        vertices = self._cpu_vertices
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        vertices[0] = ((left, top), (u1, v1), color_rgba)
        vertices[1] = ((right, top), (u2, v1), color_rgba)
        vertices[2] = ((left, bottom), (u1, v2), color_rgba)
        vertices[3] = ((right, top), (u2, v1), color_rgba)
        vertices[4] = ((left, bottom), (u1, v2), color_rgba)
        vertices[5] = ((right, bottom), (u2, v2), color_rgba)
        self.resource_manager.queue.write_buffer(
            self._vertex_buffer, 0, memoryview(vertices.tobytes())
        )

        uniform_data = self._build_uniforms(
            letterbox_geometry=letterbox_geometry
            or (0, 0, int(window_size[0]), int(window_size[1])),
            viewport_offset=viewport_offset,
            viewport_size=viewport_size,
            tile_dimensions=tile_dimensions,
            intensity=intensity,
            angle=angle,
            drop_length=drop_length,
            advection=advection,
            drop_spacing=drop_spacing,
            stream_spacing=stream_spacing,
            rain_color=rain_color,
        )
        self.resource_manager.queue.write_buffer(
            self._uniform_buffer, 0, memoryview(uniform_data)
        )

        # Reuse the bind group when the exclusion texture is unchanged.
        if rain_exclusion_texture is not self._cached_exclusion_texture:
            self._cached_bind_group = self._create_bind_group(rain_exclusion_texture)
            self._cached_exclusion_texture = rain_exclusion_texture
        bind_group = self._cached_bind_group

        render_pass.set_viewport(
            0.0, 0.0, float(window_size[0]), float(window_size[1]), 0.0, 1.0
        )
        render_pass.set_pipeline(self._pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, self._vertex_buffer)
        render_pass.draw(6, 1, 0, 0)

    def _build_uniforms(
        self,
        letterbox_geometry: PixelRect,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        tile_dimensions: TileDimensions,
        intensity: float,
        angle: float,
        drop_length: float,
        advection: TileDisplacement,
        drop_spacing: float,
        stream_spacing: float,
        rain_color: colors.Color,
    ) -> bytes:
        """Pack rain uniforms using vec4-aligned layout."""
        return struct.pack(
            "20f",
            float(letterbox_geometry[0]),
            float(letterbox_geometry[1]),
            float(letterbox_geometry[2]),
            float(letterbox_geometry[3]),
            float(viewport_offset[0]),
            float(viewport_offset[1]),
            float(viewport_size[0]),
            float(viewport_size[1]),
            float(intensity),
            float(angle),
            float(drop_length),
            float(advection[0]),
            float(rain_color[0]),
            float(rain_color[1]),
            float(rain_color[2]),
            float(advection[1]),
            float(tile_dimensions[0]),
            float(tile_dimensions[1]),
            float(drop_spacing),
            float(stream_spacing),
        )

    def _create_bind_group(
        self,
        rain_exclusion_texture: wgpu.GPUTexture,
    ) -> wgpu.GPUBindGroup:
        """Create bind group for rain resources."""
        return self.shader_manager.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buffer}},
                {
                    "binding": 1,
                    "resource": self.resource_manager.get_texture_view(
                        rain_exclusion_texture
                    ),
                },
                {"binding": 2, "resource": self._sampler},
            ],
            label="rain_bind_group",
        )
