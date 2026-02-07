"""WGPU light overlay composer.

Composes dark and light tile textures using GPU light intensity/masks. The output
matches WorldView's CPU overlay blend while keeping the composition on-GPU.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from catley.util.coordinates import Rect
from catley.util.live_vars import record_time_live_variable

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager


_VERTEX_DTYPE = np.dtype([("position", "2f4")])


class WGPULightOverlayComposer:
    """Render-pass composer for GPU light overlays."""

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
    ) -> None:
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager

        self._uniform_buffer = self.resource_manager.device.create_buffer(
            size=32,  # 2 vec4<f32>
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="light_overlay_compose_uniforms",
        )

        self._vertex_buffer = self.resource_manager.device.create_buffer(
            size=6 * _VERTEX_DTYPE.itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="light_overlay_compose_fullscreen_quad",
        )
        self._write_fullscreen_quad_vertices()

        self._bind_group_layout = self.resource_manager.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
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
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float
                    },
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
            ],
            label="light_overlay_compose_bind_group_layout",
        )

        self._pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/lighting/light_overlay_compose.wgsl",
            fragment_shader_path="wgsl/lighting/light_overlay_compose.wgsl",
            vertex_layout=[
                {
                    "array_stride": _VERTEX_DTYPE.itemsize,
                    "step_mode": wgpu.VertexStepMode.vertex,
                    "attributes": [
                        {
                            "format": wgpu.VertexFormat.float32x2,
                            "offset": 0,
                            "shader_location": 0,
                        }
                    ],
                }
            ],
            bind_group_layouts=[self._bind_group_layout],
            targets=[{"format": "rgba8unorm"}],
            cache_key="light_overlay_compose_pipeline",
        )
        self._visible_mask_texture: wgpu.GPUTexture | None = None
        self._visible_mask_size: tuple[int, int] | None = None

    def _write_fullscreen_quad_vertices(self) -> None:
        vertices = np.array(
            [
                ((-1.0, -1.0),),
                ((1.0, -1.0),),
                ((-1.0, 1.0),),
                ((1.0, -1.0),),
                ((-1.0, 1.0),),
                ((1.0, 1.0),),
            ],
            dtype=_VERTEX_DTYPE,
        )
        self.resource_manager.queue.write_buffer(
            self._vertex_buffer,
            0,
            memoryview(vertices.tobytes()),
        )

    def compose(
        self,
        *,
        dark_texture: wgpu.GPUTexture,
        light_texture: wgpu.GPUTexture,
        lightmap_texture: wgpu.GPUTexture,
        visible_mask_buffer: np.ndarray,
        viewport_bounds: Rect,
        viewport_offset: tuple[int, int],
        pad_tiles: int,
        tile_dimensions: tuple[int, int],
    ) -> wgpu.GPUTexture:
        """Compose a light overlay texture fully on the GPU."""
        output_texture = self.resource_manager.get_or_create_render_texture(
            width=dark_texture.width,
            height=dark_texture.height,
            texture_format="rgba8unorm",
            cache_key_suffix="gpu_light_overlay_compose",
        )

        visible_texture = self._upload_visible_mask_texture(visible_mask_buffer)

        uniform_data = struct.pack(
            "8f",
            float(viewport_bounds.width),
            float(viewport_bounds.height),
            float(viewport_offset[0]),
            float(viewport_offset[1]),
            float(tile_dimensions[0]),
            float(tile_dimensions[1]),
            float(pad_tiles),
            0.3,  # explored-but-not-visible spillover multiplier
        )
        self.resource_manager.queue.write_buffer(
            self._uniform_buffer, 0, memoryview(uniform_data)
        )

        bind_group = self.resource_manager.device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._uniform_buffer}},
                {"binding": 1, "resource": dark_texture.create_view()},
                {"binding": 2, "resource": light_texture.create_view()},
                {"binding": 3, "resource": lightmap_texture.create_view()},
                {"binding": 4, "resource": visible_texture.create_view()},
            ],
            label="light_overlay_compose_bind_group",
        )

        with record_time_live_variable("time.render.light_overlay_gpu_compose_ms"):
            command_encoder = self.resource_manager.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": output_texture.create_view(),
                        "resolve_target": None,
                        "clear_value": (0.0, 0.0, 0.0, 0.0),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }
                ]
            )
            render_pass.set_pipeline(self._pipeline)
            render_pass.set_bind_group(0, bind_group)
            render_pass.set_vertex_buffer(0, self._vertex_buffer)
            render_pass.draw(6, 1, 0, 0)
            render_pass.end()

            self.resource_manager.queue.submit([command_encoder.finish()])

        return output_texture

    def _upload_visible_mask_texture(
        self,
        visible_mask_buffer: np.ndarray,
    ) -> wgpu.GPUTexture:
        """Upload buffer-space visible mask (shape: [width, height]) to GPU."""
        if visible_mask_buffer.ndim != 2:
            raise ValueError("visible_mask_buffer must be a 2D array")

        width = int(visible_mask_buffer.shape[0])
        height = int(visible_mask_buffer.shape[1])
        size = (width, height)
        if self._visible_mask_texture is None or self._visible_mask_size != size:
            self._visible_mask_texture = self.resource_manager.device.create_texture(
                size=(width, height, 1),
                format=wgpu.TextureFormat.r8unorm,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
                label="light_overlay_visible_mask_texture",
            )
            self._visible_mask_size = size

        visible_data = np.ascontiguousarray(
            (visible_mask_buffer.T * 255).astype(np.uint8)
        )
        assert self._visible_mask_texture is not None
        self.resource_manager.queue.write_texture(
            {
                "texture": self._visible_mask_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(visible_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": width,
                "rows_per_image": height,
            },
            (width, height, 1),
        )
        return self._visible_mask_texture
