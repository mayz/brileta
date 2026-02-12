"""WGPU renderer for atmospheric effects (cloud shadows and ground mist)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import wgpu
from PIL import Image as PILImage

from brileta import colors, config
from brileta.types import PixelPos, PixelRect, ViewOffset, WorldTilePos

if TYPE_CHECKING:
    from .resource_manager import WGPUResourceManager
    from .shader_manager import WGPUShaderManager

# Vertex dtype matching the screen renderer for compatibility
VERTEX_DTYPE = np.dtype(
    [
        ("position", "2f4"),  # (x, y) in pixels
        ("uv", "2f4"),  # (u, v) texture coordinates
        ("color", "4f4"),  # (r, g, b, a) - unused but required for layout
    ]
)


class WGPUAtmosphericRenderer:
    """WGPU renderer for atmospheric effects (cloud shadows, ground mist).

    Renders full-screen atmospheric effects like cloud shadows and ground mist.
    Effects are masked by sky exposure and fog-of-war visibility.
    """

    def __init__(
        self,
        resource_manager: WGPUResourceManager,
        shader_manager: WGPUShaderManager,
        surface_format: str,
    ) -> None:
        """Initialize WGPU atmospheric renderer.

        Args:
            resource_manager: Shared WGPU resource manager
            shader_manager: Shared WGPU shader manager
            surface_format: Surface format for render targets
        """
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.surface_format = surface_format

        # Vertex buffer for full-screen quad (6 vertices)
        self._cpu_vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        self._vertex_buffer = self.resource_manager.device.create_buffer(
            size=6 * VERTEX_DTYPE.itemsize,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Pool of uniform buffers for atmospheric parameters.
        # WGPU requires separate buffers for each draw within a render pass,
        # because queue.write_buffer during render pass encoding may not
        # synchronize correctly when the same buffer is reused.
        self._uniform_buffer_pool: list[wgpu.GPUBuffer] = []
        self._uniform_buffer_index = 0
        for _ in range(4):  # Pool of 4 buffers (more than we'll ever need)
            self._uniform_buffer_pool.append(
                self.resource_manager.device.create_buffer(
                    size=256,  # Generous size for uniform struct with padding
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
            )

        # Load noise texture
        self._noise_texture = self._load_noise_texture()

        # Default fallback textures for explored/visible maps
        self._default_explored_texture: wgpu.GPUTexture | None = None
        self._default_visible_texture: wgpu.GPUTexture | None = None

        # Create bind group layout for atmospheric shader
        self._bind_group_layout = self._create_bind_group_layout()

        # Create render pipelines for both blend modes
        self._darken_pipeline = self._create_pipeline(blend_mode="darken")
        self._lighten_pipeline = self._create_pipeline(blend_mode="lighten")

        # Create repeating sampler for noise texture (drift offset can exceed 0-1)
        self._sampler = self.resource_manager.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.MipmapFilterMode.linear,
            address_mode_u=wgpu.AddressMode.repeat,
            address_mode_v=wgpu.AddressMode.repeat,
            label="atmospheric_noise_sampler",
        )

    def _create_bind_group_layout(self) -> wgpu.GPUBindGroupLayout:
        """Create bind group layout for atmospheric shader."""
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
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": "2d",
                    },
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": "2d",
                    },
                },
                {
                    "binding": 4,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.float,
                        "view_dimension": "2d",
                    },
                },
                {
                    "binding": 5,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": wgpu.SamplerBindingType.filtering},
                },
            ],
            label="atmospheric_bind_group_layout",
        )

    def _create_pipeline(self, blend_mode: str) -> wgpu.GPURenderPipeline:
        """Create render pipeline for the specified blend mode.

        Args:
            blend_mode: "darken" for multiplicative, "lighten" for additive
        """
        # Vertex layout matching VERTEX_DTYPE
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

        # Configure blend state based on mode
        if blend_mode == "darken":
            # Multiplicative blend: dst * src
            targets = [
                {
                    "format": self.surface_format,
                    "blend": {
                        "color": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.dst,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                        "alpha": {
                            "operation": wgpu.BlendOperation.add,
                            "src_factor": wgpu.BlendFactor.one,
                            "dst_factor": wgpu.BlendFactor.zero,
                        },
                    },
                }
            ]
        else:
            # Additive blend with alpha: src_alpha * src + (1 - src_alpha) * dst
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

        return self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/effects/cloud_layer.wgsl",
            fragment_shader_path="wgsl/effects/cloud_layer.wgsl",
            vertex_layout=vertex_layout,
            bind_group_layouts=[self._bind_group_layout],
            targets=targets,
            cache_key=f"atmospheric_renderer_{blend_mode}_pipeline",
        )

    def _load_noise_texture(self) -> wgpu.GPUTexture | None:
        """Load the static cloud noise texture."""
        if config.IS_TEST_ENVIRONMENT:
            return None

        texture_path = (
            Path(config.PROJECT_ROOT_PATH) / "assets" / "textures" / "cloud_noise.png"
        )
        if not texture_path.exists():
            return None

        img = PILImage.open(str(texture_path)).convert("RGBA")
        pixels = np.array(img, dtype="u1")

        texture = self.resource_manager.device.create_texture(
            size=(img.size[0], img.size[1], 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        self.resource_manager.queue.write_texture(
            destination={"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data=memoryview(pixels.tobytes()),
            data_layout={
                "offset": 0,
                "bytes_per_row": img.size[0] * 4,
                "rows_per_image": img.size[1],
            },
            size=(img.size[0], img.size[1], 1),
        )

        return texture

    def _get_default_explored_texture(self) -> wgpu.GPUTexture | None:
        """Return a 1x1 white texture to treat all tiles as explored."""
        if self._default_explored_texture is not None:
            return self._default_explored_texture
        if config.IS_TEST_ENVIRONMENT:
            return None

        # Create 1x1 RGBA texture with white (fully explored)
        pixels = np.array([[[255, 255, 255, 255]]], dtype="u1")
        texture = self.resource_manager.device.create_texture(
            size=(1, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.resource_manager.queue.write_texture(
            destination={"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data=memoryview(pixels.tobytes()),
            data_layout={"offset": 0, "bytes_per_row": 4, "rows_per_image": 1},
            size=(1, 1, 1),
        )
        self._default_explored_texture = texture
        return texture

    def _get_default_visible_texture(self) -> wgpu.GPUTexture | None:
        """Return a 1x1 white texture to treat all tiles as visible."""
        if self._default_visible_texture is not None:
            return self._default_visible_texture
        if config.IS_TEST_ENVIRONMENT:
            return None

        # Create 1x1 RGBA texture with white (fully visible)
        pixels = np.array([[[255, 255, 255, 255]]], dtype="u1")
        texture = self.resource_manager.device.create_texture(
            size=(1, 1, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self.resource_manager.queue.write_texture(
            destination={"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data=memoryview(pixels.tobytes()),
            data_layout={"offset": 0, "bytes_per_row": 4, "rows_per_image": 1},
            size=(1, 1, 1),
        )
        self._default_visible_texture = texture
        return texture

    def begin_frame(self) -> None:
        """Reset per-frame state. Call at the start of each frame."""
        self._uniform_buffer_index = 0

    def render(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: PixelPos,
        letterbox_geometry: PixelRect | None,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        sky_exposure_threshold: float,
        sky_exposure_texture: wgpu.GPUTexture | None,
        explored_texture: wgpu.GPUTexture | None,
        visible_texture: wgpu.GPUTexture | None,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        strength: float,
        tint_color: colors.Color,
        drift_offset: ViewOffset,
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        blend_mode: str,
        pixel_bounds: PixelRect,
    ) -> None:
        """Render an atmospheric layer.

        Args:
            render_pass: Active WGPU render pass
            window_size: Window dimensions in pixels
            letterbox_geometry: Letterbox offset and dimensions
            viewport_offset: Top-left corner of viewport in world tiles
            viewport_size: Viewport dimensions in tiles
            map_size: Full map dimensions in tiles
            sky_exposure_threshold: Threshold for outdoor masking
            sky_exposure_texture: Texture with sky exposure values per tile
            explored_texture: Fog-of-war explored mask
            visible_texture: Fog-of-war visible mask
            noise_scale: UV multiplier for noise pattern
            noise_threshold_low: Lower threshold for noise smoothstep
            noise_threshold_high: Upper threshold for noise smoothstep
            strength: Effect intensity (0.0 - 1.0)
            tint_color: RGB tint color (0-255)
            drift_offset: World-space UV offset from animation
            turbulence_offset: Time offset for turbulence sampling
            turbulence_strength: Turbulence intensity
            turbulence_scale: Scale of turbulence pattern
            blend_mode: "darken" for shadows, "lighten" for mist
            pixel_bounds: Screen pixel bounds (left, top, right, bottom)
        """
        if sky_exposure_texture is None or self._noise_texture is None:
            return

        # Use default textures if not provided
        if explored_texture is None:
            explored_texture = self._get_default_explored_texture()
        if explored_texture is None:
            return
        if visible_texture is None:
            visible_texture = self._get_default_visible_texture()
        if visible_texture is None:
            return

        left, top, right, bottom = pixel_bounds
        width_px = right - left
        height_px = bottom - top
        if width_px <= 0 or height_px <= 0:
            return

        # Build full-screen quad vertices
        vertices = self._cpu_vertices
        u1, v1, u2, v2 = 0.0, 0.0, 1.0, 1.0
        color_rgba = (1.0, 1.0, 1.0, 1.0)

        vertices[0] = ((left, top), (u1, v1), color_rgba)
        vertices[1] = ((right, top), (u2, v1), color_rgba)
        vertices[2] = ((left, bottom), (u1, v2), color_rgba)
        vertices[3] = ((right, top), (u2, v1), color_rgba)
        vertices[4] = ((left, bottom), (u1, v2), color_rgba)
        vertices[5] = ((right, bottom), (u2, v2), color_rgba)

        # Upload vertex data
        self.resource_manager.queue.write_buffer(
            self._vertex_buffer, 0, memoryview(vertices.tobytes())
        )

        # Build and upload uniform data
        uniform_data = self._build_uniforms(
            letterbox_geometry=letterbox_geometry or (0, 0, 800, 600),
            viewport_offset=viewport_offset,
            viewport_size=viewport_size,
            map_size=map_size,
            blend_mode=blend_mode,
            strength=strength,
            tint_color=tint_color,
            noise_scale=noise_scale,
            noise_threshold_low=noise_threshold_low,
            noise_threshold_high=noise_threshold_high,
            drift_offset=drift_offset,
            turbulence_offset=turbulence_offset,
            turbulence_strength=turbulence_strength,
            turbulence_scale=turbulence_scale,
            sky_exposure_threshold=sky_exposure_threshold,
        )
        # Get a fresh uniform buffer from the pool for this draw.
        # This ensures each draw within a render pass has its own buffer,
        # avoiding stale data issues with queue.write_buffer during encoding.
        uniform_buffer = self._uniform_buffer_pool[
            self._uniform_buffer_index % len(self._uniform_buffer_pool)
        ]
        self._uniform_buffer_index += 1

        self.resource_manager.queue.write_buffer(
            uniform_buffer, 0, memoryview(uniform_data)
        )

        # Create bind group with current textures and the specific uniform buffer
        bind_group = self._create_bind_group(
            uniform_buffer=uniform_buffer,
            sky_exposure_texture=sky_exposure_texture,
            explored_texture=explored_texture,
            visible_texture=visible_texture,
        )

        # Set viewport
        render_pass.set_viewport(
            0.0, 0.0, float(window_size[0]), float(window_size[1]), 0.0, 1.0
        )

        # Select pipeline based on blend mode
        pipeline = (
            self._darken_pipeline if blend_mode == "darken" else self._lighten_pipeline
        )

        # Draw
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, self._vertex_buffer)
        render_pass.draw(6, 1, 0, 0)

    def _build_uniforms(
        self,
        letterbox_geometry: PixelRect,
        viewport_offset: WorldTilePos,
        viewport_size: tuple[int, int],
        map_size: tuple[int, int],
        blend_mode: str,
        strength: float,
        tint_color: colors.Color,
        noise_scale: float,
        noise_threshold_low: float,
        noise_threshold_high: float,
        drift_offset: ViewOffset,
        turbulence_offset: float,
        turbulence_strength: float,
        turbulence_scale: float,
        sky_exposure_threshold: float,
    ) -> bytes:
        """Pack uniform data for the shader.

        Uses vec4 packing for reliable alignment across all backends.
        Each vec4 is 16 bytes, total = 6 * 16 = 96 bytes.
        """
        blend_mode_value = 0.0 if blend_mode == "darken" else 1.0

        # Pack as 6 vec4s (24 floats total = 96 bytes)
        return struct.pack(
            "24f",
            # vec4 0: letterbox (offset_x, offset_y, scaled_w, scaled_h)
            float(letterbox_geometry[0]),
            float(letterbox_geometry[1]),
            float(letterbox_geometry[2]),
            float(letterbox_geometry[3]),
            # vec4 1: viewport_data (offset_x, offset_y, size_x, size_y)
            float(viewport_offset[0]),
            float(viewport_offset[1]),
            float(viewport_size[0]),
            float(viewport_size[1]),
            # vec4 2: map_and_blend (map_w, map_h, blend_mode, strength)
            float(map_size[0]),
            float(map_size[1]),
            blend_mode_value,
            strength,
            # vec4 3: tint_and_noise (r, g, b, noise_scale)
            float(tint_color[0]),
            float(tint_color[1]),
            float(tint_color[2]),
            noise_scale,
            # vec4 4: thresholds_and_drift (low, high, drift_x, drift_y)
            noise_threshold_low,
            noise_threshold_high,
            drift_offset[0],
            drift_offset[1],
            # vec4 5: turbulence_and_sky (offset, strength, scale, sky_threshold)
            turbulence_offset,
            turbulence_strength,
            turbulence_scale,
            sky_exposure_threshold,
        )

    def _create_bind_group(
        self,
        uniform_buffer: wgpu.GPUBuffer,
        sky_exposure_texture: wgpu.GPUTexture,
        explored_texture: wgpu.GPUTexture,
        visible_texture: wgpu.GPUTexture,
    ) -> wgpu.GPUBindGroup:
        """Create bind group for the current frame's textures."""
        return self.shader_manager.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": uniform_buffer},
                },
                {
                    "binding": 1,
                    "resource": sky_exposure_texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": explored_texture.create_view(),
                },
                {
                    "binding": 3,
                    "resource": visible_texture.create_view(),
                },
                {
                    "binding": 4,
                    "resource": self._noise_texture.create_view()
                    if self._noise_texture
                    else explored_texture.create_view(),
                },
                {
                    "binding": 5,
                    "resource": self._sampler,
                },
            ],
            label="atmospheric_bind_group",
        )
