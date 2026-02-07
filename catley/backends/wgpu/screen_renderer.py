"""WGPU screen rendering - port of ModernGL version."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from catley.types import PixelPos

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
        shadow_atlas_texture: wgpu.GPUTexture,
        surface_format: str,
    ) -> None:
        """Initialize WGPU screen renderer.

        Args:
            resource_manager: WGPU resource manager for buffer/texture caching
            shader_manager: WGPU shader manager for pipeline creation
            atlas_texture: The main tileset atlas texture
            shadow_atlas_texture: Gaussian-blurred atlas for projected shadows
            surface_format: The surface format for render targets
        """
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        self.atlas_texture = atlas_texture
        self.shadow_atlas_texture = shadow_atlas_texture
        self.surface_format = surface_format

        # Main Vertex Buffer for Screen Rendering
        self.cpu_vertex_buffer = np.zeros(MAX_QUADS * 6, dtype=VERTEX_DTYPE)
        self.vertex_count = 0
        self._shadow_start = 0
        self._shadow_end = 0

        # Create vertex buffer
        buffer_size = MAX_QUADS * 6 * VERTEX_DTYPE.itemsize
        self.vertex_buffer = self.resource_manager.get_or_create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            label="screen_renderer_vertex_buffer",
        )

        # Create dedicated uniform buffer for letterbox parameters
        self.uniform_buffer = self.resource_manager.device.create_buffer(
            size=16,  # vec4<f32> = 16 bytes
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="screen_renderer_uniform_buffer",
        )

        # Create pipeline and bind groups
        self._create_pipeline()

        # Pre-create bind groups during initialization for better performance.
        # Sharp pass uses nearest sampling; shadow pass uses linear sampling.
        self._cached_bind_group = self._create_bind_group(
            texture=self.atlas_texture,
            sampler=self.resource_manager.nearest_sampler,
            label="screen_renderer_bind_group",
        )
        self._shadow_bind_group = self._create_bind_group(
            texture=self.shadow_atlas_texture,
            sampler=self.resource_manager.linear_sampler,
            label="screen_renderer_shadow_bind_group",
        )

    def _create_pipeline(self) -> None:
        """Create the WGPU render pipeline for screen rendering."""
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

        # Create render pipeline
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

        # Use proper screen shader for screen rendering
        self.pipeline = self.shader_manager.create_render_pipeline(
            vertex_shader_path="wgsl/screen/main.wgsl",
            fragment_shader_path="wgsl/screen/main.wgsl",  # Same file
            vertex_layout=vertex_layout,
            bind_group_layouts=[self.bind_group_layout],
            targets=targets,
            cache_key="screen_renderer_pipeline",
        )

    def _create_bind_group(
        self,
        texture: wgpu.GPUTexture,
        sampler: wgpu.GPUSampler,
        label: str,
    ) -> wgpu.GPUBindGroup:
        """Create a bind group for the provided texture/sampler pair."""
        return self.shader_manager.create_bind_group(
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
                    "resource": sampler,
                },
            ],
            label=label,
        )

    def begin_frame(self) -> None:
        """Reset the internal vertex count to zero at the start of a frame."""
        self.vertex_count = 0
        self._shadow_start = 0
        self._shadow_end = 0

    def mark_shadow_start(self) -> None:
        """Record where shadow vertices begin in the shared vertex buffer."""
        self._shadow_start = self.vertex_count

    def mark_shadow_end(self) -> None:
        """Record where shadow vertices end in the shared vertex buffer."""
        self._shadow_end = self.vertex_count

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

    def add_parallelogram(
        self,
        corners: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        uv_coords: tuple[float, float, float, float],
        base_color: tuple[float, float, float, float],
        tip_color: tuple[float, float, float, float],
        *,
        vertex_colors: (
            tuple[
                tuple[float, float, float, float],
                tuple[float, float, float, float],
                tuple[float, float, float, float],
                tuple[float, float, float, float],
            ]
            | None
        ) = None,
    ) -> None:
        """Add a projected glyph quad defined by four arbitrary corners.

        Corner order is:
        (base_left, base_right, tip_left, tip_right).

        UV mapping: base edge gets the texture bottom (v2), tip edge gets the
        texture top (v1), left corners get u1, right corners get u2.

        By default, base corners use *base_color* and tip corners use
        *tip_color*.  Pass *vertex_colors* as a 4-tuple
        ``(c_base_left, c_base_right, c_tip_left, c_tip_right)`` to override
        the per-vertex colours independently of the UV layout - useful when
        the alpha-fade direction differs from the base/tip axis (e.g.
        horizontal shadows where the fade runs left-right but base/tip
        represents bottom/top for correct UV orientation).
        """
        if self.vertex_count + 6 > len(self.cpu_vertex_buffer):
            return

        (base_left, base_right, tip_left, tip_right) = corners
        u1, v1, u2, v2 = uv_coords

        if vertex_colors is not None:
            c0, c1, c2, c3 = vertex_colors
        else:
            c0 = c1 = base_color
            c2 = c3 = tip_color

        # UV mapping projects glyph bottom onto the base edge and glyph top
        # onto the tip edge so shadows stretch outward from actor feet.
        vertices = np.zeros(6, dtype=VERTEX_DTYPE)
        vertices[0] = (base_left, (u1, v2), c0)
        vertices[1] = (base_right, (u2, v2), c1)
        vertices[2] = (tip_left, (u1, v1), c2)
        vertices[3] = (base_right, (u2, v2), c1)
        vertices[4] = (tip_left, (u1, v1), c2)
        vertices[5] = (tip_right, (u2, v1), c3)

        self.cpu_vertex_buffer[self.vertex_count : self.vertex_count + 6] = vertices
        self.vertex_count += 6

    def render_to_screen(
        self,
        render_pass: wgpu.GPURenderPassEncoder,
        window_size: PixelPos,
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
            scaled_w, scaled_h = map(int, window_size)

        # Update uniform buffer with letterbox parameters (same as UI renderer)
        letterbox_data = struct.pack(
            "4f", float(offset_x), float(offset_y), float(scaled_w), float(scaled_h)
        )
        self.resource_manager.queue.write_buffer(
            self.uniform_buffer, 0, memoryview(letterbox_data)
        )

        # Upload vertex data to GPU
        self.resource_manager.queue.write_buffer(
            self.vertex_buffer,
            0,
            memoryview(self.cpu_vertex_buffer[: self.vertex_count].tobytes()),
        )

        # Set the viewport to the full size of the window to ensure rendering
        # is not clipped after a resize.
        render_pass.set_viewport(
            0.0, 0.0, float(window_size[0]), float(window_size[1]), 0.0, 1.0
        )

        # Set up render pass
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, self.vertex_buffer)

        # Split one vertex buffer into three draw ranges:
        # pre-shadow sharp pass, blurred shadow pass, post-shadow sharp pass.
        shadow_start = min(max(self._shadow_start, 0), self.vertex_count)
        shadow_end = min(max(self._shadow_end, shadow_start), self.vertex_count)

        pre_shadow_count = shadow_start
        shadow_count = shadow_end - shadow_start
        post_shadow_count = self.vertex_count - shadow_end

        if pre_shadow_count > 0:
            render_pass.set_bind_group(0, self._cached_bind_group)
            render_pass.draw(pre_shadow_count, 1, 0)

        if shadow_count > 0:
            render_pass.set_bind_group(0, self._shadow_bind_group)
            render_pass.draw(shadow_count, 1, shadow_start)

        if post_shadow_count > 0:
            render_pass.set_bind_group(0, self._cached_bind_group)
            render_pass.draw(post_shadow_count, 1, shadow_end)
