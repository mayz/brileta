"""WGPU-based implementation of the lighting system using fragment shaders.

This class implements the lighting system using WGPU fragment shaders for
high-performance parallel light computation. It maintains compatibility with
the LightingSystem interface while providing significant performance gains
for scenes with many lights.

Uses fragment shaders instead of compute shaders for compatibility.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import wgpu

from catley.config import AMBIENT_LIGHT_LEVEL
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.base import LightingSystem

if TYPE_CHECKING:
    from catley.backends.wgpu.resource_manager import WGPUResourceManager
    from catley.backends.wgpu.shader_manager import WGPUShaderManager
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource
    from catley.view.render.graphics import GraphicsContext


class GPULightingSystem(LightingSystem):
    """WGPU-accelerated implementation of the lighting system using fragment shaders.

    This implementation uses WGPU fragment shaders to perform lighting calculations
    in parallel on the GPU, providing significant performance improvements over CPU
    implementations, especially for scenes with many lights.

    Features:
    - Parallel point light computation using fragment shaders
    - Hardware accelerated distance calculations
    - Compatible with existing LightingSystem interface
    - Works with WGPU on modern graphics APIs
    - Automatic fallback to CPU system if GPU unavailable
    """

    # Maximum number of lights we can handle in a single render pass
    MAX_LIGHTS = 32

    def __init__(
        self,
        game_world: GameWorld,
        graphics_context: GraphicsContext,
        fallback_system: LightingSystem | None = None,
    ) -> None:
        """Initialize the WGPU GPU lighting system.

        Args:
            game_world: The game world to query for lighting data
            graphics_context: Graphics context (WGPU required for GPU operations)
            fallback_system: Optional CPU fallback if GPU unavailable
        """
        super().__init__(game_world)

        self.graphics_context = graphics_context
        # Check if graphics context has WGPU device/queue
        if hasattr(graphics_context, "device") and hasattr(graphics_context, "queue"):
            self.device = graphics_context.device
            self.queue = graphics_context.queue
        else:
            # Test/dummy graphics context, will fail initialization and use fallback
            self.device = None
            self.queue = None
        self.fallback_system = fallback_system

        # WGPU resources
        self._render_pipeline: wgpu.GPURenderPipeline | None = None
        self._shader_manager: WGPUShaderManager | None = None
        self._resource_manager: WGPUResourceManager | None = None
        self._uniform_buffer: wgpu.GPUBuffer | None = None
        self._bind_group: wgpu.GPUBindGroup | None = None
        self._output_texture: wgpu.GPUTexture | None = None
        self._output_buffer: wgpu.GPUBuffer | None = None
        self._vertex_buffer: wgpu.GPUBuffer | None = None

        # Track time for dynamic effects
        self._time = 0.0

        # Current viewport for resource sizing
        self._current_viewport: Rect | None = None

        # Performance optimization: Track light configuration changes
        self._last_light_data_hash: int | None = None
        self._cached_light_data: list[float] | None = None
        self._cached_light_revision: int = -1

        # Cache for shadow casters
        self._cached_shadow_casters: list[float] | None = None
        self._cached_shadow_revision: int = -1

        # Sky exposure texture for directional lighting
        self._sky_exposure_texture: wgpu.GPUTexture | None = None
        self._sky_exposure_sampler: wgpu.GPUSampler | None = None
        self._cached_map_revision: int = -1

        # Initialize GPU resources
        if not self._initialize_gpu_resources():
            pass  # Will use fallback system if available

    def _initialize_gpu_resources(self) -> bool:
        """Initialize WGPU fragment shader-based lighting.

        Returns:
            True if initialization successful, False if fallback needed
        """
        try:
            # Check if we have a valid device/queue
            if self.device is None or self.queue is None:
                return False

            # Initialize fragment-based lighting
            return self._initialize_fragment_lighting()

        except Exception:
            return False

    def _initialize_fragment_lighting(self) -> bool:
        """Initialize fragment shader-based lighting."""
        try:
            # Import managers here to avoid circular imports
            from catley.backends.wgpu.resource_manager import WGPUResourceManager
            from catley.backends.wgpu.shader_manager import WGPUShaderManager

            assert self.device is not None
            assert self.queue is not None

            self._shader_manager = WGPUShaderManager(self.device)

            # Initialize resource manager - use shared one if available
            if hasattr(self.graphics_context, "resource_manager"):
                self._resource_manager = self.graphics_context.resource_manager
            else:
                self._resource_manager = WGPUResourceManager(self.device, self.queue)

            # Create shader module and render pipeline
            self._create_render_pipeline()

            # Create uniform buffer
            self._create_uniform_buffer()

            # Create fullscreen quad vertex buffer
            self._create_fullscreen_quad()

            # Create sampler for sky exposure texture
            self._create_sampler()

            return True

        except Exception as e:
            import traceback

            print(f"Failed to initialize WGPU fragment-based lighting: {e}")
            print("Full traceback:")
            traceback.print_exc()
            return False

    def _create_render_pipeline(self) -> None:
        """Create the WGPU render pipeline for lighting computation."""
        assert self._shader_manager is not None
        assert self.device is not None

        # Load WGSL lighting shader
        shader_module = self._shader_manager.create_shader_module(
            "wgsl/lighting/point_light.wgsl", "lighting_shader"
        )

        # Create bind group layout for uniforms and textures
        bind_group_layout = self.device.create_bind_group_layout(
            entries=[
                # Uniform buffer (binding 0)
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                # Sky exposure texture (binding 1)
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                # Texture sampler (binding 2)
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {},
                },
            ]
        )

        # Create pipeline layout
        pipeline_layout = self.device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create render pipeline
        self._render_pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": shader_module,
                "entry_point": "vs_main",
                "buffers": [
                    {
                        "array_stride": 8,  # 2 floats * 4 bytes
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
            },
            fragment={
                "module": shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba32float,
                        "blend": None,
                        "write_mask": wgpu.ColorWrite.ALL,
                    }
                ],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil=None,
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
                "alpha_to_coverage_enabled": False,
            },
        )

    def _create_uniform_buffer(self) -> None:
        """Create the uniform buffer for lighting data."""
        assert self.device is not None

        # Calculate uniform buffer size based on WGSL LightingUniforms struct
        # Must match the struct layout exactly with proper alignment
        uniform_size = (
            # viewport_offset: vec2i + viewport_size: vec2i = 16 bytes
            16
            +
            # light_count: i32 + _padding1: vec3i = 16 bytes
            16
            +
            # light_positions: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_radii: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_intensities: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_colors: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_flicker_enabled: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_flicker_speed: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_min_brightness: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # light_max_brightness: array<vec4f, 32> = 32 * 16 = 512 bytes
            512
            +
            # ambient_light + time + tile_aligned + _padding2 = 16 bytes
            16
            +
            # shadow uniforms: count + intensity + max_length + falloff = 16 bytes
            16
            +
            # shadow_caster_positions: array<vec4f, 64> = 64 * 16 = 1024 bytes
            1024
            +
            # sun_direction: vec2f + sun_color: vec3f + sun_intensity: f32 = 24 bytes,
            # rounded to 32
            32
            +
            # sky_exposure_power: f32 + _padding3: vec3f = 16 bytes
            16
        )

        self._uniform_buffer = self.device.create_buffer(
            size=uniform_size,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

    def _create_fullscreen_quad(self) -> None:
        """Create a full-screen quad for fragment shader rendering."""
        # Full-screen quad vertices: position (x, y) in NDC space [-1, 1]
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,  # Bottom-left
                1.0,
                -1.0,  # Bottom-right
                1.0,
                1.0,  # Top-right
                -1.0,
                -1.0,  # Bottom-left
                1.0,
                1.0,  # Top-right
                -1.0,
                1.0,  # Top-left
            ],
            dtype=np.float32,
        )

        assert self.device is not None
        self._vertex_buffer = self.device.create_buffer_with_data(
            data=quad_vertices.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_sampler(self) -> None:
        """Create sampler for sky exposure texture."""
        assert self.device is not None

        self._sky_exposure_sampler = self.device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            mipmap_filter=wgpu.FilterMode.nearest,
        )

    def _collect_light_data(self, viewport_bounds: Rect) -> list[float]:
        """Collect light data from the game world and format for GPU uniforms.

        Args:
            viewport_bounds: The viewport to collect lights for

        Returns:
            Flat list of floats representing light data (12 floats per light)
            Format: position.xy, radius, base_intensity, color.rgb,
                   flicker_enabled, flicker_speed, min_brightness, max_brightness
        """
        from catley.game.lights import DirectionalLight

        light_data = []

        for light in self.game_world.lights:
            # Skip directional lights - they are handled separately
            if isinstance(light, DirectionalLight):
                continue
            lx, ly = light.position

            # Simple frustum culling - only include lights that could affect viewport
            if (
                viewport_bounds.x1 - light.radius
                <= lx
                < viewport_bounds.x1 + viewport_bounds.width + light.radius
                and viewport_bounds.y1 - light.radius
                <= ly
                < viewport_bounds.y1 + viewport_bounds.height + light.radius
            ):
                # Get light color as RGB floats (color is a tuple of 0-255 integers)
                r, g, b = (
                    light.color[0] / 255.0,
                    light.color[1] / 255.0,
                    light.color[2] / 255.0,
                )

                # Base intensity (flicker will be applied in shader)
                base_intensity = 1.0

                # Extract flicker parameters from DynamicLight objects
                from catley.game.lights import DynamicLight

                if isinstance(light, DynamicLight):
                    flicker_enabled = 1.0 if light.flicker_enabled else 0.0
                    flicker_speed = light.flicker_speed
                    min_brightness = light.min_brightness
                    max_brightness = light.max_brightness
                else:
                    # Static lights don't flicker
                    flicker_enabled = 0.0
                    flicker_speed = 1.0
                    min_brightness = 1.0
                    max_brightness = 1.0

                # Pack light data: position.xy, radius, base_intensity, color.rgb,
                # flicker_enabled, flicker_speed, min_brightness, max_brightness
                light_data.extend(
                    [
                        float(lx),
                        float(ly),  # position
                        float(light.radius),  # radius
                        base_intensity,  # base intensity
                        r,
                        g,
                        b,  # color
                        flicker_enabled,  # flicker enabled flag
                        flicker_speed,  # flicker speed
                        min_brightness,  # minimum brightness multiplier
                        max_brightness,  # maximum brightness multiplier
                        0.0,  # padding for alignment
                    ]
                )

        return light_data

    def _collect_shadow_casters_global(self, viewport_bounds: Rect) -> list[float]:
        """Collect all shadow casters in the viewport globally.

        This method replaces the inefficient per-light approach with a single
        global collection, letting the GPU shader handle per-light relevance.

        Performance: O(N) vs O(N²) from per-light collection approach.

        Args:
            viewport_bounds: The viewport bounds for rendering

        Returns:
            Flat list of floats representing shadow caster data (2 floats per caster)
            Format: position.xy for each shadow caster in viewport
        """
        from catley.config import SHADOW_MAX_LENGTH, SHADOWS_ENABLED

        if not SHADOWS_ENABLED:
            return []

        shadow_casters = []

        # Expand viewport bounds to include potential shadow influence
        expanded_bounds = Rect.from_bounds(
            x1=viewport_bounds.x1 - SHADOW_MAX_LENGTH,
            y1=viewport_bounds.y1 - SHADOW_MAX_LENGTH,
            x2=viewport_bounds.x2 + SHADOW_MAX_LENGTH,
            y2=viewport_bounds.y2 + SHADOW_MAX_LENGTH,
        )

        # Get shadow-casting actors in expanded viewport
        if self.game_world.actor_spatial_index:
            actors = self.game_world.actor_spatial_index.get_in_bounds(
                int(expanded_bounds.x1),
                int(expanded_bounds.y1),
                int(expanded_bounds.x2),
                int(expanded_bounds.y2),
            )

            for actor in actors:
                if hasattr(actor, "blocks_movement") and actor.blocks_movement:
                    shadow_casters.extend([float(actor.x), float(actor.y)])

        # Get shadow-casting tiles in expanded viewport
        if self.game_world.game_map:
            from catley.environment import tile_types

            game_map = self.game_world.game_map

            # Calculate tile bounds within expanded viewport
            min_x = max(0, int(expanded_bounds.x1))
            max_x = min(game_map.width, int(expanded_bounds.x2) + 1)
            min_y = max(0, int(expanded_bounds.y1))
            max_y = min(game_map.height, int(expanded_bounds.y2) + 1)

            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    tile_id = game_map.tiles[x, y]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))
                    if tile_data["casts_shadows"]:
                        shadow_casters.extend([float(x), float(y)])

        return shadow_casters

    def _update_sky_exposure_texture(self) -> None:
        """Update the sky exposure texture from the game map's region data.

        This creates a texture where each pixel represents a tile's sky exposure value,
        based on the region it belongs to. Only recreates when map structure changes.
        """
        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Check if we need to update based on map revision
        if (
            self._cached_map_revision == game_map.structural_revision
            and self._sky_exposure_texture is not None
        ):
            return  # No update needed

        # Create float32 array for sky exposure data
        sky_exposure_data = np.zeros(
            (game_map.height, game_map.width), dtype=np.float32
        )

        # Populate sky exposure data from regions, respecting tile transparency
        for y in range(game_map.height):
            for x in range(game_map.width):
                region = game_map.get_region_at((x, y))
                if region:
                    # Non-transparent tiles block all sunlight
                    if game_map.transparent[x, y]:
                        sky_exposure_data[y, x] = region.sky_exposure
                    else:
                        sky_exposure_data[y, x] = 0.0  # Block all sunlight

        # Release old texture if it exists
        if self._sky_exposure_texture is not None:
            # WGPU textures don't need explicit release
            self._sky_exposure_texture = None
            self._bind_group = None  # Invalidate bind group when texture changes

        # Create new texture with sky exposure data - WGPU syntax
        assert self.device is not None
        assert self.queue is not None

        self._sky_exposure_texture = self.device.create_texture(
            size=(game_map.width, game_map.height, 1),
            format=wgpu.TextureFormat.r32float,  # Single channel 32-bit float
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        # Write sky exposure data to texture - WGPU syntax
        self.queue.write_texture(
            {
                "texture": self._sky_exposure_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            sky_exposure_data.tobytes(),
            {
                "offset": 0,
                "bytes_per_row": game_map.width * 4,  # 4 bytes per float32
                "rows_per_image": game_map.height,
            },
            (game_map.width, game_map.height, 1),
        )

        # Update cached revision
        self._cached_map_revision = game_map.structural_revision

    def _pack_uniform_data(
        self,
        light_data: list[float],
        light_count: int,
        shadow_casters: list[float],
        shadow_caster_count: int,
        viewport_bounds: Rect,
    ) -> bytes:
        """Pack all uniform data into a single buffer matching WGSL LightingUniforms
        struct.

        This replaces ModernGL's individual uniform setters with WGPU's structured
        approach.
        """
        from catley.config import (
            SHADOW_FALLOFF,
            SHADOW_INTENSITY,
            SHADOW_MAX_LENGTH,
            SKY_EXPOSURE_POWER,
        )
        from catley.game.lights import DirectionalLight

        # Find active directional light
        directional_light = None
        for light in self.game_world.lights:
            if isinstance(light, DirectionalLight):
                directional_light = light
                break

        # Start building the struct data using Python's struct module
        # Must match WGSL LightingUniforms layout exactly
        struct_data = []

        # viewport_offset: vec2i + viewport_size: vec2i = 16 bytes
        struct_data.extend(
            [
                viewport_bounds.x1,
                viewport_bounds.y1,  # viewport_offset
                viewport_bounds.width,
                viewport_bounds.height,  # viewport_size
            ]
        )

        # light_count: i32 + _padding1: vec3i = 16 bytes (4 i32s)
        struct_data.extend([light_count, 0, 0, 0])  # padding to 16 bytes

        # Prepare light data arrays (pad to MAX_LIGHTS with vec4f alignment)
        # light_positions: array<vec4f, 32> = 32 * 16 = 512 bytes
        for i in range(self.MAX_LIGHTS):
            if i < light_count:
                base_idx = i * 12
                x = light_data[base_idx]
                y = light_data[base_idx + 1]
                struct_data.extend([x, y, 0.0, 0.0])  # vec4f: xy used, zw padding
            else:
                struct_data.extend([0.0, 0.0, 0.0, 0.0])  # Empty light

        # light_radii: array<vec4f, 32> = 32 * 16 = 512 bytes
        for i in range(self.MAX_LIGHTS):
            if i < light_count:
                base_idx = i * 12
                radius = light_data[base_idx + 2]
                struct_data.extend(
                    [radius, 0.0, 0.0, 0.0]
                )  # vec4f: x used, yzw padding
            else:
                struct_data.extend([0.0, 0.0, 0.0, 0.0])

        # light_intensities: array<vec4f, 32> = 32 * 16 = 512 bytes
        for i in range(self.MAX_LIGHTS):
            if i < light_count:
                base_idx = i * 12
                intensity = light_data[base_idx + 3]
                struct_data.extend(
                    [intensity, 0.0, 0.0, 0.0]
                )  # vec4f: x used, yzw padding
            else:
                struct_data.extend([0.0, 0.0, 0.0, 0.0])

        # light_colors: array<vec4f, 32> = 32 * 16 = 512 bytes
        for i in range(self.MAX_LIGHTS):
            if i < light_count:
                base_idx = i * 12
                r = light_data[base_idx + 4]
                g = light_data[base_idx + 5]
                b = light_data[base_idx + 6]
                struct_data.extend([r, g, b, 0.0])  # vec4f: rgb used, w padding
            else:
                struct_data.extend([0.0, 0.0, 0.0, 0.0])

        # Flicker arrays: light_flicker_enabled, light_flicker_speed,
        # light_min_brightness, light_max_brightness
        # Each is array<vec4f, 32> = 32 * 16 = 512 bytes
        flicker_arrays = [7, 8, 9, 10]  # indices in light_data
        for array_idx in flicker_arrays:
            for i in range(self.MAX_LIGHTS):
                if i < light_count:
                    base_idx = i * 12
                    value = light_data[base_idx + array_idx]
                    struct_data.extend(
                        [value, 0.0, 0.0, 0.0]
                    )  # vec4f: x used, yzw padding
                else:
                    struct_data.extend([0.0, 0.0, 0.0, 0.0])

        # Global uniforms: ambient_light + time + tile_aligned + _padding2 = 16 bytes
        struct_data.extend(
            [
                AMBIENT_LIGHT_LEVEL,
                self._time,
                1.0,  # tile_aligned (always True)
                0.0,  # padding
            ]
        )

        # Shadow uniforms: shadow_caster_count + shadow_intensity +
        # shadow_max_length + shadow_falloff_enabled = 16 bytes
        struct_data.extend(
            [
                shadow_caster_count,
                SHADOW_INTENSITY,
                SHADOW_MAX_LENGTH,
                1.0 if SHADOW_FALLOFF else 0.0,
            ]
        )

        # shadow_caster_positions: array<vec4f, 64> = 64 * 16 = 1024 bytes
        MAX_SHADOW_CASTERS = 64
        actual_count = min(shadow_caster_count, MAX_SHADOW_CASTERS)
        for i in range(MAX_SHADOW_CASTERS):
            if i < actual_count:
                x = shadow_casters[i * 2]
                y = shadow_casters[i * 2 + 1]
                struct_data.extend([x, y, 0.0, 0.0])  # vec4f: xy used, zw padding
            else:
                struct_data.extend([0.0, 0.0, 0.0, 0.0])

        # Directional light uniforms: sun_direction + sun_color +
        # sun_intensity + sky_exposure_power + padding
        if directional_light:
            sun_dir_x = directional_light.direction.x
            sun_dir_y = directional_light.direction.y
            sun_color_r = directional_light.color[0] / 255.0
            sun_color_g = directional_light.color[1] / 255.0
            sun_color_b = directional_light.color[2] / 255.0
            sun_intensity = directional_light.intensity
        else:
            sun_dir_x = sun_dir_y = 0.0
            sun_color_r = sun_color_g = sun_color_b = 0.0
            sun_intensity = 0.0

        # sun_direction: vec2f + sun_color: vec3f + sun_intensity: f32 = 24 bytes,
        # pad to 32
        struct_data.extend(
            [
                sun_dir_x,
                sun_dir_y,  # sun_direction
                sun_color_r,
                sun_color_g,
                sun_color_b,  # sun_color
                sun_intensity,  # sun_intensity
                0.0,
                0.0,  # padding to 32 bytes
            ]
        )

        # sky_exposure_power: f32 + _padding3: vec3f = 16 bytes
        struct_data.extend(
            [
                SKY_EXPOSURE_POWER,
                0.0,
                0.0,
                0.0,  # padding
            ]
        )

        # Convert to bytes using struct.pack
        # Use appropriate format for each data type (i=int32, f=float32)
        # First 8 values are integers (viewport + light_count + padding)
        format_parts = ["i"] * 8 + ["f"] * (len(struct_data) - 8)
        format_string = "".join(format_parts)

        # Convert first 8 values to integers
        int_data = [int(x) for x in struct_data[:8]]
        float_data = struct_data[8:]
        all_data = int_data + float_data

        return struct.pack(format_string, *all_data)

    def _update_uniform_buffer(
        self,
        light_data: list[float],
        light_count: int,
        shadow_casters: list[float],
        shadow_caster_count: int,
        viewport_bounds: Rect,
    ) -> None:
        """Update the uniform buffer with current lighting data."""
        assert self._uniform_buffer is not None
        assert self.queue is not None

        # Pack all uniform data into structured format
        uniform_bytes = self._pack_uniform_data(
            light_data,
            light_count,
            shadow_casters,
            shadow_caster_count,
            viewport_bounds,
        )

        # Write to uniform buffer
        self.queue.write_buffer(self._uniform_buffer, 0, uniform_bytes)

    def _update_time_uniform(self) -> None:
        """Update only the time uniform for better performance."""
        if self._uniform_buffer is None or self.queue is None:
            return

        # Calculate offset to time field in uniform buffer
        # Time is at: all previous fields + ambient_light (4 bytes)
        time_offset = (
            16  # viewport_offset + viewport_size
            + 16  # light_count + padding
            + 512  # light_positions
            + 512  # light_radii
            + 512  # light_intensities
            + 512  # light_colors
            + 512  # light_flicker_enabled
            + 512  # light_flicker_speed
            + 512  # light_min_brightness
            + 512  # light_max_brightness
            + 16  # shadow uniforms
            + 1024  # shadow_caster_positions
            + 32  # sun data
            + 16  # sky_exposure_power + padding
            + 4  # ambient_light (f32, 4 bytes) - time comes after this
        )

        # Write just the time value (4 bytes)
        time_data = struct.pack("f", self._time)
        self.queue.write_buffer(self._uniform_buffer, time_offset, time_data)

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects."""
        self._time += fixed_timestep

        # Update fallback system if available
        if self.fallback_system is not None:
            self.fallback_system.update(fixed_timestep)

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap using WGPU fragment shaders.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if computation failed
        """
        # Check if GPU is available
        if self._render_pipeline is None:
            if self.fallback_system is not None:
                return self.fallback_system.compute_lightmap(viewport_bounds)
            return None

        # Try GPU computation first
        gpu_result = self._compute_lightmap_gpu(viewport_bounds)
        if gpu_result is not None:
            self.revision += 1
            return gpu_result

        # Fall back to CPU system if available
        if self.fallback_system is not None:
            return self.fallback_system.compute_lightmap(viewport_bounds)

        # No fallback available
        return None

    def _ensure_resources_for_viewport(self, viewport_bounds: Rect) -> bool:
        """Ensure GPU resources are sized appropriately for the viewport.

        Args:
            viewport_bounds: The viewport area that will be rendered

        Returns:
            True if resources are ready, False if fallback needed
        """
        if self._render_pipeline is None:
            return False

        # Check if we need to resize the output texture
        if (
            self._current_viewport is None
            or self._current_viewport.width != viewport_bounds.width
            or self._current_viewport.height != viewport_bounds.height
        ):
            try:
                # Release old texture/buffer if they exist
                self._output_texture = None
                self._output_buffer = None
                self._bind_group = None  # Invalidate bind group when texture changes

                # Create new output texture
                assert self.device is not None
                self._output_texture = self.device.create_texture(
                    size=(viewport_bounds.width, viewport_bounds.height, 1),
                    format=wgpu.TextureFormat.rgba32float,  # RGBA 32-bit float
                    # to handle shader output
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.COPY_SRC,
                )

                # Create buffer for reading back results
                self._output_buffer = self.device.create_buffer(
                    size=viewport_bounds.width
                    * viewport_bounds.height
                    * 4
                    * 4,  # 4 components * 4 bytes each
                    usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
                )

                self._current_viewport = viewport_bounds

            except Exception:
                return False

        return True

    def _compute_lightmap_gpu(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Perform the actual GPU lighting computation using fragment shaders."""
        try:
            # Ensure resources are ready
            if not self._ensure_resources_for_viewport(viewport_bounds):
                return None

            assert self._render_pipeline is not None
            assert self._output_texture is not None
            assert self._output_buffer is not None
            assert self._vertex_buffer is not None

            # Use cached light data if revision hasn't changed
            if (
                self._cached_light_revision != self.revision
                or self._cached_light_data is None
            ):
                self._cached_light_data = self._collect_light_data(viewport_bounds)
                self._cached_light_revision = self.revision

            light_data = self._cached_light_data
            light_count = min(
                len(light_data) // 12, self.MAX_LIGHTS
            )  # 12 floats per light

            if light_count > self.MAX_LIGHTS:
                print(
                    f"Warning: Too many lights ({light_count}), "
                    f"limiting to {self.MAX_LIGHTS}"
                )

            # Collect shadow casters globally
            if (
                self._cached_shadow_revision != self.revision
                or self._cached_shadow_casters is None
            ):
                self._cached_shadow_casters = self._collect_shadow_casters_global(
                    viewport_bounds
                )
                self._cached_shadow_revision = self.revision

            shadow_casters = self._cached_shadow_casters
            shadow_caster_count = len(shadow_casters) // 2  # 2 floats per caster

            # Update sky exposure texture if needed
            self._update_sky_exposure_texture()

            # Smart uniform updates - only update when lights or shadow casters change
            light_data_hash = hash(
                (
                    tuple(light_data[: light_count * 12]),
                    tuple(shadow_casters),
                    # Note: time is excluded from hash for performance -
                    # updated separately
                )
            )
            if self._last_light_data_hash != light_data_hash:
                self._update_uniform_buffer(
                    light_data,
                    light_count,
                    shadow_casters,
                    shadow_caster_count,
                    viewport_bounds,
                )
                self._last_light_data_hash = light_data_hash
            else:
                # Lights haven't changed, only update time for dynamic effects
                self._update_time_uniform()

            # Create bind group with current resources (only if needed)
            if self._bind_group is None:
                self._create_bind_group()

            # Create command encoder and render pass
            assert self.device is not None
            command_encoder = self.device.create_command_encoder()

            # Create render pass
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self._output_texture.create_view(),
                        "resolve_target": None,
                        "clear_value": (0.0, 0.0, 0.0, 1.0),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    }
                ]
            )

            # Set pipeline and resources
            render_pass.set_pipeline(self._render_pipeline)
            render_pass.set_bind_group(0, self._bind_group)
            render_pass.set_vertex_buffer(0, self._vertex_buffer)

            # Draw fullscreen quad (6 vertices for 2 triangles)
            render_pass.draw(6, 1, 0, 0)

            render_pass.end()

            # Copy render target to readback buffer
            command_encoder.copy_texture_to_buffer(
                {
                    "texture": self._output_texture,
                    "mip_level": 0,
                    "origin": (0, 0, 0),
                },
                {
                    "buffer": self._output_buffer,
                    "offset": 0,
                    "bytes_per_row": viewport_bounds.width
                    * 4
                    * 4,  # 4 components * 4 bytes
                    "rows_per_image": viewport_bounds.height,
                },
                (viewport_bounds.width, viewport_bounds.height, 1),
            )

            # Submit commands
            assert self.queue is not None
            self.queue.submit([command_encoder.finish()])

            # Map buffer for reading (without explicit sync)
            assert self._output_buffer is not None
            mapped_data = self._output_buffer.map_sync(wgpu.MapMode.READ)  # type: ignore
            assert mapped_data is not None
            result_data = np.frombuffer(mapped_data, dtype=np.float32)
            self._output_buffer.unmap()

            # Reshape to (width, height, 4) and extract RGB
            result_image = result_data.reshape(
                (viewport_bounds.height, viewport_bounds.width, 4)
            )
            rgb_result = result_image[:, :, :3]  # Extract RGB channels

            # Check for invalid values that could cause overflow
            if np.any(np.isnan(rgb_result)) or np.any(np.isinf(rgb_result)):
                return None

            if np.any(rgb_result < 0) or np.any(rgb_result > 1):
                # Clamp to valid range
                rgb_result = np.clip(rgb_result, 0.0, 1.0)

            # Transpose to match expected (width, height, 3) format
            return np.transpose(rgb_result, (1, 0, 2))

        except Exception:
            return None

    def _create_bind_group(self) -> None:
        """Create bind group with current uniform buffer and sky exposure texture."""
        assert self.device is not None
        assert self._uniform_buffer is not None
        assert self._sky_exposure_sampler is not None

        # Create a default sky exposure texture if none exists
        if self._sky_exposure_texture is None:
            self._create_default_sky_exposure_texture()

        assert self._sky_exposure_texture is not None

        assert self._render_pipeline is not None
        self._bind_group = self.device.create_bind_group(
            layout=self._render_pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self._uniform_buffer},
                },
                {
                    "binding": 1,
                    "resource": self._sky_exposure_texture.create_view(),
                },
                {
                    "binding": 2,
                    "resource": self._sky_exposure_sampler,
                },
            ],
        )

    def _create_default_sky_exposure_texture(self) -> None:
        """Create a default 1x1 sky exposure texture for when no map is available."""
        assert self.device is not None
        assert self.queue is not None

        # Create 1x1 texture with 0.0 sky exposure
        self._sky_exposure_texture = self.device.create_texture(
            size=(1, 1, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._bind_group = None  # Invalidate bind group when texture is created

        # Write zero sky exposure
        sky_data = np.array([0.0], dtype=np.float32)
        self.queue.write_texture(
            {
                "texture": self._sky_exposure_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            sky_data.tobytes(),
            {
                "offset": 0,
                "bytes_per_row": 4,  # 1 pixel * 4 bytes
                "rows_per_image": 1,
            },
            (1, 1, 1),
        )

    # LightingSystem event handlers
    def on_light_added(self, light: LightSource) -> None:
        """Notification that a light has been added."""
        self.revision += 1
        if self.fallback_system is not None:
            self.fallback_system.on_light_added(light)

    def on_light_removed(self, light: LightSource) -> None:
        """Notification that a light has been removed."""
        self.revision += 1
        if self.fallback_system is not None:
            self.fallback_system.on_light_removed(light)

    def on_light_moved(self, light: LightSource) -> None:
        """Notification that a light has moved."""
        self.revision += 1
        if self.fallback_system is not None:
            self.fallback_system.on_light_moved(light)

    def on_global_light_changed(self) -> None:
        """Notification that global lighting has changed."""
        self.revision += 1
        if self.fallback_system is not None:
            self.fallback_system.on_global_light_changed()

    def release(self) -> None:
        """Release GPU resources."""
        # WGPU resources are automatically cleaned up by Python GC
        # Just clear references
        self._render_pipeline = None
        self._uniform_buffer = None
        self._bind_group = None
        self._output_texture = None
        self._output_buffer = None
        self._vertex_buffer = None
        self._sky_exposure_texture = None
        self._sky_exposure_sampler = None

        # Clear cached data
        self._cached_light_data = None
        self._cached_shadow_casters = None
        self._last_light_data_hash = None
