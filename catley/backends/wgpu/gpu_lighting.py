"""WGPU-based implementation of the lighting system using fragment shaders.

This class implements the lighting system using WGPU fragment shaders for
high-performance parallel light computation. It maintains compatibility with
the LightingSystem interface while providing significant performance gains
for scenes with many lights.

Uses fragment shaders instead of compute shaders for compatibility.
"""

from __future__ import annotations

import math
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
    from catley.game.actors import Actor
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource
    from catley.view.render.graphics import GraphicsContext


# WGPU requires bytes_per_row to be a multiple of 256 when copying textures to buffers
WGPU_COPY_BYTES_PER_ROW_ALIGNMENT = 256


def _align_to_copy_row_alignment(size: int) -> int:
    """Round up to the next multiple of WGPU's copy row alignment (256 bytes)."""
    alignment = WGPU_COPY_BYTES_PER_ROW_ALIGNMENT
    return (size + alignment - 1) & ~(alignment - 1)


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
    """

    # Maximum number of lights we can handle in a single render pass
    MAX_LIGHTS = 32

    def __init__(
        self,
        game_world: GameWorld,
        graphics_context: GraphicsContext,
    ) -> None:
        """Initialize the WGPU GPU lighting system.

        Args:
            game_world: The game world to query for lighting data
            graphics_context: Graphics context (WGPU required for GPU operations)
        """
        super().__init__(game_world)

        self.graphics_context = graphics_context
        # Get WGPU device/queue - fail fast if unavailable
        if hasattr(graphics_context, "device") and hasattr(graphics_context, "queue"):
            self.device: wgpu.GPUDevice = graphics_context.device
            self.queue: wgpu.GPUQueue = graphics_context.queue
        else:
            # Create a standalone WGPU context for lighting calculations
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference=wgpu.PowerPreference.high_performance
            )
            self.device = adapter.request_device_sync(
                required_features=[],
                required_limits={},
                label="catley_lighting_device",
            )
            self.queue = self.device.queue

        # WGPU resources
        self._render_pipeline: wgpu.GPURenderPipeline | None = None
        self._shader_manager: WGPUShaderManager | None = None
        self._resource_manager: WGPUResourceManager | None = None
        # Single uniform buffer for simplified struct (with proper vec4f alignment)
        self._uniform_buffer: wgpu.GPUBuffer | None = None
        self._bind_group: wgpu.GPUBindGroup | None = None
        self._output_texture: wgpu.GPUTexture | None = None
        self._output_buffer: wgpu.GPUBuffer | None = None
        self._vertex_buffer: wgpu.GPUBuffer | None = None

        # Track time for dynamic effects
        self._time = 0.0

        # Current viewport for resource sizing
        self._current_viewport: Rect | None = None
        # Row stride for buffer readback (WGPU requires 256-byte alignment)
        self._padded_bytes_per_row: int = 0
        self._unpadded_bytes_per_row: int = 0

        # Performance optimization: Track light configuration changes
        self._last_light_data_hash: int | None = None
        self._cached_light_data: list[float] | None = None
        self._cached_light_revision: int = -1

        # Track first frame to force uniform update
        self._first_frame = True

        # Cache for shadow casters
        self._cached_shadow_casters: list[float] | None = None
        self._cached_shadow_revision: int = -1

        # Sky exposure texture for directional lighting
        self._sky_exposure_texture: wgpu.GPUTexture | None = None
        self._sky_exposure_sampler: wgpu.GPUSampler | None = None
        self._cached_map_revision: int = -1

        # Explored/visible textures for fog-of-war masking
        self._explored_texture: wgpu.GPUTexture | None = None
        self._cached_exploration_revision: int = -1
        self._visible_texture: wgpu.GPUTexture | None = None
        self._cached_visibility_revision: tuple[int, int] | None = None

        # Emission texture for light-emitting tiles (acid pools, hot coals, etc.)
        self._emission_texture: wgpu.GPUTexture | None = None

        # Shadow grid texture for terrain shadow casting
        self._shadow_grid_texture: wgpu.GPUTexture | None = None
        self._cached_shadow_grid_revision: int = -1

        # Initialize GPU resources
        if not self._initialize_gpu_resources():
            raise RuntimeError("Failed to initialize WGPU GPU lighting system")

    def _initialize_gpu_resources(self) -> bool:
        """Initialize WGPU fragment shader-based lighting.

        Returns:
            True if initialization successful, False if fallback needed
        """
        try:
            # Initialize fragment-based lighting
            return self._initialize_fragment_lighting()

        except Exception as e:
            print(f"Failed to initialize WGPU GPU resources: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _initialize_fragment_lighting(self) -> bool:
        """Initialize fragment shader-based lighting."""
        try:
            # Import managers here to avoid circular imports
            from catley.backends.wgpu.resource_manager import WGPUResourceManager
            from catley.backends.wgpu.shader_manager import WGPUShaderManager

            self._shader_manager = WGPUShaderManager(self.device)

            # Initialize resource manager - use shared one if available
            resource_manager = self.graphics_context.resource_manager
            if isinstance(resource_manager, WGPUResourceManager):
                self._resource_manager = resource_manager
            else:
                self._resource_manager = WGPUResourceManager(self.device, self.queue)

            # Create shader module and render pipeline
            self._create_render_pipeline()

            # Create uniform buffer
            self._create_uniform_buffer()

            # Create fullscreen quad vertex buffer
            self._create_fullscreen_quad()

            # Create samplers for textures
            self._create_samplers()

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
                # Sky exposure texture (binding 22)
                {
                    "binding": 22,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
                },
                # Texture sampler (binding 23)
                {
                    "binding": 23,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {},
                },
                # Emission texture (binding 24) - unfilterable since rgba32float
                {
                    "binding": 24,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float
                    },
                },
                # Shadow grid texture (binding 25) - r8unorm for terrain shadows
                {
                    "binding": 25,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": wgpu.TextureSampleType.float},
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
        """Create uniform buffer for simplified struct."""

        # Estimate size for simplified struct with vec4f alignment
        # Each vec4f array element takes 16 bytes regardless of actual usage
        estimated_size = (
            16  # viewport_data: vec4i (16 bytes)
            + 16  # light_count + ambient_light + time + tile_aligned (16 bytes)
            + 32 * 16 * 8  # 8 light arrays * 32 lights * 16 bytes each
            + 16  # actor shadow uniforms (16 bytes)
            + 64 * 16  # actor shadow positions: 64 actors * 16 bytes each
            + 16  # sun uniforms (16 bytes)
            + 64  # extra padding
        )

        self._uniform_buffer = self.device.create_buffer(
            size=estimated_size,
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

        self._vertex_buffer = self.device.create_buffer_with_data(
            data=quad_vertices.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )

    def _create_samplers(self) -> None:
        """Create sampler for the sky exposure texture.

        Uses nearest-neighbor filtering to prevent interpolation bleeding
        at tile boundaries (walls should have sharp sky exposure cutoff).
        This matches the ModernGL backend's behavior.
        """

        self._sky_exposure_sampler = self.device.create_sampler(
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
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

    def _collect_actor_shadow_casters(self, viewport_bounds: Rect) -> list[float]:
        """Collect actor shadow casters in the viewport.

        Terrain shadows are now handled by the shadow grid texture. This method
        only collects actors (NPCs) that cast shadows, for the small actor
        shadow uniform array in the shader.

        Args:
            viewport_bounds: The viewport bounds for rendering

        Returns:
            Flat list of floats representing actor data (3 floats per actor):
            [x, y, shadow_height, x, y, shadow_height, ...]
        """
        from catley.config import SHADOW_MAX_LENGTH, SHADOWS_ENABLED
        from catley.game.lights import DirectionalLight

        if not SHADOWS_ENABLED:
            return []

        # Maximum actors the shader can handle
        MAX_ACTOR_SHADOWS = 64

        # Compute expansion that accounts for shadow stretching at low sun
        # elevations. The shader extends shadow reach by length_scale, so we
        # must include actors whose stretched shadows could reach the viewport.
        shadow_expansion = SHADOW_MAX_LENGTH
        directional_light = next(
            (
                light
                for light in self.game_world.get_global_lights()
                if isinstance(light, DirectionalLight)
            ),
            None,
        )
        if directional_light:
            elev_rad = math.radians(max(directional_light.elevation_degrees, 0.1))
            length_scale = min(1.0 / math.tan(elev_rad), 8.0)
            shadow_expansion = int(SHADOW_MAX_LENGTH * length_scale + 0.5)

        # Expand viewport bounds to include potential shadow influence
        expanded_bounds = Rect.from_bounds(
            x1=viewport_bounds.x1 - shadow_expansion,
            y1=viewport_bounds.y1 - shadow_expansion,
            x2=viewport_bounds.x2 + shadow_expansion,
            y2=viewport_bounds.y2 + shadow_expansion,
        )

        # Collect shadow-casting actors in expanded viewport
        actor_positions: list[float] = []

        if self.game_world.actor_spatial_index:
            actors = self.game_world.actor_spatial_index.get_in_bounds(
                int(expanded_bounds.x1),
                int(expanded_bounds.y1),
                int(expanded_bounds.x2),
                int(expanded_bounds.y2),
            )

            count = 0
            for actor in actors:
                if count >= MAX_ACTOR_SHADOWS:
                    break
                shadow_h = getattr(actor, "shadow_height", 0)
                if shadow_h > 0:
                    actor_positions.extend(
                        [float(actor.x), float(actor.y), float(shadow_h)]
                    )
                    count += 1

        return actor_positions

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

        # Create RGBA8 array for sky exposure data (use red channel for exposure)
        sky_exposure_data = np.zeros(
            (game_map.height, game_map.width, 4), dtype=np.uint8
        )

        # Vectorized sky exposure calculation: iterate over regions (small dict)
        # instead of tiles (large grid). Build a lookup array mapping region_id
        # to sky_exposure, then use numpy advanced indexing.
        if game_map.regions:
            max_region_id = max(game_map.regions.keys())
            # Lookup table: index = region_id, value = sky_exposure * 255
            exposure_lookup = np.zeros(max_region_id + 1, dtype=np.uint8)
            for region_id, region in game_map.regions.items():
                exposure_lookup[region_id] = int(region.sky_exposure * 255)

            # Map tile_to_region_id to sky exposure values (clamp -1 to 0 for lookup)
            region_ids = game_map.tile_to_region_id
            clamped_ids = np.clip(region_ids, 0, max_region_id)
            sky_values = exposure_lookup[clamped_ids]

            # Tiles with no region (id < 0) get 0 exposure
            sky_values = np.where(region_ids >= 0, sky_values, 0)

            # Non-transparent tiles block all sunlight
            sky_values = np.where(game_map.transparent, sky_values, 0)

            # Store in red channel (transpose from (w,h) to (h,w) for texture)
            sky_exposure_data[:, :, 0] = sky_values.T

        # Set alpha to 255 for all pixels
        sky_exposure_data[:, :, 3] = 255

        # Release old texture if it exists
        if self._sky_exposure_texture is not None:
            # WGPU textures don't need explicit release
            self._sky_exposure_texture = None
            self._bind_group = None  # Invalidate bind group when texture changes

        # Create new texture with sky exposure data - WGPU syntax

        self._sky_exposure_texture = self.device.create_texture(
            size=(game_map.width, game_map.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        # Write sky exposure data to texture - WGPU syntax
        self.queue.write_texture(
            {
                "texture": self._sky_exposure_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(sky_exposure_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": game_map.width * 4,  # 4 bytes per float32
                "rows_per_image": game_map.height,
            },
            (game_map.width, game_map.height, 1),
        )

        # Update cached revision
        self._cached_map_revision = game_map.structural_revision

    def _update_explored_texture(self) -> None:
        """Update the explored mask texture from the game map.

        Creates a single-channel texture where each pixel represents whether
        a tile has been explored by the player (255) or not (0).
        Uses r8unorm format for WebGPU compatibility (filterable).
        """
        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Check if we need to update based on exploration revision
        if (
            self._cached_exploration_revision == game_map.exploration_revision
            and self._explored_texture is not None
        ):
            return  # No update needed

        # Convert bool array to uint8 (0 or 255), transposed to match texture coords
        explored_data = np.ascontiguousarray(
            (game_map.explored.T * 255).astype(np.uint8)
        )

        # Release old texture if it exists
        if self._explored_texture is not None:
            self._explored_texture = None

        # Create new texture with explored data - use r8unorm for filterability
        self._explored_texture = self.device.create_texture(
            size=(game_map.width, game_map.height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        # Write explored data to texture
        self.queue.write_texture(
            {
                "texture": self._explored_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(explored_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": game_map.width,  # 1 byte per uint8
                "rows_per_image": game_map.height,
            },
            (game_map.width, game_map.height, 1),
        )

        # Update cached revision
        self._cached_exploration_revision = game_map.exploration_revision

    def _update_visible_texture(self) -> None:
        """Update the visible mask texture from the game map.

        Creates a single-channel texture where each pixel represents whether
        a tile is currently visible to the player (255) or not (0).
        Uses r8unorm format for WebGPU compatibility (filterable).
        """
        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Check if we need to update based on visibility revision
        if self._visible_texture is not None:
            current_revision = (self._cached_exploration_revision, self.revision)
            if self._cached_visibility_revision == current_revision:
                return  # No update needed

        # Convert bool array to uint8 (0 or 255), transposed to match texture coords
        visible_data = np.ascontiguousarray((game_map.visible.T * 255).astype(np.uint8))

        # Release old texture if it exists
        if self._visible_texture is not None:
            self._visible_texture = None

        # Create new texture with visible data - use r8unorm for filterability
        self._visible_texture = self.device.create_texture(
            size=(game_map.width, game_map.height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        # Write visible data to texture
        self.queue.write_texture(
            {
                "texture": self._visible_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(visible_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": game_map.width,  # 1 byte per uint8
                "rows_per_image": game_map.height,
            },
            (game_map.width, game_map.height, 1),
        )

        # Update cached revision
        self._cached_visibility_revision = (
            self._cached_exploration_revision,
            self.revision,
        )

    def _update_emission_texture(self, viewport_bounds: Rect) -> None:
        """Update the emission texture with light-emitting tile data.

        Creates a texture where each texel represents a tile's emission properties:
        - RGB: emission color (0-1, pre-multiplied by intensity)
        - A: light radius (for falloff calculation in shader)

        Uses vectorized numpy operations for performance - avoids Python loops
        over every tile in the viewport.

        Args:
            viewport_bounds: The viewport area being rendered
        """
        from catley.config import TILE_EMISSION_ENABLED
        from catley.environment.tile_types import get_emission_map

        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Early exit if emission is disabled - still need a zeroed texture
        if not TILE_EMISSION_ENABLED:
            if self._emission_texture is None:
                self._emission_texture = self.device.create_texture(
                    size=(viewport_bounds.width, viewport_bounds.height, 1),
                    format=wgpu.TextureFormat.rgba32float,
                    usage=wgpu.TextureUsage.TEXTURE_BINDING
                    | wgpu.TextureUsage.COPY_DST,
                )
                self._bind_group = None
                # Write zeros once
                zeros = np.zeros(
                    (viewport_bounds.height, viewport_bounds.width, 4), dtype=np.float32
                )
                self.queue.write_texture(
                    {
                        "texture": self._emission_texture,
                        "mip_level": 0,
                        "origin": (0, 0, 0),
                    },
                    memoryview(zeros.tobytes()),
                    {
                        "offset": 0,
                        "bytes_per_row": viewport_bounds.width * 16,
                        "rows_per_image": viewport_bounds.height,
                    },
                    (viewport_bounds.width, viewport_bounds.height, 1),
                )
            return

        # Ensure emission texture matches viewport size
        if self._emission_texture is None or self._emission_texture.size != (
            viewport_bounds.width,
            viewport_bounds.height,
            1,
        ):
            # Release old texture
            self._emission_texture = None
            self._bind_group = None  # Invalidate bind group when texture changes

            # Create new emission texture (RGBA32float for precision)
            self._emission_texture = self.device.create_texture(
                size=(viewport_bounds.width, viewport_bounds.height, 1),
                format=wgpu.TextureFormat.rgba32float,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )

        # Create emission data array for the viewport
        emission_data = np.zeros(
            (viewport_bounds.height, viewport_bounds.width, 4), dtype=np.float32
        )

        # Calculate tile bounds within viewport (clamped to map bounds)
        min_x = max(0, viewport_bounds.x1)
        max_x = min(game_map.width, viewport_bounds.x2)
        min_y = max(0, viewport_bounds.y1)
        max_y = min(game_map.height, viewport_bounds.y2)

        # Get emission map for the viewport region
        tile_slice = game_map.tiles[min_x:max_x, min_y:max_y]
        emission_map = get_emission_map(tile_slice)

        # Vectorized emission data population
        # Find all tiles that emit light using boolean mask
        emits_mask = emission_map["emits_light"]

        # Only process if there are any emitting tiles
        if np.any(emits_mask):
            # Get coordinates of emitting tiles (x, y indices due to array shape)
            emitting_coords = np.argwhere(emits_mask)

            # Calculate viewport-relative coordinates for emitting tiles
            # emitting_coords are (local_x, local_y) pairs
            vp_x = min_x + emitting_coords[:, 0] - viewport_bounds.x1
            vp_y = min_y + emitting_coords[:, 1] - viewport_bounds.y1

            # Filter to valid viewport bounds
            valid_mask = (
                (vp_x >= 0)
                & (vp_x < viewport_bounds.width)
                & (vp_y >= 0)
                & (vp_y < viewport_bounds.height)
            )

            if np.any(valid_mask):
                valid_coords = emitting_coords[valid_mask]
                valid_vp_x = vp_x[valid_mask]
                valid_vp_y = vp_y[valid_mask]

                # Extract emission parameters for valid tiles
                valid_emissions = emission_map[valid_coords[:, 0], valid_coords[:, 1]]

                # Get color, intensity, and radius arrays
                colors = valid_emissions["light_color"]  # Shape: (N, 3)
                intensities = valid_emissions["light_intensity"]  # Shape: (N,)
                radii = valid_emissions["light_radius"]  # Shape: (N,)

                # Calculate pre-multiplied RGB values
                # colors is uint8, convert to float and multiply by intensity
                rgb = (colors / 255.0) * intensities[:, np.newaxis]

                # Write to emission_data using advanced indexing
                # Note: emission_data is (height, width, channels) = (y, x, c)
                emission_data[valid_vp_y, valid_vp_x, 0] = rgb[:, 0]
                emission_data[valid_vp_y, valid_vp_x, 1] = rgb[:, 1]
                emission_data[valid_vp_y, valid_vp_x, 2] = rgb[:, 2]
                emission_data[valid_vp_y, valid_vp_x, 3] = radii.astype(np.float32)

        # Upload emission data to texture
        assert self._emission_texture is not None
        self.queue.write_texture(
            {
                "texture": self._emission_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(emission_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": viewport_bounds.width
                * 4
                * 4,  # 4 components * 4 bytes
                "rows_per_image": viewport_bounds.height,
            },
            (viewport_bounds.width, viewport_bounds.height, 1),
        )

    def _update_shadow_grid_texture(self) -> None:
        """Update the shadow grid texture from the game map's shadow_heights array.

        Creates a texture where each pixel stores a tile's shadow height.
        Used by the shader for height-aware ray marching to determine terrain
        shadows. Height values are stored directly as uint8 (r8unorm format,
        so the shader reads value/255 and multiplies by 255 to recover the int).
        Only recreates when map structure changes.
        """
        game_map = self.game_world.game_map
        if game_map is None:
            return

        # Check if we need to update based on map structural revision
        if (
            self._cached_shadow_grid_revision == game_map.structural_revision
            and self._shadow_grid_texture is not None
        ):
            return  # No update needed

        # Shadow heights are already uint8. Transpose to match texture coords.
        shadow_data = np.ascontiguousarray(game_map.shadow_heights.T.astype(np.uint8))

        # Release old texture if it exists
        if self._shadow_grid_texture is not None:
            self._shadow_grid_texture = None
            self._bind_group = None  # Invalidate bind group when texture changes

        # Create new texture with shadow grid data - use r8unorm for filterability
        self._shadow_grid_texture = self.device.create_texture(
            size=(game_map.width, game_map.height, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        # Write shadow grid data to texture
        self.queue.write_texture(
            {
                "texture": self._shadow_grid_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(shadow_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": game_map.width,  # 1 byte per uint8
                "rows_per_image": game_map.height,
            },
            (game_map.width, game_map.height, 1),
        )

        # Update cached revision
        self._cached_shadow_grid_revision = game_map.structural_revision

    def _pack_uniform_data(
        self,
        light_data: list[float],
        light_count: int,
        actor_shadow_positions: list[float],
        actor_shadow_count: int,
        viewport_bounds: Rect,
    ) -> bytes:
        """Pack uniform data into a byte buffer matching the WGSL struct layout."""
        from catley.config import (
            SHADOW_FALLOFF,
            SHADOW_INTENSITY,
            SHADOW_MAX_LENGTH,
            SKY_EXPOSURE_POWER,
            SUN_SHADOW_INTENSITY,
        )
        from catley.game.lights import DirectionalLight

        buffer = bytearray()

        # viewport_data: vec4f
        buffer.extend(
            struct.pack(
                "4f",
                float(viewport_bounds.x1),
                float(viewport_bounds.y1),
                float(viewport_bounds.width),
                float(viewport_bounds.height),
            )
        )

        # light_count: i32, ambient_light: f32, time: f32, tile_aligned: u32
        buffer.extend(
            struct.pack("ifff", light_count, AMBIENT_LIGHT_LEVEL, self._time, 1.0)
        )  # tile_aligned = true

        # --- Light Arrays ---
        def pack_light_array(base_idx, data_idx, components):
            arr_buffer = bytearray()
            for i in range(self.MAX_LIGHTS):
                if i < light_count:
                    val = light_data[i * 12 + data_idx]
                    if components == 1:
                        arr_buffer.extend(struct.pack("f", val))
                    elif components == 2:
                        arr_buffer.extend(
                            struct.pack(
                                "2f",
                                light_data[i * 12 + data_idx],
                                light_data[i * 12 + data_idx + 1],
                            )
                        )
                    elif components == 3:
                        arr_buffer.extend(
                            struct.pack(
                                "3f",
                                light_data[i * 12 + data_idx],
                                light_data[i * 12 + data_idx + 1],
                                light_data[i * 12 + data_idx + 2],
                            )
                        )
                    arr_buffer.extend(
                        b"\x00" * (16 - components * 4)
                    )  # Padding to vec4f
                else:
                    arr_buffer.extend(struct.pack("4f", 0.0, 0.0, 0.0, 0.0))
            return arr_buffer

        buffer.extend(pack_light_array(0, 0, 2))  # light_positions
        buffer.extend(pack_light_array(0, 2, 1))  # light_radii
        buffer.extend(pack_light_array(0, 3, 1))  # light_intensities
        buffer.extend(pack_light_array(0, 4, 3))  # light_colors
        buffer.extend(pack_light_array(0, 7, 1))  # light_flicker_enabled
        buffer.extend(pack_light_array(0, 8, 1))  # light_flicker_speed
        buffer.extend(pack_light_array(0, 9, 1))  # light_min_brightness
        buffer.extend(pack_light_array(0, 10, 1))  # light_max_brightness

        # --- Actor Shadow Uniforms (terrain shadows use grid texture) ---
        buffer.extend(
            struct.pack(
                "ifif",
                actor_shadow_count,
                SHADOW_INTENSITY,
                SHADOW_MAX_LENGTH,
                1.0 if SHADOW_FALLOFF else 0.0,
            )
        )

        # Actor shadow positions with height (terrain shadows use texture)
        # Format per actor: x, y, shadow_height, padding
        MAX_ACTOR_SHADOWS = 64
        for i in range(MAX_ACTOR_SHADOWS):
            if i < actor_shadow_count:
                buffer.extend(
                    struct.pack(
                        "4f",
                        actor_shadow_positions[i * 3],
                        actor_shadow_positions[i * 3 + 1],
                        actor_shadow_positions[i * 3 + 2],
                        0.0,
                    )
                )  # xy, height, padding
            else:
                buffer.extend(struct.pack("4f", 0.0, 0.0, 0.0, 0.0))

        # --- Directional Light Uniforms ---
        directional_light = next(
            (
                light
                for light in self.game_world.lights
                if isinstance(light, DirectionalLight)
            ),
            None,
        )
        if directional_light:
            sun_dir_x, sun_dir_y = (
                directional_light.direction.x,
                directional_light.direction.y,
            )
            sun_r, sun_g, sun_b = [c / 255.0 for c in directional_light.color]
            sun_intensity = directional_light.intensity

            # Shadow length scale: 1/tan(elevation). Low sun = longer shadows,
            # high sun = shorter shadows. Clamped to max 8.0 near horizon.
            elev_rad = math.radians(max(directional_light.elevation_degrees, 0.1))
            shadow_length_scale = min(1.0 / math.tan(elev_rad), 8.0)
        else:
            sun_dir_x, sun_dir_y = 0.0, 0.0
            sun_r, sun_g, sun_b = 0.0, 0.0, 0.0
            sun_intensity = 0.0
            shadow_length_scale = 1.0

        buffer.extend(
            struct.pack("2f2f", sun_dir_x, sun_dir_y, 0.0, 0.0)
        )  # sun_direction + padding
        buffer.extend(
            struct.pack("3ff", sun_r, sun_g, sun_b, sun_intensity)
        )  # sun_color + sun_intensity
        buffer.extend(
            struct.pack(
                "ffff",
                SKY_EXPOSURE_POWER,
                SUN_SHADOW_INTENSITY,
                shadow_length_scale,
                0.0,
            )
        )  # sky_exposure_power + sun_shadow_intensity + sun_shadow_length_scale + padding

        # Map size for sky exposure UV calculation
        game_map = self.game_world.game_map
        map_width = float(game_map.width) if game_map else 1.0
        map_height = float(game_map.height) if game_map else 1.0
        buffer.extend(
            struct.pack("2f2f", map_width, map_height, 0.0, 0.0)
        )  # map_size + _padding3

        return bytes(buffer)

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

        # Pack uniform data into bytes
        uniform_bytes = self._pack_uniform_data(
            light_data,
            light_count,
            shadow_casters,
            shadow_caster_count,
            viewport_bounds,
        )

        # Write to GPU buffer
        self.queue.write_buffer(self._uniform_buffer, 0, memoryview(uniform_bytes))

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects."""
        self._time += fixed_timestep

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
            return None

        # Perform GPU computation
        gpu_result = self._compute_lightmap_gpu(viewport_bounds)
        if gpu_result is not None:
            self.revision += 1
            return gpu_result

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
                self._output_texture = self.device.create_texture(
                    size=(viewport_bounds.width, viewport_bounds.height, 1),
                    format=wgpu.TextureFormat.rgba32float,
                    # to handle shader output
                    usage=wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.COPY_SRC,
                )

                # Create buffer for reading back results
                # WGPU requires bytes_per_row to be aligned to 256 bytes
                bytes_per_row = viewport_bounds.width * 4 * 4  # 4 components * 4 bytes
                padded_bytes_per_row = _align_to_copy_row_alignment(bytes_per_row)
                self._output_buffer = self.device.create_buffer(
                    size=padded_bytes_per_row * viewport_bounds.height,
                    usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
                )
                # Store padded row size for readback
                self._padded_bytes_per_row = padded_bytes_per_row
                self._unpadded_bytes_per_row = bytes_per_row

                self._current_viewport = viewport_bounds

            except Exception as e:
                print(f"Failed to ensure WGPU resources for viewport: {e}")
                import traceback

                traceback.print_exc()
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

            # Collect actor shadow casters (terrain shadows now use grid texture)
            # Cache is invalidated by on_actor_moved() when actors move
            if self._cached_shadow_casters is None:
                self._cached_shadow_casters = self._collect_actor_shadow_casters(
                    viewport_bounds
                )

            actor_shadow_positions = self._cached_shadow_casters
            actor_shadow_count = len(actor_shadow_positions) // 3  # 3 floats per actor

            # Update sky exposure texture if needed
            self._update_sky_exposure_texture()

            # Update explored/visible textures for fog-of-war masking
            self._update_explored_texture()
            self._update_visible_texture()

            # Update shadow grid texture for terrain shadows
            self._update_shadow_grid_texture()

            # Update emission texture for light-emitting tiles
            self._update_emission_texture(viewport_bounds)

            # Only update uniforms when lights, casters, or viewport changes
            light_data_hash = hash(
                (
                    tuple(light_data[: light_count * 12]),
                    tuple(actor_shadow_positions),
                    viewport_bounds.x1,
                    viewport_bounds.y1,
                )
            )
            if self._last_light_data_hash != light_data_hash or self._first_frame:
                self._update_uniform_buffer(
                    light_data,
                    light_count,
                    actor_shadow_positions,
                    actor_shadow_count,
                    viewport_bounds,
                )
                self._last_light_data_hash = light_data_hash
                if self._first_frame:
                    self._first_frame = False  # Unset the flag after the first update
            else:
                # Time changes every frame, so we still need to update the buffer
                self._update_uniform_buffer(
                    light_data,
                    light_count,
                    actor_shadow_positions,
                    actor_shadow_count,
                    viewport_bounds,
                )

            # Create bind group with current resources (only if needed)
            if self._bind_group is None:
                self._create_bind_group()
            assert self._bind_group is not None  # Set by _create_bind_group()

            # Create command encoder and render pass
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

            # Copy render target to readback buffer (using aligned bytes_per_row)
            command_encoder.copy_texture_to_buffer(
                {
                    "texture": self._output_texture,
                    "mip_level": 0,
                    "origin": (0, 0, 0),
                },
                {
                    "buffer": self._output_buffer,
                    "offset": 0,
                    "bytes_per_row": self._padded_bytes_per_row,
                    "rows_per_image": viewport_bounds.height,
                },
                (viewport_bounds.width, viewport_bounds.height, 1),
            )

            # Submit commands
            self.queue.submit([command_encoder.finish()])

            # Map buffer for reading with proper error handling.
            # PERF: This GPU→CPU transfer (map_sync) is the main lighting bottleneck
            # (~20% of frame time when profiled). The lightmap must come back to CPU
            # because visibility masking, animation effects, and tile appearance
            # blending currently happen in Python. A future optimization would move
            # those operations to GPU shaders, keeping the lightmap on-GPU and only
            # reading back the final composited frame for display.
            assert self._output_buffer is not None
            try:
                # Map the buffer (this waits for GPU operations to complete)
                self._output_buffer.map_sync(wgpu.MapMode.READ)
                # Read the mapped data
                mapped_data = self._output_buffer.read_mapped()
                if mapped_data is None:
                    print("Failed to read mapped buffer data")
                    return None
                # wgpu returns memoryview but types it as ArrayLike
                result_data = np.frombuffer(mapped_data, dtype=np.float32)
            finally:
                # Always unmap the buffer, even if mapping failed
                try:
                    self._output_buffer.unmap()
                except Exception as e:
                    print(f"Error unmapping buffer: {e}")

            # Handle row padding: buffer has padded rows for WGPU alignment.
            # Each row has padded_bytes_per_row bytes; only width*4*4 are valid.
            padded_floats_per_row = self._padded_bytes_per_row // 4
            valid_floats_per_row = viewport_bounds.width * 4  # 4 components

            # Reshape to (height, padded_floats_per_row) and extract valid columns
            padded_image = result_data.reshape(
                (viewport_bounds.height, padded_floats_per_row)
            )
            # Slice to get only valid data, then reshape to (height, width, 4)
            result_image = padded_image[:, :valid_floats_per_row].reshape(
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

        except Exception as e:
            print(f"Error in WGPU lightmap computation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _create_bind_group(self) -> None:
        """Create bind group with uniform buffer and all textures."""
        assert self._uniform_buffer is not None
        assert self._sky_exposure_sampler is not None

        # Create default textures if none exist
        if self._sky_exposure_texture is None:
            self._create_default_sky_exposure_texture()
        if self._emission_texture is None:
            self._create_default_emission_texture()
        if self._shadow_grid_texture is None:
            self._create_default_shadow_grid_texture()

        assert self._sky_exposure_texture is not None
        assert self._emission_texture is not None
        assert self._shadow_grid_texture is not None

        assert self._render_pipeline is not None
        self._bind_group = self.device.create_bind_group(
            layout=self._render_pipeline.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 0,
                    "resource": {"buffer": self._uniform_buffer},
                },
                {
                    "binding": 22,
                    "resource": self._sky_exposure_texture.create_view(),
                },
                {
                    "binding": 23,
                    "resource": self._sky_exposure_sampler,
                },
                {
                    "binding": 24,
                    "resource": self._emission_texture.create_view(),
                },
                {
                    "binding": 25,
                    "resource": self._shadow_grid_texture.create_view(),
                },
            ],
        )

    def _create_default_sky_exposure_texture(self) -> None:
        """Create a default 1x1 sky exposure texture for when no map is available."""

        # Create 1x1 texture with 0.0 sky exposure
        self._sky_exposure_texture = self.device.create_texture(
            size=(1, 1, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._bind_group = None  # Invalidate bind group when texture is created

        # Write zero sky exposure (RGBA8 format)
        sky_data = np.array([0, 0, 0, 255], dtype=np.uint8)  # [R, G, B, A]
        self.queue.write_texture(
            {
                "texture": self._sky_exposure_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(sky_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": 4,  # 1 pixel * 4 bytes (RGBA8)
                "rows_per_image": 1,
            },
            (1, 1, 1),
        )

    def _create_default_emission_texture(self) -> None:
        """Create a default 1x1 emission texture for when no map is available."""

        # Create 1x1 texture with zero emission (RGBA32float format)
        self._emission_texture = self.device.create_texture(
            size=(1, 1, 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._bind_group = None  # Invalidate bind group when texture is created

        # Write zero emission (RGBA32float format - 16 bytes per pixel)
        emission_data = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.queue.write_texture(
            {
                "texture": self._emission_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(emission_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": 16,  # 1 pixel * 4 components * 4 bytes
                "rows_per_image": 1,
            },
            (1, 1, 1),
        )

    def _create_default_shadow_grid_texture(self) -> None:
        """Create a default 1x1 shadow grid texture for when no map is available."""

        # Create 1x1 texture with no shadow blocking (r8unorm format)
        self._shadow_grid_texture = self.device.create_texture(
            size=(1, 1, 1),
            format=wgpu.TextureFormat.r8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        self._bind_group = None  # Invalidate bind group when texture is created

        # Write zero shadow blocking (R8 format - 1 byte per pixel)
        shadow_data = np.array([0], dtype=np.uint8)  # No shadow
        self.queue.write_texture(
            {
                "texture": self._shadow_grid_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            memoryview(shadow_data.tobytes()),
            {
                "offset": 0,
                "bytes_per_row": 1,  # 1 pixel * 1 byte (R8)
                "rows_per_image": 1,
            },
            (1, 1, 1),
        )

    # LightingSystem event handlers
    def on_light_added(self, light: LightSource) -> None:
        """Notification that a light has been added."""
        self.revision += 1

    def on_light_removed(self, light: LightSource) -> None:
        """Notification that a light has been removed."""
        self.revision += 1

    def on_light_moved(self, light: LightSource) -> None:
        """Notification that a light has moved."""
        self.revision += 1

    def on_global_light_changed(self) -> None:
        """Notification that global lighting has changed."""
        self.revision += 1

    def on_actor_moved(self, actor: Actor) -> None:
        """Invalidate shadow caster cache when any actor moves.

        Actors cast shadows, so when they move we need to recollect shadow
        caster positions. We invalidate the cache by clearing it, which forces
        _compute_lightmap_gpu to rebuild it on the next frame.
        """
        self._cached_shadow_casters = None

    def get_sky_exposure_texture(self) -> wgpu.GPUTexture | None:
        """Return the sky exposure texture for atmospheric effects."""
        return self._sky_exposure_texture

    def get_explored_texture(self) -> wgpu.GPUTexture | None:
        """Return the explored mask texture for fog-of-war effects."""
        return self._explored_texture

    def get_visible_texture(self) -> wgpu.GPUTexture | None:
        """Return the visible mask texture for fog-of-war effects."""
        return self._visible_texture

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
        self._explored_texture = None
        self._visible_texture = None
        self._emission_texture = None
        self._shadow_grid_texture = None

        # Clear cached data
        self._cached_light_data = None
        self._cached_shadow_casters = None
        self._last_light_data_hash = None
