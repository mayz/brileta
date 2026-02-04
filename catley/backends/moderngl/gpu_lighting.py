"""GPU-based implementation of the lighting system using fragment shaders.

This class implements the lighting system using ModernGL fragment shaders for
high-performance parallel light computation. It maintains compatibility with
the LightingSystem interface while providing significant performance gains
for scenes with many lights.

Uses fragment shaders instead of compute shaders for OpenGL 4.1+ compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from catley.config import AMBIENT_LIGHT_LEVEL
from catley.types import FixedTimestep
from catley.util.coordinates import Rect
from catley.view.render.lighting.base import LightingSystem

if TYPE_CHECKING:
    from catley.backends.moderngl.resource_manager import ModernGLResourceManager
    from catley.backends.moderngl.shader_manager import ShaderManager
    from catley.game.actors import Actor
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource
    from catley.view.render.graphics import GraphicsContext


from . import get_uniform


class GPULightingSystem(LightingSystem):
    """GPU-accelerated implementation of the lighting system using fragment shaders.

    This implementation uses ModernGL fragment shaders to perform lighting calculations
    in parallel on the GPU, providing significant performance improvements over CPU
    implementations, especially for scenes with many lights.

    Features:
    - Parallel point light computation using fragment shaders
    - Hardware accelerated distance calculations
    - Compatible with existing LightingSystem interface
    - Works on OpenGL 3.3+ (including macOS OpenGL 4.1)
    """

    # Maximum number of lights we can handle in a single render pass
    MAX_LIGHTS = 32

    def __init__(
        self,
        game_world: GameWorld,
        graphics_context: GraphicsContext,
    ) -> None:
        """Initialize the GPU lighting system.

        Args:
            game_world: The game world to query for lighting data
            graphics_context: Graphics context (ModernGL required for GPU operations)
        """
        super().__init__(game_world)

        self.graphics_context = graphics_context
        # Check if graphics context has mgl_context (for real ModernGL usage)
        self.mgl_context: moderngl.Context
        if hasattr(graphics_context, "mgl_context"):
            self.mgl_context = graphics_context.mgl_context  # type: ignore[assignment]
        else:
            # Create a standalone context for lighting calculations
            self.mgl_context = moderngl.create_context(standalone=True)

        # GPU resources
        self._fragment_program: moderngl.Program | None = None
        self._shader_manager: ShaderManager | None = None
        self._resource_manager: ModernGLResourceManager | None = None
        self._output_texture: moderngl.Texture | None = None
        self._output_buffer: moderngl.Buffer | None = None
        self._fullscreen_vao: moderngl.VertexArray | None = None

        # Track time for dynamic effects
        self._time = 0.0

        # Current viewport for resource sizing
        self._current_viewport: Rect | None = None

        # Performance optimization: Track light configuration changes
        self._last_light_data_hash: int | None = None
        self._cached_light_data: list[float] | None = None
        self._cached_light_revision: int = -1

        # Cache for actor shadow casters (invalidated by on_actor_moved)
        self._cached_shadow_casters: list[float] | None = None

        # Sky exposure texture for directional lighting
        self._sky_exposure_texture: moderngl.Texture | None = None
        self._cached_map_revision: int = -1
        self._explored_texture: moderngl.Texture | None = None
        self._cached_exploration_revision: int = -1
        self._visible_texture: moderngl.Texture | None = None
        self._cached_visibility_revision: tuple[int, int] | None = None

        # Emission texture for light-emitting tiles (acid pools, hot coals, etc.)
        self._emission_texture: moderngl.Texture | None = None

        # Shadow grid texture for terrain shadow casting
        self._shadow_grid_texture: moderngl.Texture | None = None
        self._cached_shadow_grid_revision: int = -1

        # Initialize GPU resources
        if not self._initialize_gpu_resources():
            raise RuntimeError("Failed to initialize ModernGL GPU lighting system")

    def _uniform(self, name: str) -> moderngl.Uniform:
        """Get a uniform from the fragment program with proper type narrowing."""
        assert self._fragment_program is not None
        return get_uniform(self._fragment_program, name)

    def get_sky_exposure_texture(self) -> moderngl.Texture | None:
        """Return the sky exposure texture for mask-based effects."""
        return self._sky_exposure_texture

    def get_explored_texture(self) -> moderngl.Texture | None:
        """Return the explored mask texture."""
        return self._explored_texture

    def get_visible_texture(self) -> moderngl.Texture | None:
        """Return the visible mask texture."""
        return self._visible_texture

    def _initialize_gpu_resources(self) -> bool:
        """Initialize GPU fragment shader-based lighting.

        Returns:
            True if initialization successful, False if fallback needed
        """
        try:
            # Check if we have a valid mgl_context
            if self.mgl_context is None:
                return False

            # Log OpenGL version and capabilities
            self._log_opengl_capabilities()

            # Initialize fragment-based lighting
            return self._initialize_fragment_lighting()

        except Exception as e:
            print(f"Failed to initialize GPU resources: {e}")
            import traceback

            traceback.print_exc()
            return False

    def _log_opengl_capabilities(self) -> None:
        """Log OpenGL version and capabilities for debugging."""
        try:
            assert self.mgl_context is not None
            info = self.mgl_context.info

            # Check version for fragment shader compatibility
            version_info = info.get("GL_VERSION", "")
            if version_info:
                try:
                    # Extract major.minor from version string (e.g., "4.1 Metal - 86")
                    parts = version_info.split(".")
                    if len(parts) >= 2:
                        major_version = int(parts[0].split()[-1])
                        minor_version = int(parts[1].split()[0])
                        # Fragment shaders work on OpenGL 3.3+
                        if major_version < 3 or (
                            major_version == 3 and minor_version < 3
                        ):
                            print(
                                f"Warning: OpenGL {major_version}.{minor_version} "
                                f"may be too old for GPU lighting"
                            )
                except (ValueError, IndexError):
                    pass

        except Exception:
            pass

    def _initialize_fragment_lighting(self) -> bool:
        """Initialize fragment shader-based lighting."""
        try:
            # Import ShaderManager and ResourceManager here to avoid circular imports
            from catley.backends.moderngl.resource_manager import (
                ModernGLResourceManager,
            )
            from catley.backends.moderngl.shader_manager import ShaderManager

            assert self.mgl_context is not None
            self._shader_manager = ShaderManager(self.mgl_context)

            # Initialize resource manager - use shared one if available AND ModernGL
            resource_manager = getattr(self.graphics_context, "resource_manager", None)
            if isinstance(resource_manager, ModernGLResourceManager):
                self._resource_manager = resource_manager
            else:
                # Create our own ModernGL resource manager for standalone context
                self._resource_manager = ModernGLResourceManager(self.mgl_context)

            # Create fragment shader program
            self._fragment_program = self._shader_manager.create_program(
                "glsl/lighting/point_light.vert",
                "glsl/lighting/point_light.frag",
                "fragment_lighting",
            )

            # Create full-screen quad geometry
            self._create_fullscreen_quad()

            return True

        except Exception as e:
            print(f"Failed to initialize fragment-based lighting: {e}")
            import traceback

            traceback.print_exc()
            return False

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

        assert self.mgl_context is not None
        vbo = self.mgl_context.buffer(quad_vertices.tobytes())
        self._fullscreen_vao = self.mgl_context.vertex_array(
            self._fragment_program, [(vbo, "2f", "in_position")]
        )

    def _ensure_resources_for_viewport(self, viewport_bounds: Rect) -> bool:
        """Ensure GPU resources are sized appropriately for the viewport.

        Args:
            viewport_bounds: The viewport area that will be rendered

        Returns:
            True if resources are ready, False if fallback needed
        """
        if self._fragment_program is None:
            return False

        # Check if we need to resize the output texture
        if (
            self._current_viewport is None
            or self._current_viewport.width != viewport_bounds.width
            or self._current_viewport.height != viewport_bounds.height
        ):
            try:
                # Release old texture if it exists
                if self._output_texture is not None:
                    self._output_texture.release()
                if self._output_buffer is not None:
                    self._output_buffer.release()

                # Create new output texture (mgl_context guaranteed non-None here)
                assert self.mgl_context is not None
                self._output_texture = self.mgl_context.texture(
                    (viewport_bounds.width, viewport_bounds.height),
                    components=4,  # RGBA
                    dtype="f4",  # 32-bit float texture to handle shader float output
                )

                # Create buffer for reading back results
                self._output_buffer = self.mgl_context.buffer(
                    reserve=viewport_bounds.width
                    * viewport_bounds.height
                    * 4
                    * 4  # 4 components * 4 bytes each
                )

                self._current_viewport = viewport_bounds

            except Exception:
                return False

        return True

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects."""
        self._time += fixed_timestep

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap using GPU fragment shaders.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if computation failed
        """
        # Check if GPU is available
        if self._fragment_program is None:
            return None

        # Perform GPU computation
        gpu_result = self._compute_lightmap_gpu(viewport_bounds)
        if gpu_result is not None:
            self.revision += 1
            return gpu_result

        return None

    def _compute_lightmap_gpu(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Perform the actual GPU lighting computation using fragment shaders."""
        try:
            # Ensure resources are ready
            if not self._ensure_resources_for_viewport(viewport_bounds):
                return None

            assert self._fragment_program is not None
            assert self._output_texture is not None
            assert self._output_buffer is not None
            assert self._fullscreen_vao is not None

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

            actor_shadow_casters = self._cached_shadow_casters
            actor_shadow_count = len(actor_shadow_casters) // 2  # 2 floats per actor

            # Update sky exposure texture if needed
            self._update_sky_exposure_texture()
            self._update_explored_texture()
            self._update_visible_texture()

            # Update shadow grid texture for terrain shadows
            self._update_shadow_grid_texture()

            # Update emission texture for light-emitting tiles
            self._update_emission_texture(viewport_bounds)

            # Get or create cached framebuffer for rendering
            assert self._resource_manager is not None
            assert self._output_texture is not None

            # Use resource manager to get cached FBO for our texture
            fbo = self._resource_manager.get_or_create_fbo_for_texture(
                self._output_texture
            )

            # Save current FBO to restore after rendering
            previous_fbo = self.mgl_context.fbo

            # Set up rendering state
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)

            # Smart uniform updates - only update when lights or shadow casters change
            light_data_hash = hash(
                (
                    tuple(light_data[: light_count * 12]),
                    tuple(actor_shadow_casters),
                    self._time,
                )
            )
            if self._last_light_data_hash != light_data_hash:
                self._set_lighting_uniforms(light_data, light_count, viewport_bounds)
                self._set_actor_shadow_uniforms(
                    actor_shadow_casters, actor_shadow_count
                )
                self._set_directional_light_uniforms()
                self._last_light_data_hash = light_data_hash
            else:
                # Still need to update time uniform for dynamic effects
                assert self._fragment_program is not None
                self._uniform("u_time").value = self._time

            # Bind sky exposure texture to texture unit 1
            if self._sky_exposure_texture is not None:
                self._sky_exposure_texture.use(location=1)
                self._uniform("u_sky_exposure_map").value = 1

            # Bind emission texture to texture unit 2
            if self._emission_texture is not None:
                self._emission_texture.use(location=2)
                self._uniform("u_emission_map").value = 2

            # Bind shadow grid texture to texture unit 3
            if self._shadow_grid_texture is not None:
                self._shadow_grid_texture.use(location=3)
                self._uniform("u_shadow_grid").value = 3

            # Render full-screen quad
            self._fullscreen_vao.render()

            # Restore previous FBO
            if previous_fbo is not None:
                previous_fbo.use()

            # Read back results from GPU to CPU.
            # PERF: This GPU→CPU transfer is the main lighting bottleneck (~20% of
            # frame time when profiled). The lightmap must come back to CPU because
            # visibility masking, animation effects, and tile appearance blending
            # currently happen in Python. A future optimization would move those
            # operations to GPU shaders, keeping the lightmap on-GPU and only
            # reading back the final composited frame for display.
            self._output_texture.read_into(self._output_buffer)
            result_data = np.frombuffer(self._output_buffer.read(), dtype=np.float32)

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

        except Exception as e:
            print(f"Error in GPU lightmap computation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _set_lighting_uniforms(
        self, light_data: list[float], light_count: int, viewport_bounds: Rect
    ) -> None:
        """Set uniforms for the fragment shader."""
        # Prepare light data arrays (pad to MAX_LIGHTS)
        positions = [0.0] * (self.MAX_LIGHTS * 2)
        radii = [0.0] * self.MAX_LIGHTS
        intensities = [0.0] * self.MAX_LIGHTS
        colors = [(0.0, 0.0, 0.0)] * self.MAX_LIGHTS
        flicker_enabled = [0.0] * self.MAX_LIGHTS
        flicker_speed = [0.0] * self.MAX_LIGHTS
        min_brightness = [1.0] * self.MAX_LIGHTS
        max_brightness = [1.0] * self.MAX_LIGHTS

        # Fill in actual light data
        for i in range(light_count):
            base_idx = i * 12
            positions[i * 2] = light_data[base_idx]  # x
            positions[i * 2 + 1] = light_data[base_idx + 1]  # y
            radii[i] = light_data[base_idx + 2]
            intensities[i] = light_data[base_idx + 3]
            colors[i] = (
                light_data[base_idx + 4],  # r
                light_data[base_idx + 5],  # g
                light_data[base_idx + 6],  # b
            )
            flicker_enabled[i] = light_data[base_idx + 7]
            flicker_speed[i] = light_data[base_idx + 8]
            min_brightness[i] = light_data[base_idx + 9]
            max_brightness[i] = light_data[base_idx + 10]

        # Set uniforms
        assert self._fragment_program is not None
        self._uniform("u_light_count").value = light_count
        self._uniform("u_light_positions").value = positions
        self._uniform("u_light_radii").value = radii
        self._uniform("u_light_intensities").value = intensities
        self._uniform("u_light_colors").value = colors
        self._uniform("u_light_flicker_enabled").value = flicker_enabled
        self._uniform("u_light_flicker_speed").value = flicker_speed
        self._uniform("u_light_min_brightness").value = min_brightness
        self._uniform("u_light_max_brightness").value = max_brightness
        self._uniform("u_ambient_light").value = AMBIENT_LIGHT_LEVEL
        self._uniform("u_time").value = self._time
        self._uniform("u_tile_aligned").value = True

        # Set viewport uniforms for coordinate calculation
        self._uniform("u_viewport_offset").value = (
            viewport_bounds.x1,
            viewport_bounds.y1,
        )
        self._uniform("u_viewport_size").value = (
            viewport_bounds.width,
            viewport_bounds.height,
        )

    def _set_actor_shadow_uniforms(
        self,
        actor_positions: list[float],
        actor_count: int,
    ) -> None:
        """Set actor shadow uniforms for the fragment shader.

        Terrain shadows are handled by the shadow grid texture. This sets
        uniforms for dynamic actor shadows (NPCs that block light).
        """
        from catley.config import SHADOW_FALLOFF, SHADOW_INTENSITY, SHADOW_MAX_LENGTH

        # Maximum actors the shader can handle (64 actors * 2 floats = 128)
        MAX_ACTOR_SHADOWS = 64

        # Prepare actor position arrays (pad to MAX_ACTOR_SHADOWS)
        positions = [0.0] * (MAX_ACTOR_SHADOWS * 2)

        # Fill in actual actor data
        actual_count = min(actor_count, MAX_ACTOR_SHADOWS)
        for i in range(actual_count):
            positions[i * 2] = actor_positions[i * 2]  # x
            positions[i * 2 + 1] = actor_positions[i * 2 + 1]  # y

        # Set actor shadow uniforms
        assert self._fragment_program is not None
        self._uniform("u_actor_shadow_count").value = actual_count
        self._uniform("u_actor_shadow_positions").value = positions

        # Set general shadow config (used by both terrain and actor shadows)
        self._uniform("u_shadow_intensity").value = SHADOW_INTENSITY
        self._uniform("u_shadow_max_length").value = SHADOW_MAX_LENGTH
        self._uniform("u_shadow_falloff_enabled").value = SHADOW_FALLOFF

    def _set_directional_light_uniforms(self) -> None:
        """Set uniforms for directional lighting (sun/moon)."""
        from catley.config import SKY_EXPOSURE_POWER, SUN_SHADOW_INTENSITY
        from catley.game.lights import DirectionalLight

        # Find active directional light
        directional_light = None
        for light in self.game_world.lights:
            if isinstance(light, DirectionalLight):
                directional_light = light
                break

        assert self._fragment_program is not None

        # Set map size uniform for sky exposure UV calculation
        game_map = self.game_world.game_map
        if game_map:
            self._uniform("u_map_size").value = (game_map.width, game_map.height)

        if directional_light:
            # Set sun uniforms from the directional light
            self._uniform("u_sun_direction").value = (
                directional_light.direction.x,
                directional_light.direction.y,
            )
            self._uniform("u_sun_color").value = (
                directional_light.color[0] / 255.0,
                directional_light.color[1] / 255.0,
                directional_light.color[2] / 255.0,
            )
            self._uniform("u_sun_intensity").value = directional_light.intensity
        else:
            # No directional light - set "off" values
            self._uniform("u_sun_direction").value = (0.0, 0.0)
            self._uniform("u_sun_color").value = (0.0, 0.0, 0.0)
            self._uniform("u_sun_intensity").value = 0.0

        # Always set sky exposure power and sun shadow intensity
        self._uniform("u_sky_exposure_power").value = SKY_EXPOSURE_POWER
        self._uniform("u_sun_shadow_intensity").value = SUN_SHADOW_INTENSITY

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
            Flat list of floats representing actor positions (2 floats per actor)
            Format: position.xy for each shadow-casting actor, max 16 actors
        """
        from catley.config import SHADOW_MAX_LENGTH, SHADOWS_ENABLED

        if not SHADOWS_ENABLED:
            return []

        # Maximum actors the shader can handle
        MAX_ACTOR_SHADOWS = 64

        # Expand viewport bounds to include potential shadow influence
        expanded_bounds = Rect.from_bounds(
            x1=viewport_bounds.x1 - SHADOW_MAX_LENGTH,
            y1=viewport_bounds.y1 - SHADOW_MAX_LENGTH,
            x2=viewport_bounds.x2 + SHADOW_MAX_LENGTH,
            y2=viewport_bounds.y2 + SHADOW_MAX_LENGTH,
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
                if hasattr(actor, "blocks_movement") and actor.blocks_movement:
                    actor_positions.extend([float(actor.x), float(actor.y)])
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

        # Vectorized sky exposure calculation: iterate over regions (small dict)
        # instead of tiles (large grid). Build a lookup array mapping region_id
        # to sky_exposure, then use numpy advanced indexing.
        if game_map.regions:
            max_region_id = max(game_map.regions.keys())
            # Lookup table: index = region_id, value = sky_exposure
            exposure_lookup = np.zeros(max_region_id + 1, dtype=np.float32)
            for region_id, region in game_map.regions.items():
                exposure_lookup[region_id] = region.sky_exposure

            # Map tile_to_region_id to sky exposure values (clamp -1 to 0 for lookup)
            region_ids = game_map.tile_to_region_id
            clamped_ids = np.clip(region_ids, 0, max_region_id)
            sky_values = exposure_lookup[clamped_ids]

            # Tiles with no region (id < 0) get 0 exposure
            sky_values = np.where(region_ids >= 0, sky_values, 0.0)

            # Non-transparent tiles block all sunlight
            sky_values = np.where(game_map.transparent, sky_values, 0.0)

            # Transpose from (w,h) to (h,w) for texture format
            sky_exposure_data = sky_values.T.astype(np.float32)
        else:
            sky_exposure_data = np.zeros(
                (game_map.height, game_map.width), dtype=np.float32
            )

        # Release old texture if it exists
        if self._sky_exposure_texture is not None:
            self._sky_exposure_texture.release()

        # Create new texture with sky exposure data
        assert self.mgl_context is not None
        self._sky_exposure_texture = self.mgl_context.texture(
            (game_map.width, game_map.height),
            components=1,  # Single channel (R)
            dtype="f4",  # 32-bit float
        )
        # Use nearest-neighbor filtering to prevent interpolation bleeding
        # at tile boundaries (walls should have sharp sky exposure cutoff)
        self._sky_exposure_texture.filter = (
            self.mgl_context.NEAREST,
            self.mgl_context.NEAREST,
        )
        assert self._sky_exposure_texture is not None
        self._sky_exposure_texture.write(sky_exposure_data.tobytes())

        # Update cached revision
        self._cached_map_revision = game_map.structural_revision

    def _update_explored_texture(self) -> None:
        """Update the explored mask texture from the game map."""
        game_map = self.game_world.game_map
        if game_map is None:
            return

        if (
            self._cached_exploration_revision == game_map.exploration_revision
            and self._explored_texture is not None
        ):
            return

        explored_data = np.ascontiguousarray(game_map.explored.T, dtype=np.float32)

        if self._explored_texture is not None:
            self._explored_texture.release()

        assert self.mgl_context is not None
        self._explored_texture = self.mgl_context.texture(
            (game_map.width, game_map.height),
            components=1,
            dtype="f4",
        )
        assert self._explored_texture is not None
        self._explored_texture.write(explored_data.tobytes())

        self._cached_exploration_revision = game_map.exploration_revision

    def _update_visible_texture(self) -> None:
        """Update the visible mask texture from the game map."""
        game_map = self.game_world.game_map
        if game_map is None:
            return

        if self._visible_texture is not None:
            current_revision = (self._cached_exploration_revision, self.revision)
            if self._cached_visibility_revision == current_revision:
                return

        visible_data = np.ascontiguousarray(game_map.visible.T, dtype=np.float32)

        if self._visible_texture is not None:
            self._visible_texture.release()

        assert self.mgl_context is not None
        self._visible_texture = self.mgl_context.texture(
            (game_map.width, game_map.height),
            components=1,
            dtype="f4",
        )
        assert self._visible_texture is not None
        self._visible_texture.write(visible_data.tobytes())

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
                assert self.mgl_context is not None
                self._emission_texture = self.mgl_context.texture(
                    (viewport_bounds.width, viewport_bounds.height),
                    components=4,
                    dtype="f4",
                )
                # Write zeros once
                zeros = np.zeros(
                    (viewport_bounds.height, viewport_bounds.width, 4), dtype=np.float32
                )
                self._emission_texture.write(zeros.tobytes())
            return

        # Ensure emission texture matches viewport size
        if self._emission_texture is None or self._emission_texture.size != (
            viewport_bounds.width,
            viewport_bounds.height,
        ):
            # Release old texture if it exists
            if self._emission_texture is not None:
                self._emission_texture.release()

            # Create new emission texture
            assert self.mgl_context is not None
            self._emission_texture = self.mgl_context.texture(
                (viewport_bounds.width, viewport_bounds.height),
                components=4,  # RGBA
                dtype="f4",  # 32-bit float for precision
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
        self._emission_texture.write(emission_data.tobytes())

    def _update_shadow_grid_texture(self) -> None:
        """Update the shadow grid texture from the game map's casts_shadows array.

        Creates a texture where each pixel indicates whether a tile blocks light.
        Used by the shader for ray marching to determine terrain shadows.
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

        # Convert boolean array to uint8 (0 or 255)
        # Transpose to match texture coordinate system (width, height)
        shadow_data = (game_map.casts_shadows.T * 255).astype(np.uint8)

        # Release old texture if it exists
        if self._shadow_grid_texture is not None:
            self._shadow_grid_texture.release()

        # Create new texture with shadow grid data
        assert self.mgl_context is not None
        self._shadow_grid_texture = self.mgl_context.texture(
            (game_map.width, game_map.height),
            components=1,  # Single channel (R)
            dtype="f1",  # unsigned byte normalized to 0.0-1.0
        )
        # CRITICAL: Use nearest-neighbor filtering for pixel-exact shadow boundaries
        # Linear filtering would cause shadow edges to bleed between tiles
        self._shadow_grid_texture.filter = (
            self.mgl_context.NEAREST,
            self.mgl_context.NEAREST,
        )
        self._shadow_grid_texture.write(shadow_data.tobytes())

        # Update cached revision
        self._cached_shadow_grid_revision = game_map.structural_revision

    def on_light_added(self, light: LightSource) -> None:
        """Notification that a light has been added."""
        # GPU system doesn't need caching invalidation like CPU system
        # but we update revision to trigger view refresh
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

    def release(self) -> None:
        """Release GPU resources."""
        if self._fragment_program is not None:
            self._fragment_program.release()
        if self._output_texture is not None:
            self._output_texture.release()
        if self._output_buffer is not None:
            self._output_buffer.release()
        if self._fullscreen_vao is not None:
            self._fullscreen_vao.release()
        if self._sky_exposure_texture is not None:
            self._sky_exposure_texture.release()
        if self._explored_texture is not None:
            self._explored_texture.release()
        if self._visible_texture is not None:
            self._visible_texture.release()
        if self._emission_texture is not None:
            self._emission_texture.release()
        if self._shadow_grid_texture is not None:
            self._shadow_grid_texture.release()

        # Clear cached data
        self._cached_light_data = None
        self._cached_shadow_casters = None
        self._last_light_data_hash = None
