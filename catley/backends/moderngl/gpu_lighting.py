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
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource
    from catley.view.render.graphics import GraphicsContext


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
        if hasattr(graphics_context, "mgl_context"):
            self.mgl_context = graphics_context.mgl_context
        else:
            # Test/dummy graphics context, will fail initialization and use fallback
            self.mgl_context = None

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

        # Cache for shadow casters
        self._cached_shadow_casters: list[float] | None = None
        self._cached_shadow_revision: int = -1

        # Sky exposure texture for directional lighting
        self._sky_exposure_texture: moderngl.Texture | None = None
        self._cached_map_revision: int = -1

        # Initialize GPU resources
        if not self._initialize_gpu_resources():
            raise RuntimeError("Failed to initialize ModernGL GPU lighting system")

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

        except Exception:
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

            # Initialize resource manager - use shared one if available
            if hasattr(self.graphics_context, "resource_manager"):
                self._resource_manager = self.graphics_context.resource_manager
            else:
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

            # Collect shadow casters globally (more efficient than per-light collection)
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

            # Get or create cached framebuffer for rendering
            assert self._resource_manager is not None
            assert self._output_texture is not None

            # Use resource manager to get cached FBO for our texture
            fbo = self._resource_manager.get_or_create_fbo_for_texture(
                self._output_texture
            )

            # Set up rendering state
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)

            # Smart uniform updates - only update when lights or shadow casters change
            light_data_hash = hash(
                (
                    tuple(light_data[: light_count * 12]),
                    tuple(shadow_casters),
                    self._time,
                )
            )
            if self._last_light_data_hash != light_data_hash:
                self._set_lighting_uniforms(light_data, light_count, viewport_bounds)
                self._set_shadow_uniforms(
                    shadow_casters, shadow_caster_count, viewport_bounds
                )
                self._set_directional_light_uniforms()
                self._last_light_data_hash = light_data_hash
            else:
                # Still need to update time uniform for dynamic effects
                assert self._fragment_program is not None
                self._fragment_program["u_time"].value = self._time

            # Bind sky exposure texture to texture unit 1
            if self._sky_exposure_texture is not None:
                self._sky_exposure_texture.use(location=1)
                self._fragment_program["u_sky_exposure_map"].value = 1

            # Render full-screen quad
            self._fullscreen_vao.render()

            # Read back results
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

        except Exception:
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
        self._fragment_program["u_light_count"].value = light_count
        self._fragment_program["u_light_positions"].value = positions
        self._fragment_program["u_light_radii"].value = radii
        self._fragment_program["u_light_intensities"].value = intensities
        self._fragment_program["u_light_colors"].value = colors
        self._fragment_program["u_light_flicker_enabled"].value = flicker_enabled
        self._fragment_program["u_light_flicker_speed"].value = flicker_speed
        self._fragment_program["u_light_min_brightness"].value = min_brightness
        self._fragment_program["u_light_max_brightness"].value = max_brightness
        self._fragment_program["u_ambient_light"].value = AMBIENT_LIGHT_LEVEL
        self._fragment_program["u_time"].value = self._time
        self._fragment_program["u_tile_aligned"].value = True

        # Set viewport uniforms for coordinate calculation
        self._fragment_program["u_viewport_offset"].value = (
            viewport_bounds.x1,
            viewport_bounds.y1,
        )
        self._fragment_program["u_viewport_size"].value = (
            viewport_bounds.width,
            viewport_bounds.height,
        )

    def _set_shadow_uniforms(
        self,
        shadow_casters: list[float],
        shadow_caster_count: int,
        viewport_bounds: Rect,
    ) -> None:
        """Set shadow caster uniforms for the fragment shader."""
        from catley.config import SHADOW_FALLOFF, SHADOW_INTENSITY, SHADOW_MAX_LENGTH

        # Maximum number of shadow casters we can handle
        MAX_SHADOW_CASTERS = 64

        # Prepare shadow caster position arrays (pad to MAX_SHADOW_CASTERS)
        shadow_positions = [0.0] * (MAX_SHADOW_CASTERS * 2)

        # Fill in actual shadow caster data
        actual_count = min(shadow_caster_count, MAX_SHADOW_CASTERS)
        for i in range(actual_count):
            shadow_positions[i * 2] = shadow_casters[i * 2]  # x
            shadow_positions[i * 2 + 1] = shadow_casters[i * 2 + 1]  # y

        # Set shadow uniforms
        assert self._fragment_program is not None
        self._fragment_program["u_shadow_caster_count"].value = actual_count
        self._fragment_program["u_shadow_caster_positions"].value = shadow_positions
        self._fragment_program["u_shadow_intensity"].value = SHADOW_INTENSITY
        self._fragment_program["u_shadow_max_length"].value = SHADOW_MAX_LENGTH
        self._fragment_program["u_shadow_falloff_enabled"].value = SHADOW_FALLOFF

    def _set_directional_light_uniforms(self) -> None:
        """Set uniforms for directional lighting (sun/moon)."""
        from catley.config import SKY_EXPOSURE_POWER
        from catley.game.lights import DirectionalLight

        # Find active directional light
        directional_light = None
        for light in self.game_world.lights:
            if isinstance(light, DirectionalLight):
                directional_light = light
                break

        assert self._fragment_program is not None

        if directional_light:
            # Set sun uniforms from the directional light
            self._fragment_program["u_sun_direction"].value = (
                directional_light.direction.x,
                directional_light.direction.y,
            )
            self._fragment_program["u_sun_color"].value = (
                directional_light.color[0] / 255.0,
                directional_light.color[1] / 255.0,
                directional_light.color[2] / 255.0,
            )
            self._fragment_program[
                "u_sun_intensity"
            ].value = directional_light.intensity
        else:
            # No directional light - set "off" values
            self._fragment_program["u_sun_direction"].value = (0.0, 0.0)
            self._fragment_program["u_sun_color"].value = (0.0, 0.0, 0.0)
            self._fragment_program["u_sun_intensity"].value = 0.0

        # Always set sky exposure power
        self._fragment_program["u_sky_exposure_power"].value = SKY_EXPOSURE_POWER

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
            self._sky_exposure_texture.release()

        # Create new texture with sky exposure data
        assert self.mgl_context is not None
        self._sky_exposure_texture = self.mgl_context.texture(
            (game_map.width, game_map.height),
            components=1,  # Single channel (R)
            dtype="f4",  # 32-bit float
        )
        assert self._sky_exposure_texture is not None
        self._sky_exposure_texture.write(sky_exposure_data.tobytes())

        # Update cached revision
        self._cached_map_revision = game_map.structural_revision

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

        # Clear cached data
        self._cached_light_data = None
        self._cached_shadow_casters = None
        self._last_light_data_hash = None
