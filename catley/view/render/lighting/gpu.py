"""GPU-based implementation of the lighting system.

This class implements the lighting system using ModernGL compute shaders for
high-performance parallel light computation. It maintains compatibility with
the LightingSystem interface while providing significant performance gains
for scenes with many lights.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import moderngl
import numpy as np

from catley.config import AMBIENT_LIGHT_LEVEL
from catley.types import FixedTimestep
from catley.util.coordinates import Rect

from .base import LightingSystem

if TYPE_CHECKING:
    from catley.backends.moderngl.graphics import ModernGLGraphicsContext
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource

logger = logging.getLogger(__name__)


class GPULightingSystem(LightingSystem):
    """GPU-accelerated implementation of the lighting system using compute shaders.

    This implementation uses ModernGL compute shaders to perform lighting calculations
    in parallel on the GPU, providing significant performance improvements over CPU
    implementations, especially for scenes with many lights.

    Features:
    - Parallel point light computation
    - Hardware accelerated distance calculations
    - Compatible with existing LightingSystem interface
    - Automatic fallback to CPU system if compute shaders unavailable
    """

    # Maximum number of lights we can handle in a single compute dispatch
    MAX_LIGHTS = 256

    def __init__(
        self,
        game_world: GameWorld,
        graphics_context: ModernGLGraphicsContext,
        fallback_system: LightingSystem | None = None,
    ) -> None:
        """Initialize the GPU lighting system.

        Args:
            game_world: The game world to query for lighting data
            graphics_context: ModernGL graphics context for GPU operations
            fallback_system: Optional CPU fallback if GPU compute unavailable
        """
        super().__init__(game_world)

        self.graphics_context = graphics_context
        self.mgl_context = graphics_context.mgl_context
        self.fallback_system = fallback_system

        # GPU resources
        self._compute_program: moderngl.ComputeShader | None = None
        self._light_buffer: moderngl.Buffer | None = None
        self._output_texture: moderngl.Texture | None = None
        self._output_buffer: moderngl.Buffer | None = None

        # Track time for dynamic effects
        self._time = 0.0

        # Current viewport for resource sizing
        self._current_viewport: Rect | None = None

        # Initialize GPU resources
        if not self._initialize_gpu_resources():
            logger.warning("GPU compute shaders not available, will use fallback")

    def _initialize_gpu_resources(self) -> bool:
        """Initialize GPU compute shader and buffers.

        Returns:
            True if initialization successful, False if fallback needed
        """
        try:
            # Check if compute shaders are supported
            if not hasattr(self.mgl_context, "compute_shader"):
                logger.info("Compute shaders not supported by OpenGL context")
                return False

            # Create basic point light compute shader
            compute_source = self._create_point_light_compute_shader()
            self._compute_program = self.mgl_context.compute_shader(compute_source)

            # Create light data buffer (will be resized as needed)
            self._light_buffer = self.mgl_context.buffer(
                reserve=self.MAX_LIGHTS * 32
            )  # 32 bytes per light

            logger.info("GPU lighting system initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize GPU lighting resources: {e}")
            return False

    def _create_point_light_compute_shader(self) -> str:
        """Create the GLSL compute shader source for point light calculations."""
        return """#version 430

// Work group size - process 8x8 tiles at a time
layout(local_size_x = 8, local_size_y = 8) in;

// Output lightmap texture
layout(rgba32f, binding = 0) uniform writeonly image2D lightmap;

// Light data buffer
layout(std430, binding = 1) readonly buffer LightBuffer {
    // Each light: position.xy, radius, intensity, color.rgb, padding
    float lights[];
};

// Uniforms
uniform int u_light_count;
uniform float u_ambient_light;
uniform ivec2 u_viewport_offset;  // Offset from world to viewport coordinates

void main() {
    ivec2 pixel_coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 image_size = imageSize(lightmap);

    // Check bounds
    if (pixel_coord.x >= image_size.x || pixel_coord.y >= image_size.y) {
        return;
    }

    // Convert pixel coordinate to world coordinate
    vec2 world_pos = vec2(pixel_coord + u_viewport_offset);

    // Start with ambient lighting
    vec3 final_color = vec3(u_ambient_light);

    // Add contribution from each point light
    for (int i = 0; i < u_light_count; i++) {
        int base_idx = i * 8;  // 8 floats per light

        vec2 light_pos = vec2(lights[base_idx], lights[base_idx + 1]);
        float light_radius = lights[base_idx + 2];
        float light_intensity = lights[base_idx + 3];
        vec3 light_color = vec3(
            lights[base_idx + 4],
            lights[base_idx + 5],
            lights[base_idx + 6]
        );

        // Calculate distance from light
        vec2 light_vec = world_pos - light_pos;
        float distance = length(light_vec);

        // Skip if outside light radius
        if (distance > light_radius) {
            continue;
        }

        // Calculate attenuation (linear falloff for now)
        float attenuation = max(0.0, 1.0 - (distance / light_radius));
        attenuation *= light_intensity;

        // Add light contribution
        final_color += light_color * attenuation;
    }

    // Clamp to valid range and write to output
    final_color = clamp(final_color, 0.0, 1.0);
    imageStore(lightmap, pixel_coord, vec4(final_color, 1.0));
}"""

    def _ensure_resources_for_viewport(self, viewport_bounds: Rect) -> bool:
        """Ensure GPU resources are sized appropriately for the viewport.

        Args:
            viewport_bounds: The viewport area that will be rendered

        Returns:
            True if resources are ready, False if fallback needed
        """
        if self._compute_program is None:
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

                # Create new output texture
                self._output_texture = self.mgl_context.texture(
                    (viewport_bounds.width, viewport_bounds.height),
                    components=4,  # RGBA
                )

                # Create buffer for reading back results
                self._output_buffer = self.mgl_context.buffer(
                    reserve=viewport_bounds.width
                    * viewport_bounds.height
                    * 4
                    * 4  # 4 components * 4 bytes each
                )

                self._current_viewport = viewport_bounds
                logger.debug(
                    f"Resized GPU lighting resources to "
                    f"{viewport_bounds.width}x{viewport_bounds.height}"
                )

            except Exception as e:
                logger.error(f"Failed to resize GPU resources: {e}")
                return False

        return True

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects."""
        self._time += fixed_timestep

        # Update fallback system if available
        if self.fallback_system is not None:
            self.fallback_system.update(fixed_timestep)

    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap using GPU compute shaders.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if computation failed
        """
        # Try GPU computation first
        gpu_result = self._compute_lightmap_gpu(viewport_bounds)
        if gpu_result is not None:
            self.revision += 1
            return gpu_result

        # Fall back to CPU system if available
        if self.fallback_system is not None:
            logger.debug("Falling back to CPU lighting system")
            return self.fallback_system.compute_lightmap(viewport_bounds)

        # No fallback available
        logger.error("GPU lighting failed and no fallback system available")
        return None

    def _compute_lightmap_gpu(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Perform the actual GPU lighting computation."""
        try:
            # Ensure resources are ready
            if not self._ensure_resources_for_viewport(viewport_bounds):
                return None

            assert self._compute_program is not None
            assert self._light_buffer is not None
            assert self._output_texture is not None
            assert self._output_buffer is not None

            # Collect light data from game world
            light_data = self._collect_light_data(viewport_bounds)
            light_count = len(light_data) // 8  # 8 floats per light

            if light_count > self.MAX_LIGHTS:
                logger.warning(
                    f"Too many lights ({light_count}), limiting to {self.MAX_LIGHTS}"
                )
                light_count = self.MAX_LIGHTS
                light_data = light_data[: self.MAX_LIGHTS * 8]

            # Upload light data to GPU
            self._light_buffer.write(np.array(light_data, dtype=np.float32).tobytes())

            # Bind resources
            self._output_texture.bind_to_image(0, read=False, write=True)
            self._light_buffer.bind_to_storage_buffer(1)

            # Set uniforms
            self._compute_program["u_light_count"].value = light_count
            self._compute_program["u_ambient_light"].value = AMBIENT_LIGHT_LEVEL
            self._compute_program["u_viewport_offset"].value = (
                viewport_bounds.x1,
                viewport_bounds.y1,
            )

            # Dispatch compute shader
            group_x = (viewport_bounds.width + 7) // 8  # Round up to nearest 8
            group_y = (viewport_bounds.height + 7) // 8
            self._compute_program.run(group_x, group_y)

            # Read back results
            self._output_texture.read_into(self._output_buffer)
            result_data = np.frombuffer(self._output_buffer.read(), dtype=np.float32)

            # Reshape to (width, height, 4) and extract RGB
            result_image = result_data.reshape(
                (viewport_bounds.height, viewport_bounds.width, 4)
            )
            rgb_result = result_image[:, :, :3]  # Extract RGB channels

            # Transpose to match expected (width, height, 3) format
            return np.transpose(rgb_result, (1, 0, 2))

        except Exception as e:
            logger.error(f"GPU lighting computation failed: {e}")
            return None

    def _collect_light_data(self, viewport_bounds: Rect) -> list[float]:
        """Collect light data from the game world and format for GPU buffer.

        Args:
            viewport_bounds: The viewport to collect lights for

        Returns:
            Flat list of floats representing light data (8 floats per light)
        """
        light_data = []

        for light in self.game_world.lights:
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
                # Get light color as RGB floats
                r, g, b = light.color.as_rgb_floats()

                # For dynamic lights, we might apply flicker here
                intensity = 1.0  # Base intensity

                # Pack light data: position.xy, radius, intensity, color.rgb, padding
                light_data.extend(
                    [
                        float(lx),
                        float(ly),  # position
                        float(light.radius),  # radius
                        intensity,  # intensity
                        r,
                        g,
                        b,  # color
                        0.0,  # padding
                    ]
                )

        return light_data

    def on_light_added(self, light: LightSource) -> None:
        """Notification that a light has been added."""
        # GPU system doesn't need caching invalidation like CPU system
        # but we update revision to trigger view refresh
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
        if self._compute_program is not None:
            self._compute_program.release()
        if self._light_buffer is not None:
            self._light_buffer.release()
        if self._output_texture is not None:
            self._output_texture.release()
        if self._output_buffer is not None:
            self._output_buffer.release()
