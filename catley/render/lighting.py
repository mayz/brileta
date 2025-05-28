from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import tcod

from catley import colors

if TYPE_CHECKING:
    from catley.game.entities import Entity

# Preset configurations for different light types
TORCH_PRESET = {
    "radius": 10,
    "color": (0.7, 0.5, 0.3),  # Warm orange/yellow
    "light_type": "dynamic",
    "flicker_enabled": True,
    "flicker_speed": 3.0,
    "movement_scale": 0.0,
    "min_brightness": 1.15,
    "max_brightness": 1.35,
}


@dataclass
class LightingConfig:
    """Configuration for the lighting system"""

    fov_radius: int = 15
    ambient_light: float = 0.1  # Base light level


@dataclass
class LightSource:
    """Represents a light source in the game world"""

    radius: int  # Required parameter must come first
    _pos: tuple[int, int] | None = None  # Internal position storage
    color: colors.Color = (1.0, 1.0, 1.0)  # RGB
    light_type: Literal["static", "dynamic"] = "static"
    flicker_enabled: bool = False
    flicker_speed: float = 3.0  # Moderate flicker speed
    movement_scale: float = 0.0  # No movement
    min_brightness: float = 1.15  # Higher base brightness
    max_brightness: float = 1.35  # Still ~17% variation but at a higher level

    @classmethod
    def create_torch(cls, position: tuple[int, int] | None = None) -> "LightSource":
        """Create a torch light source with preset values"""
        return cls(_pos=position, **TORCH_PRESET)

    def __post_init__(self) -> None:
        self._owner = None
        self._lighting_system = None

    @property
    def position(self) -> tuple[int, int]:
        """Returns position from owner if attached, otherwise stored position"""
        if self._owner:
            return (self._owner.x, self._owner.y)
        if self._pos is None:
            raise ValueError("Position not set and no owner attached")
        return self._pos

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._pos = value

    def attach(self, owner: "Entity", lighting_system: "LightingSystem") -> None:
        """Attach this light source to an entity and lighting system"""
        self._owner = owner
        self._lighting_system = lighting_system
        lighting_system.add_light(self)

    def detach(self) -> None:
        """Detach from current owner and lighting system"""
        if self._lighting_system:
            self._lighting_system.remove_light(self)
        self._owner = None
        self._lighting_system = None


class LightingSystem:
    def __init__(self, config: LightingConfig | None = None) -> None:
        self.config = config if config is not None else LightingConfig()
        self.noise = tcod.noise.Noise(
            dimensions=2,  # 2D noise for position and time
            algorithm=tcod.noise.Algorithm.SIMPLEX,
            implementation=tcod.noise.Implementation.SIMPLE,
        )
        self.time = 0.0
        self.light_sources: list[LightSource] = []

    def add_light(self, light: LightSource) -> None:
        """Add a light source to the system"""
        self.light_sources.append(light)

    def remove_light(self, light: LightSource) -> None:
        """Remove a light source from the system"""
        if light in self.light_sources:
            self.light_sources.remove(light)

    def update(self, delta_time: float) -> None:
        """Update lighting effects based on elapsed time"""
        self.time += delta_time  # Just track raw time, light sources will scale it

    def compute_lighting(self, map_width: int, map_height: int) -> np.ndarray:
        """Compute lighting values for the entire map using bounded calculations."""
        # Start with ambient light - shape (width, height, 3) for RGB.
        # Use float32 for precision.
        light_map = np.full(
            (map_width, map_height, 3), self.config.ambient_light, dtype=np.float32
        )

        for light in self.light_sources:
            pos_x, pos_y = light.position
            radius = light.radius

            # Skip lights with zero or negative radius, as they illuminate nothing.
            if radius <= 0:
                continue

            # For dynamic lights, get the current flicker value
            flicker = 1.0
            if light.light_type == "dynamic" and light.flicker_enabled:
                # Sample a single noise value for uniform flicker
                flicker_grid = tcod.noise.grid(
                    shape=(1, 1),
                    scale=0.5,  # Lower scale = smoother transitions
                    # The type hint for `tcod.noise.grid` is ignored here because
                    # passing an int quantizes the value and results in jerky flicker.
                    # Moreover, the function's own docstring shows that it accepts
                    # a float for `origin`.
                    origin=(self.time * light.flicker_speed, 0),  # type: ignore[arg-type]
                    indexing="ij",
                )
                flicker_noise = self.noise[flicker_grid][0, 0]
                # Convert from [-1, 1] to [min_brightness, max_brightness]
                flicker = light.min_brightness + (
                    (flicker_noise + 1.0)
                    * 0.5
                    * (light.max_brightness - light.min_brightness)
                )

            # Determine the bounding box for the light's influence,
            # clamped to map boundaries. Ensure integer results for slicing.
            min_bx = max(0, int(np.floor(pos_x - radius)))
            max_bx = min(map_width - 1, int(np.ceil(pos_x + radius)))
            min_by = max(0, int(np.floor(pos_y - radius)))
            max_by = min(map_height - 1, int(np.ceil(pos_y + radius)))

            # If the bounding box is invalid (e.g., light entirely off-map or zero
            # effective area), skip.
            if min_bx > max_bx or min_by > max_by:
                continue

            # Create coordinate grid for the bounding box only
            # np.mgrid uses [inclusive:exclusive] for the stop value in slices, so +1
            tile_x_coords, tile_y_coords = np.mgrid[
                min_bx : max_bx + 1, min_by : max_by + 1
            ]

            # Calculate distances from the light source to each tile in the bounding box
            dx = tile_x_coords - pos_x
            dy = tile_y_coords - pos_y

            # Calculate squared distance first
            distance_sq_in_bounds = dx * dx + dy * dy

            # Create a mask for tiles within the circular radius
            # (tiles outside this are not actually lit by this light,
            # even if in bounding box).
            radius_sq = radius * radius
            in_circle_mask = distance_sq_in_bounds <= radius_sq

            # Only proceed with tiles that are actually within the circle
            if not np.any(in_circle_mask):
                continue

            # Calculate actual distance only for relevant tiles within the circle
            # Add a small epsilon to distance_sq_in_bounds before sqrt to avoid sqrt(0)
            # issues if needed, though for light falloff, sqrt(0) giving 0 is fine.
            distance_in_bounds = np.sqrt(distance_sq_in_bounds[in_circle_mask])

            # Calculate light intensity with normalized distance for these tiles
            # Intensity falls off linearly to the edge of the radius
            # Ensure radius isn't zero if we got here
            # (already checked, but good practice)
            norm_distance = np.clip(1.0 - (distance_in_bounds / radius), 0.0, 1.0)
            scalar_intensity_values = (
                norm_distance * flicker
            )  # 1D array of scalar intensities

            # Get the slice of the main light_map we'll be updating
            # We need to apply the 1D scalar_intensity_values to the 2D masked region
            # of the slice.
            light_map_slice = light_map[min_bx : max_bx + 1, min_by : max_by + 1, :]

            # Reshape scalar_intensity_values to be compatible with broadcasting over
            # the color channels for the masked elements.
            # The shape of light_map_slice[in_circle_mask] is (N, 3) where N is num True
            # in in_circle_mask.
            # scalar_intensity_values is (N,). We need (N, 1) to broadcast with (3,).
            light_color_array = (
                np.array(light.color, dtype=np.float32) / 255.0
                if np.max(light.color) > 1.0
                else np.array(light.color, dtype=np.float32)
            )

            colored_intensity_contribution = (
                scalar_intensity_values[:, np.newaxis] * light_color_array
            )

            # Apply to the specific (masked) elements within the slice
            # This directly modifies the `light_map_slice` (and thus `light_map`)
            # Get current values in the masked part of the slice
            current_values_in_masked_slice = light_map_slice[in_circle_mask]
            # Combine by taking the maximum
            updated_values = np.maximum(
                current_values_in_masked_slice, colored_intensity_contribution
            )
            # Assign back to the masked part of the slice
            light_map_slice[in_circle_mask] = updated_values

        return np.clip(light_map, 0.0, 1.0)  # Final clip to [0,1] range
