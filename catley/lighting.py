from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import tcod

from . import colors

if TYPE_CHECKING:
    from .model import Entity

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
        """Compute lighting values for the entire map"""
        # Start with ambient light - shape (width, height, 3) for RGB
        light_map = np.full((map_width, map_height, 3), self.config.ambient_light)

        for light in self.light_sources:
            light_intensity = np.zeros((map_width, map_height, 3))  # RGB channels
            pos_x, pos_y = light.position

            # For dynamic lights, get the current flicker value
            flicker = 1.0
            if light.light_type == "dynamic" and light.flicker_enabled:
                # Sample a single noise value for uniform flicker
                flicker_grid = tcod.noise.grid(
                    shape=(1, 1),
                    scale=0.5,  # Lower scale = smoother transitions
                    origin=(int(self.time * light.flicker_speed), 0),
                    indexing="ij",
                )
                flicker_noise = self.noise[flicker_grid][0, 0]
                # Convert from [-1, 1] to [min_brightness, max_brightness]
                flicker = light.min_brightness + (
                    (flicker_noise + 1.0)
                    * 0.5
                    * (light.max_brightness - light.min_brightness)
                )

            # Calculate distances using numpy broadcasting
            x_coords, y_coords = np.mgrid[0:map_width, 0:map_height]
            dx = x_coords - pos_x
            dy = y_coords - pos_y
            distance = np.sqrt(dx * dx + dy * dy)

            # Calculate brightness
            brightness = flicker  # Use the uniform flicker value

            # Calculate light intensity with normalized distance
            norm_distance = np.clip(1.0 - (distance / light.radius), 0, 1)
            intensity = norm_distance * brightness

            # Apply color using broadcasting
            for i, color_component in enumerate(light.color):
                light_intensity[..., i] = intensity * color_component

            # Combine lights using max function
            light_map = np.maximum(light_map, light_intensity)

        return np.clip(light_map, 0, 1)
