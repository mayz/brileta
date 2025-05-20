from dataclasses import dataclass
from typing import Literal

import numpy as np
import tcod


@dataclass
class LightSource:
    """Represents a light source in the game world"""

    radius: int  # Required parameter must come first
    _pos: tuple[int, int] | None = None  # Internal position storage
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB
    light_type: Literal["static", "dynamic"] = "static"
    flicker_enabled: bool = False
    flicker_speed: float = 0.2
    movement_scale: float = 1.5
    min_brightness: float = 0.6
    max_brightness: float = 1.4

    def __post_init__(self):
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
    def position(self, value: tuple[int, int]):
        self._pos = value

    def attach(self, owner, lighting_system) -> None:
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


@dataclass
class LightingConfig:
    """Configuration for the lighting system"""

    fov_radius: int = 15
    ambient_light: float = 0.1  # Base light level


class LightingSystem:
    def __init__(self, config: LightingConfig | None = None):
        self.config = config if config is not None else LightingConfig()
        self.noise = tcod.noise.Noise(1)
        self.torch_time = 0.0
        self.light_sources: list[LightSource] = []

    def add_light(self, light: LightSource) -> None:
        """Add a light source to the system"""
        self.light_sources.append(light)

    def remove_light(self, light: LightSource) -> None:
        """Remove a light source from the system"""
        if light in self.light_sources:
            self.light_sources.remove(light)

    def compute_lighting(self, map_width: int, map_height: int) -> np.ndarray:
        """Compute lighting values for the entire map

        Args:
            map_width: Width of the game map
            map_height: Height of the game map
        """
        # Start with ambient light
        light_map = np.full((map_width, map_height), self.config.ambient_light)

        # Update time for flickering lights
        self.torch_time += 0.1

        for light in self.light_sources:
            light_intensity = np.zeros((map_width, map_height))

            # Calculate light position with noise if dynamic
            pos_x, pos_y = light.position
            if light.light_type == "dynamic" and light.flicker_enabled:
                noise_x = self.noise.get_point(self.torch_time)
                noise_y = self.noise.get_point(self.torch_time + 11)
                pos_x += noise_x * light.movement_scale
                pos_y += noise_y * light.movement_scale

            # Calculate distances
            x_coords, y_coords = np.mgrid[0:map_width, 0:map_height]
            dx = x_coords - pos_x
            dy = y_coords - pos_y
            distance = np.sqrt(dx * dx + dy * dy)

            # Calculate brightness with flicker if enabled
            brightness = 1.0
            if light.flicker_enabled:
                noise_power = self.noise.get_point(self.torch_time + 17)
                brightness_range = light.max_brightness - light.min_brightness
                brightness = light.min_brightness + noise_power * brightness_range

            # Calculate light intensity
            intensity = np.clip(1.0 - (distance / light.radius), 0, 1) * brightness

            # Apply color
            for color in light.color:
                light_intensity += intensity * color

            # Combine lights using max function
            light_map = np.maximum(light_map, light_intensity)

        return np.clip(light_map, 0, 1)
