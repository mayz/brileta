from dataclasses import dataclass

import numpy as np
import tcod


@dataclass
class LightingConfig:
    """Configuration for the lighting system"""
    fov_radius: int = 10
    torch_enabled: bool = True
    torch_radius: int = 8  # Separate from FOV radius
    torch_flicker_speed: float = 0.2
    torch_movement_scale: float = 1.5
    torch_min_brightness: float = 0.6
    torch_max_brightness: float = 1.4


class LightingSystem:
    def __init__(self, config: LightingConfig | None = None):
        self.config = config if config is not None else LightingConfig()
        self.noise = tcod.noise.Noise(1)
        self.torch_time = 0.0

    def compute_lighting(self, player_pos: tuple[int, int],
                        map_width: int, map_height: int) -> np.ndarray:
        """Compute lighting values for the entire map"""
        if not self.config.torch_enabled:
            return np.ones((map_width, map_height))

        # Update torch state
        self.torch_time += self.config.torch_flicker_speed
        noise_x = self.noise.get_point(self.torch_time)
        noise_y = self.noise.get_point(self.torch_time + 11)
        noise_power = self.noise.get_point(self.torch_time + 17)

        torch_dx = noise_x * self.config.torch_movement_scale
        torch_dy = noise_y * self.config.torch_movement_scale

        brightness_range = (
            self.config.torch_max_brightness - self.config.torch_min_brightness
        )
        torch_power = self.config.torch_min_brightness + noise_power * brightness_range

        # Calculate distances
        x_coords, y_coords = np.mgrid[0:map_width, 0:map_height]
        dx = x_coords - (player_pos[0] + torch_dx)
        dy = y_coords - (player_pos[1] + torch_dy)
        distance = np.sqrt(dx * dx + dy * dy)

        # Calculate light intensity
        light_intensity = np.clip(
            1.0 - (distance / self.config.torch_radius),
            0,
            1
        ) * torch_power

        return light_intensity