"""Light source hierarchy for the game lighting system.

This module defines the data containers for lights in the game world.
They are simple data structures that represent what lights exist and where
they are, but have no knowledge of how they are rendered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from catley import colors, config
from catley.util.coordinates import WorldTilePos

if TYPE_CHECKING:
    from catley.game.actors import Actor, Character


class Vec2:
    """Simple 2D vector for directional lighting."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def normalized(self) -> Vec2:
        """Return a normalized version of this vector."""
        magnitude = (self.x**2 + self.y**2) ** 0.5
        if magnitude == 0:
            return Vec2(0, 0)
        return Vec2(self.x / magnitude, self.y / magnitude)

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"


class LightSource(ABC):
    """Abstract base class for all light sources in the game world.

    Defines the core properties that any light must have: position, radius, and color.
    This is a simple data container with no rendering logic.
    """

    def __init__(
        self, position: WorldTilePos, radius: int, color: colors.Color
    ) -> None:
        """Initialize a light source.

        Args:
            position: The world tile position of the light
            radius: The radius of illumination in tiles
            color: The RGB color of the light
        """
        self.position = position
        self.radius = radius
        self.color = color

    @abstractmethod
    def is_static(self) -> bool:
        """Return True if this light does not move or change over time."""
        pass


class StaticLight(LightSource):
    """A light that does not move or change over time.

    Examples: glowing crystals on walls, eternal flames, magical light sources
    embedded in the environment.
    """

    def is_static(self) -> bool:
        """Static lights never change."""
        return True


class DynamicLight(LightSource):
    """A light that can move or change intensity/color each frame.

    Examples: flickering torches, lights held by actors, magical effects
    that pulse or change color.
    """

    def __init__(
        self,
        position: WorldTilePos,
        radius: int,
        color: colors.Color,
        flicker_enabled: bool = False,
        flicker_speed: float = 1.0,
        min_brightness: float = 1.0,
        max_brightness: float = 1.0,
        owner: Actor | None = None,
    ) -> None:
        """Initialize a dynamic light source.

        Args:
            position: The world tile position of the light
            radius: The radius of illumination in tiles
            color: The RGB color of the light
            flicker_enabled: Whether this light flickers
            flicker_speed: How fast the light flickers (higher = faster)
            min_brightness: Minimum brightness multiplier for flicker
            max_brightness: Maximum brightness multiplier for flicker
            owner: The actor that owns this light (if any)
        """
        super().__init__(position, radius, color)
        self.flicker_enabled = flicker_enabled
        self.flicker_speed = flicker_speed
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.owner = owner

    def is_static(self) -> bool:
        """Dynamic lights can change."""
        return False

    @staticmethod
    def create_player_torch(owner: Character) -> DynamicLight:
        """Factory method to create a standard player torch."""
        return DynamicLight(
            position=(owner.x, owner.y),
            radius=config.TORCH_RADIUS,
            color=config.TORCH_COLOR,
            flicker_enabled=True,
            flicker_speed=config.TORCH_FLICKER_SPEED,
            min_brightness=config.TORCH_MIN_BRIGHTNESS,
            max_brightness=config.TORCH_MAX_BRIGHTNESS,
            owner=owner,
        )


class GlobalLight(LightSource):
    """Abstract base class for lights that affect the entire world.

    Global lights like sunlight or moonlight illuminate based on direction
    and sky exposure rather than distance from a point.
    """

    def is_static(self) -> bool:
        """Global lights change slowly if at all, cached with static lights."""
        return True


class DirectionalLight(GlobalLight):
    """Sunlight or moonlight with direction and intensity.

    Provides illumination based on a directional vector rather than
    radiating from a point. Intensity is modulated by sky exposure.
    """

    def __init__(
        self,
        direction: Vec2,
        color: colors.Color,
        intensity: float = 1.0,
    ) -> None:
        """Initialize a directional light source.

        Args:
            direction: Normalized direction vector (e.g., Vec2(0.6, -0.8) for sun)
            color: The RGB color of the light
            intensity: Base intensity multiplier
        """
        # Global lights don't have a traditional position or radius
        super().__init__(position=(0, 0), radius=0, color=color)
        self.direction = direction.normalized()
        self.intensity = intensity

    @staticmethod
    def create_sun(
        elevation_degrees: float = 45.0,
        azimuth_degrees: float = 135.0,  # Southeast
        intensity: float = 0.8,
        color: colors.Color = (255, 243, 204),  # Warm sunlight (converted to 0-255)
    ) -> DirectionalLight:
        """Factory method to create sunlight.

        Args:
            elevation_degrees: Sun elevation above horizon (0-90 degrees)
            azimuth_degrees: Sun direction (0=North, 90=East, 180=South, 270=West)
            intensity: Base sun intensity
            color: Sun color (warm white by default)
        """
        import math

        # Convert to radians
        elevation_rad = math.radians(elevation_degrees)
        azimuth_rad = math.radians(azimuth_degrees)

        # Convert spherical to cartesian (2D projection)
        # Y component is negative because down is positive in screen coordinates
        direction_x = math.sin(azimuth_rad) * math.cos(elevation_rad)
        direction_y = -math.sin(elevation_rad)  # Negative for downward light

        return DirectionalLight(
            direction=Vec2(direction_x, direction_y),
            color=color,
            intensity=intensity,
        )
