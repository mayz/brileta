"""Abstract interface for lighting systems.

This module defines the contract that any lighting engine must implement.
This enables swapping out CPU implementations for GPU ones in the future
without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from catley.config import (
    AMBIENT_LIGHT_LEVEL,
    SKY_EXPOSURE_POWER,
    SUN_AZIMUTH_DEGREES,
    SUN_COLOR,
    SUN_ELEVATION_DEGREES,
    SUN_ENABLED,
    SUN_INTENSITY,
)
from catley.types import FixedTimestep
from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.game.actors import Actor
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource


@dataclass(frozen=True)
class LightingConfig:
    """Shared configuration defaults for lighting systems."""

    ambient_light: float = AMBIENT_LIGHT_LEVEL

    # Global sunlight configuration
    sun_enabled: bool = SUN_ENABLED
    sun_color: tuple[int, int, int] = SUN_COLOR
    sun_elevation_degrees: float = SUN_ELEVATION_DEGREES
    sun_azimuth_degrees: float = SUN_AZIMUTH_DEGREES
    sun_intensity: float = SUN_INTENSITY
    sky_exposure_power: float = SKY_EXPOSURE_POWER


class LightingSystem(ABC):
    """Abstract base class that defines the contract for any lighting engine.

    This interface allows the game to work with different lighting implementations
    (e.g., CPU-based, GPU-accelerated) without changing the client code.

    **Subclassing Contract:**
    A concrete implementation of LightingSystem MUST adhere to the following:
    1.  It must initialize `self.revision` to 0 via `super().__init__()`.
    2.  It must increment `self.revision` whenever the lighting state changes
        in a way that requires a visual update. This is critical for cache
        invalidation in the rendering views.
    3.  Key places to increment `revision` include after a full `compute_lightmap`
        pass and within the `on_light_moved` handler.
    """

    def __init__(self, game_world: GameWorld) -> None:
        """Initialize the lighting system with a reference to the game world.

        Args:
            game_world: The game world to query for lighting data
        """
        self.game_world = game_world
        self.revision: int = 0  # Incremented when lighting state changes

    @abstractmethod
    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update internal time-based state for dynamic effects.

        This method is called each frame to update time-dependent lighting
        effects like flickering torches.

        Args:
            fixed_timestep: Fixed time step duration for deterministic lighting timing
        """
        pass

    @abstractmethod
    def compute_lightmap(self, viewport_bounds: Rect) -> np.ndarray | None:
        """Compute the final lightmap for the visible area.

        This is the primary API method that clients use to get lighting data.
        Takes the visible area and returns a renderer-agnostic NumPy array
        of float RGB intensity values.

        Args:
            viewport_bounds: The visible area to compute lighting for

        Returns:
            A (width, height, 3) NumPy array of float RGB intensity values,
            or None if no lighting data is available
        """
        pass

    @abstractmethod
    def on_light_added(self, light: LightSource) -> None:
        """Notification that a light has been added to the game world.

        Allows the lighting system to invalidate caches or update internal state.

        Args:
            light: The light source that was added
        """
        pass

    @abstractmethod
    def on_light_removed(self, light: LightSource) -> None:
        """Notification that a light has been removed from the game world.

        Allows the lighting system to invalidate caches or update internal state.

        Args:
            light: The light source that was removed
        """
        pass

    @abstractmethod
    def on_light_moved(self, light: LightSource) -> None:
        """Notification that a light has moved.

        Allows the lighting system to invalidate caches or update internal state.

        Args:
            light: The light source that moved
        """
        pass

    def on_global_light_changed(self) -> None:
        """Notification that global lighting has changed (e.g., time of day).

        Default implementation does nothing. Subclasses can override to
        invalidate caches when global lighting conditions change.
        """
        # Default implementation does nothing
        return

    def on_actor_moved(self, actor: Actor) -> None:
        """Notification that an actor has moved (invalidates shadow caster cache).

        Actors cast shadows, so when they move their cached shadow positions
        become stale. Subclasses should override to invalidate shadow caster
        caches.

        Args:
            actor: The actor that moved
        """
        # Default implementation does nothing
        return
