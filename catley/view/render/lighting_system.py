"""Abstract interface for lighting systems.

This module defines the contract that any lighting engine must implement.
This enables swapping out CPU implementations for GPU ones in the future
without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld
    from catley.game.lights import LightSource


class LightingSystem(ABC):
    """Abstract base class that defines the contract for any lighting engine.

    This interface allows the game to work with different lighting implementations
    (CPU-based, GPU-accelerated, etc.) without changing the client code.
    """

    def __init__(self, game_world: GameWorld) -> None:
        """Initialize the lighting system with a reference to the game world.

        Args:
            game_world: The game world to query for lighting data
        """
        self.game_world = game_world

    @abstractmethod
    def update(self, delta_time: float) -> None:
        """Update internal time-based state for dynamic effects.

        This method is called each frame to update time-dependent lighting
        effects like flickering torches.

        Args:
            delta_time: Time elapsed since last frame in seconds
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
