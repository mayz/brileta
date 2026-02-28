"""
A generic spatial hash grid for efficient 2D spatial queries.

This module provides the `SpatialIndex` ABC and `HasPosition` protocol,
plus a `SpatialHashGrid` class backed by a native C extension for
performance-critical spatial lookups.
"""

import abc
from typing import Protocol, TypeVar

from brileta.types import WorldTileCoord
from brileta.util._native import SpatialHashGrid as _CSpatialHashGrid
from brileta.util.coordinates import Rect


class HasPosition(Protocol):
    """A protocol for objects that have integer x and y attributes."""

    x: int
    y: int


# A TypeVar bound to the HasPosition protocol. This ensures that any object
# stored in the spatial index has .x and .y attributes.
T = TypeVar("T", bound=HasPosition)


class SpatialIndex[T: HasPosition](abc.ABC):
    """Abstract base class for a spatial indexing data structure."""

    @abc.abstractmethod
    def add(self, obj: T) -> None:
        """Add an object to the index."""
        pass

    @abc.abstractmethod
    def remove(self, obj: T) -> None:
        """Remove an object from the index."""
        pass

    @abc.abstractmethod
    def update(self, obj: T) -> None:
        """Update the position of an object that has moved."""
        pass

    @abc.abstractmethod
    def get_at_point(self, x: WorldTileCoord, y: WorldTileCoord) -> list[T]:
        """Get all objects at a specific tile (x, y)."""
        pass

    @abc.abstractmethod
    def get_in_radius(
        self, x: WorldTileCoord, y: WorldTileCoord, radius: int
    ) -> list[T]:
        """Get all objects within a certain Chebyshev distance (radius) of a point."""
        pass

    @abc.abstractmethod
    def get_in_bounds(self, x1: int, y1: int, x2: int, y2: int) -> list[T]:
        """Get all objects within a rectangular bounding box."""
        pass

    def get_in_rect(self, bounds: Rect) -> list[T]:
        """Get all objects within a Rect bounds."""
        return self.get_in_bounds(bounds.x1, bounds.y1, bounds.x2, bounds.y2)

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all objects from the index."""
        pass


class SpatialHashGrid[T: HasPosition](_CSpatialHashGrid, SpatialIndex[T]):
    """A spatial hash grid backed by a native C extension.

    Inherits all method implementations from the C extension while
    satisfying the SpatialIndex ABC for type-checking compatibility.
    Objects must have integer .x and .y attributes.
    """

    pass
