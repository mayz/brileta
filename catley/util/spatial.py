"""
A generic spatial hash grid for efficient 2D spatial queries.

This module provides a `SpatialHashGrid` class that implements a `SpatialIndex`
interface. It is used to accelerate queries for objects based on their
x, y coordinates, replacing O(n) linear scans with O(1) average-time lookups.
"""

import abc
from collections import defaultdict
from typing import Generic, Protocol, TypeVar

# A type alias for coordinate tuples to improve readability.
type Coord = tuple[int, int]


class HasPosition(Protocol):
    """A protocol for objects that have integer x and y attributes."""

    x: int
    y: int


# A TypeVar bound to the HasPosition protocol. This ensures that any object
# stored in the spatial index has .x and .y attributes.
T = TypeVar("T", bound=HasPosition)


class SpatialIndex(abc.ABC, Generic[T]):
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
    def get_at_point(self, x: int, y: int) -> list[T]:
        """Get all objects at a specific tile (x, y)."""
        pass

    @abc.abstractmethod
    def get_in_radius(self, x: int, y: int, radius: int) -> list[T]:
        """Get all objects within a certain Chebyshev distance (radius) of a point."""
        pass

    @abc.abstractmethod
    def get_in_bounds(self, x1: int, y1: int, x2: int, y2: int) -> list[T]:
        """Get all objects within a rectangular bounding box."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Remove all objects from the index."""
        pass


class SpatialHashGrid(SpatialIndex[T]):
    """
    A spatial hash grid for efficient spatial queries of objects.

    The grid divides the world space into cells of a fixed size. Objects are
    stored in a dictionary mapping cell coordinates to a list of objects
    within that cell. This makes queries for objects in a specific area
    very fast by only checking relevant cells.
    """

    def __init__(self, cell_size: int = 16):
        if cell_size <= 0:
            raise ValueError("Cell size must be a positive integer.")
        self.cell_size = cell_size
        self.grid: dict[Coord, list[T]] = defaultdict(list)
        self._obj_to_cell: dict[T, Coord] = {}

    def _hash(self, x: int, y: int) -> Coord:
        """Converts world coordinates to grid cell coordinates."""
        return x // self.cell_size, y // self.cell_size

    def add(self, obj: T) -> None:
        """Add an object to the grid."""
        cell_xy = self._hash(obj.x, obj.y)
        self.grid[cell_xy].append(obj)
        self._obj_to_cell[obj] = cell_xy

    def remove(self, obj: T) -> None:
        """Remove an object from the grid."""
        cell_xy = self._obj_to_cell.get(obj)
        if cell_xy is None:
            return  # Object not in the grid.

        try:
            self.grid[cell_xy].remove(obj)
            # If the cell is now empty, remove it from the grid to save memory.
            if not self.grid[cell_xy]:
                del self.grid[cell_xy]
        except ValueError:
            # Object was not in the list, which is fine. It might have been
            # removed already or state is out of sync.
            pass

        if obj in self._obj_to_cell:
            del self._obj_to_cell[obj]

    def update(self, obj: T) -> None:
        """Update an object's position after it has moved."""
        old_cell_xy = self._obj_to_cell.get(obj)
        new_cell_xy = self._hash(obj.x, obj.y)

        if old_cell_xy is None:
            # Object wasn't tracked; treat as a fresh add
            self.grid[new_cell_xy].append(obj)
            self._obj_to_cell[obj] = new_cell_xy
            return

        if old_cell_xy == new_cell_xy:
            return  # Fast path: object stayed in the same cell

        cell = self.grid.get(old_cell_xy)
        if cell is not None:
            try:
                cell.remove(obj)
                if not cell:
                    del self.grid[old_cell_xy]
            except ValueError:
                pass

        self.grid[new_cell_xy].append(obj)
        self._obj_to_cell[obj] = new_cell_xy

    def get_at_point(self, x: int, y: int) -> list[T]:
        """Get all objects at a specific tile (x, y)."""
        cell_xy = self._hash(x, y)
        cell_contents = self.grid.get(cell_xy, [])
        # A cell contains a region; we must filter for the exact coordinates.
        return [obj for obj in cell_contents if obj.x == x and obj.y == y]

    def get_in_bounds(self, x1: int, y1: int, x2: int, y2: int) -> list[T]:
        """Get all objects within a rectangular bounding box."""
        cx1, cy1 = self._hash(x1, y1)
        cx2, cy2 = self._hash(x2, y2)

        results: list[T] = []
        seen = set()  # Use a set to prevent adding duplicates

        for cx in range(cx1, cx2 + 1):
            for cy in range(cy1, cy2 + 1):
                for obj in self.grid.get((cx, cy), []):
                    # Check if object is within the precise bounds and not already added
                    if obj not in seen and x1 <= obj.x <= x2 and y1 <= obj.y <= y2:
                        results.append(obj)
                        seen.add(obj)
        return results

    def get_in_radius(self, x: int, y: int, radius: int) -> list[T]:
        """Get all objects within a certain Chebyshev distance (square radius)."""
        bounds_x1, bounds_y1 = x - radius, y - radius
        bounds_x2, bounds_y2 = x + radius, y + radius

        # Get all potential matches within the bounding box of the radius.
        potential_matches = self.get_in_bounds(
            bounds_x1, bounds_y1, bounds_x2, bounds_y2
        )

        # Perform the final, more precise distance check.
        return [
            obj
            for obj in potential_matches
            if max(abs(obj.x - x), abs(obj.y - y)) <= radius
        ]

    def clear(self) -> None:
        """Remove all objects from the index."""
        self.grid.clear()
        self._obj_to_cell.clear()
