"""
Instrumented version of SpatialHashGrid for performance benchmarking.

This module provides an instrumented wrapper around the base SpatialHashGrid
that adds performance measurement decorators to all methods. Use this only
for benchmarking and profiling - it has overhead that shouldn't be in the
main game loop.

Usage:
    # In benchmark scripts only
    from catley.scripts.instrumented_spatial import InstrumentedSpatialHashGrid

    grid = InstrumentedSpatialHashGrid(cell_size=16)
    # All operations will now be automatically timed
"""

from typing import TypeVar

from catley.util.performance import measure
from catley.util.spatial import HasPosition, SpatialHashGrid

# Re-export the TypeVar from the base module
T = TypeVar("T", bound=HasPosition)


class InstrumentedSpatialHashGrid(SpatialHashGrid[T]):
    """
    Performance-instrumented version of SpatialHashGrid.

    This class wraps all SpatialHashGrid methods with performance measurement
    decorators. It provides identical functionality to the base class but
    automatically tracks timing statistics for all operations.

    Only use this for benchmarking and profiling. The measurement overhead
    makes it unsuitable for production game code.
    """

    def __init__(self, cell_size: int = 16):
        """Initialize instrumented spatial hash grid.

        Args:
            cell_size: Size of each grid cell in world units.
        """
        super().__init__(cell_size)

    @measure("spatial_add")
    def add(self, obj: T) -> None:
        """Add an object to the grid with performance measurement."""
        return super().add(obj)

    @measure("spatial_remove")
    def remove(self, obj: T) -> None:
        """Remove an object from the grid with performance measurement."""
        return super().remove(obj)

    @measure("spatial_update")
    def update(self, obj: T) -> None:
        """Update an object's position with performance measurement."""
        return super().update(obj)

    @measure("spatial_get_at_point")
    def get_at_point(self, x: int, y: int) -> list[T]:
        """Get all objects at a specific point with performance measurement."""
        return super().get_at_point(x, y)

    @measure("spatial_get_in_radius")
    def get_in_radius(self, x: int, y: int, radius: int) -> list[T]:
        """Get objects within radius with performance measurement."""
        return super().get_in_radius(x, y, radius)

    @measure("spatial_clear")
    def clear(self) -> None:
        """Clear all objects with performance measurement."""
        return super().clear()

    # Don't instrument _hash() since it's a simple calculation
    # and would add overhead without much insight


def create_instrumented_grid(cell_size: int = 16) -> InstrumentedSpatialHashGrid:
    """Factory function to create an instrumented spatial hash grid.

    Args:
        cell_size: Size of each grid cell in world units.

    Returns:
        New InstrumentedSpatialHashGrid instance ready for use with any
        HasPosition objects.

    Example:
        grid = create_instrumented_grid(16)
        grid.add(my_actor)  # my_actor must have .x and .y attributes
        results = grid.get_in_radius(x, y, radius)
    """
    return InstrumentedSpatialHashGrid(cell_size)
