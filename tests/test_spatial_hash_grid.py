from dataclasses import dataclass

import pytest

from catley.util.spatial import SpatialHashGrid


@dataclass(frozen=True)
class Pointy:
    """A simple, hashable object with position for testing."""

    x: int
    y: int
    name: str


@dataclass(eq=True)
class MutablePointy:
    """A mutable version for testing updates."""

    x: int
    y: int
    name: str
    __hash__ = object.__hash__


def test_grid_initialization() -> None:
    """Test that the grid initializes correctly."""
    grid = SpatialHashGrid(cell_size=10)
    assert grid.cell_size == 10
    assert not grid.grid
    assert not grid._obj_to_cell


def test_grid_init_invalid_cell_size() -> None:
    """Test that initializing with a non-positive cell size raises an error."""
    with pytest.raises(ValueError):
        SpatialHashGrid(cell_size=0)
    with pytest.raises(ValueError):
        SpatialHashGrid(cell_size=-10)


def test_add_and_get_at_point() -> None:
    """Test adding an object and retrieving it by its exact point."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.add(obj1)

    assert grid.get_at_point(5, 5) == [obj1]
    assert not grid.get_at_point(6, 6)


def test_add_multiple_at_same_point() -> None:
    """Test adding multiple objects to the exact same coordinates."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(7, 8, "obj1")
    obj2 = Pointy(7, 8, "obj2")
    grid.add(obj1)
    grid.add(obj2)

    results = grid.get_at_point(7, 8)
    assert len(results) == 2
    assert obj1 in results
    assert obj2 in results


def test_remove_object() -> None:
    """Test removing an object from the grid."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.add(obj1)
    assert grid.get_at_point(5, 5) == [obj1]

    grid.remove(obj1)
    assert not grid.get_at_point(5, 5)
    assert not grid.grid  # The cell should be deleted as it's now empty
    assert not grid._obj_to_cell


def test_remove_nonexistent_object() -> None:
    """Test that removing an object not in the grid does not raise an error."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.remove(obj1)  # Should not fail


def test_update_object_within_same_cell() -> None:
    """Test updating an object's position but keeping it in the same cell."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = MutablePointy(2, 2, "obj1")
    grid.add(obj1)
    cell = grid._hash(2, 2)
    assert grid.grid[cell] == {obj1}

    obj1.x = 3
    obj1.y = 3
    grid.update(obj1)

    assert not grid.get_at_point(2, 2)
    assert grid.get_at_point(3, 3) == [obj1]
    assert grid.grid[cell] == {obj1}  # Should still be in the same cell


def test_update_object_across_cells() -> None:
    """Test moving an object to a different cell."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = MutablePointy(5, 5, "obj1")
    grid.add(obj1)
    old_cell = grid._hash(5, 5)

    obj1.x = 15
    obj1.y = 15
    new_cell = grid._hash(15, 15)
    grid.update(obj1)

    assert old_cell not in grid.grid  # Old cell should be gone
    assert grid.grid[new_cell] == {obj1}
    assert not grid.get_at_point(5, 5)
    assert grid.get_at_point(15, 15) == [obj1]


def test_get_in_bounds() -> None:
    """Test querying for objects within a rectangular area."""
    grid = SpatialHashGrid(cell_size=10)
    p1 = Pointy(5, 5, "p1")  # In bounds
    p2 = Pointy(15, 15, "p2")  # In bounds
    p3 = Pointy(25, 25, "p3")  # Out of bounds
    p4 = Pointy(8, 8, "p4")  # In bounds
    p5 = Pointy(0, 0, "p5")  # On edge, in bounds

    for p in [p1, p2, p3, p4, p5]:
        grid.add(p)

    results = grid.get_in_bounds(0, 0, 20, 20)
    assert len(results) == 4
    assert p1 in results
    assert p2 in results
    assert p4 in results
    assert p5 in results
    assert p3 not in results


def test_get_in_radius() -> None:
    """Test querying for objects within a radius (Chebyshev distance)."""
    grid = SpatialHashGrid(cell_size=10)
    center_x, center_y = 10, 10
    radius = 5

    p_center = Pointy(10, 10, "center")  # dist 0
    p_inside = Pointy(12, 12, "inside")  # dist 2
    p_edge = Pointy(15, 15, "edge")  # dist 5
    p_outside = Pointy(16, 16, "outside")  # dist 6
    p_far = Pointy(30, 30, "far")

    for p in [p_center, p_inside, p_edge, p_outside, p_far]:
        grid.add(p)

    results = grid.get_in_radius(center_x, center_y, radius)
    assert len(results) == 3
    assert p_center in results
    assert p_inside in results
    assert p_edge in results
    assert p_outside not in results
    assert p_far not in results


def test_clear_grid() -> None:
    """Test that clear removes all objects from the grid."""
    grid = SpatialHashGrid()
    grid.add(Pointy(1, 1, "p1"))
    grid.add(Pointy(2, 2, "p2"))

    assert grid.grid and grid._obj_to_cell
    grid.clear()
    assert not grid.grid
    assert not grid._obj_to_cell
