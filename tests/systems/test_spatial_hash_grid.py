from dataclasses import dataclass

import pytest

from brileta.util.coordinates import Rect
from brileta.util.spatial import SpatialHashGrid


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


# ---- Construction --------------------------------------------------------


def test_grid_initialization() -> None:
    """Test that the grid initializes correctly with a given cell size."""
    grid = SpatialHashGrid(cell_size=10)
    assert grid.cell_size == 10


def test_grid_default_cell_size() -> None:
    """Test that the grid uses 16 as default cell size."""
    grid = SpatialHashGrid()
    assert grid.cell_size == 16


def test_grid_init_invalid_cell_size() -> None:
    """Test that initializing with a non-positive cell size raises an error."""
    with pytest.raises(ValueError):
        SpatialHashGrid(cell_size=0)
    with pytest.raises(ValueError):
        SpatialHashGrid(cell_size=-10)


# ---- add / get_at_point -------------------------------------------------


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


# ---- remove --------------------------------------------------------------


def test_remove_object() -> None:
    """Test removing an object from the grid."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.add(obj1)
    assert grid.get_at_point(5, 5) == [obj1]

    grid.remove(obj1)
    assert not grid.get_at_point(5, 5)


def test_remove_nonexistent_object() -> None:
    """Test that removing an object not in the grid does not raise an error."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.remove(obj1)  # Should not fail


def test_remove_then_query() -> None:
    """Removing an object should make it invisible to all query methods."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = Pointy(5, 5, "obj1")
    grid.add(obj1)
    grid.remove(obj1)

    assert not grid.get_at_point(5, 5)
    assert not grid.get_in_bounds(0, 0, 10, 10)
    assert not grid.get_in_radius(5, 5, 5)


# ---- update --------------------------------------------------------------


def test_update_object_within_same_cell() -> None:
    """Test updating an object's position but keeping it in the same cell."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = MutablePointy(2, 2, "obj1")
    grid.add(obj1)

    obj1.x = 3
    obj1.y = 3
    grid.update(obj1)

    assert not grid.get_at_point(2, 2)
    assert grid.get_at_point(3, 3) == [obj1]


def test_update_object_across_cells() -> None:
    """Test moving an object to a different cell."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = MutablePointy(5, 5, "obj1")
    grid.add(obj1)

    obj1.x = 15
    obj1.y = 15
    grid.update(obj1)

    assert not grid.get_at_point(5, 5)
    assert grid.get_at_point(15, 15) == [obj1]


def test_update_untracked_object_acts_as_add() -> None:
    """Calling update on an object not in the grid should add it."""
    grid = SpatialHashGrid(cell_size=10)
    obj1 = MutablePointy(5, 5, "obj1")
    grid.update(obj1)
    assert grid.get_at_point(5, 5) == [obj1]


# ---- get_in_bounds -------------------------------------------------------


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


def test_get_in_bounds_swapped_coordinates() -> None:
    """get_in_bounds should handle swapped coordinate pairs."""
    grid = SpatialHashGrid(cell_size=10)
    obj = Pointy(5, 5, "obj")
    grid.add(obj)

    # Pass x2 < x1 and y2 < y1 - should still work.
    results = grid.get_in_bounds(10, 10, 0, 0)
    assert obj in results


# ---- get_in_radius -------------------------------------------------------


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


# ---- get_in_rect ---------------------------------------------------------


def test_get_in_rect() -> None:
    """Test querying via a Rect object."""
    grid = SpatialHashGrid(cell_size=10)
    p_in = Pointy(3, 3, "in")
    p_out = Pointy(20, 20, "out")
    grid.add(p_in)
    grid.add(p_out)

    rect = Rect(0, 0, 10, 10)  # x1=0, y1=0, x2=10, y2=10
    results = grid.get_in_rect(rect)
    assert p_in in results
    assert p_out not in results


# ---- clear ---------------------------------------------------------------


def test_clear_grid() -> None:
    """Test that clear removes all objects from the grid."""
    grid = SpatialHashGrid()
    grid.add(Pointy(1, 1, "p1"))
    grid.add(Pointy(2, 2, "p2"))

    grid.clear()
    assert not grid.get_at_point(1, 1)
    assert not grid.get_at_point(2, 2)
    assert not grid.get_in_bounds(-1000, -1000, 1000, 1000)


# ---- Edge cases ----------------------------------------------------------


def test_negative_coordinates() -> None:
    """Objects at negative coordinates should work correctly."""
    grid = SpatialHashGrid(cell_size=10)
    obj = Pointy(-5, -10, "neg")
    grid.add(obj)

    assert grid.get_at_point(-5, -10) == [obj]
    assert obj in grid.get_in_bounds(-20, -20, 0, 0)
    assert obj in grid.get_in_radius(-5, -10, 3)


def test_objects_straddling_cell_boundaries() -> None:
    """Objects exactly on cell boundaries should be found correctly."""
    grid = SpatialHashGrid(cell_size=10)
    # Objects at cell boundary (10 is the start of cell 1)
    at_boundary = Pointy(10, 10, "boundary")
    just_before = Pointy(9, 9, "before")
    grid.add(at_boundary)
    grid.add(just_before)

    # A query that includes both should find both.
    results = grid.get_in_bounds(9, 9, 10, 10)
    assert len(results) == 2
    assert at_boundary in results
    assert just_before in results

    # A query that only covers the first cell should find just_before.
    results = grid.get_in_bounds(0, 0, 9, 9)
    assert just_before in results
    assert at_boundary not in results


def test_large_radius_query() -> None:
    """A large radius should still return the correct results."""
    grid = SpatialHashGrid(cell_size=4)
    objs = [Pointy(i, i, f"p{i}") for i in range(20)]
    for o in objs:
        grid.add(o)

    results = grid.get_in_radius(10, 10, 100)
    assert len(results) == 20


def test_many_objects_same_cell() -> None:
    """Multiple objects in the same cell should all be tracked."""
    grid = SpatialHashGrid(cell_size=100)
    objs = [Pointy(i, i, f"p{i}") for i in range(50)]
    for o in objs:
        grid.add(o)

    # All should be found in a large enough bounds query.
    results = grid.get_in_bounds(0, 0, 49, 49)
    assert len(results) == 50


def test_add_remove_add_same_object() -> None:
    """Adding, removing, then re-adding the same object should work."""
    grid = SpatialHashGrid(cell_size=10)
    obj = MutablePointy(5, 5, "obj")
    grid.add(obj)
    grid.remove(obj)
    assert not grid.get_at_point(5, 5)

    obj.x = 15
    obj.y = 15
    grid.add(obj)
    assert grid.get_at_point(15, 15) == [obj]
    assert not grid.get_at_point(5, 5)


def test_update_after_move_multiple_times() -> None:
    """Repeated updates should correctly track the object's position."""
    grid = SpatialHashGrid(cell_size=5)
    obj = MutablePointy(0, 0, "mover")
    grid.add(obj)

    for i in range(1, 20):
        obj.x = i * 3
        obj.y = i * 3
        grid.update(obj)
        assert grid.get_at_point(i * 3, i * 3) == [obj]
        if i > 0:
            assert not grid.get_at_point((i - 1) * 3, (i - 1) * 3)
