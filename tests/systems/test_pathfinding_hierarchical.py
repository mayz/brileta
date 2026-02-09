"""Tests for hierarchical pathfinding (HPA*) - region graph traversal."""

from __future__ import annotations

import numpy as np
import pytest

from brileta.environment.generators import GeneratedMapData
from brileta.environment.map import GameMap, MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.util.coordinates import Rect
from brileta.util.pathfinding import find_region_path


def make_connected_regions() -> dict[int, MapRegion]:
    """Create a simple 3-region graph: 0 -- 1 -- 2 (linear connection)."""
    return {
        0: MapRegion(
            id=0,
            region_type="room",
            bounds=[Rect(0, 0, 5, 5)],
            connections={1: (4, 2)},  # Door to region 1 at (4, 2)
        ),
        1: MapRegion(
            id=1,
            region_type="hallway",
            bounds=[Rect(5, 0, 5, 5)],
            connections={0: (5, 2), 2: (9, 2)},  # Doors to regions 0 and 2
        ),
        2: MapRegion(
            id=2,
            region_type="room",
            bounds=[Rect(10, 0, 5, 5)],
            connections={1: (10, 2)},  # Door to region 1
        ),
    }


def make_branching_regions() -> dict[int, MapRegion]:
    """
    Create a branching region graph:
        0 -- 1 -- 2
             |
             3
    """
    return {
        0: MapRegion(
            id=0,
            region_type="room",
            bounds=[Rect(0, 0, 5, 5)],
            connections={1: (4, 2)},
        ),
        1: MapRegion(
            id=1,
            region_type="hallway",
            bounds=[Rect(5, 0, 5, 10)],
            connections={0: (5, 2), 2: (9, 2), 3: (7, 9)},
        ),
        2: MapRegion(
            id=2,
            region_type="room",
            bounds=[Rect(10, 0, 5, 5)],
            connections={1: (10, 2)},
        ),
        3: MapRegion(
            id=3,
            region_type="room",
            bounds=[Rect(5, 10, 5, 5)],
            connections={1: (7, 10)},
        ),
    }


def make_disconnected_regions() -> dict[int, MapRegion]:
    """Create two disconnected region groups: {0, 1} and {2}."""
    return {
        0: MapRegion(
            id=0,
            region_type="room",
            bounds=[Rect(0, 0, 5, 5)],
            connections={1: (4, 2)},
        ),
        1: MapRegion(
            id=1,
            region_type="room",
            bounds=[Rect(5, 0, 5, 5)],
            connections={0: (5, 2)},
        ),
        2: MapRegion(
            id=2,
            region_type="room",
            bounds=[Rect(20, 0, 5, 5)],
            connections={},  # No connections - isolated
        ),
    }


@pytest.fixture
def linear_map() -> GameMap:
    """Create a map with 3 linearly connected regions."""
    width, height = 15, 5
    tiles = np.full((width, height), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions = make_connected_regions()

    # Build tile_to_region_id mapping
    tile_to_region_id = np.full((width, height), -1, dtype=np.int16, order="F")
    for region_id, region in regions.items():
        for rect in region.bounds:
            for x in range(rect.x1, rect.x2):
                for y in range(rect.y1, rect.y2):
                    if 0 <= x < width and 0 <= y < height:
                        tile_to_region_id[x, y] = region_id

    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=tile_to_region_id,
    )
    gm = GameMap(width, height, map_data)
    gm.invalidate_property_caches()
    return gm


@pytest.fixture
def branching_map() -> GameMap:
    """Create a map with branching region connections."""
    width, height = 15, 15
    tiles = np.full((width, height), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions = make_branching_regions()

    tile_to_region_id = np.full((width, height), -1, dtype=np.int16, order="F")
    for region_id, region in regions.items():
        for rect in region.bounds:
            for x in range(rect.x1, rect.x2):
                for y in range(rect.y1, rect.y2):
                    if 0 <= x < width and 0 <= y < height:
                        tile_to_region_id[x, y] = region_id

    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=tile_to_region_id,
    )
    gm = GameMap(width, height, map_data)
    gm.invalidate_property_caches()
    return gm


@pytest.fixture
def disconnected_map() -> GameMap:
    """Create a map with disconnected regions."""
    width, height = 25, 5
    tiles = np.full((width, height), TileTypeID.FLOOR, dtype=np.uint8, order="F")
    regions = make_disconnected_regions()

    tile_to_region_id = np.full((width, height), -1, dtype=np.int16, order="F")
    for region_id, region in regions.items():
        for rect in region.bounds:
            for x in range(rect.x1, rect.x2):
                for y in range(rect.y1, rect.y2):
                    if 0 <= x < width and 0 <= y < height:
                        tile_to_region_id[x, y] = region_id

    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=tile_to_region_id,
    )
    gm = GameMap(width, height, map_data)
    gm.invalidate_property_caches()
    return gm


# --- find_region_path tests ---


class TestFindRegionPath:
    """Tests for the high-level region graph pathfinding."""

    def test_same_region_returns_single_element(self, linear_map: GameMap) -> None:
        """Path within the same region returns just that region."""
        result = find_region_path(linear_map, 0, 0)
        assert result == [0]

    def test_adjacent_regions_returns_both(self, linear_map: GameMap) -> None:
        """Path between adjacent regions returns both."""
        result = find_region_path(linear_map, 0, 1)
        assert result == [0, 1]

    def test_multi_hop_path(self, linear_map: GameMap) -> None:
        """Path through multiple regions returns full sequence."""
        result = find_region_path(linear_map, 0, 2)
        assert result == [0, 1, 2]

    def test_reverse_path(self, linear_map: GameMap) -> None:
        """Path works in reverse direction."""
        result = find_region_path(linear_map, 2, 0)
        assert result == [2, 1, 0]

    def test_branching_path_to_branch(self, branching_map: GameMap) -> None:
        """Path to a branch region uses shortest route."""
        result = find_region_path(branching_map, 0, 3)
        assert result == [0, 1, 3]

    def test_branching_path_across_branches(self, branching_map: GameMap) -> None:
        """Path between branches goes through hub."""
        result = find_region_path(branching_map, 2, 3)
        assert result == [2, 1, 3]

    def test_disconnected_regions_returns_none(self, disconnected_map: GameMap) -> None:
        """No path between disconnected regions."""
        result = find_region_path(disconnected_map, 0, 2)
        assert result is None

    def test_connected_in_island(self, disconnected_map: GameMap) -> None:
        """Path exists within connected island."""
        result = find_region_path(disconnected_map, 0, 1)
        assert result == [0, 1]

    def test_nonexistent_start_region(self, linear_map: GameMap) -> None:
        """Returns None for nonexistent start region."""
        result = find_region_path(linear_map, 999, 0)
        assert result is None

    def test_nonexistent_end_region(self, linear_map: GameMap) -> None:
        """Returns None for nonexistent end region."""
        result = find_region_path(linear_map, 0, 999)
        assert result is None


class TestRegionLookup:
    """Tests for region lookup functionality used by HPA*."""

    def test_get_region_at_returns_correct_region(self, linear_map: GameMap) -> None:
        """get_region_at returns the correct region for a tile."""
        # Tile at (2, 2) should be in region 0
        region = linear_map.get_region_at((2, 2))
        assert region is not None
        assert region.id == 0

        # Tile at (7, 2) should be in region 1
        region = linear_map.get_region_at((7, 2))
        assert region is not None
        assert region.id == 1

        # Tile at (12, 2) should be in region 2
        region = linear_map.get_region_at((12, 2))
        assert region is not None
        assert region.id == 2

    def test_tile_to_region_id_mapping(self, linear_map: GameMap) -> None:
        """tile_to_region_id gives O(1) lookup for region membership."""
        assert linear_map.tile_to_region_id[2, 2] == 0
        assert linear_map.tile_to_region_id[7, 2] == 1
        assert linear_map.tile_to_region_id[12, 2] == 2

    def test_connection_points_are_valid(self, linear_map: GameMap) -> None:
        """Region connection points are walkable tiles."""
        for region_id, region in linear_map.regions.items():
            for neighbor_id, conn_pos in region.connections.items():
                x, y = conn_pos
                assert linear_map.walkable[x, y], (
                    f"Connection from region {region_id} to {neighbor_id} "
                    f"at {conn_pos} is not walkable"
                )
