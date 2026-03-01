"""Generation context for the pipeline map generator.

The GenerationContext is a mutable container that holds all state during map
generation. Each layer in the pipeline receives the same context and modifies
it in place. This avoids copying large numpy arrays between layers.

Layers should use rng.get("map.<layer>") to get isolated RNG streams.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from brileta.environment.generators.base import GeneratedMapData
from brileta.environment.map import MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.types import WorldTilePos
from brileta.util import rng
from brileta.util.coordinates import Rect

if TYPE_CHECKING:
    from brileta.environment.generators.buildings.building import Building


@dataclass
class StreetData:
    """Data about the street network, shared between layers.

    Attributes:
        streets: List of Rect objects representing street areas.
        zones: List of Rect objects representing building zones adjacent to streets.
        style: Street layout style ("single", "cross", "grid").
    """

    streets: list[Rect] = field(default_factory=list)
    zones: list[Rect] = field(default_factory=list)
    style: str = "single"


@dataclass
class GenerationContext:
    """Mutable state container passed through the generation pipeline.

    Each layer receives this context and modifies it in place. The context
    contains all the data needed to generate a map, including the tile array,
    regions, and building objects.

    Attributes:
        width: Map width in tiles.
        height: Map height in tiles.
        tiles: 2D numpy array of TileTypeID values. Shape: (width, height).
        regions: Dictionary mapping region IDs to MapRegion objects.
        tile_to_region_id: 2D numpy array mapping each tile to its region ID.
            Value of -1 means no region assigned. Shape: (width, height).
        buildings: List of Building objects placed on the map.
        street_data: Street network data for building placement and door orientation.
        tree_positions: World positions where tree actors should be spawned.
        boulder_positions: World positions where boulder actors should be spawned.
    """

    width: int
    height: int
    tiles: np.ndarray
    regions: dict[int, MapRegion]
    tile_to_region_id: np.ndarray
    buildings: list[Building] = field(default_factory=list)
    street_data: StreetData = field(default_factory=StreetData)
    tree_positions: list[WorldTilePos] = field(default_factory=list)
    boulder_positions: list[WorldTilePos] = field(default_factory=list)
    _next_region_id: int = 0

    @classmethod
    def create_empty(
        cls,
        width: int,
        height: int,
        fill_tile: TileTypeID = TileTypeID.WALL,
    ) -> GenerationContext:
        """Create an empty generation context with default values.

        Args:
            width: Map width in tiles.
            height: Map height in tiles.
            fill_tile: Tile type to fill the initial map with.

        Returns:
            A new GenerationContext ready for layer processing.
        """
        tiles = np.full(
            (width, height),
            fill_value=fill_tile,
            dtype=np.uint8,
            order="F",
        )

        tile_to_region_id = np.full(
            (width, height),
            -1,
            dtype=np.int16,
            order="F",
        )

        return cls(
            width=width,
            height=height,
            tiles=tiles,
            regions={},
            tile_to_region_id=tile_to_region_id,
            buildings=[],
            _next_region_id=0,
        )

    def next_region_id(self) -> int:
        """Allocate and return the next available region ID.

        Each call returns a unique ID, incrementing the internal counter.
        """
        region_id = self._next_region_id
        self._next_region_id += 1
        return region_id

    def add_region(self, region: MapRegion) -> None:
        """Add a region to the context's region dictionary.

        Args:
            region: The MapRegion to add.
        """
        self.regions[region.id] = region

    # ------------------------------------------------------------------
    # Tile-set helpers used by multiple generation layers
    # ------------------------------------------------------------------

    def collect_building_tiles(self) -> set[WorldTilePos]:
        """Collect all tiles covered by building shapes."""
        tiles: set[WorldTilePos] = set()
        for building in self.buildings:
            tiles.update(building.tile_set())
        return tiles

    def collect_street_tiles(self) -> set[WorldTilePos]:
        """Collect all tiles covered by street rectangles."""
        map_bounds = Rect(0, 0, self.width, self.height)
        tiles: set[WorldTilePos] = set()
        for street in self.street_data.streets:
            clipped = street.clip(map_bounds)
            if clipped is not None:
                for x in range(clipped.x1, clipped.x2):
                    for y in range(clipped.y1, clipped.y2):
                        tiles.add((x, y))
        return tiles

    def expand_tiles(
        self,
        tiles: set[WorldTilePos],
        radius: int,
    ) -> set[WorldTilePos]:
        """Expand a tile set outward by Chebyshev radius.

        Creates a buffer zone around the input tiles. Used to build exclusion
        zones around buildings and proximity zones around streets.

        Args:
            tiles: Source tile positions to expand from.
            radius: Expansion distance in tiles (Chebyshev/chessboard metric).

        Returns:
            New set containing all tiles within radius of any source tile.
        """
        if radius <= 0 or not tiles:
            return set(tiles)

        # Use numpy for efficient expansion: stamp a boolean grid, dilate,
        # read back.
        grid = np.zeros((self.width, self.height), dtype=bool)
        for x, y in tiles:
            if 0 <= x < self.width and 0 <= y < self.height:
                grid[x, y] = True

        # Cumulative sum dilation: for each axis, smear True values outward
        # by ``radius`` in both directions.  Equivalent to a box filter but
        # avoids scipy/ndimage dependency.
        padded = np.pad(grid.astype(np.int8), radius, mode="constant")
        cs = np.cumsum(padded, axis=0)
        dilated_x = cs[2 * radius :, :] - cs[: -2 * radius, :]
        cs2 = np.cumsum(dilated_x, axis=1)
        dilated = cs2[:, 2 * radius :] - cs2[:, : -2 * radius :]

        expanded_coords = np.argwhere(dilated[: self.width, : self.height] > 0)
        return {(int(x), int(y)) for x, y in expanded_coords}

    def to_generated_map_data(self) -> GeneratedMapData:
        """Convert this context to a GeneratedMapData for use with GameMap.

        Generates a decoration seed from the terrain RNG so per-tile visual
        decoration is deterministic for a given world seed.

        Returns:
            A GeneratedMapData instance containing the final map data.
        """
        _terrain_rng = rng.get("map.terrain_decoration")
        decoration_seed = _terrain_rng.getrandbits(32)

        return GeneratedMapData(
            tiles=self.tiles,
            regions=self.regions,
            tile_to_region_id=self.tile_to_region_id,
            buildings=self.buildings,
            streets=self.street_data.streets,
            tree_positions=self.tree_positions,
            boulder_positions=self.boulder_positions,
            decoration_seed=decoration_seed,
        )
