"""Generation context for the pipeline map generator.

The GenerationContext is a mutable container that holds all state during map
generation. Each layer in the pipeline receives the same context and modifies
it in place. This avoids copying large numpy arrays between layers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from catley.environment.generators.base import GeneratedMapData
from catley.environment.map import MapRegion
from catley.environment.tile_types import TileTypeID
from catley.types import RandomSeed
from catley.util.coordinates import Rect

if TYPE_CHECKING:
    from catley.environment.generators.buildings.building import Building


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
        rng: Random number generator for deterministic generation.
    """

    width: int
    height: int
    tiles: np.ndarray
    regions: dict[int, MapRegion]
    tile_to_region_id: np.ndarray
    buildings: list[Building] = field(default_factory=list)
    street_data: StreetData = field(default_factory=StreetData)
    rng: random.Random = field(default_factory=random.Random)
    _next_region_id: int = 0

    @classmethod
    def create_empty(
        cls,
        width: int,
        height: int,
        seed: RandomSeed = None,
        fill_tile: TileTypeID = TileTypeID.WALL,
    ) -> GenerationContext:
        """Create an empty generation context with default values.

        Args:
            width: Map width in tiles.
            height: Map height in tiles.
            seed: Optional random seed for deterministic generation.
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

        rng = random.Random(seed)

        return cls(
            width=width,
            height=height,
            tiles=tiles,
            regions={},
            tile_to_region_id=tile_to_region_id,
            buildings=[],
            rng=rng,
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

    def to_generated_map_data(self) -> GeneratedMapData:
        """Convert this context to a GeneratedMapData for use with GameMap.

        Returns:
            A GeneratedMapData instance containing the final map data.
        """
        return GeneratedMapData(
            tiles=self.tiles,
            regions=self.regions,
            tile_to_region_id=self.tile_to_region_id,
            buildings=self.buildings,
            streets=self.street_data.streets,
        )
