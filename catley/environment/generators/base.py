"""Base classes for map generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from catley.environment.map import MapRegion

if TYPE_CHECKING:
    from catley.environment.generators.buildings.building import Building
    from catley.types import TileCoord
    from catley.util.coordinates import Rect


@dataclass
class GeneratedMapData:
    """A container for all raw data produced by a map generator.

    Attributes:
        tiles: 2D numpy array of tile type IDs.
        regions: Dictionary mapping region IDs to MapRegion objects.
        tile_to_region_id: 2D numpy array mapping tiles to region IDs.
        buildings: List of Building objects (for settlement maps).
        streets: List of Rect objects representing street areas (for settlement maps).
    """

    tiles: np.ndarray
    regions: dict[int, MapRegion]
    tile_to_region_id: np.ndarray
    buildings: list[Building] = field(default_factory=list)
    streets: list[Rect] = field(default_factory=list)


class BaseMapGenerator(abc.ABC):
    """Abstract base class for map generation algorithms."""

    def __init__(self, map_width: TileCoord, map_height: TileCoord) -> None:
        self.map_width = map_width
        self.map_height = map_height

    @abc.abstractmethod
    def generate(self) -> GeneratedMapData:
        """Generate the map layout and its structural data."""
        raise NotImplementedError
