"""Base classes for map generation."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from catley.environment.map import MapRegion

if TYPE_CHECKING:
    from catley.types import TileCoord


@dataclass
class GeneratedMapData:
    """A container for all raw data produced by a map generator."""

    tiles: np.ndarray
    regions: dict[int, MapRegion]
    tile_to_region_id: np.ndarray


class BaseMapGenerator(abc.ABC):
    """Abstract base class for map generation algorithms."""

    def __init__(self, map_width: TileCoord, map_height: TileCoord) -> None:
        self.map_width = map_width
        self.map_height = map_height

    @abc.abstractmethod
    def generate(self) -> GeneratedMapData:
        """Generate the map layout and its structural data."""
        raise NotImplementedError
