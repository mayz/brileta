from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from catley.environment import tile_types
from catley.util.coordinates import Rect, TileCoord

if TYPE_CHECKING:
    from catley.game.game_world import GameWorld

    from .generators import GeneratedMapData


@dataclass
class MapRegion:
    """Represents a single logical area of the map, like a room or hallway."""

    id: int
    region_type: str  # e.g., "room", "hallway"
    bounds: list[Rect] = field(default_factory=list)
    # Mapping of neighbor_region_id -> (x, y) door tile connecting them
    connections: dict[int, tuple[int, int]] = field(default_factory=dict)


class GameMap:
    """The game map."""

    def __init__(
        self, width: TileCoord, height: TileCoord, map_data: GeneratedMapData
    ) -> None:
        self.width: TileCoord = width
        self.height: TileCoord = height
        self.revision: int = 0
        self.gw: GameWorld | None = None

        # Directly use the generated tile map and region data.
        self.tiles = map_data.tiles
        self.regions = map_data.regions
        self.tile_to_region_id = map_data.tile_to_region_id

        # Field of view (FOV) state tracking.
        # Which tiles are currently visible.
        self.visible = np.full((width, height), fill_value=False, dtype=bool, order="F")

        # Which tiles have been previously explored
        # (whether they're currently visible or not).
        self.explored = np.full(
            (width, height), fill_value=False, dtype=bool, order="F"
        )

        # Cached property arrays for performance.
        # These are populated on-demand by the respective properties.
        self._walkable_map_cache: np.ndarray | None = None
        self._transparent_map_cache: np.ndarray | None = None
        self._dark_appearance_map_cache: np.ndarray | None = None
        self._light_appearance_map_cache: np.ndarray | None = None

    def invalidate_property_caches(self) -> None:
        """Call this whenever `self.tiles` changes to clear cached property maps."""
        self._walkable_map_cache = None
        self._transparent_map_cache = None
        self._dark_appearance_map_cache = None
        self._light_appearance_map_cache = None
        self.revision += 1

    @property
    def walkable(self) -> np.ndarray:
        """Boolean array of shape (width, height) where True means tile is walkable."""
        if self._walkable_map_cache is None:
            self._walkable_map_cache = tile_types.get_walkable_map(self.tiles)
        return self._walkable_map_cache

    @property
    def transparent(self) -> np.ndarray:
        """Boolean array of shape (width, height) where True means tile is transparent
        (for FOV)."""
        if self._transparent_map_cache is None:
            self._transparent_map_cache = tile_types.get_transparent_map(self.tiles)
        return self._transparent_map_cache

    @property
    def dark_appearance_map(self) -> np.ndarray:
        """A map of 'dark' TileTypeAppearance structs derived from self.tiles."""
        if self._dark_appearance_map_cache is None:
            self._dark_appearance_map_cache = tile_types.get_dark_appearance_map(
                self.tiles
            )
        return self._dark_appearance_map_cache

    @property
    def light_appearance_map(self) -> np.ndarray:
        """A map of 'light' TileTypeAppearance structs derived from self.tiles."""
        if self._light_appearance_map_cache is None:
            self._light_appearance_map_cache = tile_types.get_light_appearance_map(
                self.tiles
            )
        return self._light_appearance_map_cache
