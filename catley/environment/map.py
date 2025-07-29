from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from catley.environment import tile_types
from catley.types import WorldTilePos
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
    connections: dict[int, WorldTilePos] = field(default_factory=dict)
    # Sky exposure for global lighting (0.0 = no sky, 1.0 = full sky exposure)
    sky_exposure: float = 0.0

    @classmethod
    def create_outdoor_region(
        cls,
        map_region_id: int,
        region_type: str = "outdoor",
        sky_exposure: float = 1.0,
        **kwargs,
    ) -> MapRegion:
        """Factory for outdoor regions with full sky exposure."""
        return cls(
            id=map_region_id,
            region_type=region_type,
            sky_exposure=sky_exposure,
            **kwargs,
        )

    @classmethod
    def create_indoor_region(
        cls,
        map_region_id: int,
        region_type: str = "indoor",
        sky_exposure: float = 0.0,
        **kwargs,
    ) -> MapRegion:
        """Factory for indoor regions with no sky exposure."""
        return cls(
            id=map_region_id,
            region_type=region_type,
            sky_exposure=sky_exposure,
            **kwargs,
        )


class GameMap:
    """The game map."""

    def __init__(
        self, width: TileCoord, height: TileCoord, map_data: GeneratedMapData
    ) -> None:
        self.width: TileCoord = width
        self.height: TileCoord = height
        self.structural_revision: int = 0
        self.exploration_revision: int = 0
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
        self.structural_revision += 1

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
            self._dark_appearance_map_cache = self._get_region_aware_appearance_map(
                is_light=False
            )
        return self._dark_appearance_map_cache

    @property
    def light_appearance_map(self) -> np.ndarray:
        """A map of 'light' TileTypeAppearance structs derived from self.tiles."""
        if self._light_appearance_map_cache is None:
            self._light_appearance_map_cache = self._get_region_aware_appearance_map(
                is_light=True
            )
        return self._light_appearance_map_cache

    def _get_region_aware_appearance_map(self, is_light: bool) -> np.ndarray:
        """Get appearance map with region-aware background colors for certain tiles."""
        # Start with the base appearance map from tile types
        if is_light:
            appearance_map = tile_types.get_light_appearance_map(self.tiles)
        else:
            appearance_map = tile_types.get_dark_appearance_map(self.tiles)

        # Create a copy so we can modify it
        appearance_map = appearance_map.copy()

        # Apply region-aware background colors for specific tile types
        for y in range(self.height):
            for x in range(self.width):
                tile_id = self.tiles[x, y]

                # Modify boulders and floor tiles - inherit regional ground colors
                # Doors should use their standard tile colors to match surrounding walls
                if tile_id in (
                    tile_types.TILE_TYPE_ID_BOULDER,
                    tile_types.TILE_TYPE_ID_FLOOR,
                ):
                    region = self.get_region_at((x, y))
                    if region:
                        # Boulders and floor tiles inherit ground color from region
                        ground_bg_color = self._get_region_ground_color(
                            region, is_light
                        )

                        # Update background color preserving char and foreground
                        current_appearance = appearance_map[x, y]
                        appearance_map[x, y] = (
                            current_appearance["ch"],
                            current_appearance["fg"],
                            ground_bg_color,
                        )

        return appearance_map

    def _get_region_ground_color(
        self, region: MapRegion, is_light: bool
    ) -> tuple[int, int, int]:
        """Get the appropriate ground background color for a region."""
        from catley import colors

        # Determine if this is an outdoor region based on sky exposure
        is_outdoor = region.sky_exposure > 0.5

        if is_outdoor:
            return (
                colors.OUTDOOR_LIGHT_GROUND if is_light else colors.OUTDOOR_DARK_GROUND
            )
        return colors.LIGHT_GROUND if is_light else colors.DARK_GROUND

    def _get_region_wall_color(
        self, region: MapRegion, is_light: bool
    ) -> tuple[int, int, int]:
        """Get the appropriate wall background color for a region."""
        from catley import colors

        # Determine if this is an outdoor region based on sky exposure
        is_outdoor = region.sky_exposure > 0.5

        if is_outdoor:
            return colors.OUTDOOR_LIGHT_WALL if is_light else colors.OUTDOOR_DARK_WALL
        return colors.LIGHT_WALL if is_light else colors.DARK_WALL

    def get_region_at(self, world_pos: WorldTilePos) -> MapRegion | None:
        """Get the MapRegion at the given world position, if any."""
        x, y = world_pos
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None

        region_id = self.tile_to_region_id[x, y]
        if region_id < 0:  # No region assigned
            return None

        # Look up the region by ID in the dictionary
        return self.regions.get(region_id)

    def invalidate_appearance_caches(self) -> None:
        """Invalidate appearance caches when regions change."""
        self._dark_appearance_map_cache = None
        self._light_appearance_map_cache = None
