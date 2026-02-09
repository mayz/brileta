from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from brileta import colors
from brileta.environment import tile_types
from brileta.environment.tile_types import TileTypeID
from brileta.types import WorldTilePos
from brileta.util.coordinates import Rect, TileCoord

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld

    from .generators import GeneratedMapData

# Per-cell animation state for tiles with dynamic color/glyph effects.
# Values range from 0-1000 and are used to modulate tile colors via random walk.
TileAnimationState = np.dtype(
    [
        ("fg_values", "3h"),  # RGB modulation values for foreground (0-1000)
        ("bg_values", "3h"),  # RGB modulation values for background (0-1000)
        ("show_glyph", np.bool_),  # Whether to show the glyph this frame
    ]
)


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
        self.decoration_seed = map_data.decoration_seed

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
        self._animation_params_cache: np.ndarray | None = None
        self._shadow_heights_map_cache: np.ndarray | None = None

        # Per-cell animation state for tiles that animate (color oscillation, flicker)
        self.animation_state = self._init_animation_state()

    def invalidate_property_caches(self) -> None:
        """Call this whenever `self.tiles` changes to clear cached property maps."""
        self._walkable_map_cache = None
        self._transparent_map_cache = None
        self._dark_appearance_map_cache = None
        self._light_appearance_map_cache = None
        self._animation_params_cache = None
        self._shadow_heights_map_cache = None
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
        """Get appearance map with region-aware background colors for certain tiles.

        Uses vectorized numpy operations: iterates over regions (small dict)
        instead of tiles (large grid).
        """
        # Start with the base appearance map from tile types
        if is_light:
            appearance_map = tile_types.get_light_appearance_map(self.tiles)
        else:
            appearance_map = tile_types.get_dark_appearance_map(self.tiles)

        # Create a copy so we can modify it
        appearance_map = appearance_map.copy()

        # Apply region-aware background colors for specific tile types
        # Boulders and floor tiles inherit ground color from their region.
        # Doors use standard tile colors to match surrounding walls.
        if self.regions:
            # Mask for tiles that need regional coloring
            special_mask = np.isin(self.tiles, [TileTypeID.BOULDER, TileTypeID.FLOOR])

            # Apply colors per-region (iterate small dict, not large grid)
            for region_id, region in self.regions.items():
                # Combine tile type mask with region membership mask
                region_mask = (self.tile_to_region_id == region_id) & special_mask
                if not np.any(region_mask):
                    continue

                # Compute color once per region, apply to all matching tiles
                ground_bg_color = self._get_region_ground_color(region, is_light)
                appearance_map["bg"][region_mask] = ground_bg_color

        return appearance_map

    def _get_region_ground_color(
        self, region: MapRegion, is_light: bool
    ) -> tuple[int, int, int]:
        """Get the appropriate ground background color for a region."""
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

    @property
    def animation_params(self) -> np.ndarray:
        """A map of TileAnimationParams structs derived from self.tiles."""
        if self._animation_params_cache is None:
            self._animation_params_cache = tile_types.get_animation_map(self.tiles)
        return self._animation_params_cache

    @property
    def shadow_heights(self) -> np.ndarray:
        """uint8 array of shape (width, height) with per-tile shadow height.

        0 means the tile casts no shadow. Higher values mean the tile is taller
        and casts a longer shadow (e.g., 4 for walls, 2 for boulders).

        Used by the lighting system for height-aware shadow casting.
        """
        if self._shadow_heights_map_cache is None:
            self._shadow_heights_map_cache = tile_types.get_shadow_height_map(
                self.tiles
            )
        return self._shadow_heights_map_cache

    def _init_animation_state(self) -> np.ndarray:
        """Initialize animation state array with random values.

        Each cell gets random RGB modulation values (0-1000) for both
        foreground and background, and glyph visibility starts as True.
        """
        state = np.zeros((self.width, self.height), dtype=TileAnimationState, order="F")

        # Initialize with random values in the 0-1000 range
        rng = np.random.default_rng()
        state["fg_values"] = rng.integers(0, 1001, size=(self.width, self.height, 3))
        state["bg_values"] = rng.integers(0, 1001, size=(self.width, self.height, 3))
        state["show_glyph"] = True  # Start with all glyphs visible

        return state
