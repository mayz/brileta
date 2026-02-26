from __future__ import annotations

import numpy as np

from brileta.environment.generators.base import GeneratedMapData
from brileta.environment.map import GameMap, MapRegion
from brileta.environment.tile_types import TileTypeID


def _build_game_map_for_sun_shadow_grid() -> GameMap:
    width = 4
    height = 4
    tiles = np.full((width, height), int(TileTypeID.FLOOR), dtype=np.uint8, order="F")
    tile_to_region_id = np.full((width, height), -1, dtype=np.int16, order="F")

    tile_to_region_id[0:2, :] = 0
    tile_to_region_id[2:4, :] = 1

    regions = {
        0: MapRegion.create_outdoor_region(0, region_type="exterior", sky_exposure=1.0),
        1: MapRegion.create_indoor_region(1, region_type="room", sky_exposure=1.0),
    }

    map_data = GeneratedMapData(
        tiles=tiles,
        regions=regions,
        tile_to_region_id=tile_to_region_id,
        decoration_seed=0,
    )
    return GameMap(width, height, map_data)


def test_sun_shadow_eligibility_grid_is_cached_and_invalidated() -> None:
    game_map = _build_game_map_for_sun_shadow_grid()
    outdoor_region_types = frozenset({"exterior"})
    outdoor_tile_ids = frozenset({int(TileTypeID.GRASS)})

    grid1 = game_map.get_sun_shadow_eligibility_grid(
        outdoor_region_types=outdoor_region_types,
        outdoor_tile_ids=outdoor_tile_ids,
    )
    grid2 = game_map.get_sun_shadow_eligibility_grid(
        outdoor_region_types=outdoor_region_types,
        outdoor_tile_ids=outdoor_tile_ids,
    )

    assert grid2 is grid1
    assert bool(grid1[0, 0]) is True
    assert bool(grid1[3, 0]) is False

    # Structural change: indoor fallback tile becomes outdoor-looking ground.
    game_map.tiles[3, 0] = int(TileTypeID.GRASS)
    game_map.invalidate_property_caches()
    grid3 = game_map.get_sun_shadow_eligibility_grid(
        outdoor_region_types=outdoor_region_types,
        outdoor_tile_ids=outdoor_tile_ids,
    )

    assert grid3 is not grid1
    assert bool(grid3[3, 0]) is True

    # Region metadata change (debug path) should also invalidate.
    game_map.regions[0].sky_exposure = 0.0
    game_map.invalidate_appearance_caches()
    grid4 = game_map.get_sun_shadow_eligibility_grid(
        outdoor_region_types=outdoor_region_types,
        outdoor_tile_ids=outdoor_tile_ids,
    )

    assert grid4 is not grid3
    assert bool(grid4[0, 0]) is False
