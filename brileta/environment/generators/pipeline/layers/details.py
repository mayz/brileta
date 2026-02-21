"""Detail layer for adding environmental decoration placement.

This layer currently records boulder spawn positions for later actor
materialization. The underlying terrain tile remains unchanged.
"""

from __future__ import annotations

from brileta.environment.generators.pipeline.context import GenerationContext
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.tile_types import TileTypeID
from brileta.types import WorldTilePos
from brileta.util import rng

_rng = rng.get("map.details")


class DetailLayer(GenerationLayer):
    """Adds environmental details like boulders to outdoor areas.

    This layer records boulder actor spawn positions across outdoor terrain
    while avoiding buildings and existing tree placements.

    Features:
    - Boulder actor spawn positions in outdoor areas
    - Future: vegetation, debris, etc.
    """

    def __init__(
        self,
        boulder_density: float = 0.02,
        min_distance_from_buildings: int = 2,
    ) -> None:
        """Initialize the detail layer.

        Args:
            boulder_density: Probability of placing a boulder on each valid tile.
            min_distance_from_buildings: Minimum distance from buildings for details.
        """
        self.boulder_density = boulder_density
        self.min_distance_from_buildings = min_distance_from_buildings

    def apply(self, ctx: GenerationContext) -> None:
        """Add details to the map.

        Args:
            ctx: The generation context to modify.
        """
        # Build a set of tiles near buildings to avoid
        avoided_tiles: set[WorldTilePos] = set()

        for building in ctx.buildings:
            fp = building.footprint
            # Expand footprint by min_distance
            for x in range(
                fp.x1 - self.min_distance_from_buildings,
                fp.x2 + self.min_distance_from_buildings,
            ):
                for y in range(
                    fp.y1 - self.min_distance_from_buildings,
                    fp.y2 + self.min_distance_from_buildings,
                ):
                    avoided_tiles.add((x, y))

        # Place boulders on outdoor tiles.
        outdoor_tiles = {
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT_PATH,
            TileTypeID.GRAVEL,
        }
        # Seed with existing tree positions so boulders never land on a tree.
        # (boulder_positions is empty when DetailLayer runs; new positions are
        # added incrementally below as each boulder is chosen.)
        reserved_positions: set[WorldTilePos] = set(ctx.tree_positions)

        for x in range(ctx.width):
            for y in range(ctx.height):
                pos: WorldTilePos = (x, y)

                # Skip if near buildings
                if pos in avoided_tiles:
                    continue

                # Only place on outdoor tiles
                if ctx.tiles[x, y] not in outdoor_tiles:
                    continue

                # Keep boulders from overlapping other decoration actors.
                if pos in reserved_positions:
                    continue

                # Random chance to place boulder
                if _rng.random() < self.boulder_density:
                    ctx.boulder_positions.append(pos)
                    reserved_positions.add(pos)
