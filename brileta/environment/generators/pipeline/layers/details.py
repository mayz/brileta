"""Detail layer for adding environmental decoration placement.

This layer currently records boulder spawn positions for later actor
materialization. The underlying terrain tile remains unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from brileta import config
from brileta.environment.generators.pipeline.context import GenerationContext
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.tile_types import TileTypeID
from brileta.types import WorldTilePos
from brileta.util import rng
from brileta.util.noise import FractalType, NoiseGenerator, NoiseType

_rng = rng.get("map.details")


class DetailLayer(GenerationLayer):
    """Adds environmental details like boulders to outdoor areas.

    Boulder placement is split into two passes:

    - Wild boulders: higher density far from streets.
    - Settlement boulders: very sparse density near streets.

    Boulders are never placed on street tiles and avoid building exclusion
    zones and existing tree/boulder reservation coordinates.

    Features:
    - Street-aware boulder distribution
    - Distinct settlement and wilderness densities
    - Future: vegetation, debris, etc.
    """

    def __init__(
        self,
        wild_density: float = config.SETTLEMENT_WILD_BOULDER_DENSITY,
        settlement_density: float = config.SETTLEMENT_SETTLEMENT_BOULDER_DENSITY,
        min_distance_from_buildings: int = 2,
        street_buffer: int = config.SETTLEMENT_BOULDER_STREET_BUFFER,
    ) -> None:
        """Initialize the detail layer.

        Args:
            wild_density: Probability of placing a boulder on eligible wild tiles.
            settlement_density: Probability on eligible settlement tiles.
            min_distance_from_buildings: Minimum distance from buildings for details.
            street_buffer: Distance from streets considered part of settlement.
        """
        self.wild_density = wild_density
        self.settlement_density = settlement_density
        self.min_distance_from_buildings = min_distance_from_buildings
        self.street_buffer = street_buffer

    def apply(self, ctx: GenerationContext) -> None:
        """Add details to the map.

        Args:
            ctx: The generation context to modify.
        """
        # Low-frequency noise creates natural boulder bands and clearings
        # without changing global average density too aggressively.
        density_noise = NoiseGenerator(
            seed=_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=config.BOULDER_DENSITY_NOISE_FREQUENCY,
            fractal_type=FractalType.FBM,
            octaves=config.BOULDER_DENSITY_NOISE_OCTAVES,
        )

        building_tiles = ctx.collect_building_tiles()
        street_tiles = ctx.collect_street_tiles()

        # Keep boulders away from walls/doors so settlement flow and entries
        # remain clear.
        building_buffer = ctx.expand_tiles(
            building_tiles, self.min_distance_from_buildings
        )

        # Tiles within this radius of streets are treated as "settlement".
        settlement_zone = ctx.expand_tiles(street_tiles, self.street_buffer)

        outdoor_tiles = {
            TileTypeID.COBBLESTONE,
            TileTypeID.GRASS,
            TileTypeID.DIRT,
            TileTypeID.GRAVEL,
        }

        # Start with existing decorative actors so no two actors overlap.
        reserved_positions: set[WorldTilePos] = set(ctx.tree_positions) | set(
            ctx.boulder_positions
        )

        # Pass 1: wild boulders (outside settlement zone).
        self._place_boulders(
            ctx=ctx,
            density=self.wild_density,
            density_noise=density_noise,
            outdoor_tiles=outdoor_tiles,
            reserved_positions=reserved_positions,
            eligible=lambda pos: (
                pos not in settlement_zone
                and pos not in building_buffer
                and pos not in street_tiles
            ),
        )

        # Pass 2: settlement boulders (sparse, near streets but never on them).
        self._place_boulders(
            ctx=ctx,
            density=self.settlement_density,
            density_noise=density_noise,
            outdoor_tiles=outdoor_tiles,
            reserved_positions=reserved_positions,
            eligible=lambda pos: (
                pos in settlement_zone
                and pos not in building_buffer
                and pos not in street_tiles
            ),
        )

    def _place_boulders(
        self,
        ctx: GenerationContext,
        density: float,
        density_noise: NoiseGenerator,
        outdoor_tiles: set[TileTypeID],
        reserved_positions: set[WorldTilePos],
        eligible: Callable[[WorldTilePos], bool],
    ) -> None:
        """Place boulders on valid outdoor tiles using noise-modulated density.

        Args:
            ctx: The generation context to modify.
            density: Per-tile placement chance for this pass.
            density_noise: Noise generator used to modulate local density.
            outdoor_tiles: Terrain IDs where boulders may appear.
            reserved_positions: Global set of occupied decoration coordinates.
            eligible: Additional zone/exclusion predicate.
        """
        if density <= 0:
            return

        for x in range(ctx.width):
            for y in range(ctx.height):
                pos: WorldTilePos = (x, y)

                # Only place on outdoor tiles
                if ctx.tiles[x, y] not in outdoor_tiles:
                    continue

                # Keep boulders from overlapping other decoration actors.
                if pos in reserved_positions:
                    continue

                if not eligible(pos):
                    continue

                noise_val = density_noise.sample(float(x), float(y))
                density_multiplier = self._density_multiplier(noise_val)

                # Random chance to place boulder
                if _rng.random() < density * density_multiplier:
                    ctx.boulder_positions.append(pos)
                    reserved_positions.add(pos)

    @staticmethod
    def _density_multiplier(noise_val: float) -> float:
        """Map noise in [-1, 1] to a subtle density multiplier in [0.5, 1.5]."""
        return max(0.5, min(1.5, 1.0 + noise_val * 0.5))
