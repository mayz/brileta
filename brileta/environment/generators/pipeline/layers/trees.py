"""Tree placement layer for adding trees to settlements.

Places trees in two distinct passes to create natural-looking settlements:

1. **Wild trees** fill open land far from streets and buildings, creating
   dense forested borders around the settlement. These use higher density
   and cluster naturally.

2. **Settlement trees** are sparser and placed deliberately near buildings
   and along streets - yard trees, hedgerows, and landscaping that make
   the built environment feel lived-in.

A low-frequency noise field modulates placement density per tile, creating
organic groves and clearings instead of uniform random scatter.

The split creates the visual effect of dense tree borders surrounding a
more open, deliberately planted settlement interior.
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

_rng = rng.get("map.trees")

# Terrain types where trees can grow.
_PLANTABLE_TERRAIN: set[int] = {
    TileTypeID.GRASS,
    TileTypeID.DIRT,
    TileTypeID.GRAVEL,
}


class TreePlacementLayer(GenerationLayer):
    """Places trees in wild areas and around settlement buildings.

    Two placement passes create distinct wild and settlement tree zones:

    - Wild trees: dense clusters on grass/dirt far from streets and buildings.
    - Settlement trees: sparse, deliberate placement near buildings and streets
      (yard trees, hedgerows, property-line trees).

    Trees are only placed on plantable outdoor terrain (grass, dirt, gravel)
    and never on streets, inside buildings, or on reserved boulder tiles.
    """

    def __init__(
        self,
        wild_density: float = config.SETTLEMENT_WILD_TREE_DENSITY,
        yard_density: float = config.SETTLEMENT_YARD_TREE_DENSITY,
        min_distance_from_buildings: int = 2,
        street_buffer: int = 12,
    ) -> None:
        """Initialize the tree placement layer.

        Args:
            wild_density: Probability of placing a tree on each eligible wild tile.
            yard_density: Probability of placing a tree on each eligible settlement tile.
            min_distance_from_buildings: Minimum distance from building walls for
                wild trees. Settlement trees use a tighter 1-tile buffer.
            street_buffer: Distance from streets that separates "wild" from
                "settlement" zones. Tiles farther than this from any street
                are considered wild.
        """
        self.wild_density = wild_density
        self.yard_density = yard_density
        self.min_distance_from_buildings = min_distance_from_buildings
        self.street_buffer = street_buffer

    def apply(self, ctx: GenerationContext) -> None:
        """Place trees on the map in wild and settlement zones.

        Args:
            ctx: The generation context to modify.
        """
        # Create density noise field for grove/clearing modulation.
        # Seed is derived from the tree RNG stream so it's deterministic
        # from the master seed but independent of placement roll order.
        density_noise = NoiseGenerator(
            seed=_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=config.TREE_DENSITY_NOISE_FREQUENCY,
            fractal_type=FractalType.FBM,
            octaves=config.TREE_DENSITY_NOISE_OCTAVES,
        )

        # Pre-compute tile sets for fast lookup during placement.
        building_tiles = ctx.collect_building_tiles()
        street_tiles = ctx.collect_street_tiles()

        # Building exclusion zone: tiles within min_distance of any building wall.
        building_buffer = ctx.expand_tiles(
            building_tiles, self.min_distance_from_buildings
        )

        # Street proximity zone: tiles within street_buffer of any street.
        # Tiles inside this zone are "settlement"; tiles outside are "wild".
        settlement_zone = ctx.expand_tiles(street_tiles, self.street_buffer)

        # Tight building adjacency zone for settlement trees (1 tile from walls).
        # Settlement trees can be placed here but wild trees cannot.
        building_adjacent = ctx.expand_tiles(building_tiles, 1)

        # Pass 1: Wild trees (dense, outside settlement zone)
        self._place_trees(
            ctx,
            density=self.wild_density,
            density_noise=density_noise,
            eligible=lambda pos: (
                pos not in settlement_zone
                and pos not in building_buffer
                and pos not in street_tiles
            ),
        )

        # Pass 2: Settlement trees (sparse, inside settlement zone)
        self._place_trees(
            ctx,
            density=self.yard_density,
            density_noise=density_noise,
            eligible=lambda pos: (
                pos in settlement_zone
                and pos not in building_tiles
                and pos not in building_adjacent
                and pos not in street_tiles
            ),
        )

    def _place_trees(
        self,
        ctx: GenerationContext,
        density: float,
        density_noise: NoiseGenerator,
        eligible: Callable[[WorldTilePos], bool],
    ) -> None:
        """Place trees on eligible tiles with noise-modulated density.

        The base density is scaled per tile by a noise field so that high-noise
        areas get up to 3x density (dense groves) and low-noise areas drop to
        0 (natural clearings). The 1.5x scaling lifts average density ~50%
        above the base, which is intentional - groves need enough peak density
        to read as clusters at the base rates used in config.

        Args:
            ctx: The generation context to modify.
            density: Base probability of placing a tree on each eligible tile.
            density_noise: Noise generator for spatial density modulation.
            eligible: Callable that takes a WorldTilePos and returns True
                if a tree can be placed there.
        """
        existing_positions = set(ctx.tree_positions) | set(ctx.boulder_positions)
        for x in range(ctx.width):
            for y in range(ctx.height):
                if ctx.tiles[x, y] not in _PLANTABLE_TERRAIN:
                    continue

                pos: WorldTilePos = (x, y)
                if pos in existing_positions:
                    continue
                if not eligible(pos):
                    continue

                # Sample noise in [-1, 1] and remap to a density multiplier
                # in [0, 3]. The +1 shift centers the range at 1.5x, but
                # the extra headroom lets grove peaks reach 3x base density
                # while clearings still bottom out at 0.
                noise_val = density_noise.sample(float(x), float(y))
                density_multiplier = max(0.0, (noise_val + 1.0) * 1.5)

                if _rng.random() < density * density_multiplier:
                    ctx.tree_positions.append(pos)
                    existing_positions.add(pos)
