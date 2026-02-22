"""Terrain generation layers.

These layers handle the base terrain setup and natural landscape:
- OpenFieldLayer: Creates a flat outdoor area with a single region
- NaturalTerrainLayer: Generates the pre-settlement natural landscape using noise
- RandomTerrainLayer: Fallback that adds variety using simple weighted random selection

The natural terrain represents what the land looked like *before* any settlement
was built. Dirt is the base, with two noise fields layered together: a low-frequency
field for large grass regions, and a high-frequency field for small isolated patches
(microhabitat pockets). Gravel is not placed by this layer - it's a transitional
material added by feature layers (street margins, future riverbeds, cliff bases).
Streets and buildings are carved into this terrain by later layers, reflecting the
causal history: terrain first, then human infrastructure.

Future terrain types this layer could gain:
- Water features (ponds, streams) via a dedicated water noise field
- Exposed rock/bedrock patches (noting boulders already exist as actors,
  and rock terrain could inform their placement density)

TODO: Implement CellularAutomataTerrainLayer for wilderness maps.
Cellular automata creates organic terrain (caves, rubble fields, rocky outcroppings)
using iterative neighbor-counting rules. Good for wilderness/ruins maps where
smooth noise-based terrain isn't appropriate.

Algorithm sketch:
1. Initialize grid with random noise (initial_density % walls)
2. For each iteration:
   - Count wall neighbors in 8-directional Moore neighborhood
   - Cell becomes wall if neighbors >= birth_limit
   - Cell becomes floor if neighbors < death_limit
3. Flood-fill to ensure connectivity

Tuning guide:
- initial_density=0.45, iterations=4 -> balanced caves
- initial_density=0.35, iterations=5 -> more open areas
- initial_density=0.55, iterations=3 -> tighter, more enclosed
"""

from __future__ import annotations

import numpy as np

from brileta import config
from brileta.environment.generators.pipeline.context import GenerationContext
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.map import MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect
from brileta.util.noise import FractalType, NoiseGenerator, NoiseType

_random_terrain_rng = rng.get("map.random_terrain")
_natural_terrain_rng = rng.get("map.natural_terrain")


class OpenFieldLayer(GenerationLayer):
    """Creates a flat outdoor field as the base terrain.

    This layer fills the entire map with COBBLESTONE tiles and creates
    a single outdoor region covering the whole map. This provides a base
    for subsequent layers to build upon.

    The outdoor region has full sky exposure (1.0) for sunlight effects.
    """

    def apply(self, ctx: GenerationContext) -> None:
        """Fill the map with cobblestone and create an outdoor region.

        Args:
            ctx: The generation context to modify.
        """
        # Fill entire map with cobblestone as a neutral starting tile.
        # NaturalTerrainLayer will overwrite these with actual terrain.
        ctx.tiles[:, :] = TileTypeID.COBBLESTONE

        # Create a single outdoor region covering the entire map
        region_id = ctx.next_region_id()
        outdoor_region = MapRegion.create_outdoor_region(
            map_region_id=region_id,
            region_type="exterior",
            bounds=[Rect(0, 0, ctx.width, ctx.height)],
            sky_exposure=1.0,
        )
        ctx.add_region(outdoor_region)

        # Assign all tiles to this region
        ctx.tile_to_region_id[:, :] = region_id


class RandomTerrainLayer(GenerationLayer):
    """Adds visual terrain variety using simple weighted random selection.

    This layer modifies outdoor floor tiles to create terrain variety
    using pure random selection with configurable weights. It does NOT
    use noise or any adjacency constraints.

    Primarily used as a fallback when other terrain generation fails.

    Note: This layer only changes tile appearances - it does NOT modify
    regions. Buildings placed later will overwrite terrain tiles.
    """

    def __init__(
        self,
        grass_weight: float = 0.5,
        path_weight: float = 0.3,
        gravel_weight: float = 0.2,
    ) -> None:
        """Initialize the terrain layer with tile weights.

        Args:
            grass_weight: Probability weight for GRASS tiles.
            path_weight: Probability weight for DIRT tiles.
            gravel_weight: Probability weight for GRAVEL tiles.
        """
        self.grass_weight = grass_weight
        self.path_weight = path_weight
        self.gravel_weight = gravel_weight

    def apply(self, ctx: GenerationContext) -> None:
        """Apply terrain variety to outdoor tiles using random selection.

        Args:
            ctx: The generation context to modify.
        """
        for x in range(ctx.width):
            for y in range(ctx.height):
                if ctx.tiles[x, y] != TileTypeID.COBBLESTONE:
                    continue

                # Simple terrain variety based on random roll
                roll = _random_terrain_rng.random()

                # Normalize weights
                total = self.grass_weight + self.path_weight + self.gravel_weight

                if roll < self.grass_weight / total:
                    ctx.tiles[x, y] = TileTypeID.GRASS
                elif roll < (self.grass_weight + self.path_weight) / total:
                    ctx.tiles[x, y] = TileTypeID.DIRT
                elif roll < 1.0:
                    ctx.tiles[x, y] = TileTypeID.GRAVEL
                # else keep as COBBLESTONE


# =============================================================================
# Noise-based Natural Terrain
# =============================================================================


class NaturalTerrainLayer(GenerationLayer):
    """Generates the pre-settlement natural landscape using noise fields.

    This represents what the land looked like before anyone built anything.
    The base terrain is dirt, with two noise fields layered together:

    - **Grass noise**: low-frequency, smooth patches of grass over dirt.
      These are the large-scale terrain regions - meadows, clearings, etc.
    - **Island noise**: high-frequency, high-threshold noise that creates
      small 1-3 tile patches. Asymmetric thresholds reflect ecology: grass
      colonizes dirt easily (seeds blow in, favorable pockets) so grass-in-
      dirt islands are common. Bare dirt surviving inside grass is rarer
      (requires something actively preventing growth) so the threshold for
      grass-to-dirt flips is higher.

    Gravel is not placed by this layer. It's a transitional material added
    by feature layers (street margins, future riverbeds, cliff bases).

    Later layers (streets, buildings) carve into this terrain, reflecting
    the causal history: natural terrain existed first, then the settlement
    was built on top of it.
    """

    def __init__(
        self,
        grass_frequency: float = config.GRASS_NOISE_FREQUENCY,
        grass_octaves: int = config.GRASS_NOISE_OCTAVES,
        grass_threshold: float = config.GRASS_NOISE_THRESHOLD,
        island_frequency: float = config.GRASS_ISLAND_FREQUENCY,
        island_octaves: int = config.GRASS_ISLAND_OCTAVES,
        island_threshold: float = config.GRASS_ISLAND_THRESHOLD,
        island_bare_threshold: float = config.GRASS_ISLAND_BARE_THRESHOLD,
    ) -> None:
        """Initialize the natural terrain layer.

        Args:
            grass_frequency: Noise frequency for large grass patches.
            grass_octaves: FBM octaves for grass noise.
            grass_threshold: Noise values above this become grass.
            island_frequency: Noise frequency for small island patches.
                Higher values produce smaller features.
            island_octaves: FBM octaves for island noise.
            island_threshold: Noise peaks above this flip dirt to grass.
            island_bare_threshold: Noise peaks above this flip grass to
                bare dirt. Higher than island_threshold because grass
                colonizes dirt more easily than dirt persists in grass.
        """
        self.grass_frequency = grass_frequency
        self.grass_octaves = grass_octaves
        self.grass_threshold = grass_threshold
        self.island_frequency = island_frequency
        self.island_octaves = island_octaves
        self.island_threshold = island_threshold
        self.island_bare_threshold = island_bare_threshold

    def apply(self, ctx: GenerationContext) -> None:
        """Generate natural terrain over all outdoor tiles.

        All operations are vectorized with numpy and the native noise
        batch sampler. The steps:

        1. Fill all COBBLESTONE tiles with DIRT (the base terrain).
        2. Sample grass noise and place grass where noise exceeds threshold.
        3. Sample island noise and flip tiles where peaks exceed the island
           threshold, creating small isolated patches in terrain interiors.

        Args:
            ctx: The generation context to modify.
        """
        tiles = ctx.tiles

        # Step 1: fill all cobblestone (the OpenFieldLayer placeholder) with
        # dirt as the base natural terrain.
        outdoor_mask = tiles == TileTypeID.COBBLESTONE
        tiles[outdoor_mask] = TileTypeID.DIRT

        # Build coordinate arrays for all outdoor tiles (now dirt).
        xs, ys = np.where(outdoor_mask)
        if len(xs) == 0:
            return
        xs_f = xs.astype(np.float32)
        ys_f = ys.astype(np.float32)

        # Step 2: place grass where noise exceeds threshold.
        grass_noise = NoiseGenerator(
            seed=_natural_terrain_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=self.grass_frequency,
            fractal_type=FractalType.FBM,
            octaves=self.grass_octaves,
        )
        grass_vals = grass_noise.sample_array(xs_f, ys_f)
        grass_mask = grass_vals > self.grass_threshold
        tiles[xs[grass_mask], ys[grass_mask]] = TileTypeID.GRASS

        # Step 3: island noise pass. A separate high-frequency noise field
        # whose peaks flip tiles to the opposite type, creating small organic
        # patches. Asymmetric thresholds: grass colonizes dirt easily (lower
        # threshold) but bare dirt persisting in grass is rarer (higher
        # threshold), reflecting real ecology.
        island_noise = NoiseGenerator(
            seed=_natural_terrain_rng.getrandbits(32),
            noise_type=NoiseType.OPENSIMPLEX2,
            frequency=self.island_frequency,
            fractal_type=FractalType.FBM,
            octaves=self.island_octaves,
        )
        island_vals = island_noise.sample_array(xs_f, ys_f)

        # Apply per-tile with direction-dependent thresholds.
        is_dirt = tiles[xs, ys] == TileTypeID.DIRT
        is_grass = tiles[xs, ys] == TileTypeID.GRASS

        # Dirt tiles flip to grass at the lower threshold (grass colonizes).
        grass_islands = is_dirt & (island_vals > self.island_threshold)
        tiles[xs[grass_islands], ys[grass_islands]] = TileTypeID.GRASS

        # Grass tiles flip to dirt only at the higher threshold (rarer).
        bare_islands = is_grass & (island_vals > self.island_bare_threshold)
        tiles[xs[bare_islands], ys[bare_islands]] = TileTypeID.DIRT
