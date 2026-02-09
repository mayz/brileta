"""Terrain generation layers.

These layers handle the base terrain setup and visual variety:
- OpenFieldLayer: Creates a flat outdoor area with a single region
- RandomTerrainLayer: Adds visual variety using simple weighted random selection
- WFCTerrainLayer: Adds visual variety using Wave Function Collapse

TODO: Implement CellularAutomataTerrainLayer for wilderness maps.
Cellular automata creates organic terrain (caves, rubble fields, rocky outcroppings)
using iterative neighbor-counting rules. Good for wilderness/ruins maps where
WFC's smooth transitions aren't needed.

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

from enum import IntEnum, auto

from brileta.environment.generators.pipeline.context import GenerationContext
from brileta.environment.generators.pipeline.layer import GenerationLayer
from brileta.environment.generators.wfc_solver import (
    DIRECTIONS,
    OPPOSITE_DIR,
    WFCContradiction,
    WFCPattern,
    WFCSolver,
)
from brileta.environment.map import MapRegion
from brileta.environment.tile_types import TileTypeID
from brileta.util import rng
from brileta.util.coordinates import Rect

_random_terrain_rng = rng.get("map.random_terrain")
_wfc_terrain_rng = rng.get("map.wfc_terrain")


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
        # Fill entire map with cobblestone
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
    use Wave Function Collapse or any adjacency constraints.

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
            path_weight: Probability weight for DIRT_PATH tiles.
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
                    ctx.tiles[x, y] = TileTypeID.DIRT_PATH
                elif roll < 1.0:
                    ctx.tiles[x, y] = TileTypeID.GRAVEL
                # else keep as COBBLESTONE


# =============================================================================
# WFC Terrain Patterns
# =============================================================================


class TerrainPatternID(IntEnum):
    """Pattern types for WFC terrain generation."""

    GRASS = 0
    DIRT = auto()
    GRAVEL = auto()
    COBBLESTONE = auto()


def create_terrain_patterns() -> dict[TerrainPatternID, WFCPattern[TerrainPatternID]]:
    """Define adjacency rules for natural terrain transitions.

    Adjacency philosophy:
    - GRASS <-> DIRT (natural transition)
    - DIRT <-> GRAVEL (path edges)
    - DIRT <-> COBBLESTONE (path meets road)
    - GRAVEL <-> COBBLESTONE (road edges)
    - GRASS cannot directly touch COBBLESTONE (no abrupt grass-to-road)

    This creates natural-looking transitions where grass blends into dirt,
    dirt into gravel or cobblestone, creating coherent terrain zones.
    """
    patterns: dict[TerrainPatternID, WFCPattern[TerrainPatternID]] = {}

    # GRASS: natural ground, transitions to DIRT only
    # Weight: high (most common)
    patterns[TerrainPatternID.GRASS] = WFCPattern(
        pattern_id=TerrainPatternID.GRASS,
        tile_type=TileTypeID.GRASS,
        weight=4.0,
        valid_neighbors={
            "N": {TerrainPatternID.GRASS, TerrainPatternID.DIRT},
            "E": {TerrainPatternID.GRASS, TerrainPatternID.DIRT},
            "S": {TerrainPatternID.GRASS, TerrainPatternID.DIRT},
            "W": {TerrainPatternID.GRASS, TerrainPatternID.DIRT},
        },
    )

    # DIRT: transition terrain, connects grass to paths/roads
    # Weight: medium-high (common in paths)
    patterns[TerrainPatternID.DIRT] = WFCPattern(
        pattern_id=TerrainPatternID.DIRT,
        tile_type=TileTypeID.DIRT_PATH,
        weight=3.0,
        valid_neighbors={
            "N": {
                TerrainPatternID.GRASS,
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "E": {
                TerrainPatternID.GRASS,
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "S": {
                TerrainPatternID.GRASS,
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "W": {
                TerrainPatternID.GRASS,
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
        },
    )

    # GRAVEL: rough path surface, between dirt and cobblestone
    # Weight: medium (less common than dirt)
    patterns[TerrainPatternID.GRAVEL] = WFCPattern(
        pattern_id=TerrainPatternID.GRAVEL,
        tile_type=TileTypeID.GRAVEL,
        weight=2.0,
        valid_neighbors={
            "N": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "E": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "S": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "W": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
        },
    )

    # COBBLESTONE: paved road, can't touch grass directly
    # Weight: lower (roads are less frequent than natural ground)
    patterns[TerrainPatternID.COBBLESTONE] = WFCPattern(
        pattern_id=TerrainPatternID.COBBLESTONE,
        tile_type=TileTypeID.COBBLESTONE,
        weight=1.5,
        valid_neighbors={
            "N": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "E": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "S": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
            "W": {
                TerrainPatternID.DIRT,
                TerrainPatternID.GRAVEL,
                TerrainPatternID.COBBLESTONE,
            },
        },
    )

    return patterns


def _verify_terrain_patterns_symmetric(
    patterns: dict[TerrainPatternID, WFCPattern[TerrainPatternID]],
) -> None:
    """Verify terrain patterns have symmetric adjacency rules.

    Raises AssertionError if any asymmetry is found. Called during module load
    to catch pattern definition errors early.
    """
    for pattern_id, pattern in patterns.items():
        for direction in DIRECTIONS:
            opposite = OPPOSITE_DIR[direction]
            for neighbor_id in pattern.valid_neighbors[direction]:
                neighbor = patterns[neighbor_id]
                assert pattern_id in neighbor.valid_neighbors[opposite], (
                    f"Asymmetric terrain adjacency: {pattern_id.name} allows "
                    f"{neighbor_id.name} to {direction}, but {neighbor_id.name} "
                    f"does not allow {pattern_id.name} to {opposite}"
                )


# Verify patterns are symmetric at module load time
_verify_terrain_patterns_symmetric(create_terrain_patterns())


# =============================================================================
# WFC Terrain Layer
# =============================================================================


class WFCTerrainLayer(GenerationLayer):
    """Adds terrain variety using Wave Function Collapse.

    Uses adjacency rules to create natural-looking terrain transitions
    where grass blends into dirt, dirt into paths, etc. This produces
    coherent terrain zones rather than random noise.

    When street data is available, constrains street tiles to remain
    as cobblestone while allowing natural terrain around them.

    Note: This layer only changes tile appearances - it does NOT modify
    regions. Buildings placed later will overwrite terrain tiles.
    """

    def __init__(self, max_attempts: int = 5) -> None:
        """Initialize the WFC terrain layer.

        Args:
            max_attempts: Number of times to retry if WFC contradicts.
                Falls back to RandomTerrainLayer after exhausting attempts.
        """
        self.max_attempts = max_attempts
        self.patterns = create_terrain_patterns()

    def apply(self, ctx: GenerationContext) -> None:
        """Apply terrain variety to outdoor tiles using WFC.

        Args:
            ctx: The generation context to modify.
        """
        for _attempt in range(self.max_attempts):
            try:
                self._apply_wfc(ctx)
                return
            except WFCContradiction:
                continue

        # Fallback to random terrain if WFC keeps failing
        RandomTerrainLayer().apply(ctx)

    def _apply_wfc(self, ctx: GenerationContext) -> None:
        """Run WFC to generate terrain patterns.

        When street data is available, constrains street tiles to
        COBBLESTONE pattern before solving.

        Args:
            ctx: The generation context to modify.

        Raises:
            WFCContradiction: If WFC fails to find a valid solution.
        """
        solver = WFCSolver(ctx.width, ctx.height, self.patterns, _wfc_terrain_rng)

        # Constrain street tiles to cobblestone if street data is available
        if ctx.street_data.streets:
            street_constraints: list[tuple[int, int, set[TerrainPatternID]]] = [
                (x, y, {TerrainPatternID.COBBLESTONE})
                for street in ctx.street_data.streets
                for x in range(max(0, street.x1), min(ctx.width, street.x2))
                for y in range(max(0, street.y1), min(ctx.height, street.y2))
            ]

            # Apply constraints in batch
            if street_constraints:
                solver.constrain_cells(street_constraints)

        result = solver.solve()

        # Apply the result to the context
        for x in range(ctx.width):
            for y in range(ctx.height):
                # Only modify outdoor floor tiles
                if ctx.tiles[x, y] == TileTypeID.COBBLESTONE:
                    pattern = self.patterns[result[x][y]]
                    ctx.tiles[x, y] = pattern.tile_type
