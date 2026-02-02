"""Pipeline generator that orchestrates layer-based map generation.

The PipelineGenerator runs a sequence of GenerationLayers, each transforming
a shared GenerationContext. This enables compositional map generation where
each layer focuses on one aspect of the map.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.environment.generators.base import BaseMapGenerator, GeneratedMapData
from catley.environment.tile_types import TileTypeID

from .context import GenerationContext

if TYPE_CHECKING:
    from .layer import GenerationLayer

from catley.types import RandomSeed, TileCoord


class PipelineGenerator(BaseMapGenerator):
    """Map generator that runs layers sequentially on a shared context.

    The pipeline creates an empty GenerationContext and passes it through
    each layer in order. Layers modify the context in place, building up
    the final map.

    Example:
        generator = PipelineGenerator(
            layers=[
                OpenFieldLayer(),
                RandomTerrainLayer(),
                BuildingPlacementLayer(templates=...),
                DetailLayer(),
            ],
            width=80,
            height=43,
            seed=12345,
        )
        map_data = generator.generate()

    Attributes:
        layers: List of GenerationLayer instances to apply.
        seed: Optional random seed for reproducible generation.
    """

    def __init__(
        self,
        layers: list[GenerationLayer],
        map_width: TileCoord,
        map_height: TileCoord,
        seed: RandomSeed = None,
    ) -> None:
        """Initialize the pipeline generator.

        Args:
            layers: List of GenerationLayer instances to apply in order.
            map_width: Width of the map in tiles.
            map_height: Height of the map in tiles.
            seed: Optional random seed for deterministic generation.
        """
        super().__init__(map_width, map_height)
        self.layers = layers
        self.seed = seed

    def generate(self) -> GeneratedMapData:
        """Generate a map by running all layers in sequence.

        Creates an empty GenerationContext and applies each layer to it.
        Returns the final map data suitable for use with GameMap.

        Returns:
            GeneratedMapData containing tiles, regions, and tile-to-region mapping.
        """
        # Create empty context - layers will fill it
        ctx = GenerationContext.create_empty(
            width=self.map_width,
            height=self.map_height,
            seed=self.seed,
            fill_tile=TileTypeID.WALL,
        )

        # Apply each layer in sequence
        for layer in self.layers:
            layer.apply(ctx)

        # Convert to the standard output format
        return ctx.to_generated_map_data()
