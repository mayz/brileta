"""Map generation algorithms for Brileta.

This package provides various map generators:
- RoomsAndCorridorsGenerator: Classic dungeon-style rooms and corridors
- PipelineGenerator: Layered pipeline architecture for compositional maps

Pipeline-based generation is the primary system. Different map types (settlement,
wilderness) are created by composing different layers:
- Settlement: OpenFieldLayer + StreetNetworkLayer + WFCTerrainLayer + buildings
- Wilderness (TODO): CellularAutomataTerrainLayer + RuinScatterLayer + details

And a reusable WFC solver for constraint-based generation:
- WFCSolver: Generic Wave Function Collapse solver
- WFCPattern: Pattern definition with adjacency rules
"""

from .base import BaseMapGenerator, GeneratedMapData
from .dungeon import RoomsAndCorridorsGenerator
from .pipeline import (
    BuildingPlacementLayer,
    DetailLayer,
    GenerationContext,
    GenerationLayer,
    OpenFieldLayer,
    PipelineGenerator,
    RandomTerrainLayer,
    StreetNetworkLayer,
    WFCTerrainLayer,
    create_pipeline,
    create_settlement_pipeline,
)
from .wfc_solver import (
    WFCContradiction,
    WFCPattern,
    WFCSolver,
)

__all__ = [
    "BaseMapGenerator",
    "BuildingPlacementLayer",
    "DetailLayer",
    "GeneratedMapData",
    "GenerationContext",
    "GenerationLayer",
    "OpenFieldLayer",
    "PipelineGenerator",
    "RandomTerrainLayer",
    "RoomsAndCorridorsGenerator",
    "StreetNetworkLayer",
    "WFCContradiction",
    "WFCPattern",
    "WFCSolver",
    "WFCTerrainLayer",
    "create_pipeline",
    "create_settlement_pipeline",
]
