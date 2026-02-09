"""Pipeline-based map generation system.

This package provides a layered architecture for compositional map
generation. Each layer transforms a shared GenerationContext, and the
pipeline outputs GeneratedMapData compatible with existing code.

Example usage:
    from brileta.environment.generators.pipeline import create_pipeline

    generator = create_pipeline("settlement", width=80, height=43)
    map_data = generator.generate()

The pipeline can also be assembled manually for custom configurations:
    from brileta.environment.generators.pipeline import (
        PipelineGenerator,
        OpenFieldLayer,
        StreetNetworkLayer,
        BuildingPlacementLayer,
    )

    generator = PipelineGenerator(
        layers=[
            OpenFieldLayer(),
            StreetNetworkLayer(style="cross"),
            BuildingPlacementLayer(),
        ],
        map_width=80,
        map_height=43,
    )
"""

from .context import GenerationContext, StreetData
from .factory import create_pipeline, create_settlement_pipeline
from .layer import GenerationLayer
from .layers import (
    BuildingPlacementLayer,
    DetailLayer,
    OpenFieldLayer,
    RandomTerrainLayer,
    StreetNetworkLayer,
    WFCTerrainLayer,
)
from .pipeline import PipelineGenerator

__all__ = [
    "BuildingPlacementLayer",
    "DetailLayer",
    "GenerationContext",
    "GenerationLayer",
    "OpenFieldLayer",
    "PipelineGenerator",
    "RandomTerrainLayer",
    "StreetData",
    "StreetNetworkLayer",
    "WFCTerrainLayer",
    "create_pipeline",
    "create_settlement_pipeline",
]
