"""Factory functions for creating pre-configured pipelines.

These functions provide convenient ways to create common pipeline
configurations without needing to manually assemble layers.

Currently implemented:
- "settlement": Outdoor settlement with streets, buildings, WFC terrain

TODO: Future map types to implement:
- "wilderness": Outdoor map with CellularAutomataTerrainLayer, RuinScatterLayer
  for organic caves/ruins, scattered damaged buildings, wasteland terrain
- "interior": Pure indoor map (bunker, large building) using adapted dungeon
  generation with the pipeline architecture
"""

from __future__ import annotations

from brileta import config
from brileta.environment.generators.buildings.templates import get_default_templates
from brileta.types import RandomSeed

from .layers import (
    BuildingPlacementLayer,
    DetailLayer,
    OpenFieldLayer,
    StreetNetworkLayer,
    WFCTerrainLayer,
)
from .pipeline import PipelineGenerator


def create_pipeline(
    name: str,
    width: int,
    height: int,
    seed: RandomSeed = None,
) -> PipelineGenerator:
    """Create a pre-configured pipeline by name.

    Available pipelines:
    - "settlement": Outdoor settlement with buildings and terrain variety

    Args:
        name: Name of the pipeline configuration to use.
        width: Map width in tiles.
        height: Map height in tiles.
        seed: Optional random seed for deterministic generation.

    Returns:
        A configured PipelineGenerator ready to generate maps.

    Raises:
        ValueError: If the pipeline name is not recognized.
    """
    if name == "settlement":
        return create_settlement_pipeline(width, height, seed)
    raise ValueError(f"Unknown pipeline name: {name!r}")


def create_settlement_pipeline(
    width: int,
    height: int,
    seed: RandomSeed = None,
    street_style: str | None = None,
) -> PipelineGenerator:
    """Create a settlement pipeline with default configuration.

    The settlement pipeline generates:
    1. Base outdoor terrain (OpenFieldLayer)
    2. Street network with building zones (StreetNetworkLayer)
    3. Terrain variety with WFC around streets (WFCTerrainLayer)
    4. Buildings in zones with doors facing streets (BuildingPlacementLayer)
    5. Environmental details like boulders (DetailLayer)

    Args:
        width: Map width in tiles.
        height: Map height in tiles.
        seed: Optional random seed for deterministic generation.
        street_style: Street layout ("single", "cross", "grid").
            If None, uses config.SETTLEMENT_STREET_STYLE.

    Returns:
        A configured PipelineGenerator.
    """
    if street_style is None:
        street_style = config.SETTLEMENT_STREET_STYLE

    layers = [
        # 1. Fill with outdoor floor as base
        OpenFieldLayer(),
        # 2. Create street network and define building zones
        StreetNetworkLayer(style=street_style),
        # 3. Add terrain variety (WFC respects streets as cobblestone)
        WFCTerrainLayer(),
        # 4. Place buildings in zones, doors face streets
        BuildingPlacementLayer(
            templates=get_default_templates(),
            lot_min_size=config.SETTLEMENT_LOT_MIN_SIZE,
            lot_max_size=config.SETTLEMENT_LOT_MAX_SIZE,
            building_density=config.SETTLEMENT_BUILDING_DENSITY,
            max_buildings=config.SETTLEMENT_MAX_BUILDINGS,
        ),
        # 5. Add environmental details
        DetailLayer(),
    ]

    return PipelineGenerator(
        layers=layers,
        map_width=width,
        map_height=height,
        seed=seed,
    )
