"""Generation layers for the pipeline map generator.

Each layer transforms the GenerationContext in a specific way:
- Terrain layers: Set up base terrain and add visual variety
- Street layers: Create street networks and building zones
- Building layers: Place buildings and create interior spaces
- Detail layers: Add environmental details like boulders
"""

from .buildings import BuildingPlacementLayer
from .details import DetailLayer
from .streets import StreetNetworkLayer
from .terrain import OpenFieldLayer, RandomTerrainLayer, WFCTerrainLayer
from .trees import TreePlacementLayer

__all__ = [
    "BuildingPlacementLayer",
    "DetailLayer",
    "OpenFieldLayer",
    "RandomTerrainLayer",
    "StreetNetworkLayer",
    "TreePlacementLayer",
    "WFCTerrainLayer",
]
