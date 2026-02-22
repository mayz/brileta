"""Generation layers for the pipeline map generator.

Each layer transforms the GenerationContext in a specific way:
- Terrain layers: Set up base terrain and generate natural landscape
- Street layers: Create street networks and building zones
- Building layers: Place buildings and create interior spaces
- Detail layers: Add environmental details like boulders and trees
"""

from .buildings import BuildingPlacementLayer
from .details import DetailLayer
from .streets import StreetNetworkLayer
from .terrain import NaturalTerrainLayer, OpenFieldLayer, RandomTerrainLayer
from .trees import TreePlacementLayer

__all__ = [
    "BuildingPlacementLayer",
    "DetailLayer",
    "NaturalTerrainLayer",
    "OpenFieldLayer",
    "RandomTerrainLayer",
    "StreetNetworkLayer",
    "TreePlacementLayer",
]
