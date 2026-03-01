"""Building system for procedural settlement generation.

This package provides dataclasses and templates for generating buildings
with rooms, doors, and semantic meaning.
"""

from .building import Building, ChimneyType, RoofProfile, RoofStyle, Room
from .templates import BuildingTemplate, get_default_templates

__all__ = [
    "Building",
    "BuildingTemplate",
    "ChimneyType",
    "RoofProfile",
    "RoofStyle",
    "Room",
    "get_default_templates",
]
