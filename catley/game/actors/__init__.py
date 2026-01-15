"""
This package contains all modules related to the definition, behavior,
and state of game actors.
"""

# Export the submodules themselves for access to concrete implementations.
# Expose the core actor classes and base classes directly.
# These form the main, stable API of the package.
from . import ai, components, conditions, container, status_effects
from .ai import AIComponent
from .conditions import Condition
from .container import (
    Container,
    create_bookcase,
)
from .core import NPC, PC, Actor, Character
from .status_effects import StatusEffect

# Note: We don't export *every* concrete class here.
# Only the most fundamental ones that define the API.

__all__ = [
    # Core Classes  # ruff: noqa: RUF022
    "Actor",
    "Character",
    "PC",
    "NPC",
    "Container",
    # Factory Functions
    "create_bookcase",
    # Base/Interface Classes
    "AIComponent",
    "Condition",
    "StatusEffect",
    # Submodules for concrete implementations
    "ai",
    "components",
    "conditions",
    "container",
    "status_effects",
]
