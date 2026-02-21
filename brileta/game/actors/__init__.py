"""
This package contains all modules related to the definition, behavior,
and state of game actors.
"""

# Export the submodules themselves for access to concrete implementations.
# Expose the core actor classes and base classes directly.
# These form the main, stable API of the package.
from . import (
    ai,
    boulder,
    boulder_sprites,
    components,
    conditions,
    container,
    status_effects,
    tree_sprites,
    trees,
)
from .ai import AIComponent
from .boulder import Boulder
from .boulder_sprites import BoulderArchetype
from .conditions import Condition
from .container import (
    Container,
    ItemPile,
    create_bookcase,
)
from .core import NPC, PC, Actor, Character
from .status_effects import StatusEffect
from .tree_sprites import (
    TreeArchetype,
    generate_tree_sprite,
    generate_tree_sprite_for_position,
    sprite_visual_scale_for_shadow_height,
    tree_sprite_seed,
    visual_scale_with_height_jitter,
)
from .trees import Tree, create_conifer_tree, create_deciduous_tree

# Note: We don't export *every* concrete class here.
# Only the most fundamental ones that define the API.

__all__ = [
    # Core Classes  # ruff: noqa: RUF022
    "Actor",
    "Character",
    "PC",
    "NPC",
    "Container",
    "ItemPile",
    "Boulder",
    "BoulderArchetype",
    "Tree",
    # Sprite Generation
    "TreeArchetype",
    "generate_tree_sprite",
    "generate_tree_sprite_for_position",
    "sprite_visual_scale_for_shadow_height",
    "tree_sprite_seed",
    "visual_scale_with_height_jitter",
    # Factory Functions
    "create_bookcase",
    "create_deciduous_tree",
    "create_conifer_tree",
    # Base/Interface Classes
    "AIComponent",
    "Condition",
    "StatusEffect",
    # Submodules for concrete implementations
    "ai",
    "boulder",
    "boulder_sprites",
    "components",
    "conditions",
    "container",
    "status_effects",
    "tree_sprites",
    "trees",
]
