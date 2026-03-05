"""Procedural humanoid character sprite generation with directional output.

This package provides deterministic sprite generation for humanoid actors. A
single seed rolls one character appearance, then that appearance is rendered in
four directional standing poses (front/back/left/right).
"""

from __future__ import annotations

from brileta.types import Facing

from .appearance import (
    CHARACTER_DIRECTIONAL_POSE_COUNT,
    CLOTHING_PALETTES,
    HAIR_PALETTES,
    HUMANOID_GLYPHS,
    PANTS_PALETTES,
    POSE_BACK_STAND,
    POSE_FRONT_STAND,
    POSE_LEFT_STAND,
    POSE_RIGHT_STAND,
    POSE_STAND,
    POSES,
    SKIN_PALETTES,
    BodyParams,
    CharacterAppearance,
    Pose,
    character_sprite_seed,
)
from .clothing import CLOTHING_DEFS, CLOTHING_STYLE_NAMES
from .hair import HAIR_STYLE_NAMES
from .renderer import (
    CARICATURE_TEMPLATES,
    draw_character_pose,
    generate_caricature_belly,
    generate_caricature_broad,
    generate_caricature_child,
    generate_caricature_standard,
    generate_caricature_thin,
    generate_character,
    generate_character_pose_set,
    generate_character_sprite,
    roll_character_appearance,
)

__all__ = [
    "CARICATURE_TEMPLATES",
    "CHARACTER_DIRECTIONAL_POSE_COUNT",
    "CLOTHING_DEFS",
    "CLOTHING_PALETTES",
    "CLOTHING_STYLE_NAMES",
    "HAIR_PALETTES",
    "HAIR_STYLE_NAMES",
    "HUMANOID_GLYPHS",
    "PANTS_PALETTES",
    "POSES",
    "POSE_BACK_STAND",
    "POSE_FRONT_STAND",
    "POSE_LEFT_STAND",
    "POSE_RIGHT_STAND",
    "POSE_STAND",
    "SKIN_PALETTES",
    "BodyParams",
    "CharacterAppearance",
    "Facing",
    "Pose",
    "character_sprite_seed",
    "draw_character_pose",
    "generate_caricature_belly",
    "generate_caricature_broad",
    "generate_caricature_child",
    "generate_caricature_standard",
    "generate_caricature_thin",
    "generate_character",
    "generate_character_pose_set",
    "generate_character_sprite",
    "roll_character_appearance",
]
