"""Procedural humanoid character sprite generation with directional output.

This package provides deterministic sprite generation for humanoid actors. A
single seed rolls one character appearance, then that appearance is rendered in
four directional standing poses (front/back/left/right).
"""

from __future__ import annotations

from .appearance import (
    CHARACTER_DIRECTIONAL_POSE_COUNT,
    CLOTHING_PALETTES,
    HAIR_PALETTES,
    HUMANOID_GLYPHS,
    PANTS_PALETTES,
    POSE_BACK_STAND,
    POSE_LEFT_STAND,
    POSE_RIGHT_STAND,
    POSE_STAND,
    POSES,
    SKIN_PALETTES,
    CharacterAppearance,
    character_sprite_seed,
)
from .clothing import CLOTHING_STYLE_NAMES
from .hair import HAIR_STYLE_NAMES
from .renderer import (
    CARICATURE_TEMPLATES,
    draw_character_pose,
    generate_character_pose_set,
    generate_character_sprite,
    roll_character_appearance,
)

__all__ = [
    "CARICATURE_TEMPLATES",
    "CHARACTER_DIRECTIONAL_POSE_COUNT",
    "CLOTHING_PALETTES",
    "CLOTHING_STYLE_NAMES",
    "HAIR_PALETTES",
    "HAIR_STYLE_NAMES",
    "HUMANOID_GLYPHS",
    "PANTS_PALETTES",
    "POSES",
    "POSE_BACK_STAND",
    "POSE_LEFT_STAND",
    "POSE_RIGHT_STAND",
    "POSE_STAND",
    "SKIN_PALETTES",
    "CharacterAppearance",
    "character_sprite_seed",
    "draw_character_pose",
    "generate_character_pose_set",
    "generate_character_sprite",
    "roll_character_appearance",
]
