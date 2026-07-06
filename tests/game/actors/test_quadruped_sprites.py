"""Tests for procedural quadruped (critter) sprite generation and wiring."""

from __future__ import annotations

import hashlib

import numpy as np

from brileta.sprites.characters import CHARACTER_POSE_COUNT, CHARACTER_POSES
from brileta.sprites.quadrupeds import (
    DOG_PRESET,
    QUADRUPED_POSE_COUNT,
    QUADRUPED_POSES,
    QuadrupedAppearance,
    draw_quadruped_pose,
    generate_quadruped_pose_set,
)
from brileta.types import Facing


def _digest(pose_set: list[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for sprite in pose_set:
        hasher.update(sprite.tobytes())
    return hasher.hexdigest()


def test_pose_set_is_twelve_rgba_sprites() -> None:
    """Generation yields the full 12-frame RGBA pose set, none empty."""
    pose_set = generate_quadruped_pose_set(42)
    assert len(pose_set) == QUADRUPED_POSE_COUNT == 12
    for sprite in pose_set:
        assert sprite.shape == (20, 20, 4)
        assert int(np.count_nonzero(sprite[:, :, 3])) > 0


def test_pose_layout_matches_character_layout() -> None:
    """Quadruped poses must match the character facing/frame layout exactly.

    The runtime pose selection (``Character.update_sprite_pose``) is shared, so
    dogs only render correctly if their pose set is the same length and facing
    order as the character set.
    """
    assert QUADRUPED_POSE_COUNT == CHARACTER_POSE_COUNT
    assert [p.facing for p in QUADRUPED_POSES] == [p.facing for p in CHARACTER_POSES]
    # Index 0 is the front (SOUTH) standing frame, the resting sprite.
    assert QUADRUPED_POSES[0].facing is Facing.SOUTH


def test_generation_is_deterministic() -> None:
    """Same seed produces byte-identical output (guards visual regressions)."""
    a = generate_quadruped_pose_set(12345)
    b = generate_quadruped_pose_set(12345)
    for sprite_a, sprite_b in zip(a, b, strict=True):
        assert np.array_equal(sprite_a, sprite_b)


def test_different_seeds_differ() -> None:
    """Different seeds roll visibly different dogs."""
    assert _digest(generate_quadruped_pose_set(111)) != _digest(
        generate_quadruped_pose_set(222)
    )


def test_appearance_from_seed_is_consistent() -> None:
    """Appearance roll is deterministic for a given seed."""
    a = QuadrupedAppearance.from_seed(9001, DOG_PRESET, 20)
    b = QuadrupedAppearance.from_seed(9001, DOG_PRESET, 20)
    assert a == b


def test_alpha_is_hardened_to_zero_or_255() -> None:
    """Every pose has a 1-bit alpha channel (crisp pixel-art edges)."""
    for sprite in generate_quadruped_pose_set(7):
        unique = set(np.unique(sprite[:, :, 3]).tolist())
        assert unique <= {0, 255}


def test_east_is_mirror_of_west() -> None:
    """EAST poses are horizontal mirrors of the authored WEST poses."""
    appearance = QuadrupedAppearance.from_seed(73, DOG_PRESET, 20)
    for west, east in (
        (QUADRUPED_POSES[6], QUADRUPED_POSES[9]),  # stand
        (QUADRUPED_POSES[7], QUADRUPED_POSES[10]),  # walk-a
        (QUADRUPED_POSES[8], QUADRUPED_POSES[11]),  # walk-b
    ):
        assert west.facing is Facing.WEST
        assert east.facing is Facing.EAST
        west_sprite = draw_quadruped_pose(appearance, west)
        east_sprite = draw_quadruped_pose(appearance, east)
        assert np.array_equal(
            east_sprite, np.ascontiguousarray(west_sprite[:, ::-1, :])
        )


def test_walk_frames_differ_from_stand() -> None:
    """Side walk frames must change pixels vs the stand, else motion is dead."""
    pose_set = generate_quadruped_pose_set(8)
    left_stand, left_a, left_b = pose_set[6], pose_set[7], pose_set[8]
    assert not np.array_equal(left_stand, left_a)
    assert not np.array_equal(left_a, left_b)


def test_dog_type_opts_into_critter_sprites() -> None:
    """DOG_TYPE carries the dog preset and its NPCs inherit it (the wiring
    the controller keys off - never a glyph or id match)."""
    from brileta.game.actors.npc_types import DOG_TYPE

    assert DOG_TYPE.critter_preset is DOG_PRESET
    dog = DOG_TYPE.create(0, 0, "Rex")
    assert dog.critter_preset is DOG_PRESET
