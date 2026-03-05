"""Tests for actor-facing pose UV selection."""

from __future__ import annotations

from brileta.game.actors.core import NPC, Actor
from brileta.types import Facing, SpriteUV


def test_directional_pose_uv_selects_explicit_east_and_west_slots() -> None:
    """Directional pose set should use dedicated UV slots for WEST and EAST."""
    actor = Actor(x=0, y=0, ch="@", color=(255, 255, 255))
    actor.character_sprite_uvs = (
        SpriteUV(0.00, 0.00, 0.01, 0.01),  # front_stand
        SpriteUV(0.01, 0.00, 0.02, 0.01),  # back_stand
        SpriteUV(0.10, 0.20, 0.30, 0.40),  # left_stand
        SpriteUV(0.11, 0.21, 0.31, 0.41),  # right_stand
    )

    actor.facing = Facing.WEST
    actor._update_active_sprite_uv(moving=False)
    assert actor.sprite_uv == SpriteUV(0.10, 0.20, 0.30, 0.40)

    actor.facing = Facing.EAST
    actor._update_active_sprite_uv(moving=False)
    assert actor.sprite_uv == SpriteUV(0.11, 0.21, 0.31, 0.41)


def test_npc_move_updates_facing_for_cardinal_directions() -> None:
    """NPC movement should update facing via the shared Actor.move() path."""
    npc = NPC(x=0, y=0, ch="@", color=(255, 255, 255), name="Test NPC")

    npc.move(-1, 0)
    assert npc.facing is Facing.WEST

    npc.move(1, 0)
    assert npc.facing is Facing.EAST

    npc.move(0, -1)
    assert npc.facing is Facing.NORTH

    npc.move(0, 1)
    assert npc.facing is Facing.SOUTH
