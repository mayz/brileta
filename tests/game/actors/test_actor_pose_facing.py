"""Tests for actor-facing pose UV selection."""

from __future__ import annotations

from brileta.game.actors.core import NPC, Actor
from brileta.types import Facing, SpriteUV


def _pose_set() -> tuple[SpriteUV, ...]:
    """Build a 12-slot pose set with a unique UV per (facing x frame)."""
    return tuple(
        SpriteUV(float(i), float(i), float(i) + 0.5, float(i) + 0.5) for i in range(12)
    )


def test_directional_pose_uv_selects_explicit_east_and_west_slots() -> None:
    """Idle selection should pick the standing frame per facing.

    Layout is grouped by facing (S, N, W, E), each facing = [stand, walk_a,
    walk_b], so stand indices are 0, 3, 6, 9.
    """
    actor = Actor(x=0, y=0, ch="@", color=(255, 255, 255))
    poses = _pose_set()
    actor.character_sprite_uvs = poses

    actor.facing = Facing.WEST
    actor._update_active_sprite_uv(moving=False)
    assert actor.sprite_uv == poses[6]  # west stand

    actor.facing = Facing.EAST
    actor._update_active_sprite_uv(moving=False)
    assert actor.sprite_uv == poses[9]  # east stand


def test_moving_selection_alternates_walk_frames_by_walk_frame() -> None:
    """While moving, selection uses the two walk frames per facing."""
    actor = Actor(x=0, y=0, ch="@", color=(255, 255, 255))
    poses = _pose_set()
    actor.character_sprite_uvs = poses
    actor.facing = Facing.SOUTH  # stand=0, walk_a=1, walk_b=2

    actor.walk_frame = 0
    actor._update_active_sprite_uv(moving=True)
    assert actor.sprite_uv == poses[1]  # walk_a

    actor.walk_frame = 1
    actor._update_active_sprite_uv(moving=True)
    assert actor.sprite_uv == poses[2]  # walk_b

    actor._update_active_sprite_uv(moving=False)
    assert actor.sprite_uv == poses[0]  # back to stand


def test_walk_frame_holds_past_step_then_settles_to_stand_after_idle_delay() -> None:
    """A stopped actor keeps its last walk frame, settling to stand only after
    the idle-settle delay.

    Two behaviors in one: (1) a completed tile move does NOT flicker to stand
    (the "hopping" regression), and (2) continuous walking never reaches idle
    because each step re-arms the timer well before the ~2s delay lapses.
    """
    from types import SimpleNamespace
    from typing import Any, cast

    from brileta.game.actors.core import _IDLE_SETTLE_DELAY_S
    from brileta.types import FixedTimestep
    from brileta.view.animation import AnimationManager

    step = FixedTimestep(1 / 60)
    manager = AnimationManager()
    controller = cast(Any, SimpleNamespace(animation_manager=manager))
    actor = Actor(x=0, y=0, ch="@", color=(255, 255, 255))
    poses = _pose_set()
    actor.character_sprite_uvs = poses
    actor.facing = Facing.SOUTH  # south stand=poses[0], walk frames poses[1]/[2]
    walk_frames = (poses[1], poses[2])

    actor.move(0, 1, controller, duration=0.1)  # one tile step (0.1s glide)
    assert actor.sprite_uv in walk_frames  # walking

    # Tick well past the move completion but within the idle delay: the actor
    # holds its walk frame (no flicker to stand between tiles / while wandering).
    hold_steps = int((_IDLE_SETTLE_DELAY_S * 0.5) * 60)
    for _ in range(hold_steps):
        manager.update(step)
    assert manager.is_queue_empty()  # the tile's move animation finished
    assert actor.sprite_uv in walk_frames  # still holding the walk frame

    # Keep ticking past the idle delay with no further steps: settles to stand.
    for _ in range(int(_IDLE_SETTLE_DELAY_S * 60) + 5):
        manager.update(step)
    assert actor.sprite_uv == poses[0]  # standing


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
