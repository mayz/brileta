"""Tests for RoutineAction / RoutineGoal daily-routine behavior.

Validates:
- routine_target picks the workplace anchor by day and home by night, shifted
  by the NPC's per-NPC schedule offset.
- A resident with an assigned routine adopts a RoutineGoal when safe, while an
  unassigned one falls back to wandering (routine outscores wander only when a
  routine exists).
- The goal walks toward a far target and mills once close.
- Settlement assignment anchors a fraction of residents, spreads workplaces
  under a per-tile cap, and leaves the rest wandering.
"""

from typing import cast

from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC
from brileta.game.actors.ai.behaviors.routine import RoutineGoal, routine_target
from brileta.game.actors.ai.behaviors.wander import WanderGoal
from brileta.game.actors.npc_types import RESIDENT_TYPE
from brileta.game.clock import GameClock
from brileta.game.game_world import GameWorld
from tests.helpers import make_ai_world


def _make_resident(gw: GameWorld, x: int, y: int) -> NPC:
    resident = RESIDENT_TYPE.create(x, y, "Resident", game_world=gw)
    gw.add_actor(resident)
    return resident


def test_routine_target_switches_with_time_of_day() -> None:
    """Daytime points at the anchor; night points home."""
    controller, _player, _npc = make_ai_world(disposition=0)
    resident = _make_resident(cast(GameWorld, controller.gw), 10, 10)
    resident.home_pos = (10, 10)
    resident.anchor_pos = (20, 20)

    # Mid-morning (default start) is a workday: head to the anchor.
    assert routine_target(0.35, resident) == (20, 20)
    # Deep night wraps to home.
    assert routine_target(0.0, resident) == (10, 10)
    assert routine_target(0.95, resident) == (10, 10)


def test_routine_target_falls_back_to_other_anchor() -> None:
    """A missing anchor falls back to whichever one is set."""
    controller, _player, _npc = make_ai_world(disposition=0)
    resident = _make_resident(cast(GameWorld, controller.gw), 10, 10)
    resident.home_pos = None
    resident.anchor_pos = (20, 20)
    assert routine_target(0.0, resident) == (20, 20)  # night, but no home set


def test_resident_with_routine_adopts_routine_goal() -> None:
    """A safe resident with a routine should pursue it over wandering."""
    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()  # default mid-morning: work phase

    resident = _make_resident(gw, 40, 40)
    resident.home_pos = (40, 40)
    resident.anchor_pos = (41, 40)  # within arrival radius: mill in place

    resident.ai.get_action(controller, resident)
    assert isinstance(resident.current_goal, RoutineGoal)


def test_resident_without_routine_wanders() -> None:
    """Without home/anchor the routine is inert and wander wins instead."""
    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()

    resident = _make_resident(gw, 40, 40)
    # No home_pos / anchor_pos assigned.

    resident.ai.get_action(controller, resident)
    assert isinstance(resident.current_goal, WanderGoal)


def test_routine_goal_walks_toward_far_target() -> None:
    """When far from the target the goal issues a walk plan toward it."""
    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()  # work phase -> anchor is the target

    resident = _make_resident(gw, 40, 40)
    resident.home_pos = (40, 40)
    resident.anchor_pos = (50, 40)

    goal = RoutineGoal()
    resident.current_goal = goal
    action = goal.get_next_action(resident, controller)

    assert action is None
    assert resident.active_plan is not None
    assert resident.active_plan.context.target_position == (50, 40)


def test_routine_goal_mills_when_at_target() -> None:
    """At the target the goal idles or takes a small step, never pathfinds."""
    import brileta.game.actors.ai.behaviors.routine as routine_module

    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()

    resident = _make_resident(gw, 40, 40)
    resident.home_pos = (40, 40)
    resident.anchor_pos = (40, 40)  # standing exactly on the anchor

    goal = RoutineGoal()
    resident.current_goal = goal

    # Force a mill step: expect a one-tile move, no walk plan.
    from unittest.mock import patch

    with patch.object(routine_module._rng, "random", return_value=0.0):
        action = goal.get_next_action(resident, controller)
    assert isinstance(action, MoveIntent)
    assert resident.active_plan is None


def test_routine_goal_starts_errand_to_feature_tile() -> None:
    """At the anchor, an NPC may leave on an errand toward a feature tile."""
    from unittest.mock import patch

    import brileta.game.actors.ai.behaviors.routine as routine_module

    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()  # work phase
    gw.routine_errand_pool = [(50, 40)]  # a feature tile far from the anchor

    resident = _make_resident(gw, 40, 40)
    resident.home_pos = (40, 40)
    resident.anchor_pos = (40, 40)  # standing on the anchor

    goal = RoutineGoal()
    resident.current_goal = goal

    # Force the errand roll to fire.
    with patch.object(routine_module._rng, "random", return_value=0.0):
        goal.get_next_action(resident, controller)

    assert goal._errand_target == (50, 40)
    assert resident.active_plan is not None
    assert resident.active_plan.context.target_position == (50, 40)


def test_routine_goal_errand_lingers_then_clears() -> None:
    """Arriving at an errand tile lingers, then clears so the NPC heads back."""
    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()

    resident = _make_resident(gw, 50, 40)  # standing on the errand destination
    resident.home_pos = (40, 40)
    resident.anchor_pos = (40, 40)

    goal = RoutineGoal()
    goal._errand_target = (50, 40)
    goal._errand_linger = 1
    resident.current_goal = goal

    # Tick 1: at the destination, spend the one linger tick.
    assert goal.get_next_action(resident, controller) is None
    assert goal._errand_linger == 0
    assert goal._errand_target == (50, 40)

    # Tick 2: linger exhausted, errand clears (routine resumes next tick).
    assert goal.get_next_action(resident, controller) is None
    assert goal._errand_target is None


def test_routine_goal_errand_gives_up_when_stuck() -> None:
    """An errand that makes no progress is abandoned, not retried forever."""
    import brileta.game.actors.ai.behaviors.routine as routine_module

    controller, _player, _npc = make_ai_world(npc_x=0, npc_y=0, disposition=0)
    gw = cast(GameWorld, controller.gw)
    gw.clock = GameClock()

    resident = _make_resident(gw, 40, 40)
    resident.home_pos = (40, 40)
    resident.anchor_pos = (40, 40)

    goal = RoutineGoal()
    goal._errand_target = (55, 40)  # far destination the NPC never reaches
    goal._errand_linger = 3
    resident.current_goal = goal

    # Drive the goal without ever executing its walk plan, so the NPC makes no
    # progress each tick - the same state an unreachable destination produces.
    for _ in range(routine_module._ERRAND_MAX_STUCK_TICKS + 1):
        goal.get_next_action(resident, controller)

    assert goal._errand_target is None


def test_routine_offset_shifts_schedule() -> None:
    """A positive offset delays the work window; a negative one advances it."""
    controller, _player, _npc = make_ai_world(disposition=0)
    resident = _make_resident(cast(GameWorld, controller.gw), 10, 10)
    resident.home_pos = (10, 10)
    resident.anchor_pos = (20, 20)

    # Base workday ends at _WORK_END (0.75). At t=0.80 an unshifted NPC is home.
    resident.routine_offset = 0.0
    assert routine_target(0.80, resident) == (10, 10)
    # A +0.10 offset pushes the workday end to 0.85, so at 0.80 it's still at work.
    resident.routine_offset = 0.10
    assert routine_target(0.80, resident) == (20, 20)


def test_settlement_assignment_spreads_and_partially_anchors() -> None:
    """Batch assignment anchors ~75% of residents, capped and spread out."""
    from collections import Counter

    from brileta import config
    from brileta.game.enums import GeneratorType
    from brileta.game.game_world import (
        _ROUTINE_ANCHOR_TILE_CAP,
        _ROUTINE_ANCHORED_FRACTION,
    )

    gw = GameWorld(
        config.MAP_WIDTH,
        config.MAP_HEIGHT,
        generator_type=GeneratorType.SETTLEMENT,
        seed="thud-cause-sleet",
    )
    population = gw._pending_routine_npcs
    anchored = [r for r in population if r.anchor_pos is not None]
    unanchored = [r for r in population if r.anchor_pos is None]

    # Exactly the configured fraction is anchored; the rest keep wandering.
    assert len(anchored) == round(len(population) * _ROUTINE_ANCHORED_FRACTION)
    assert unanchored, "expected some residents to stay unanchored and wander"

    # Anchored residents have both a home and an anchor; unanchored have neither.
    for r in anchored:
        assert r.home_pos is not None and r.anchor_pos is not None
    for r in unanchored:
        assert r.home_pos is None

    # No workplace tile is crowded beyond the cap.
    anchor_usage = Counter(r.anchor_pos for r in anchored)
    assert max(anchor_usage.values()) <= _ROUTINE_ANCHOR_TILE_CAP

    # Schedule offsets vary rather than all firing at once.
    assert len({round(r.routine_offset, 3) for r in anchored}) > 1
