"""Tests for hostile AI behavior, disposition, and combat awareness.

Validates:
- Hostile NPCs chase, attack, and flee based on utility scoring.
- Hazard escape overrides normal combat behavior.
- Disposition labels, thresholds, and modification clamping.
- Neutral, unfriendly, and wary NPC behavior tiers.
- Wander goal lifecycle for non-hostile NPCs.
- set_hostile transitions and NPC-vs-NPC targeting.
- ai.force_hostile live variable override.
- Combat awareness: NPCs react to attacks from outside aggro range.
- Threat disappearance and attacker death clear combat state.
"""

from contextlib import contextmanager
from typing import cast

import pytest

from brileta import colors
from brileta.events import FloatingTextEvent, reset_event_bus_for_testing
from brileta.game import ranges
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.movement import MoveIntent
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai import HOSTILE_UPPER, disposition_label, escalate_hostility
from brileta.game.actors.ai.behaviors.flee import _FLEE_SAFE_DISTANCE, FleeGoal
from brileta.game.actors.ai.behaviors.wander import WanderGoal
from brileta.game.actors.ai.goals import GoalState
from brileta.game.game_world import GameWorld
from brileta.types import DIRECTIONS, DIRECTIONS_AND_CENTER
from tests.helpers import DummyController, DummyGameWorld, make_ai_world

# ---------------------------------------------------------------------------
# Core Hostile Behavior Tests
# ---------------------------------------------------------------------------


def test_hostile_ai_sets_active_plan() -> None:
    """AIComponent creates an active_plan to walk toward player."""
    controller, player, npc = make_ai_world()
    action = npc.ai.get_action(controller, npc)
    assert action is None  # Returns None because plan was set
    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None
    tx, ty = plan.context.target_position
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


def test_hostile_ai_attacks_when_adjacent() -> None:
    controller, _player, npc = make_ai_world()
    npc.x = 1
    npc.y = 0
    action = npc.ai.get_action(controller, npc)
    assert isinstance(action, AttackIntent)
    assert npc.active_plan is None


def test_hostile_ai_flees_when_low_health() -> None:
    """Low-health hostile NPCs should flee instead of attacking."""
    controller, _player, npc = make_ai_world()
    npc.x = 1
    npc.y = 0

    # Reduce health to make fleeing score higher than attacking
    npc.take_damage(4)  # Max HP is 5 for default toughness

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert action.dx == 1
    assert action.dy == 0


def test_hostile_ai_avoids_hazardous_destination_tiles() -> None:
    """AI prefers non-hazardous tiles when selecting destination adjacent to player."""
    from brileta.environment.tile_types import TileTypeID

    controller, player, npc = make_ai_world()

    # Player at (0, 0), NPC at (3, 0)
    # The closest adjacent tile to player from NPC's perspective is (1, 0)
    # Make (1, 0) hazardous - AI should pick a different adjacent tile
    controller.gw.game_map.tiles[1, 0] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)
    assert action is None  # Returns None because plan was set

    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None

    # The destination should NOT be the hazardous tile
    tx, ty = plan.context.target_position
    assert (tx, ty) != (1, 0), "AI should avoid hazardous destination tile"

    # But it should still be adjacent to the player
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


def test_hostile_ai_uses_hazardous_tile_when_no_alternative() -> None:
    """AI will use hazardous destination tile if all options are hazardous."""
    from brileta.environment.tile_types import TileTypeID

    controller, player, npc = make_ai_world()

    # Make ALL tiles adjacent to the player hazardous
    for dx, dy in DIRECTIONS:
        tx, ty = player.x + dx, player.y + dy
        if 0 <= tx < 80 and 0 <= ty < 80:
            controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)
    assert action is None

    plan = npc.active_plan
    assert plan is not None
    assert plan.context.target_position is not None

    # AI should still pick a destination (the least bad option)
    tx, ty = plan.context.target_position
    assert ranges.calculate_distance(player.x, player.y, tx, ty) == 1


# ---------------------------------------------------------------------------
# Hazard Escape Tests
# ---------------------------------------------------------------------------


def test_npc_escapes_hazard_before_attacking() -> None:
    """NPC on hazard should escape before pursuing player."""
    from brileta.environment.tile_types import TileTypeID

    controller, _player, npc = make_ai_world()

    # Place NPC on acid pool
    controller.gw.game_map.tiles[npc.x, npc.y] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    # Should return escape MoveIntent, not attack/pathfind
    assert isinstance(action, MoveIntent)


def test_npc_escapes_to_nearest_safe_tile() -> None:
    """NPC should escape to nearest non-hazardous tile, preferring orthogonal."""
    from brileta.environment.tile_types import TileTypeID

    controller, _player, npc = make_ai_world()
    npc.x, npc.y = 5, 5

    # Place NPC on hazard
    controller.gw.game_map.tiles[5, 5] = TileTypeID.ACID_POOL

    # Surround with hazards except one safe tile at (4, 5)
    for dx, dy in DIRECTIONS:
        tx, ty = npc.x + dx, npc.y + dy
        if (tx, ty) != (4, 5):
            controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL

    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    # Should escape to (4, 5) - the only safe tile
    assert action.dx == -1
    assert action.dy == 0


def test_npc_stays_if_all_adjacent_hazardous() -> None:
    """NPC stays put if all adjacent tiles are also hazards."""
    from brileta.environment.tile_types import TileTypeID

    controller, _player, npc = make_ai_world()
    npc.x, npc.y = 5, 5

    # Surround entirely with hazards (including the NPC's tile)
    for dx, dy in DIRECTIONS_AND_CENTER:
        tx, ty = npc.x + dx, npc.y + dy
        controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    # Should fall through to normal behavior (setting active_plan)
    # since there's no escape, returns None and sets active plan
    assert action is None
    assert npc.active_plan is not None


def test_npc_skips_blocked_safe_tile() -> None:
    """NPC should skip safe tiles blocked by other actors."""
    from brileta.environment.tile_types import TileTypeID

    controller, _player, npc = make_ai_world()
    npc.x, npc.y = 5, 5

    # Place a blocking actor at (4, 5) - would be the closest safe tile
    blocker = NPC(
        4,
        5,
        "b",
        colors.RED,
        "Blocker",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(blocker)
    assert controller.gw.player is not None
    blocker.ai.set_hostile(controller.gw.player)

    # Place NPC on hazard
    controller.gw.game_map.tiles[5, 5] = TileTypeID.ACID_POOL

    # Surround with hazards except blocked (4, 5) and one other safe tile (5, 4)
    for dx, dy in DIRECTIONS:
        tx, ty = npc.x + dx, npc.y + dy
        if (tx, ty) not in [(4, 5), (5, 4)]:
            controller.gw.game_map.tiles[tx, ty] = TileTypeID.ACID_POOL
    controller.gw.game_map.invalidate_property_caches()

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    # Should escape to (5, 4), not blocked (4, 5)
    assert action.dx == 0
    assert action.dy == -1


# ---------------------------------------------------------------------------
# Disposition & Behavior Tier Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-100, "Hostile"),
        (-52, "Hostile"),
        (-51, "Hostile"),
        (-50, "Unfriendly"),
        (-21, "Unfriendly"),
        (-20, "Wary"),
        (0, "Approachable"),
        (20, "Approachable"),
        (21, "Friendly"),
        (100, "Ally"),
    ],
)
def test_disposition_label_boundaries(value: int, expected: str) -> None:
    """Disposition labels should match inclusive boundary expectations."""
    assert disposition_label(value) == expected


def test_neutral_npc_in_range_does_not_attack() -> None:
    """Disposition 0 should not trigger attack behavior on proximity alone."""
    controller, _player, npc = make_ai_world(disposition=0)
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert not isinstance(action, AttackIntent)
    assert isinstance(npc.current_goal, WanderGoal)
    assert isinstance(action, MoveIntent)


def test_numeric_hostile_disposition_attacks_in_range() -> None:
    """Numeric hostile disposition should attack when adjacent."""
    controller, _player, npc = make_ai_world(disposition=-75)  # Hostile
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, AttackIntent)


def test_avoid_action_moves_unfriendly_npc_away() -> None:
    """Unfriendly NPCs should choose AvoidAction and step away from player."""
    controller, _player, npc = make_ai_world(disposition=-35)  # Unfriendly
    npc.x = 1
    npc.y = 0

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert (action.dx, action.dy) == (1, 0)


def test_watch_action_returns_none_for_wary_npc() -> None:
    """Wary NPCs should specifically pick WatchAction."""
    controller, _player, npc = make_ai_world(disposition=-20)  # Wary boundary
    npc.x = 1
    npc.y = 0

    # Remove all flee/avoid options (including lateral tiles at same distance)
    # so Watch vs Idle is the relevant decision.
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False
    gm.walkable[0, 1] = False

    action = npc.ai.get_action(controller, npc)

    assert action is None
    assert npc.active_plan is None
    assert npc.ai.last_chosen_action is not None
    assert "Watch" in npc.ai.last_chosen_action


def test_avoid_action_returns_none_when_cornered() -> None:
    """Cornered unfriendly NPCs should fall back cleanly when avoid has no step."""
    controller, _player, npc = make_ai_world(disposition=-35)  # Unfriendly
    npc.x = 1
    npc.y = 0

    # Block all tiles that would increase or maintain distance from player.
    gm = controller.gw.game_map
    gm.walkable[2, 0] = False
    gm.walkable[2, 1] = False
    gm.walkable[1, 1] = False
    gm.walkable[0, 1] = False

    action = npc.ai.get_action(controller, npc)

    assert action is None
    assert npc.ai.last_chosen_action in {"Watch", "Idle"}


# ---------------------------------------------------------------------------
# Wander Goal (Non-Hostile NPC) Tests
# ---------------------------------------------------------------------------


def test_wander_creates_goal_when_no_threat() -> None:
    """Neutral NPC should start wander as a persistent goal when safe."""
    controller, player, npc = make_ai_world(disposition=0)
    player.teleport(20, 20)
    npc.teleport(5, 5)

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, MoveIntent)
    assert isinstance(npc.current_goal, WanderGoal)


def test_wander_goal_continues_across_ticks() -> None:
    """Existing WanderGoal should be continued instead of recreated each tick."""
    controller, player, npc = make_ai_world(disposition=0)
    player.teleport(20, 20)
    npc.teleport(5, 5)

    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, WanderGoal)
    first_goal = npc.current_goal

    npc.ai.get_action(controller, npc)

    assert npc.current_goal is first_goal


# ---------------------------------------------------------------------------
# set_hostile / Disposition Modification Tests
# ---------------------------------------------------------------------------


def test_set_hostile_transitions_passive_to_aggressive() -> None:
    """set_hostile() should switch a non-hostile NPC into attack behavior."""
    controller, player, npc = make_ai_world(disposition=40)  # Friendly
    npc.x = 1
    npc.y = 0

    first_action = npc.ai.get_action(controller, npc)
    assert not isinstance(first_action, AttackIntent)

    npc.ai.set_hostile(player)
    second_action = npc.ai.get_action(controller, npc)
    assert isinstance(second_action, AttackIntent)


def test_modify_disposition_clamps_to_valid_range() -> None:
    """modify_disposition() should clamp numeric disposition to [-100, 100]."""
    _controller, player, npc = make_ai_world(disposition=0)

    npc.ai.modify_disposition(player, 999)
    assert npc.ai.disposition_toward(player) == 100

    npc.ai.modify_disposition(player, -999)
    assert npc.ai.disposition_toward(player) == -100


def test_is_hostile_toward_uses_hostile_threshold_boundary() -> None:
    """is_hostile_toward() should flip exactly at HOSTILE_UPPER."""
    _controller, player, npc = make_ai_world(disposition=0)

    # Neutral should not be hostile.
    assert not npc.ai.is_hostile_toward(player)

    # Typical hostile value should be hostile.
    npc.ai.set_hostile(player)
    assert npc.ai.is_hostile_toward(player)

    # Exact threshold is hostile; one above is not.
    npc.ai.modify_disposition(player, HOSTILE_UPPER - npc.ai.disposition_toward(player))
    assert npc.ai.is_hostile_toward(player)
    npc.ai.modify_disposition(player, 1)
    assert not npc.ai.is_hostile_toward(player)


def test_hostile_npc_can_target_another_npc() -> None:
    """Hostile relationship should allow NPC-vs-NPC targeting."""
    controller, player, npc = make_ai_world(disposition=40)  # Friendly
    npc2 = NPC(
        1,
        0,
        "r",
        colors.RED,
        "Rival",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc2)
    npc2.ai.modify_disposition(player, 40)  # Friendly toward player

    player.teleport(10, 10)
    npc.teleport(2, 0)

    npc.ai.set_hostile(npc2)

    action = npc.ai.get_action(controller, npc)

    assert isinstance(action, AttackIntent)
    assert action.defender is npc2


def test_escalate_hostility_noop_when_defender_has_no_ai() -> None:
    """Escalation should no-op when the defender is the AI-less player."""
    controller, player, npc = make_ai_world(disposition=-75)
    assert player.ai is None

    attacker_awareness_before = npc.ai._last_attacker_id
    disposition_before = npc.ai.disposition_toward(player)

    escalate_hostility(npc, player, controller)

    assert npc.ai._last_attacker_id == attacker_awareness_before
    assert npc.ai.disposition_toward(player) == disposition_before


# ---------------------------------------------------------------------------
# ai.force_hostile Tests
# ---------------------------------------------------------------------------


@contextmanager
def _override_live_variable(name: str, value: object):
    """Temporarily override (or register) a live variable to return *value*.

    Restores the original getter on exit so that test failures don't leak
    state into other tests.
    """
    from brileta.util.live_vars import live_variable_registry

    var = live_variable_registry.get_variable(name)
    if var is None:
        live_variable_registry.register(
            name,
            getter=lambda: value,
            setter=lambda v: None,
            description=f"Test override for {name}",
        )
        original_getter = None
    else:
        original_getter = var.getter
        var.getter = lambda: value

    try:
        yield
    finally:
        var = live_variable_registry.get_variable(name)
        if var is not None:
            if original_getter is not None:
                var.getter = original_getter
            else:
                var.getter = lambda: False


def test_force_hostile_makes_wary_npc_use_hostile_behavior() -> None:
    """With ai.force_hostile on, a Wary NPC should score hostile actions."""
    with _override_live_variable("ai.force_hostile", True):
        gw = DummyGameWorld()
        player = Character(
            0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        npc = NPC(
            3,
            0,
            "g",
            colors.RED,
            "Guard",
            game_world=cast(GameWorld, gw),
        )
        gw.player = player
        gw.add_actor(player)
        gw.add_actor(npc)
        npc.ai.modify_disposition(player, -10)  # Wary
        controller = DummyController(gw)

        # With the player within aggro range, force_hostile should make
        # hostile combat behavior eligible (attack/chase).
        result = npc.ai.get_action(controller, npc)
        assert result is not None or npc.active_plan is not None


# ---------------------------------------------------------------------------
# Combat Awareness Tests
# ---------------------------------------------------------------------------


def test_combat_awareness_selects_attacker_outside_aggro_range() -> None:
    """An NPC shot from outside aggro range should target the attacker."""
    controller, player, npc = make_ai_world(
        npc_x=20, npc_y=0, npc_hp_damage=0, disposition=0
    )

    # Player is at (0,0), NPC at (20,0) - well outside aggro_radius (10).
    # Without combat awareness, the NPC would just see the player as a
    # neutral fallback target with threat_level 0.
    context_before = npc.ai._build_context(controller, npc)
    assert context_before.threat_level == 0.0

    # Simulate being attacked: set hostile and notify.
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # Now the NPC should perceive the attacker as a threat.
    context_after = npc.ai._build_context(controller, npc)
    assert context_after.threat_level > 0.0
    assert context_after.target is player


def test_combat_awareness_threat_decays_with_distance() -> None:
    """Awareness threat should be higher when closer, zero at flee distance."""
    controller, player, npc = make_ai_world(
        npc_x=15, npc_y=0, npc_hp_damage=0, disposition=0
    )
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # At distance 15 (outside aggro 10, inside flee distance 50)
    context_15 = npc.ai._build_context(controller, npc)
    threat_at_15 = context_15.threat_level

    # Teleport further away
    npc.teleport(40, 0)
    context_40 = npc.ai._build_context(controller, npc)
    threat_at_40 = context_40.threat_level

    assert threat_at_15 > threat_at_40 > 0.0

    # At or beyond _FLEE_SAFE_DISTANCE, threat should be 0
    npc.teleport(_FLEE_SAFE_DISTANCE, 0)
    context_safe = npc.ai._build_context(controller, npc)
    assert context_safe.threat_level == 0.0


def test_combat_awareness_npc_flees_from_ranged_attacker() -> None:
    """NPC attacked from range should flee, not keep wandering."""
    controller, player, npc = make_ai_world(
        npc_x=15, npc_y=0, npc_hp_damage=4, disposition=0
    )

    # Before being attacked: NPC wanders (no threat)
    npc.ai.get_action(controller, npc)
    # NPC should have a WanderGoal (neutral, no threat)
    assert npc.current_goal is None or isinstance(npc.current_goal, WanderGoal)

    # Clear goal state for clean test
    if npc.current_goal is not None:
        npc.current_goal.abandon()
        npc.current_goal = None
    npc.active_plan = None

    # Simulate ranged attack
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    # NPC should now flee
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)


def test_hostile_flip_abandons_existing_wander_goal() -> None:
    """A hostile transition should invalidate an active wander goal immediately."""
    controller, player, npc = make_ai_world(
        npc_x=_FLEE_SAFE_DISTANCE + 10,
        npc_y=0,
        npc_hp_damage=0,
        disposition=0,
    )

    # Start in neutral/no-threat state so wander begins.
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, WanderGoal)
    initial_goal = npc.current_goal

    # Simulate a ranged aggression event that flips disposition to hostile.
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    npc.ai.get_action(controller, npc)

    # Wander must be dropped immediately after hostility flip.
    assert initial_goal.state == GoalState.ABANDONED
    assert npc.current_goal is None


def test_threat_disappearance_abandons_flee_goal() -> None:
    """FleeGoal should be abandoned when threat precondition no longer holds.

    When the player moves far enough away that threat_level drops to zero,
    ContinueGoalAction's precondition (is_threat_present) fails and the flee
    goal should be replaced by a non-combat action on the next tick.
    """
    controller, player, npc = make_ai_world(
        npc_x=3,
        npc_y=0,
        npc_hp_damage=4,
        disposition=-75,
    )

    # Hostile NPC near the player should start fleeing (low health).
    npc.ai.get_action(controller, npc)
    assert isinstance(npc.current_goal, FleeGoal)
    flee_goal = npc.current_goal

    # Move the player beyond safe distance so threat drops to zero.
    player.teleport(_FLEE_SAFE_DISTANCE + 20, _FLEE_SAFE_DISTANCE + 20)

    npc.ai.get_action(controller, npc)

    # Flee goal should have been abandoned via precondition failure.
    assert flee_goal.state in (GoalState.ABANDONED, GoalState.COMPLETED)
    assert not isinstance(npc.current_goal, FleeGoal)


def test_combat_awareness_clears_when_attacker_dies() -> None:
    """Awareness should clear when the attacker is no longer alive."""
    controller, player, npc = make_ai_world(
        npc_x=20, npc_y=0, npc_hp_damage=0, disposition=0
    )
    npc.ai.set_hostile(player)
    npc.ai.notify_attacked(player)

    assert npc.ai._last_attacker_id is not None

    # Kill the attacker
    player.health._hp = 0

    # Target selection should clear the awareness
    npc.ai._select_target_actor(controller, npc)
    assert npc.ai._last_attacker_id is None


# ---------------------------------------------------------------------------
# Action Transition Indicator Tests
# ---------------------------------------------------------------------------


def _collect_floating_texts() -> list[FloatingTextEvent]:
    """Subscribe to FloatingTextEvent and return a list that accumulates events."""
    from brileta.events import subscribe_to_event

    collected: list[FloatingTextEvent] = []
    subscribe_to_event(FloatingTextEvent, collected.append)
    return collected


def test_attack_transition_emits_red_exclamation() -> None:
    """Switching to attack should emit a red '!' floating text."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0)

    npc.ai.get_action(controller, npc)

    attack_events = [e for e in events if e.text == "!"]
    assert len(attack_events) == 1
    event = attack_events[0]
    assert event.color == colors.RED
    assert event.target_actor_id == npc.actor_id


def test_flee_transition_emits_pale_yellow_exclamation() -> None:
    """Switching to flee should emit a pale-yellow '!' floating text."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0, npc_hp_damage=4)

    npc.ai.get_action(controller, npc)

    # Both attack and flee use "!" but with different colors. The flee
    # indicator should be pale yellow, not red.
    flee_events = [e for e in events if e.text == "!" and e.color == (255, 255, 150)]
    assert len(flee_events) == 1
    assert flee_events[0].target_actor_id == npc.actor_id


def test_continuing_same_action_does_not_re_emit_indicator() -> None:
    """Staying in the same action across ticks should not fire the indicator again."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0, npc_hp_damage=4)

    # First tick: transition to flee emits pale-yellow "!"
    action = npc.ai.get_action(controller, npc)
    if isinstance(action, MoveIntent):
        npc.move(action.dx, action.dy)

    # Second tick: continuing flee should not emit again
    npc.ai.get_action(controller, npc)

    flee_events = [e for e in events if e.text == "!" and e.color == (255, 255, 150)]
    assert len(flee_events) == 1


def test_transition_indicator_not_emitted_when_npc_not_visible() -> None:
    """Indicators should be suppressed when the NPC is outside the player's FOV."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0)

    # Mark the NPC's tile as not visible (outside player FOV)
    controller.gw.game_map.visible[npc.x, npc.y] = False

    npc.ai.get_action(controller, npc)

    # The NPC should still attack, but no floating text should appear
    assert npc.ai.last_chosen_action == "Attack"
    assert len(events) == 0


def test_switching_between_tracked_actions_emits_both_indicators() -> None:
    """Transitioning attack -> flee should emit both a red and pale-yellow '!'."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    # Start with a full-health hostile NPC adjacent to the player so attack wins.
    controller, _player, npc = make_ai_world(npc_x=1, npc_y=0)

    npc.ai.get_action(controller, npc)
    assert npc.ai._last_action_id == "attack"

    # Now drop health so flee wins on the next tick.
    npc.take_damage(4)
    npc.ai.get_action(controller, npc)
    assert npc.ai._last_action_id == "flee"

    red_events = [e for e in events if e.color == colors.RED]
    yellow_events = [e for e in events if e.color == (255, 255, 150)]
    assert len(red_events) == 1
    assert len(yellow_events) == 1


def test_mundane_action_transition_emits_no_indicator() -> None:
    """Switching to wander/idle/watch/avoid should not emit any floating text."""
    reset_event_bus_for_testing()
    events = _collect_floating_texts()

    # Neutral NPC far from player - should pick wander (no threat).
    controller, player, npc = make_ai_world(disposition=0)
    player.teleport(20, 20)
    npc.teleport(5, 5)

    npc.ai.get_action(controller, npc)

    assert isinstance(npc.current_goal, WanderGoal)
    assert len(events) == 0
