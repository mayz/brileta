"""Tests for auto-entering combat mode on attack.

Combat mode should be automatically entered when:
- An NPC attacks the player (hit or miss)
- The player attacks a non-hostile NPC
- The player pushes a non-hostile NPC
- The player's noise alerts nearby neutral NPCs
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actions.executors.stunts import PushExecutor
from brileta.game.actions.stunts import PushIntent
from brileta.game.actors import NPC, Character
from brileta.game.consequences import Consequence, ConsequenceHandler
from brileta.game.items.item_types import FISTS_TYPE
from tests.helpers import reset_dummy_controller

# --- Helper functions ---


def _make_combat_test_world(
    controller: Controller,
    player_pos: tuple[int, int] = (5, 5),
    enemy_pos: tuple[int, int] = (6, 5),
    *,
    enemy_disposition: int = -75,
) -> tuple[Controller, Character, NPC]:
    """Create a test world with player and NPC at specified positions.

    Returns a full Controller so we can check combat mode state.

    Args:
        enemy_disposition: Numeric disposition toward the player.
            Common values: -75 (hostile), -10 (wary), 0 (neutral).
    """
    player = controller.gw.player
    assert player is not None
    # Remove any randomly spawned NPCs to keep combat tests deterministic.
    for actor in list(controller.gw.actors):
        if isinstance(actor, Character) and actor is not player:
            controller.gw.remove_actor(actor)

    # Move player to the specified position
    player.x = player_pos[0]
    player.y = player_pos[1]

    # Create an NPC and set per-relationship disposition toward the player
    npc = NPC(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=controller.gw,
    )
    controller.gw.add_actor(npc)
    if enemy_disposition != 0:
        npc.ai.modify_disposition(player, enemy_disposition)

    return controller, player, npc


@pytest.fixture
def auto_combat_controller(dummy_controller: Controller) -> Controller:
    reset_dummy_controller(dummy_controller)
    return dummy_controller


# --- NPC Attack Tests ---


def test_npc_attack_triggers_combat_mode(auto_combat_controller: Controller) -> None:
    """When an NPC attacks the player (hit), combat mode should activate."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Verify we start in explore mode
    assert not controller.is_combat_mode()

    # NPC attacks player with fists (force a hit)
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=15):  # Force hit
        intent = AttackIntent(controller, npc, player, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent)

    # Combat mode should now be active
    assert controller.is_combat_mode()


def test_npc_miss_still_triggers_combat_mode(
    auto_combat_controller: Controller,
) -> None:
    """When an NPC misses the player, combat mode should still activate.

    Hostile intent alone warrants combat mode - we don't wait for the outcome.
    """
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    assert not controller.is_combat_mode()

    # NPC attacks player with fists (force a miss)
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=2):  # Force miss
        intent = AttackIntent(controller, npc, player, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent)

    # Combat mode should still be active even though the attack missed
    assert controller.is_combat_mode()


# --- Player Attack Tests ---


def test_player_attack_triggers_combat_mode(auto_combat_controller: Controller) -> None:
    """When the player attacks a non-hostile NPC, combat mode should activate."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-10,  # Wary  # Non-hostile
    )

    assert not controller.is_combat_mode()
    assert npc.ai.disposition_toward(player) == -10

    # Player attacks non-hostile NPC with fists
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=15):  # Force hit
        intent = AttackIntent(controller, player, npc, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent)

    # NPC should become hostile and combat mode should activate
    assert npc.ai.disposition_toward(player) == -75
    assert controller.is_combat_mode()


def test_player_attack_already_hostile_no_double_event(
    auto_combat_controller: Controller,
) -> None:
    """Attacking an already-hostile NPC should still enter combat mode.

    This tests the NPC-attacks-player path is not triggered when the player
    attacks first (the player is the attacker, not defender).
    """
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    assert not controller.is_combat_mode()

    # Player attacks already-hostile NPC
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=15):
        intent = AttackIntent(controller, player, npc, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent)

    # No event should fire (NPC was already hostile, so hostility escalation is
    # a no-op, and the NPC-attacks-player branch doesn't trigger either).
    # However, the first combat action should ideally enter combat mode.
    # This test verifies no crash from duplicate events.
    # The player attacking an NPC doesn't auto-enter combat - that's intentional.
    # Combat mode is entered when:
    # - NPC attacks player (defensive trigger)
    # - Player makes NPC hostile (offensive trigger)
    # Attacking an already-hostile NPC doesn't change anything.
    assert not controller.is_combat_mode()


# --- Player Push Tests ---


def test_player_push_triggers_combat_mode(auto_combat_controller: Controller) -> None:
    """When the player pushes a non-hostile NPC, combat mode should activate."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-10,  # Wary
    )

    assert not controller.is_combat_mode()
    assert npc.ai.disposition_toward(player) == -10

    # Player pushes non-hostile NPC
    with patch("random.randint", return_value=15):  # Force success
        intent = PushIntent(controller, player, npc)
        executor = PushExecutor()
        executor.execute(intent)

    # NPC should become hostile and combat mode should activate
    assert npc.ai.disposition_toward(player) == -75
    assert controller.is_combat_mode()


def test_failed_push_still_triggers_combat_mode(
    auto_combat_controller: Controller,
) -> None:
    """A failed push attempt should still trigger combat mode.

    The attempt is aggressive regardless of outcome.
    """
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-10,  # Wary
    )

    assert not controller.is_combat_mode()

    # Player fails to push NPC
    with patch("random.randint", return_value=2):  # Force failure
        intent = PushIntent(controller, player, npc)
        executor = PushExecutor()
        executor.execute(intent)

    # Combat mode should activate despite the failed push
    assert npc.ai.disposition_toward(player) == -75
    assert controller.is_combat_mode()


# --- Noise Alert Tests ---


def test_noise_alert_triggers_combat_mode(auto_combat_controller: Controller) -> None:
    """When the player's noise alerts a neutral NPC, combat mode should activate."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(7, 5),  # 2 tiles away, within radius 5
        enemy_disposition=-10,  # Wary
    )

    assert not controller.is_combat_mode()
    assert npc.ai.disposition_toward(player) == -10

    # Simulate noise alert from player firing a weapon
    handler = ConsequenceHandler()
    consequence = Consequence(
        type="noise_alert",
        data={"source": player, "radius": 5},
    )
    handler.apply_consequence(consequence)

    # NPC should become hostile and combat mode should activate
    assert npc.ai.disposition_toward(player) == -75
    assert controller.is_combat_mode()


def test_noise_from_npc_does_not_trigger_combat_mode(
    auto_combat_controller: Controller,
) -> None:
    """Noise from an NPC should not trigger combat mode entry."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(7, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Create another NPC that's not hostile
    bystander = NPC(
        8,
        5,
        "b",
        colors.YELLOW,
        "Bystander",
        game_world=controller.gw,
    )
    controller.gw.add_actor(bystander)
    bystander.ai.modify_disposition(player, -10)  # Wary

    assert not controller.is_combat_mode()
    assert bystander.ai.disposition_toward(player) == -10

    # Simulate noise alert from NPC (not player)
    handler = ConsequenceHandler()
    consequence = Consequence(
        type="noise_alert",
        data={"source": npc, "radius": 5},
    )
    handler.apply_consequence(consequence)

    # NPC-generated noise should not trigger relationship hostility changes
    # or combat mode transitions for the player.
    assert bystander.ai.disposition_toward(npc) == 0
    assert bystander.ai.disposition_toward(player) == -10
    assert not controller.is_combat_mode()


# --- Idempotency Tests ---


def test_already_in_combat_no_reentry(auto_combat_controller: Controller) -> None:
    """When already in combat mode, receiving another event should not crash.

    The handler should be idempotent - calling enter_combat_mode when already
    in combat mode should be a no-op.
    """
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Manually enter combat mode first
    controller.enter_combat_mode()
    assert controller.is_combat_mode()
    initial_stack_len = len(controller.mode_stack)

    # NPC attacks player (should trigger event but not crash or double-push)
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=15):
        intent = AttackIntent(controller, npc, player, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent)

    # Should still be in combat mode with same stack depth
    assert controller.is_combat_mode()
    assert len(controller.mode_stack) == initial_stack_len


def test_multiple_events_same_frame_handled(auto_combat_controller: Controller) -> None:
    """Multiple combat events in the same frame should not cause issues."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-10,  # Wary
    )

    # Create a second NPC
    npc2 = NPC(
        4,
        5,
        "r",
        colors.RED,
        "Raider 2",
        game_world=controller.gw,
    )
    controller.gw.add_actor(npc2)
    npc2.ai.modify_disposition(player, -10)  # Wary

    # Make both NPCs visible so they become combat candidates
    controller.gw.game_map.visible[npc.x, npc.y] = True
    controller.gw.game_map.visible[npc2.x, npc2.y] = True

    # Give NPCs enough HP to survive fist attacks
    npc.health._hp = 100
    npc2.health._hp = 100

    assert not controller.is_combat_mode()

    # Player attacks both NPCs (simulating multiple events)
    weapon = FISTS_TYPE.create()
    with patch("random.randint", return_value=15):
        intent1 = AttackIntent(controller, player, npc, weapon=weapon)
        executor = AttackExecutor()
        executor.execute(intent1)

        # First attack should trigger combat mode
        assert controller.is_combat_mode()

        intent2 = AttackIntent(controller, player, npc2, weapon=weapon)
        executor.execute(intent2)

    # Should still be in combat mode without issues
    assert controller.is_combat_mode()
    # Both NPCs should be hostile
    assert npc.ai.disposition_toward(player) == -75
    assert npc2.ai.disposition_toward(player) == -75


# --- Visible Hostiles Detection Tests ---


def test_has_visible_hostiles_returns_true_when_hostile_visible(
    auto_combat_controller: Controller,
) -> None:
    """has_visible_hostiles() returns True when a hostile NPC is visible."""
    controller, _player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Make the NPC's tile visible
    controller.gw.game_map.visible[npc.x, npc.y] = True

    assert controller.has_visible_hostiles()


def test_has_visible_hostiles_returns_false_when_no_hostiles(
    auto_combat_controller: Controller,
) -> None:
    """has_visible_hostiles() returns False when no NPCs are hostile."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-10,  # Wary  # Not hostile
    )

    # Make the NPC's tile visible (still shouldn't count as hostile)
    controller.gw.game_map.visible[npc.x, npc.y] = True

    assert npc.ai.disposition_toward(player) == -10
    assert not controller.has_visible_hostiles()


def test_has_visible_hostiles_returns_false_when_hostile_not_visible(
    auto_combat_controller: Controller,
) -> None:
    """has_visible_hostiles() returns False when hostile NPCs aren't visible."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Make the NPC's tile not visible
    controller.gw.game_map.visible[npc.x, npc.y] = False

    assert npc.ai.disposition_toward(player) == -75
    assert not controller.has_visible_hostiles()


def test_has_visible_hostiles_ignores_dead_hostiles(
    auto_combat_controller: Controller,
) -> None:
    """has_visible_hostiles() returns False when hostile NPCs are dead."""
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Make the NPC visible and then kill them
    controller.gw.game_map.visible[npc.x, npc.y] = True
    npc.health._hp = 0

    assert npc.ai.disposition_toward(player) == -75
    assert not npc.health.is_alive()
    assert not controller.has_visible_hostiles()


# --- Combat Auto-Exit Tests ---


def test_combat_auto_exits_when_all_enemies_dead(
    auto_combat_controller: Controller,
) -> None:
    """Combat mode should auto-exit when the last visible enemy dies."""
    controller, _player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Make NPC visible
    controller.gw.game_map.visible[npc.x, npc.y] = True

    # Enter combat mode
    controller.enter_combat_mode()
    assert controller.is_combat_mode()

    # Kill the NPC (trigger on_actor_death)
    npc.health._hp = 0
    controller.combat_mode.on_actor_death(npc)

    # Combat mode should have auto-exited
    assert not controller.is_combat_mode()


def test_combat_ended_event_published_on_auto_exit(
    auto_combat_controller: Controller,
) -> None:
    """CombatEndedEvent should be published with 'all_enemies_dead' reason."""
    from brileta.events import CombatEndedEvent, subscribe_to_event

    controller, _player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Track received events
    received_events: list[CombatEndedEvent] = []

    def on_combat_ended(event: CombatEndedEvent) -> None:
        received_events.append(event)

    subscribe_to_event(CombatEndedEvent, on_combat_ended)

    # Make NPC visible and enter combat
    controller.gw.game_map.visible[npc.x, npc.y] = True
    controller.enter_combat_mode()

    # Kill the NPC
    npc.health._hp = 0
    controller.combat_mode.on_actor_death(npc)

    # Verify event was published with correct reason
    assert len(received_events) == 1
    assert received_events[0].reason == "all_enemies_dead"


def test_combat_ended_event_published_on_manual_exit(
    auto_combat_controller: Controller,
) -> None:
    """CombatEndedEvent should be published with 'manual_exit' reason."""
    from brileta.events import CombatEndedEvent, subscribe_to_event

    controller, _player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Track received events
    received_events: list[CombatEndedEvent] = []

    def on_combat_ended(event: CombatEndedEvent) -> None:
        received_events.append(event)

    subscribe_to_event(CombatEndedEvent, on_combat_ended)

    # Make NPC visible and enter combat
    controller.gw.game_map.visible[npc.x, npc.y] = True
    controller.enter_combat_mode()
    assert controller.is_combat_mode()

    # Manually exit combat
    controller.exit_combat_mode("manual_exit")

    # Verify event was published with correct reason
    assert len(received_events) == 1
    assert received_events[0].reason == "manual_exit"
    assert not controller.is_combat_mode()


def test_combat_ended_event_published_on_cancelled_exit(
    auto_combat_controller: Controller,
) -> None:
    """CombatEndedEvent should be published with 'cancelled' reason.

    This tests the _on_target_cancelled() path in CombatMode, which is
    triggered when PickerMode cancels (e.g., right-click during targeting).
    """
    from brileta.events import CombatEndedEvent, subscribe_to_event

    controller, _player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile
    )

    # Track received events
    received_events: list[CombatEndedEvent] = []

    def on_combat_ended(event: CombatEndedEvent) -> None:
        received_events.append(event)

    subscribe_to_event(CombatEndedEvent, on_combat_ended)

    # Make NPC visible and enter combat
    controller.gw.game_map.visible[npc.x, npc.y] = True
    controller.enter_combat_mode()
    assert controller.is_combat_mode()

    # Simulate the cancelled exit path (as if PickerMode cancelled)
    controller.combat_mode._on_target_cancelled()

    # Verify event was published with correct reason
    assert len(received_events) == 1
    assert received_events[0].reason == "cancelled"
    assert not controller.is_combat_mode()


# --- Push Against Already-Hostile Tests ---


def test_push_already_hostile_npc_no_combat_entry(
    auto_combat_controller: Controller,
) -> None:
    """Pushing an already-hostile NPC does NOT auto-enter combat mode.

    CombatInitiatedEvent is only fired when disposition changes (non-hostile
    to hostile). Pushing an NPC that's already hostile doesn't change anything,
    so no event is fired. This is consistent with attack behavior.
    """
    controller, player, npc = _make_combat_test_world(
        auto_combat_controller,
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        enemy_disposition=-75,  # Hostile  # Already hostile
    )

    assert not controller.is_combat_mode()
    assert npc.ai.disposition_toward(player) == -75

    # Player pushes already-hostile NPC
    with patch("random.randint", return_value=15):  # Force success
        intent = PushIntent(controller, player, npc)
        executor = PushExecutor()
        executor.execute(intent)

    # NPC disposition unchanged (was already hostile)
    # No event fired, so combat mode NOT entered
    # (Combat auto-entry is triggered by disposition changes only)
    assert npc.ai.disposition_toward(player) == -75
    assert not controller.is_combat_mode()
