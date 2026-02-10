"""Tests for the Trip combat stunt action."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from brileta import colors
from brileta.controller import Controller
from brileta.events import reset_event_bus_for_testing
from brileta.game.actions.executors.stunts import TripExecutor
from brileta.game.actions.stunts import TripIntent
from brileta.game.actors import Character
from brileta.game.actors.status_effects import (
    OffBalanceEffect,
    TrippedEffect,
)
from brileta.game.game_world import GameWorld
from brileta.game.resolution.d20_system import D20System
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from tests.conftest import CombatRNGPatcher, D20RNGPatcher


@dataclass
class DummyController:
    """Minimal controller for testing trip actions."""

    gw: DummyGameWorld
    frame_manager: object | None = None

    def create_resolver(
        self,
        ability_score: int,
        roll_to_exceed: int,
        has_advantage: bool = False,
        has_disadvantage: bool = False,
    ) -> D20System:
        return D20System(
            ability_score=ability_score,
            roll_to_exceed=roll_to_exceed,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )


def _make_world_with_enemy(
    player_pos: tuple[int, int] = (5, 5),
    enemy_pos: tuple[int, int] = (6, 5),
) -> tuple[DummyController, Character, Character]:
    """Create a test world with player and enemy at specified positions."""
    gw = DummyGameWorld()
    player = Character(
        player_pos[0],
        player_pos[1],
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    enemy = Character(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw)
    return controller, player, enemy


# --- Basic Trip Tests ---


def test_trip_success_applies_tripped_effect(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A successful trip applies TrippedEffect to the target."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    original_x, original_y = enemy.x, enemy.y

    # Force a successful roll (not critical)
    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should NOT have moved (trip doesn't reposition)
    assert enemy.x == original_x
    assert enemy.y == original_y
    # Enemy should be tripped
    assert enemy.status_effects.has_status_effect(TrippedEffect)


def test_trip_success_does_not_move_target(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Unlike push, trip does not move the target."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    original_x, original_y = enemy.x, enemy.y

    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        executor.execute(intent)

    # Enemy should be in the exact same position
    assert enemy.x == original_x
    assert enemy.y == original_y


def test_trip_critical_success_deals_damage(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """A critical success trip applies TrippedEffect and deals 1d4 damage."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    initial_hp = enemy.health.hp

    # Force a natural 20 for d20, and 3 for d4 impact damage
    with patch_combat_rng([20], [3]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should be tripped
    assert enemy.status_effects.has_status_effect(TrippedEffect)
    # Enemy should have taken 3 damage from hard landing
    assert enemy.health.hp == initial_hp - 3


def test_trip_partial_success_trips_but_attacker_off_balance(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A partial success (tie) trips target but attacker is off-balance."""
    reset_event_bus_for_testing()
    # Create characters with specific agility values for a tie scenario
    # roll_to_exceed = defender_agility + 10 = 10 + 10 = 20
    # total = roll + attacker_agility = 10 + 10 = 20 (tie)
    gw = DummyGameWorld()
    player = Character(
        5,
        5,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
        agility=10,
    )
    enemy = Character(
        6,
        5,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        agility=10,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw)

    # Roll of 10 + ability 10 = 20, which equals roll_to_exceed (10 + 10 = 20)
    with patch_d20_rng([10]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Target should be tripped
    assert enemy.status_effects.has_status_effect(TrippedEffect)
    # Attacker should be off-balance
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_trip_failure_gives_attacker_off_balance(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A failed trip gives the attacker OffBalanceEffect."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()

    # Force a low roll that will fail (not nat 1)
    with patch_d20_rng([2]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Target should NOT be tripped
    assert not enemy.status_effects.has_status_effect(TrippedEffect)
    # Attacker should be off-balance from the failed attempt
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_trip_critical_failure_trips_attacker(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A critical failure (nat 1) trips the attacker instead."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()

    # Force a natural 1
    with patch_d20_rng([1]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Target should NOT be tripped
    assert not enemy.status_effects.has_status_effect(TrippedEffect)
    # Attacker trips over their own feet
    assert player.status_effects.has_status_effect(TrippedEffect)


# --- Adjacency Validation ---


def test_trip_fails_if_not_adjacent() -> None:
    """Trip should fail if target is not adjacent."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(8, 5),  # 3 tiles away
    )

    intent = TripIntent(cast(Controller, controller), player, enemy)
    executor = TripExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "not_adjacent"


def test_trip_works_diagonally(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Trip should work on diagonally adjacent targets."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(6, 6),  # Diagonal from player
    )

    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), player, enemy)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.status_effects.has_status_effect(TrippedEffect)


# --- Trip Hostility Tests ---


def test_trip_makes_non_hostile_npc_hostile(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Tripping a non-hostile NPC should make them hostile."""
    from brileta.game.actors import NPC

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    # Create a wary (non-hostile) NPC
    npc = NPC(
        6,
        5,
        "n",
        colors.YELLOW,
        "Wary NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.ai.modify_disposition(player, -10)  # Wary
    controller = DummyController(gw=gw)

    assert npc.ai.disposition_toward(player) == -10

    # Force successful trip
    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), player, npc)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert npc.ai.disposition_toward(player) == -75


def test_failed_trip_still_triggers_hostility(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A failed trip should still make the NPC hostile - the attempt is aggressive."""
    from brileta.game.actors import NPC

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        6,
        5,
        "n",
        colors.YELLOW,
        "Wary NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.ai.modify_disposition(player, -10)  # Wary
    controller = DummyController(gw=gw)

    assert npc.ai.disposition_toward(player) == -10

    # Force failed trip (low roll, not nat 1)
    with patch_d20_rng([2]):
        intent = TripIntent(cast(Controller, controller), player, npc)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # The attempt was aggressive, so NPC becomes hostile
    assert npc.ai.disposition_toward(player) == -75


def test_trip_does_not_change_already_hostile(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Tripping an already hostile NPC should not change their disposition."""
    from brileta.game.actors import NPC

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc = NPC(
        6,
        5,
        "n",
        colors.RED,
        "Hostile NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    npc.ai.set_hostile(player)
    controller = DummyController(gw=gw)

    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), player, npc)
        executor = TripExecutor()
        executor.execute(intent)

    assert npc.ai.disposition_toward(player) == -75


def test_npc_tripping_npc_updates_hostility_between_npcs(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """NPC tripping another NPC should create hostility toward the attacker only."""
    from brileta.game.actors import NPC

    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    npc1 = NPC(
        5,
        5,
        "a",
        colors.RED,
        "Attacker NPC",
        game_world=cast(GameWorld, gw),
    )
    npc2 = NPC(
        6,
        5,
        "d",
        colors.YELLOW,
        "Defender NPC",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc1)
    gw.add_actor(npc2)
    npc1.ai.set_hostile(player)
    npc2.ai.modify_disposition(player, -10)  # Wary toward player
    controller = DummyController(gw=gw)

    # NPC1 trips NPC2
    with patch_d20_rng([15]):
        intent = TripIntent(cast(Controller, controller), npc1, npc2)
        executor = TripExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # NPC-vs-NPC aggression should not alter NPC2's relationship to the player.
    assert npc2.ai.disposition_toward(player) == -10
    # But NPC2 should become hostile toward the attacking NPC.
    assert npc2.ai.disposition_toward(npc1) == -75
