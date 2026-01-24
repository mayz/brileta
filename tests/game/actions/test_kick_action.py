"""Tests for the Kick combat stunt action."""

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.environment.tile_types import TileTypeID
from catley.events import reset_event_bus_for_testing
from catley.game.actions.executors.stunts import KickExecutor
from catley.game.actions.stunts import KickIntent
from catley.game.actors import Character
from catley.game.actors.status_effects import (
    OffBalanceEffect,
    TrippedEffect,
)
from catley.game.game_world import GameWorld
from catley.game.resolution.d20_system import D20System
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller for testing kick actions."""

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
    player_strength: int = 10,
    enemy_agility: int = 10,
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
        strength=player_strength,
    )
    enemy = Character(
        enemy_pos[0],
        enemy_pos[1],
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        agility=enemy_agility,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw)
    return controller, player, enemy


# --- Basic Kick Tests ---


def test_kick_success_deals_damage_and_pushes() -> None:
    """A successful kick deals d4 damage and pushes the target one tile."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    initial_hp = enemy.health.hp

    # Force a successful roll (not critical) and d4 damage of 3
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:  # d20 roll
            return 15  # Success
        if b == 4:  # d4 damage
            return 3
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have moved from (6, 5) to (7, 5) - pushed right
    assert enemy.x == 7
    assert enemy.y == 5
    # Enemy should have taken 3 damage
    assert enemy.health.hp == initial_hp - 3
    # No TrippedEffect (that's critical only)
    assert not enemy.status_effects.has_status_effect(TrippedEffect)


def test_kick_critical_success_also_trips() -> None:
    """A critical success kick deals damage, pushes, and trips the target."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    initial_hp = enemy.health.hp

    # Force a natural 20 and d4 damage of 4
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:  # d20 roll
            return 20  # Critical
        if b == 4:  # d4 damage
            return 4
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.x == 7  # Pushed
    assert enemy.health.hp == initial_hp - 4
    # Critical kick also trips the target
    assert enemy.status_effects.has_status_effect(TrippedEffect)


def test_kick_partial_success_attacker_off_balance() -> None:
    """A partial success kick deals damage and pushes, but attacker stumbles."""
    reset_event_bus_for_testing()
    # Create characters for a tie scenario (partial success)
    # roll_to_exceed = defender_agility + 10 = 10 + 10 = 20
    # total = roll + attacker_strength = 10 + 10 = 20 (tie)
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(6, 5),
        player_strength=10,
        enemy_agility=10,
    )
    initial_hp = enemy.health.hp

    # Roll of 10 gives a tie (partial success)
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 10
        if b == 4:
            return 2
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.x == 7  # Still pushed
    assert enemy.health.hp == initial_hp - 2  # Still deals damage
    # Attacker should be off-balance
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_kick_failure_attacker_off_balance() -> None:
    """A failed kick gives the attacker OffBalanceEffect, no damage dealt."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a low roll that will fail (not nat 1)
    with patch("random.randint", return_value=2):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Enemy should not have moved
    assert enemy.x == 6
    assert enemy.y == 5
    # No damage dealt
    assert enemy.health.hp == initial_hp
    # Attacker should be off-balance
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_kick_critical_failure_trips_attacker() -> None:
    """A critical failure (nat 1) trips the attacker."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()
    initial_hp = enemy.health.hp

    # Force a natural 1
    with patch("random.randint", return_value=1):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert enemy.x == 6  # Enemy didn't move
    assert enemy.health.hp == initial_hp  # No damage
    # Attacker should be tripped
    assert player.status_effects.has_status_effect(TrippedEffect)


# --- Stat Check Tests ---


def test_kick_uses_strength_vs_agility() -> None:
    """Kick should use attacker's Strength vs defender's Agility."""
    reset_event_bus_for_testing()
    # High strength attacker vs low agility defender - should succeed easily
    controller, player, enemy = _make_world_with_enemy(
        player_strength=18,  # +4 modifier
        enemy_agility=8,  # roll_to_exceed = 8 + 10 = 18
    )

    # Roll of 12 + 18 = 30, vs roll_to_exceed 18 - clear success
    with patch("random.randint", return_value=12):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded


# --- Adjacency Validation ---


def test_kick_fails_if_not_adjacent() -> None:
    """Kick should fail if target is not adjacent."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(8, 5),  # 3 tiles away
    )

    intent = KickIntent(cast(Controller, controller), player, enemy)
    executor = KickExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "not_adjacent"


# --- Environmental Interactions ---


def test_kick_into_wall_deals_impact_damage() -> None:
    """Kicking into a wall deals 1d4 extra impact damage and OffBalance."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    # Place a wall at the destination
    controller.gw.game_map.tiles[7, 5] = TileTypeID.WALL
    controller.gw.game_map.walkable[7, 5] = False

    initial_hp = enemy.health.hp

    # Force success, kick damage of 3, and wall impact damage of 2
    damage_rolls = iter([3, 2])  # kick damage, wall impact

    def fixed_randint(a: int, b: int) -> int:
        if b == 20:  # d20 roll
            return 15  # Success
        if b == 4:  # d4 damage
            return next(damage_rolls)
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should NOT have moved (wall blocked)
    assert enemy.x == 6
    assert enemy.y == 5
    # Enemy should have taken 3 (kick) + 2 (wall impact) = 5 damage
    assert enemy.health.hp == initial_hp - 5
    # Enemy should be off-balance from wall impact
    assert enemy.status_effects.has_status_effect(OffBalanceEffect)


def test_kick_into_hazard_moves_target() -> None:
    """Kicking onto a hazard tile moves the target (damage via turn system)."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    # Place acid pool at destination
    controller.gw.game_map.tiles[7, 5] = TileTypeID.ACID_POOL

    # Force success
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 15
        if b == 4:
            return 3
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have moved onto the hazard
    assert enemy.x == 7
    assert enemy.y == 5
    # Note: Actual hazard damage is applied by the turn system, not the kick


def test_kick_into_another_actor_makes_both_off_balance() -> None:
    """Kicking into another actor makes both actors off-balance."""
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    enemy1 = Character(6, 5, "r", colors.RED, "Raider", game_world=cast(GameWorld, gw))
    enemy2 = Character(7, 5, "r", colors.RED, "Bandit", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy1)
    gw.add_actor(enemy2)
    controller = DummyController(gw=gw)

    # Force success
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 15
        if b == 4:
            return 3
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, enemy1)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy1 should NOT have moved (blocked by enemy2)
    assert enemy1.x == 6
    # Both should be off-balance
    assert enemy1.status_effects.has_status_effect(OffBalanceEffect)
    assert enemy2.status_effects.has_status_effect(OffBalanceEffect)


# --- Kick Hostility Tests ---


def test_kick_makes_non_hostile_npc_hostile() -> None:
    """Kicking a non-hostile NPC should make them hostile."""
    from catley.game.actors import NPC
    from catley.game.enums import Disposition

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
        disposition=Disposition.WARY,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)

    assert npc.ai.disposition == Disposition.WARY

    # Force successful kick
    def fixed_randint(a: int, b: int) -> int:
        if b == 20:
            return 15
        if b == 4:
            return 3
        return a

    with patch("random.randint", fixed_randint):
        intent = KickIntent(cast(Controller, controller), player, npc)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert npc.ai.disposition == Disposition.HOSTILE


def test_failed_kick_still_triggers_hostility() -> None:
    """A failed kick should still make the NPC hostile - the attempt is aggressive."""
    from catley.game.actors import NPC
    from catley.game.enums import Disposition

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
        disposition=Disposition.WARY,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)

    assert npc.ai.disposition == Disposition.WARY

    # Force failed kick (low roll, not nat 1)
    with patch("random.randint", return_value=2):
        intent = KickIntent(cast(Controller, controller), player, npc)
        executor = KickExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # The attempt was aggressive, so NPC becomes hostile
    assert npc.ai.disposition == Disposition.HOSTILE
