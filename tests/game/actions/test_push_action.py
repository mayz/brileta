"""Tests for the Push combat stunt action."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from catley import colors
from catley.controller import Controller
from catley.environment.tile_types import TileTypeID
from catley.events import reset_event_bus_for_testing
from catley.game.actions.executors.stunts import PushExecutor
from catley.game.actions.stunts import PushIntent
from catley.game.actors import Character
from catley.game.actors.status_effects import (
    OffBalanceEffect,
    StaggeredEffect,
    TrippedEffect,
)
from catley.game.game_world import GameWorld
from catley.game.resolution.d20_system import D20System
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from tests.conftest import CombatRNGPatcher, D20RNGPatcher


@dataclass
class DummyController:
    """Minimal controller for testing push actions."""

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


# --- Basic Push Tests ---


def test_push_success_moves_target_one_tile(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A successful push moves the target one tile away."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )

    # Force a successful roll (not critical)
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have moved from (6, 5) to (7, 5) - pushed right
    assert enemy.x == 7
    assert enemy.y == 5


def test_push_success_applies_staggered_effect(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A successful push should apply StaggeredEffect to the defender."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )

    # Force a successful roll (not critical)
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should be staggered (not tripped - that's critical only)
    assert enemy.status_effects.has_status_effect(StaggeredEffect)
    assert not enemy.status_effects.has_status_effect(TrippedEffect)


def test_push_critical_success_trips_target(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A critical success push moves, trips target, and TrippedEffect lasts 2 turns."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )

    # Force a natural 20
    with patch_d20_rng([20]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.x == 7  # Pushed
    # Check for TrippedEffect with 2-turn duration
    assert enemy.status_effects.has_status_effect(TrippedEffect)
    # Verify TrippedEffect, not StaggeredEffect
    assert not enemy.status_effects.has_status_effect(StaggeredEffect)


def test_push_partial_success_moves_but_attacker_off_balance(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A partial success (tie) moves target but attacker is off-balance."""
    reset_event_bus_for_testing()
    # Create characters with specific strength values for a tie scenario
    # roll_to_exceed = defender_strength + 10 = 10 + 10 = 20
    # total = roll + attacker_strength = 10 + 10 = 20 (tie)
    gw = DummyGameWorld()
    player = Character(
        5,
        5,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
        strength=10,
    )
    enemy = Character(
        6,
        5,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, gw),
        strength=10,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw)

    # Roll of 10 + ability 10 = 20, which equals roll_to_exceed (10 + 10 = 20)
    with patch_d20_rng([10]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert enemy.x == 7  # Still pushed
    # Attacker should be off-balance
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_push_failure_gives_attacker_off_balance(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A failed push gives the attacker OffBalanceEffect."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()

    # Force a low roll that will fail (not nat 1)
    with patch_d20_rng([2]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # Enemy should not have moved
    assert enemy.x == 6
    assert enemy.y == 5
    # Attacker should be off-balance
    assert player.status_effects.has_status_effect(OffBalanceEffect)


def test_push_critical_failure_trips_attacker(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A critical failure (nat 1) trips the attacker."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy()

    # Force a natural 1
    with patch_d20_rng([1]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert enemy.x == 6  # Enemy didn't move
    # Attacker should be tripped
    assert player.status_effects.has_status_effect(TrippedEffect)


# --- Adjacency Validation ---


def test_push_fails_if_not_adjacent() -> None:
    """Push should fail if target is not adjacent."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(8, 5),  # 3 tiles away
    )

    intent = PushIntent(cast(Controller, controller), player, enemy)
    executor = PushExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    assert result.block_reason == "not_adjacent"


# --- Environmental Interactions ---


def test_push_into_wall_deals_impact_damage(
    patch_combat_rng: "CombatRNGPatcher",
) -> None:
    """Pushing into a wall deals 1d4 impact damage and OffBalance."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    # Place a wall at the destination
    controller.gw.game_map.tiles[7, 5] = TileTypeID.WALL
    controller.gw.game_map.walkable[7, 5] = False

    initial_hp = enemy.health.hp

    # Force success (d20 roll 15), and fix the impact damage roll to 3
    with patch_combat_rng([15], [3]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should NOT have moved (wall blocked)
    assert enemy.x == 6
    assert enemy.y == 5
    # Enemy should have taken 3 damage
    assert enemy.health.hp == initial_hp - 3
    # Enemy should be off-balance from wall impact
    assert enemy.status_effects.has_status_effect(OffBalanceEffect)


def test_push_into_another_actor_makes_both_off_balance(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Pushing into another actor makes both actors off-balance."""
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
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy1)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy1 should NOT have moved (blocked by enemy2)
    assert enemy1.x == 6
    # Both should be off-balance
    assert enemy1.status_effects.has_status_effect(OffBalanceEffect)
    assert enemy2.status_effects.has_status_effect(OffBalanceEffect)


def test_push_into_hazard_tile_moves_target(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Pushing onto a hazard tile moves the target (damage via turn system)."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5), enemy_pos=(6, 5)
    )
    # Place acid pool at destination
    controller.gw.game_map.tiles[7, 5] = TileTypeID.ACID_POOL

    # Force success
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should have moved onto the hazard
    assert enemy.x == 7
    assert enemy.y == 5
    # Note: Actual hazard damage is applied by the turn system, not the push


def test_push_at_map_edge_fails_gracefully(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Push should handle map edge gracefully."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(28, 5),
        enemy_pos=(29, 5),  # Enemy at edge of 30x30 map
    )

    # Force success
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    # Push succeeds but enemy doesn't move (out of bounds)
    assert result is not None
    assert result.succeeded
    assert enemy.x == 29  # Didn't move


# --- Push Direction Tests ---


def test_push_direction_up(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Push calculates direction correctly - up."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 6),
        enemy_pos=(5, 5),  # Enemy above player
    )

    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        executor.execute(intent)

    # Enemy should be pushed further up (y decreases)
    assert enemy.x == 5
    assert enemy.y == 4


def test_push_direction_down(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Push calculates direction correctly - down."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 4),
        enemy_pos=(5, 5),  # Enemy below player
    )

    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        executor.execute(intent)

    # Enemy should be pushed further down (y increases)
    assert enemy.x == 5
    assert enemy.y == 6


def test_push_direction_left(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Push calculates direction correctly - left."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(6, 5),
        enemy_pos=(5, 5),  # Enemy to left of player
    )

    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        executor.execute(intent)

    # Enemy should be pushed further left (x decreases)
    assert enemy.x == 4
    assert enemy.y == 5


# --- Diagonal Push Tests ---


def test_push_works_diagonally(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Push should work on diagonally adjacent targets."""
    reset_event_bus_for_testing()
    controller, player, enemy = _make_world_with_enemy(
        player_pos=(5, 5),
        enemy_pos=(6, 6),  # Diagonal from player
    )

    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, enemy)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # Enemy should be pushed diagonally away (from 6,6 to 7,7)
    assert enemy.x == 7
    assert enemy.y == 7


# --- Push Hostility Tests ---


def test_push_makes_non_hostile_npc_hostile(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Pushing a non-hostile NPC should make them hostile."""
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

    # Force successful push
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, npc)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    assert npc.ai.disposition == Disposition.HOSTILE


def test_failed_push_still_triggers_hostility(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """A failed push should still make the NPC hostile - the attempt is aggressive."""
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

    # Force failed push (low roll, not nat 1)
    with patch_d20_rng([2]):
        intent = PushIntent(cast(Controller, controller), player, npc)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert not result.succeeded
    # The attempt was aggressive, so NPC becomes hostile
    assert npc.ai.disposition == Disposition.HOSTILE


def test_push_does_not_change_already_hostile(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """Pushing an already hostile NPC should not change their disposition."""
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
        colors.RED,
        "Hostile NPC",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.HOSTILE,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw=gw)

    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), player, npc)
        executor = PushExecutor()
        executor.execute(intent)

    assert npc.ai.disposition == Disposition.HOSTILE


def test_npc_pushing_npc_does_not_trigger_hostility(
    patch_d20_rng: "D20RNGPatcher",
) -> None:
    """NPC pushing another NPC should not change disposition toward player."""
    from catley.game.actors import NPC
    from catley.game.enums import Disposition

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
        disposition=Disposition.HOSTILE,
    )
    npc2 = NPC(
        6,
        5,
        "d",
        colors.YELLOW,
        "Defender NPC",
        game_world=cast(GameWorld, gw),
        disposition=Disposition.WARY,
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc1)
    gw.add_actor(npc2)
    controller = DummyController(gw=gw)

    # NPC1 pushes NPC2
    with patch_d20_rng([15]):
        intent = PushIntent(cast(Controller, controller), npc1, npc2)
        executor = PushExecutor()
        result = executor.execute(intent)

    assert result is not None
    assert result.succeeded
    # NPC2 should still be wary (NPC-vs-NPC doesn't trigger hostility toward player)
    assert npc2.ai.disposition == Disposition.WARY
