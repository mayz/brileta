from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.combat import AttackIntent
from catley.game.actions.executors.combat import AttackExecutor
from catley.game.actors import PC, Character, status_effects
from catley.game.game_world import GameWorld
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None

    def __post_init__(self) -> None:
        self.update_fov_called = False
        if self.message_log is None:
            self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()
        self.action_cost = 100
        self.turn_manager = TurnManager(cast(Controller, self))

    def update_fov(self) -> None:  # pragma: no cover - simple flag
        self.update_fov_called = True

    def run_one_turn(self) -> None:
        # Start of Turn phase: All actors regenerate energy and process status effects
        for actor in self.gw.actors:
            actor.update_turn(cast(Controller, self))
            if actor.energy is not None:
                actor.energy.regenerate()

        # Player action (check if there's a queued action)
        player_action = self.turn_manager.dequeue_player_action()
        if player_action and self.gw.player:
            self.turn_manager.execute_intent(player_action)
            if self.gw.player.energy is not None:
                self.gw.player.energy.spend(self.action_cost)

        # NPC Action Resolution: Process all NPCs with sufficient energy
        for actor in list(self.gw.actors):
            if actor is self.gw.player:
                continue
            if actor.energy is not None and actor.energy.can_afford(self.action_cost):
                action = actor.get_next_action(cast(Controller, self))
                if action is not None:
                    self.turn_manager.execute_intent(action)
                    actor.energy.spend(self.action_cost)


def make_combat_world() -> tuple[
    DummyController, Character, Character, AttackIntent, AttackExecutor
]:
    gw = DummyGameWorld()
    attacker = Character(
        1,
        1,
        "A",
        colors.WHITE,
        "Att",
        game_world=cast(GameWorld, gw),
        strength=5,
        agility=5,
    )
    defender = Character(
        2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw), agility=5
    )
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = attacker
    controller = DummyController(gw=gw)
    weapon = FISTS_TYPE.create()
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    executor = AttackExecutor()
    return controller, attacker, defender, intent, executor


def test_offbalance_gives_disadvantage(
    patch_d20_rng: Callable[[list[int]], AbstractContextManager[Any]],
) -> None:
    _controller, attacker, _defender, intent, executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.status_effects.apply_status_effect(status_effects.OffBalanceEffect())
    # Disadvantage takes the lower of 2 rolls: 2 and 18 -> 2
    with patch_d20_rng([2, 18]):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(intent, attack, weapon, {}),
        )
    assert result.has_disadvantage
    assert result.final_roll_used == 2


def test_focused_gives_advantage(
    patch_d20_rng: Callable[[list[int]], AbstractContextManager[Any]],
) -> None:
    _controller, attacker, _defender, intent, executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.status_effects.apply_status_effect(status_effects.FocusedEffect())
    # Advantage takes the higher of 2 rolls: 5 and 17 -> 17
    with patch_d20_rng([5, 17]):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(intent, attack, weapon, {}),
        )
    assert result.has_advantage
    assert result.final_roll_used == 17


def test_modifier_combination_cancels(
    patch_d20_rng: Callable[[list[int]], AbstractContextManager[Any]],
) -> None:
    _controller, attacker, _defender, intent, executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.status_effects.apply_status_effect(status_effects.FocusedEffect())
    # Advantage + disadvantage cancel out, single roll
    with patch_d20_rng([11]):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(
                intent,
                attack,
                weapon,
                {"has_disadvantage": True},
            ),
        )
    assert not result.has_advantage
    assert not result.has_disadvantage
    assert result.final_roll_used == 11


def test_strength_boost_applies_to_roll(
    patch_d20_rng: Callable[[list[int]], AbstractContextManager[Any]],
) -> None:
    _controller, attacker, _defender, intent, executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.status_effects.apply_status_effect(
        status_effects.StrengthBoostEffect(duration=1)
    )
    with patch_d20_rng([10]):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(intent, attack, weapon, {}),
        )
    assert result.total_value == 10 + 7  # strength boosted to 7


def test_tripped_skips_turn() -> None:
    """TrippedEffect should prevent the actor's action for 2 turns."""
    gw = DummyGameWorld()
    player = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.add_actor(player)
    controller = DummyController(gw=gw)
    tm = controller.turn_manager

    player.status_effects.apply_status_effect(status_effects.TrippedEffect())
    tm.queue_action(
        AttackIntent(cast(Controller, controller), player, player, FISTS_TYPE.create())
    )
    controller.run_one_turn()
    # First turn: action was prevented, but TrippedEffect lasts 2 turns
    assert not controller.update_fov_called
    assert player.energy.accumulated_energy == player.energy.speed
    # Effect still active after 1 turn (duration=2)
    assert player.status_effects.has_status_effect(status_effects.TrippedEffect)

    # Second turn: effect expires
    controller.update_fov_called = False
    tm.queue_action(
        AttackIntent(cast(Controller, controller), player, player, FISTS_TYPE.create())
    )
    controller.run_one_turn()
    assert not controller.update_fov_called  # Still prevented on second turn
    assert not player.status_effects.has_status_effect(status_effects.TrippedEffect)
