from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.combat import AttackAction
from catley.game.actors import PC, Character
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.game.status_effects import (
    FocusedEffect,
    OffBalanceEffect,
    StrengthBoostEffect,
    TrippedEffect,
)
from catley.turn_manager import TurnManager
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld(GameWorld):
    def __init__(self) -> None:
        self.game_map = GameMap(5, 5)
        self.game_map.transparent[:] = True
        self.actors: list[Character] = []
        self.player: Character | None = None


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

    def update_fov(self) -> None:  # pragma: no cover - simple flag
        self.update_fov_called = True


def make_combat_world() -> tuple[DummyController, Character, Character, AttackAction]:
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
    gw.actors.extend([attacker, defender])
    gw.player = attacker
    controller = DummyController(gw=gw)
    weapon = FISTS_TYPE.create()
    action = AttackAction(cast(Controller, controller), attacker, defender, weapon)
    return controller, attacker, defender, action


def test_offbalance_gives_disadvantage() -> None:
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.apply_status_effect(OffBalanceEffect())
    with patch("random.randint", side_effect=[2, 18]):
        result = cast(
            D20ResolutionResult,
            action._execute_attack_roll(attack, weapon, {}),
        )
    assert result.has_disadvantage
    assert result.final_roll_used == 2


def test_focused_gives_advantage() -> None:
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.apply_status_effect(FocusedEffect())
    with patch("random.randint", side_effect=[5, 17]):
        result = cast(
            D20ResolutionResult,
            action._execute_attack_roll(attack, weapon, {}),
        )
    assert result.has_advantage
    assert result.final_roll_used == 17


def test_modifier_combination_cancels() -> None:
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.apply_status_effect(FocusedEffect())
    with patch("random.randint", return_value=11):
        result = cast(
            D20ResolutionResult,
            action._execute_attack_roll(
                attack,
                weapon,
                {"has_disadvantage": True},
            ),
        )
    assert not result.has_advantage
    assert not result.has_disadvantage
    assert result.final_roll_used == 11


def test_strength_boost_applies_to_roll() -> None:
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    attacker.apply_status_effect(StrengthBoostEffect(duration=1))
    with patch("random.randint", return_value=10):
        result = cast(
            D20ResolutionResult,
            action._execute_attack_roll(attack, weapon, {}),
        )
    assert result.total_value == 10 + 7  # strength boosted to 7


def test_tripped_skips_turn() -> None:
    gw = DummyGameWorld()
    player = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.player = player
    gw.actors.append(player)
    controller = DummyController(gw=gw)
    tm = TurnManager(cast(Controller, controller))

    player.apply_status_effect(TrippedEffect())
    tm.queue_action(
        AttackAction(cast(Controller, controller), player, player, FISTS_TYPE.create())
    )
    tm.process_unified_round()
    assert not controller.update_fov_called
    assert player.accumulated_energy == player.speed
    assert not player.has_status_effect(TrippedEffect)
