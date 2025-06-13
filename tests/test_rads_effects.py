from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character, conditions
from catley.game.game_world import GameWorld
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


def make_actor() -> Character:
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    return actor


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
        self.turn_manager = TurnManager(self)

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
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = attacker
    controller = DummyController(gw=gw)
    weapon = FISTS_TYPE.create()
    action = AttackAction(controller, attacker, defender, weapon)
    return controller, attacker, defender, action


def test_rads_melee_attack_disadvantage() -> None:
    controller, attacker, defender, action = make_combat_world()
    attacker.conditions.add_condition(conditions.Rads())
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    assert attack is not None
    with patch("random.randint", side_effect=[2, 18]):
        result = cast(
            D20ResolutionResult, action._execute_attack_roll(attack, weapon, {})
        )
    assert result.has_disadvantage


def test_rads_apply_physical_disadvantage() -> None:
    actor = make_actor()
    actor.inventory.add_to_inventory(conditions.Rads())
    for stat in ("strength", "toughness", "agility"):
        mods = actor.modifiers.get_resolution_modifiers(stat)
        assert mods["has_disadvantage"]


def test_rads_not_affect_mental_stats() -> None:
    actor = make_actor()
    actor.inventory.add_to_inventory(conditions.Rads())
    for stat in ("observation", "intelligence", "demeanor", "weirdness"):
        mods = actor.modifiers.get_resolution_modifiers(stat)
        assert not mods.get("has_disadvantage", False)


def test_multiple_rads_stack() -> None:
    actor = make_actor()
    actor.inventory.add_to_inventory(conditions.Rads())
    actor.inventory.add_to_inventory(conditions.Rads())
    mods = actor.modifiers.get_resolution_modifiers("strength")
    assert mods["has_disadvantage"]
    assert len(actor.conditions.get_conditions_by_type(conditions.Rads)) == 2
