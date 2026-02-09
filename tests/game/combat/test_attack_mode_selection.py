from dataclasses import dataclass
from typing import cast

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from brileta.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    PISTOL_TYPE,
    SNIPER_RIFLE_TYPE,
)
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character, Character]:
    gw = DummyGameWorld()
    attacker = Character(
        1,
        1,
        "A",
        colors.WHITE,
        "Attacker",
        game_world=cast(GameWorld, gw),
    )
    defender = Character(
        1,
        2,
        "D",
        colors.WHITE,
        "Defender",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(attacker)
    gw.add_actor(defender)
    controller = DummyController(gw=gw, frame_manager=object())
    return controller, attacker, defender


def test_sniper_prefers_ranged_adjacent() -> None:
    controller, attacker, defender = make_world()
    weapon = SNIPER_RIFLE_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)

    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    attack, _ = AttackExecutor()._determine_attack_method(intent)

    assert attack is weapon.ranged_attack


def test_pistol_prefers_ranged_adjacent() -> None:
    controller, attacker, defender = make_world()
    weapon = PISTOL_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)

    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    attack, _ = AttackExecutor()._determine_attack_method(intent)

    assert attack is weapon.ranged_attack


def test_knife_prefers_melee_adjacent() -> None:
    controller, attacker, defender = make_world()
    weapon = COMBAT_KNIFE_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)

    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    attack, _ = AttackExecutor()._determine_attack_method(intent)

    assert attack is weapon.melee_attack


def test_manual_mode_overrides_preferences() -> None:
    controller, attacker, defender = make_world()
    weapon = SNIPER_RIFLE_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)

    intent = AttackIntent(
        cast(Controller, controller),
        attacker,
        defender,
        weapon,
        attack_mode="melee",
    )
    attack, _ = AttackExecutor()._determine_attack_method(intent)

    assert attack is weapon.melee_attack
