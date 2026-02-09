from __future__ import annotations

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
    FISTS_TYPE,
    HUNTING_RIFLE_TYPE,
    PISTOL_TYPE,
)
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character, Character]:
    gw = DummyGameWorld()
    player = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    enemy = Character(
        1,
        0,
        "E",
        colors.WHITE,
        "Enemy",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(enemy)
    controller = DummyController(gw=gw, frame_manager=object())
    return controller, player, enemy


def test_ramming_uses_melee_weapon_when_available() -> None:
    controller, player, enemy = make_world()

    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(knife, 1)
    player.inventory.switch_to_slot(0)

    intent = AttackIntent(
        cast(Controller, controller),
        player,
        enemy,
        weapon=None,
        attack_mode="melee",
    )
    attack, weapon = AttackExecutor()._determine_attack_method(intent)

    assert weapon is knife
    assert attack is knife.melee_attack


def test_ramming_falls_back_to_fists_when_no_melee_weapon() -> None:
    controller, player, enemy = make_world()

    pistol = PISTOL_TYPE.create()
    rifle = HUNTING_RIFLE_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)
    player.inventory.equip_to_slot(rifle, 1)
    player.inventory.switch_to_slot(0)

    intent = AttackIntent(
        cast(Controller, controller),
        player,
        enemy,
        weapon=None,
        attack_mode="melee",
    )
    attack, weapon = AttackExecutor()._determine_attack_method(intent)

    assert weapon.item_type is FISTS_TYPE
    assert attack is weapon.melee_attack
