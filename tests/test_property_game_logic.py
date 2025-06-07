from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game import range_system
from catley.game.actions.actions import AreaEffectAction, AttackAction
from catley.game.actors import Character
from catley.game.items.item_types import (
    GRENADE_TYPE,
    HUNTING_RIFLE_TYPE,
    SNIPER_RIFLE_TYPE,
    SUBMACHINE_GUN_TYPE,
)
from catley.game.items.properties import TacticalProperty
from catley.util import dice
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(10, 10)
        self.game_map.transparent[:] = True
        self.actors: list[Character] = []
        self.player: Character | None = None


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []

    def add_message(self, text: str, *_args, **_kwargs) -> None:
        self.messages.append(text)


@dataclass
class DummyFrameManager:
    effects: list[str]

    def __init__(self) -> None:
        self.effects = []

    def create_effect(self, name: str, *args, **kwargs) -> None:
        self.effects.append(name)


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def _make_world(weapon_key: str):
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    gw.actors.extend([attacker, defender])
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    if weapon_key == "hunting":
        weapon = HUNTING_RIFLE_TYPE.create()
    elif weapon_key == "sniper":
        weapon = SNIPER_RIFLE_TYPE.create()
    elif weapon_key == "smg":
        weapon = SUBMACHINE_GUN_TYPE.create()
    else:
        weapon = GRENADE_TYPE.create()
    return controller, attacker, defender, weapon


def test_awkward_weapon_miss_effect() -> None:
    controller, attacker, defender, weapon = _make_world("hunting")
    action = AttackAction(cast(Controller, controller), attacker, defender, weapon)
    attack = weapon.melee_attack
    assert attack is not None
    result = dice.CheckResult(success=False)
    action._handle_attack_miss(result, attack, weapon)
    assert any("off balance" in msg for msg in controller.message_log.messages)


def test_automatic_weapon_ammo_consumption() -> None:
    controller, attacker, defender, weapon = _make_world("smg")
    ranged = weapon.ranged_attack
    assert ranged is not None
    action = AreaEffectAction(cast(Controller, controller), attacker, 3, 3, weapon)
    ranged.current_ammo = 5
    action._consume_ammo(ranged)
    assert ranged.current_ammo == 2


def test_scoped_weapon_range_modifier() -> None:
    controller, attacker, defender, weapon = _make_world("sniper")
    ranged = weapon.ranged_attack
    assert ranged is not None
    mod = range_system.get_range_modifier(weapon, "far")
    assert mod == {"has_advantage": True}


def test_explosive_visual_effect() -> None:
    controller, attacker, defender, weapon = _make_world("grenade")
    effect = weapon.area_effect
    assert effect is not None
    action = AreaEffectAction(cast(Controller, controller), attacker, 5, 5, weapon)
    action._trigger_visual_effect(effect)
    assert controller.frame_manager.effects == ["explosion"]


def test_smoke_visual_effect() -> None:
    # Create custom smoke grenade
    from catley.game.enums import ItemSize
    from catley.game.items.capabilities import AreaEffectSpec
    from catley.game.items.item_core import ItemType

    smoke_type = ItemType(
        name="Smoke",
        description="",
        size=ItemSize.TINY,
        area_effect=AreaEffectSpec(
            "d4", "circle", 1, properties={TacticalProperty.SMOKE}
        ),
    )
    weapon = smoke_type.create()
    controller, attacker, defender, _ = _make_world("grenade")
    action = AreaEffectAction(cast(Controller, controller), attacker, 2, 2, weapon)
    effect = weapon.area_effect
    assert effect is not None
    action._trigger_visual_effect(effect)
    assert controller.frame_manager.effects == ["smoke_cloud"]
