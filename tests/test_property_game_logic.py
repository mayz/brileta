from dataclasses import dataclass
from typing import cast

from game import ranges
from game.game_world import GameWorld

from catley import colors
from catley.controller import Controller
from catley.environment.map import GameMap
from catley.events import (
    EffectEvent,
    MessageEvent,
    reset_event_bus_for_testing,
    subscribe_to_event,
)
from catley.game.actions.area_effects import AreaEffectAction
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.items.item_types import (
    GRENADE_TYPE,
    HUNTING_RIFLE_TYPE,
    SNIPER_RIFLE_TYPE,
    SUBMACHINE_GUN_TYPE,
)
from catley.game.items.properties import TacticalProperty
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.util.spatial import SpatialHashGrid


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(10, 10)
        self.game_map.transparent[:] = True
        self.actors: list[Character] = []
        self.player: Character | None = None
        self.actor_spatial_index = SpatialHashGrid(cell_size=16)

    def add_actor(self, actor: Character) -> None:
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Character) -> None:
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []
        subscribe_to_event(MessageEvent, lambda e: self.add_message(e.text))

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
    result = D20ResolutionResult(outcome_tier=OutcomeTier.FAILURE)
    action._handle_attack_miss(result, attack, weapon)
    assert any("off balance" in msg for msg in controller.message_log.messages)


def test_awkward_weapon_ranged_miss_no_effect() -> None:
    controller, attacker, defender, weapon = _make_world("sniper")
    action = AttackAction(cast(Controller, controller), attacker, defender, weapon)
    attack = weapon.ranged_attack
    assert attack is not None
    result = D20ResolutionResult(outcome_tier=OutcomeTier.FAILURE)
    action._handle_attack_miss(result, attack, weapon)
    assert not any("off balance" in msg for msg in controller.message_log.messages)


def test_automatic_weapon_ammo_consumption() -> None:
    controller, attacker, defender, weapon = _make_world("smg")
    ranged = weapon.ranged_attack
    assert ranged is not None
    action = AreaEffectAction(
        controller.gw.game_map, controller.gw.actors, attacker, 3, 3, weapon
    )
    ranged.current_ammo = 5
    action._consume_ammo(ranged)
    assert ranged.current_ammo == 2


def test_scoped_weapon_range_modifier() -> None:
    controller, attacker, defender, weapon = _make_world("sniper")
    ranged = weapon.ranged_attack
    assert ranged is not None
    mod = ranges.get_range_modifier(weapon, "far")
    assert mod == {"has_advantage": True}


def test_explosive_visual_effect() -> None:
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, defender, weapon = _make_world("grenade")
    effect = weapon.area_effect
    assert effect is not None
    action = AreaEffectAction(
        controller.gw.game_map, controller.gw.actors, attacker, 5, 5, weapon
    )
    action._trigger_visual_effect(effect)
    assert effects == ["explosion"]


def test_smoke_visual_effect() -> None:
    # Create custom smoke grenade
    from catley.game.enums import AreaType, ItemSize
    from catley.game.items.capabilities import AreaEffectSpec
    from catley.game.items.item_core import ItemType

    smoke_type = ItemType(
        name="Smoke",
        description="",
        size=ItemSize.TINY,
        area_effect=AreaEffectSpec(
            "d4", AreaType.CIRCLE, 1, properties={TacticalProperty.SMOKE}
        ),
    )
    weapon = smoke_type.create()
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, defender, _ = _make_world("grenade")
    action = AreaEffectAction(
        controller.gw.game_map, controller.gw.actors, attacker, 2, 2, weapon
    )
    effect = weapon.area_effect
    assert effect is not None
    action._trigger_visual_effect(effect)
    assert effects == ["smoke_cloud"]
