from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from catley import colors
from catley.controller import Controller
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap

if TYPE_CHECKING:
    from catley.environment.map import MapRegion
from catley.events import reset_event_bus_for_testing
from catley.game.actions.area_effects import AreaEffectIntent
from catley.game.actions.combat import AttackIntent
from catley.game.actions.executors.area_effects import WeaponAreaEffectExecutor
from catley.game.actions.executors.combat import AttackExecutor
from catley.game.actors import Character, conditions
from catley.game.enums import OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.item_types import DIRTY_BOMB_TYPE, RADIUM_GUN_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.game.resolution.outcomes import CombatOutcome
from tests.helpers import DummyGameWorld


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []

    def add_message(self, text: str, *_args, **_kwargs) -> None:
        self.messages.append(text)


@dataclass
class DummyFrameManager:
    def create_effect(self, *args, **kwargs) -> None:
        pass

    def trigger_screen_shake(self, *_args, **_kwargs) -> None:
        pass


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def make_weapon_world():
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)
    gw.add_actor(defender)
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    weapon = RADIUM_GUN_TYPE.create()
    return controller, attacker, defender, weapon


class DummyMap(GameMap):
    def __init__(self, width: int, height: int) -> None:
        tiles = np.full((width, height), 1, dtype=np.uint8)
        regions: dict[int, MapRegion] = {}
        map_data = GeneratedMapData(
            tiles=tiles,
            regions=regions,
            tile_to_region_id=np.full((width, height), -1, dtype=np.int16),
        )
        super().__init__(width, height, map_data)
        self._transparent_map_cache = np.ones((width, height), dtype=bool)


def make_bomb_world():
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld(game_map=game_map)
    gw.actors = []
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    target = Character(
        5, 4, "T", colors.YELLOW, "Target", game_world=cast(GameWorld, gw)
    )
    gw.add_actor(attacker)
    gw.add_actor(target)
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    bomb = DIRTY_BOMB_TYPE.create()
    intent = AreaEffectIntent(controller, attacker, 5, 5, bomb)
    executor = WeaponAreaEffectExecutor()
    return controller, attacker, target, intent, executor


def test_radiation_damage_direct() -> None:
    gw = DummyGameWorld()
    actor = Character(0, 0, "T", colors.WHITE, "Target", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    actor.health.ap = 5
    actor.take_damage(3, damage_type="radiation")
    assert actor.health.hp == actor.health.max_hp - 3
    assert actor.health.ap == 5
    assert len(actor.conditions.get_conditions_by_type(conditions.Rads)) == 3


def test_radiation_weapon_attack() -> None:
    controller, attacker, defender, weapon = make_weapon_world()
    defender.health.ap = 4
    attack = weapon.ranged_attack
    assert attack is not None
    result = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
    outcome = CombatOutcome(damage_dealt=2)
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    executor = AttackExecutor()
    damage = executor._apply_combat_outcome(intent, result, outcome, attack, weapon)
    assert damage == 2
    assert defender.health.hp == defender.health.max_hp - 2
    assert defender.health.ap == 4
    assert len(defender.conditions.get_conditions_by_type(conditions.Rads)) == 2


def test_dirty_bomb_area_effect() -> None:
    reset_event_bus_for_testing()
    _controller, _attacker, target, intent, executor = make_bomb_world()
    effect = intent.weapon.area_effect
    assert effect is not None
    tiles = executor._circle_tiles(intent, effect)

    def fixed_randint(_a: int, _b: int) -> int:
        return 6

    with patch("random.randint", fixed_randint):
        hits = executor._apply_damage(intent, tiles, effect)
    assert hits == [(target, 4)]
    assert target.health.hp == target.health.max_hp - 4
    assert len(target.conditions.get_conditions_by_type(conditions.Rads)) == 4
