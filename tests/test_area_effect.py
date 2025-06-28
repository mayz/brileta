from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import numpy as np

from catley import colors
from catley.controller import Controller
from catley.environment.generators import GeneratedMapData
from catley.environment.map import GameMap

if TYPE_CHECKING:
    from catley.environment.map import MapRegion
from catley.events import (
    EffectEvent,
    MessageEvent,
    reset_event_bus_for_testing,
    subscribe_to_event,
)
from catley.game.actions.area_effects import AreaEffectIntent
from catley.game.actions.executors.area_effects import AreaEffectExecutor
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_types import GRENADE_TYPE
from catley.view.frame_manager import FrameManager
from catley.view.render.graphics import GraphicsContext
from tests.helpers import DummyGameWorld


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


class DummyMessageLog:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.revision = 0
        subscribe_to_event(MessageEvent, self._on_message_event)

    def _on_message_event(self, event: MessageEvent) -> None:
        self.add_message(event.text, event.color, stack=event.stack)

    def add_message(
        self, text: str, _fg: colors.Color = colors.WHITE, *, stack: bool = True
    ) -> None:
        self.messages.append(text)
        self.revision += 1


class DummyFrameManager(FrameManager):
    def __init__(self, controller: DummyController) -> None:
        # We don't need the full FrameManager setup.
        # We will mock the parts we don't want to initialize.
        self.controller = controller
        # Create a mock Renderer that satisfies the type hint.
        # We don't need a real TCOD context for this test.
        self.renderer = MagicMock(spec=GraphicsContext)

    def create_effect(
        self,
        effect_name: str,
        x: int,
        y: int,
        intensity: float = 1.0,
        direction_x: float = 0.0,
        direction_y: float = 0.0,
    ) -> None:
        pass


class DummyController(Controller):
    def __init__(self, gw: GameWorld) -> None:
        # Skip Controller.__init__ to avoid creating a full game.
        self.gw = gw
        self.message_log = DummyMessageLog()
        self.frame_manager = DummyFrameManager(self)


def make_world() -> tuple[
    DummyController,
    Character,
    Character,
    AreaEffectIntent,
    AreaEffectExecutor,
    GameMap,
]:
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld()
    gw.game_map = game_map
    gw.actors = []
    attacker = Character(
        1, 1, "A", colors.WHITE, "Attacker", game_world=cast(GameWorld, gw)
    )
    target = Character(
        5, 4, "T", colors.YELLOW, "Target", game_world=cast(GameWorld, gw)
    )
    target.health.ap = 0
    gw.add_actor(attacker)
    gw.add_actor(target)
    controller = DummyController(cast(GameWorld, gw))
    grenade = GRENADE_TYPE.create()
    intent = AreaEffectIntent(controller, attacker, 5, 5, grenade)
    executor = AreaEffectExecutor()
    return controller, attacker, target, intent, executor, game_map


def test_grenade_circle_tiles() -> None:
    controller, attacker, target, intent, executor, _ = make_world()
    effect = intent.weapon.area_effect
    assert effect is not None
    tiles = executor._circle_tiles(intent, effect)
    assert len(tiles) == 49
    assert tiles[(5, 5)] == 0
    assert tiles[(8, 5)] == 3


def test_grenade_damage_calculation() -> None:
    controller, attacker, target, intent, executor, _ = make_world()
    effect = intent.weapon.area_effect
    assert effect is not None
    tiles = executor._circle_tiles(intent, effect)

    def fixed_randint(_a: int, _b: int) -> int:
        return 6

    with patch("random.randint", fixed_randint):
        hits = executor._apply_damage(intent, tiles, effect)
    assert hits == [(target, 4)]
    assert target.health.hp == target.health.max_hp - 4


def test_grenade_action_execution() -> None:
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, target, intent, executor, _ = make_world()

    def fixed_randint(_a: int, _b: int) -> int:
        return 6

    with patch("random.randint", fixed_randint):
        executor.execute(intent)
    assert target.health.hp == target.health.max_hp - 4
    assert controller.message_log.messages[-1] == "Target takes 4 damage."
    assert effects == ["explosion"]
