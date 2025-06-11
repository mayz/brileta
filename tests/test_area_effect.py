from unittest.mock import MagicMock, patch

import numpy as np
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
from catley.game.actors import Actor, Character
from catley.game.items.item_types import GRENADE_TYPE
from catley.util.spatial import SpatialHashGrid
from catley.view.frame_manager import FrameManager
from catley.view.renderer import Renderer


class DummyMap(GameMap):
    def __init__(self, width: int, height: int) -> None:
        super().__init__(width, height)
        self._transparent_map_cache = np.ones((width, height), dtype=bool)


class DummyGameWorld(GameWorld):
    def __init__(self, game_map: GameMap, actors: list[Actor]):
        # Do not call super().__init__ to avoid heavy setup.
        self.game_map = game_map
        self.actors = actors
        self.actor_spatial_index = SpatialHashGrid(cell_size=16)

    def add_actor(self, actor: Actor) -> None:
        self.actors.append(actor)
        self.actor_spatial_index.add(actor)

    def remove_actor(self, actor: Actor) -> None:
        try:
            self.actors.remove(actor)
            self.actor_spatial_index.remove(actor)
        except ValueError:
            pass


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
    def __init__(self, controller: "DummyController") -> None:
        # We don't need the full FrameManager setup.
        # We will mock the parts we don't want to initialize.
        self.controller = controller
        # Create a mock Renderer that satisfies the type hint.
        # We don't need a real TCOD context for this test.
        self.renderer = MagicMock(spec=Renderer)

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
    DummyController, Character, Character, AreaEffectAction, GameMap
]:
    game_map = DummyMap(10, 10)
    gw = DummyGameWorld(game_map=game_map, actors=[])
    attacker = Character(1, 1, "A", colors.WHITE, "Attacker", game_world=gw)
    target = Character(5, 4, "T", colors.YELLOW, "Target", game_world=gw)
    target.health.ap = 0
    gw.actors.extend([attacker, target])
    controller = DummyController(gw)
    grenade = GRENADE_TYPE.create()
    action = AreaEffectAction(game_map, gw.actors, attacker, 5, 5, grenade)
    return controller, attacker, target, action, game_map


def test_grenade_circle_tiles() -> None:
    controller, attacker, target, action, _ = make_world()
    effect = action.weapon.area_effect
    assert effect is not None
    tiles = action._circle_tiles(effect)
    assert len(tiles) == 49
    assert tiles[(5, 5)] == 0
    assert tiles[(8, 5)] == 3


def test_grenade_damage_calculation() -> None:
    controller, attacker, target, action, _ = make_world()
    effect = action.weapon.area_effect
    assert effect is not None
    tiles = action._circle_tiles(effect)

    def fixed_randint(_a: int, _b: int) -> int:
        return 6

    with patch("random.randint", fixed_randint):
        hits = action._apply_damage(tiles, effect)
    assert hits == [(target, 4)]
    assert target.health.hp == target.health.max_hp - 4


def test_grenade_action_execution() -> None:
    reset_event_bus_for_testing()
    effects: list[str] = []
    subscribe_to_event(EffectEvent, lambda e: effects.append(e.effect_name))
    controller, attacker, target, action, _ = make_world()

    def fixed_randint(_a: int, _b: int) -> int:
        return 6

    with patch("random.randint", fixed_randint):
        action.execute()
    assert target.health.hp == target.health.max_hp - 4
    assert controller.message_log.messages[-1] == "Target takes 4 damage."
    assert effects == ["explosion"]
