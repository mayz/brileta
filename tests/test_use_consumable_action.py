from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.recovery import UseConsumableAction
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.items.item_types import STIM_TYPE
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    actor = Character(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    gw.player = actor
    controller = DummyController(gw=gw)
    return controller, actor


def test_use_consumable_action_heals_and_consumes_item() -> None:
    controller, actor = make_world()
    actor.health.hp -= 5
    item = STIM_TYPE.create()
    actor.inventory.add_to_inventory(item)

    action = UseConsumableAction(controller, actor, item)
    result = action.execute()

    assert result is not None
    assert actor.health.hp == actor.health.max_hp
    assert item not in actor.inventory


def test_item_type_hashable() -> None:
    s = {STIM_TYPE}
    assert STIM_TYPE in s
