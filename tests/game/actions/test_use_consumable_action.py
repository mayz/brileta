from dataclasses import dataclass
from typing import cast

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.executors.recovery import UseConsumableExecutor
from brileta.game.actions.recovery import UseConsumableIntent
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from brileta.game.items.item_types import STIM_TYPE
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
    actor.health._hp -= 5
    item = STIM_TYPE.create()
    actor.inventory.add_to_inventory(item)

    intent = UseConsumableIntent(controller, actor, item)
    executor = UseConsumableExecutor()
    result = executor.execute(intent)

    assert result is not None
    assert actor.health.hp == actor.health.max_hp
    assert item not in actor.inventory


def test_item_type_hashable() -> None:
    s = {STIM_TYPE}
    assert STIM_TYPE in s
