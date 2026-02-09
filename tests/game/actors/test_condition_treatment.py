from dataclasses import dataclass
from typing import cast

from brileta import colors
from brileta.controller import Controller
from brileta.game.actors import Character, conditions
from brileta.game.enums import ConsumableEffectType, InjuryLocation
from brileta.game.game_world import GameWorld
from brileta.game.items.capabilities import ConsumableEffectSpec
from brileta.game.items.item_core import ItemType
from brileta.game.items.item_types import ItemSize
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.player = actor
    gw.add_actor(actor)
    controller = DummyController(gw=gw)
    return controller, actor


def test_remove_conditions_by_type() -> None:
    _, actor = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Injury(InjuryLocation.HEAD, "Wound"))

    removed = actor.conditions.remove_conditions_by_type({conditions.Exhaustion})
    assert len(removed) == 2
    assert all(isinstance(r, conditions.Exhaustion) for r in removed)
    remaining = actor.conditions.get_all_conditions()
    assert len(remaining) == 1
    assert isinstance(remaining[0], conditions.Injury)


def test_consumable_effect_batch_removal() -> None:
    controller, actor = make_world()
    actor.health._hp = actor.health.max_hp - 5
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())

    spec = ConsumableEffectSpec(
        effect_type=ConsumableEffectType.HEAL,
        target_condition_types={conditions.Exhaustion},
    )
    item_type = ItemType(
        name="Test", description="", size=ItemSize.NORMAL, consumable_effect=spec
    )
    item = item_type.create()

    assert item.consumable_effect is not None
    item.consumable_effect.consume(actor, controller)

    assert actor.health.hp == actor.health.max_hp
    assert not actor.conditions.get_conditions_by_type(conditions.Exhaustion)
