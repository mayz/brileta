from dataclasses import dataclass
from typing import cast

from game.game_world import GameWorld

from catley import colors
from catley.controller import Controller
from catley.environment.map import GameMap
from catley.game.actors import Character
from catley.game.conditions import Injury
from catley.game.status_effects import (
    FocusedEffect,
    OffBalanceEffect,
    StrengthBoostEffect,
    TrippedEffect,
)
from catley.util.spatial import SpatialHashGrid


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(1, 1)
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
class DummyController:
    gw: DummyGameWorld


def make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    actor = Character(
        0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw), strength=5
    )
    gw.player = actor
    gw.actors.append(actor)
    controller = DummyController(gw=gw)
    return controller, actor


def test_offbalance_effect_expires() -> None:
    controller, actor = make_world()
    actor.apply_status_effect(OffBalanceEffect())
    assert actor.has_status_effect(OffBalanceEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.has_status_effect(OffBalanceEffect)


def test_focused_effect_expires() -> None:
    controller, actor = make_world()
    actor.apply_status_effect(FocusedEffect())
    assert actor.has_status_effect(FocusedEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.has_status_effect(FocusedEffect)


def test_tripped_effect_expires() -> None:
    controller, actor = make_world()
    actor.apply_status_effect(TrippedEffect())
    assert actor.has_status_effect(TrippedEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.has_status_effect(TrippedEffect)


def test_strength_boost_duration() -> None:
    controller, actor = make_world()
    actor.apply_status_effect(StrengthBoostEffect(duration=2))
    assert actor.stats.strength == 7
    actor.update_turn(cast(Controller, controller))
    assert actor.stats.strength == 7
    actor.update_turn(cast(Controller, controller))
    assert actor.stats.strength == 5
    assert not actor.has_status_effect(StrengthBoostEffect)


def test_multiple_effects_coexist() -> None:
    controller, actor = make_world()
    actor.apply_status_effect(OffBalanceEffect())
    actor.apply_status_effect(StrengthBoostEffect(duration=2))
    assert actor.has_status_effect(OffBalanceEffect)
    assert actor.has_status_effect(StrengthBoostEffect)
    actor.update_turn(cast(Controller, controller))
    assert not actor.has_status_effect(OffBalanceEffect)
    assert actor.has_status_effect(StrengthBoostEffect)


def test_condition_management_methods() -> None:
    controller, actor = make_world()
    injury = Injury()
    assert actor.add_condition(injury)
    assert actor.has_condition(Injury)
    assert injury in actor.get_conditions()
    assert actor.get_conditions_by_type(Injury) == [injury]
    assert actor.remove_condition(injury)
    assert not actor.has_condition(Injury)
