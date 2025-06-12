from dataclasses import dataclass
from typing import cast

from catley import colors
from catley.controller import Controller
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import PC, Character
from catley.game.conditions import Injury
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.game.status_effects import (
    FocusedEffect,
    OffBalanceEffect,
    StrengthBoostEffect,
    TrippedEffect,
)
from catley.game.turn_manager import TurnManager
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.action_cost = 100
        self.turn_manager = TurnManager(cast(Controller, self))


def make_world() -> tuple[DummyController, PC]:
    gw = DummyGameWorld()
    actor = PC(
        0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw), strength=5
    )
    gw.player = actor
    gw.add_actor(actor)
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


def test_offbalance_persists_until_next_round() -> None:
    controller, actor = make_world()
    tm = controller.turn_manager

    class DummyAction(GameAction):
        def __init__(
            self, controller: Controller, actor: Character, apply: bool
        ) -> None:
            super().__init__(controller, actor)
            self.apply = apply

        def execute(self) -> GameActionResult | None:  # pragma: no cover - simple
            if self.apply:
                actor.apply_status_effect(OffBalanceEffect())
            return None

    # First round applies the effect
    tm.queue_action(DummyAction(cast(Controller, controller), actor, True))
    tm.process_unified_round()
    assert actor.has_status_effect(OffBalanceEffect)

    # Next round should remove it before acting again
    tm.queue_action(DummyAction(cast(Controller, controller), actor, False))
    tm.process_unified_round()
    assert not actor.has_status_effect(OffBalanceEffect)


def test_condition_management_methods() -> None:
    controller, actor = make_world()
    injury = Injury(InjuryLocation.LEFT_ARM, "Test")
    assert actor.add_condition(injury)
    assert actor.has_condition(Injury)
    assert injury in actor.get_conditions()
    assert actor.get_conditions_by_type(Injury) == [injury]
    assert actor.remove_condition(injury)
    assert not actor.has_condition(Injury)
