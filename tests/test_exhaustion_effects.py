from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.constants.movement import MovementConstants
from catley.controller import Controller
from catley.game.actions.movement import MoveAction
from catley.game.actors import Character
from catley.game.conditions import Exhaustion, Injury
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.game.resolution.d20_system import D20ResolutionResult, D20Resolver
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None


def make_world() -> tuple[DummyController, Character]:
    gw = DummyGameWorld()
    actor = Character(
        0,
        0,
        "@",
        colors.WHITE,
        "Act",
        game_world=cast(GameWorld, gw),
    )
    gw.player = actor
    gw.add_actor(actor)
    controller = DummyController(gw=gw)
    return controller, actor


def test_single_exhaustion_energy_reduction() -> None:
    controller, actor = make_world()
    actor.accumulated_energy = 0
    actor.add_condition(Exhaustion())
    expected = int(
        actor.calculate_effective_speed() * actor.get_exhaustion_energy_multiplier()
    )
    actor.regenerate_energy()
    assert actor.accumulated_energy == expected
    assert not actor.has_exhaustion_disadvantage()


def test_double_exhaustion_disadvantage_and_energy() -> None:
    controller, actor = make_world()
    actor.accumulated_energy = 0
    actor.add_condition(Exhaustion())
    actor.add_condition(Exhaustion())
    expected = int(
        actor.calculate_effective_speed() * actor.get_exhaustion_energy_multiplier()
    )
    actor.regenerate_energy()
    assert actor.accumulated_energy == expected
    assert actor.has_exhaustion_disadvantage()

    target = Character(
        1, 0, "T", colors.WHITE, "Tar", game_world=cast(GameWorld, controller.gw)
    )
    with patch("random.randint", side_effect=[4, 18]):
        result = D20Resolver(0, 10).resolve(actor, target)
    assert isinstance(result, D20ResolutionResult)
    assert result.has_disadvantage
    assert result.final_roll_used == 4


def test_movement_stumble_with_high_exhaustion() -> None:
    controller, actor = make_world()
    actor.add_condition(Exhaustion())
    actor.add_condition(Exhaustion())
    actor.add_condition(Exhaustion())
    actor.add_condition(Exhaustion())

    with patch("random.random", return_value=0.05):
        action = MoveAction(controller, actor, dx=1, dy=0)
        result = action.execute()
    assert result is None
    assert (actor.x, actor.y) == (0, 0)


def test_exhaustion_removal_restores_effects() -> None:
    controller, actor = make_world()
    actor.add_condition(Exhaustion())
    actor.add_condition(Exhaustion())
    # Remove one stack
    exhaustion = actor.get_conditions_by_type(Exhaustion)[0]
    actor.remove_condition(exhaustion)
    assert not actor.has_exhaustion_disadvantage()
    actor.accumulated_energy = 0
    expected = int(
        actor.calculate_effective_speed() * actor.get_exhaustion_energy_multiplier()
    )
    actor.regenerate_energy()
    assert actor.accumulated_energy == expected


def test_injury_and_exhaustion_stack() -> None:
    controller, actor = make_world()
    actor.add_condition(Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    actor.add_condition(Exhaustion())
    expected_speed = int(
        actor.speed * 0.75 * MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK
    )
    assert actor.calculate_effective_speed() == expected_speed
    expected_energy = int(expected_speed * actor.get_exhaustion_energy_multiplier())
    actor.accumulated_energy = 0
    actor.regenerate_energy()
    assert actor.accumulated_energy == expected_energy
