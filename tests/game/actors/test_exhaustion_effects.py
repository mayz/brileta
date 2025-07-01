from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.constants.movement import MovementConstants
from catley.controller import Controller
from catley.game.actions.executors.movement import MoveExecutor
from catley.game.actions.movement import MoveIntent
from catley.game.actors import Character, conditions
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.game.resolution.d20_system import D20ResolutionResult, D20System
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
    actor.energy.accumulated_energy = 0
    actor.conditions.add_condition(conditions.Exhaustion())
    expected = int(actor.energy.speed * actor.modifiers.get_movement_speed_multiplier())
    expected = int(expected * actor.modifiers.get_exhaustion_energy_multiplier())
    actor.energy.regenerate()
    assert actor.energy.accumulated_energy == expected
    assert not actor.modifiers.has_disadvantage_from_exhaustion()


def test_double_exhaustion_disadvantage_and_energy() -> None:
    controller, actor = make_world()
    actor.energy.accumulated_energy = 0
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    # In RAF, use the actor's own energy calculation
    expected = actor.energy.get_speed_based_energy_amount()
    actor.energy.regenerate()
    assert abs(actor.energy.accumulated_energy - expected) < 0.01  # Float comparison
    assert actor.modifiers.has_disadvantage_from_exhaustion()

    target = Character(
        1, 0, "T", colors.WHITE, "Tar", game_world=cast(GameWorld, controller.gw)
    )
    with patch("random.randint", side_effect=[4, 18]):
        result = D20System(0, 10).resolve(actor, target)
    assert isinstance(result, D20ResolutionResult)
    assert result.has_disadvantage
    assert result.final_roll_used == 4


def test_movement_stumble_with_high_exhaustion() -> None:
    controller, actor = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())

    with patch("random.random", return_value=0.05):
        intent = MoveIntent(controller, actor, dx=1, dy=0)
        result = MoveExecutor().execute(intent)
    assert result is not None
    assert not result.succeeded
    assert (actor.x, actor.y) == (0, 0)


def test_exhaustion_removal_restores_effects() -> None:
    controller, actor = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    # Remove one stack
    exhaustion = actor.conditions.get_conditions_by_type(conditions.Exhaustion)[0]
    actor.conditions.remove_condition(exhaustion)
    assert not actor.modifiers.has_disadvantage_from_exhaustion()
    actor.energy.accumulated_energy = 0
    expected = int(actor.energy.speed * actor.modifiers.get_movement_speed_multiplier())
    expected = int(expected * actor.modifiers.get_exhaustion_energy_multiplier())
    actor.energy.regenerate()
    assert actor.energy.accumulated_energy == expected


def test_injury_and_exhaustion_stack() -> None:
    controller, actor = make_world()
    actor.conditions.add_condition(conditions.Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    actor.conditions.add_condition(conditions.Exhaustion())
    expected_speed = int(
        actor.energy.speed * actor.modifiers.get_movement_speed_multiplier()
    )
    assert expected_speed == int(
        actor.energy.speed
        * MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK
        * 0.75
    )
    assert expected_speed == int(
        actor.energy.speed * actor.modifiers.get_movement_speed_multiplier()
    )
    # In RAF, use the actor's own energy calculation
    expected_energy = actor.energy.get_speed_based_energy_amount()
    actor.energy.accumulated_energy = 0
    actor.energy.regenerate()
    assert (
        abs(actor.energy.accumulated_energy - expected_energy) < 0.01
    )  # Float comparison
