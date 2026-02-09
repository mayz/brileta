from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from brileta import colors
from brileta.controller import Controller
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actors import PC, Character, conditions
from brileta.game.enums import InjuryLocation, OutcomeTier
from brileta.game.game_world import GameWorld
from brileta.game.items.capabilities import Attack
from brileta.game.items.item_core import Item
from brileta.game.items.item_types import FISTS_TYPE
from brileta.game.resolution import combat_arbiter, d20_system
from brileta.game.resolution.combat_arbiter import determine_outcome
from brileta.game.resolution.d20_system import D20ResolutionResult
from brileta.game.turn_manager import TurnManager
from tests.conftest import FixedRandom
from tests.helpers import DummyGameWorld


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    frame_manager: object | None = None
    message_log: object | None = None

    def __post_init__(self) -> None:
        self.update_fov_called = False
        if self.message_log is None:
            self.message_log = type("Log", (), {"add_message": lambda *a, **kw: None})()
        self.action_cost = 100
        self.turn_manager = TurnManager(self)

    def update_fov(self) -> None:  # pragma: no cover - simple flag
        self.update_fov_called = True


def make_combat_world() -> tuple[
    DummyController, Character, Character, AttackIntent, AttackExecutor
]:
    gw = DummyGameWorld()
    attacker = Character(
        1,
        1,
        "A",
        colors.WHITE,
        "Att",
        game_world=cast(GameWorld, gw),
        strength=5,
        agility=5,
    )
    defender = Character(
        2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw), agility=5
    )
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = attacker
    controller = DummyController(gw=gw)
    weapon = FISTS_TYPE.create()
    intent = AttackIntent(controller, attacker, defender, weapon)
    executor = AttackExecutor()
    return controller, attacker, defender, intent, executor


def test_head_injury_disadvantage_on_intelligence() -> None:
    injury = conditions.Injury(InjuryLocation.HEAD, "Wound")
    args = {
        "has_advantage": False,
        "has_disadvantage": False,
        "stat_name": "intelligence",
    }
    result = injury.apply_to_resolution(args)
    assert result["has_disadvantage"]


def test_arm_injury_gives_attack_disadvantage() -> None:
    _controller, attacker, _defender, intent, executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    assert attack is not None
    if attacker.conditions is not None:
        attacker.conditions.add_condition(
            conditions.Injury(InjuryLocation.LEFT_ARM, "Sprain")
        )
    with patch.object(d20_system._rng, "randint", FixedRandom([2, 18])):
        result = cast(
            D20ResolutionResult,
            executor._execute_attack_roll(intent, attack, weapon, {}),
        )
    assert isinstance(result, D20ResolutionResult)
    assert result.has_disadvantage


def test_leg_injury_reduces_speed() -> None:
    gw = DummyGameWorld()
    actor = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    actor.conditions.add_condition(conditions.Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    assert int(
        actor.energy.speed * actor.modifiers.get_movement_speed_multiplier()
    ) == int(actor.energy.speed * 0.75)


def test_multiple_leg_injuries_stack() -> None:
    gw = DummyGameWorld()
    actor = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    actor.conditions.add_condition(conditions.Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    actor.conditions.add_condition(
        conditions.Injury(InjuryLocation.RIGHT_LEG, "Sprain")
    )
    assert int(
        actor.energy.speed * actor.modifiers.get_movement_speed_multiplier()
    ) == int(actor.energy.speed * 0.75 * 0.75)


def test_random_injury_location_assigned() -> None:
    """Critical hit on unarmored target assigns random injury location."""
    _controller, attacker, defender, intent, _executor = make_combat_world()
    weapon = cast(Item, intent.weapon)
    attack = cast(Attack, weapon.melee_attack)
    assert attack is not None
    # No armor equipped - critical hit will inflict injury
    with patch.object(combat_arbiter._rng, "random", return_value=0.8):
        res = D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_SUCCESS)
        outcome = determine_outcome(res, attacker, defender, weapon)
    assert outcome.injury_inflicted is not None
    assert outcome.injury_inflicted.injury_location == InjuryLocation.TORSO
