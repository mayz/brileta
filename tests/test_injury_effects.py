from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.game.actions.combat import AttackAction
from catley.game.actors import PC, Character, conditions
from catley.game.enums import InjuryLocation, OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.capabilities import Attack
from catley.game.items.item_core import Item
from catley.game.items.item_types import FISTS_TYPE
from catley.game.resolution.combat_arbiter import determine_outcome
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.game.turn_manager import TurnManager
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


def make_combat_world() -> tuple[DummyController, Character, Character, AttackAction]:
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
    action = AttackAction(controller, attacker, defender, weapon)
    return controller, attacker, defender, action


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
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    assert attack is not None
    attacker.add_condition(conditions.Injury(InjuryLocation.LEFT_ARM, "Sprain"))
    with patch("random.randint", side_effect=[2, 18]):
        result = cast(
            D20ResolutionResult, action._execute_attack_roll(attack, weapon, {})
        )
    assert isinstance(result, D20ResolutionResult)
    assert result.has_disadvantage


def test_leg_injury_reduces_speed() -> None:
    gw = DummyGameWorld()
    actor = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    actor.add_condition(conditions.Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    assert actor.calculate_effective_speed() == int(actor.speed * 0.75)


def test_multiple_leg_injuries_stack() -> None:
    gw = DummyGameWorld()
    actor = PC(0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw))
    gw.add_actor(actor)
    actor.add_condition(conditions.Injury(InjuryLocation.LEFT_LEG, "Bruise"))
    actor.add_condition(conditions.Injury(InjuryLocation.RIGHT_LEG, "Sprain"))
    assert actor.calculate_effective_speed() == int(actor.speed * 0.75 * 0.75)


def test_random_injury_location_assigned() -> None:
    controller, attacker, defender, action = make_combat_world()
    weapon = cast(Item, action.weapon)
    attack = cast(Attack, weapon.melee_attack)
    assert attack is not None
    defender.health.ap = 0
    with patch("random.random", return_value=0.8):
        res = D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_SUCCESS)
        outcome = determine_outcome(res, attacker, defender, weapon)
    assert outcome.injury_inflicted is not None
    injury = cast(conditions.Injury, outcome.injury_inflicted)
    assert injury.injury_location == InjuryLocation.TORSO
