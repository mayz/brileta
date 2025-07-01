from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.events import (
    MessageEvent,
    ScreenShakeEvent,
    reset_event_bus_for_testing,
    subscribe_to_event,
)
from catley.game.actions.combat import AttackIntent
from catley.game.actions.executors.combat import AttackExecutor
from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.game_world import GameWorld
from catley.game.items.item_types import FISTS_TYPE, PISTOL_TYPE
from catley.game.resolution.combat_arbiter import determine_outcome
from catley.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld


@dataclass
class DummyMessageLog:
    def __init__(self) -> None:
        subscribe_to_event(MessageEvent, self.add_message)

    def add_message(self, *_args, **_kwargs) -> None:  # pragma: no cover - simple stub
        pass


@dataclass
class DummyFrameManager:
    def __init__(self) -> None:
        self.trigger_screen_shake = MagicMock()

    def create_effect(self, *args, **kwargs) -> None:
        pass


@dataclass
class DummyController:
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def make_world() -> tuple[
    DummyController, Character, Character, AttackIntent, AttackExecutor
]:
    gw = DummyGameWorld()
    attacker = Character(
        1,
        1,
        "A",
        colors.WHITE,
        "Attacker",
        game_world=cast(GameWorld, gw),
    )
    defender = Character(
        2,
        1,
        "D",
        colors.WHITE,
        "Defender",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = defender
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    weapon = FISTS_TYPE.create()
    attack = weapon.melee_attack
    assert attack is not None
    return (
        controller,
        attacker,
        defender,
        AttackIntent(
            cast(Controller, controller),
            attacker,
            defender,
            weapon,
        ),
        AttackExecutor(),
    )


def make_world_ranged() -> tuple[
    DummyController, Character, Character, AttackIntent, AttackExecutor
]:
    gw = DummyGameWorld()
    attacker = Character(
        1,
        1,
        "A",
        colors.WHITE,
        "Attacker",
        game_world=cast(GameWorld, gw),
    )
    defender = Character(
        3,
        1,
        "D",
        colors.WHITE,
        "Defender",
        game_world=cast(GameWorld, gw),
    )
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.player = defender
    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    weapon = PISTOL_TYPE.create()
    attack = weapon.ranged_attack
    assert attack is not None
    return (
        controller,
        attacker,
        defender,
        AttackIntent(
            cast(Controller, controller),
            attacker,
            defender,
            weapon,
        ),
        AttackExecutor(),
    )


def test_screen_shake_uses_damage_once() -> None:
    reset_event_bus_for_testing()
    intensities: list[float] = []
    subscribe_to_event(ScreenShakeEvent, lambda e: intensities.append(e.intensity))
    controller, attacker, defender, intent, executor = make_world()
    weapon = intent.weapon
    assert weapon and weapon.melee_attack
    attack = weapon.melee_attack
    attack.damage_dice.roll = MagicMock(return_value=4)
    check = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
    outcome = determine_outcome(check, attacker, defender, weapon)
    damage = executor._apply_combat_outcome(intent, check, outcome, attack, weapon)
    executor._handle_post_attack_effects(intent, check, attack, weapon, damage)

    # Damage dice should have been rolled only once
    assert attack.damage_dice.roll.call_count == 1
    assert intensities and intensities[0] == 0.6


def test_screen_shake_ranged_attack_intensity() -> None:
    reset_event_bus_for_testing()
    intensities: list[float] = []
    subscribe_to_event(ScreenShakeEvent, lambda e: intensities.append(e.intensity))
    controller, attacker, defender, intent, executor = make_world_ranged()
    weapon = intent.weapon
    assert weapon and weapon.ranged_attack
    attack = weapon.ranged_attack
    attack.damage_dice.roll = MagicMock(return_value=4)
    check = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
    outcome = determine_outcome(check, attacker, defender, weapon)
    damage = executor._apply_combat_outcome(intent, check, outcome, attack, weapon)
    executor._handle_post_attack_effects(intent, check, attack, weapon, damage)

    assert intensities and intensities[0] == 0.32
