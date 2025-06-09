from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character
from catley.game.enums import OutcomeTier
from catley.game.items.item_types import FISTS_TYPE, PISTOL_TYPE
from catley.game.resolution.combat_arbiter import determine_outcome
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.world import tile_types
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld:
    def __init__(self) -> None:
        self.game_map = GameMap(5, 5)
        self.game_map.tiles[:] = tile_types.TILE_TYPE_ID_FLOOR  # type: ignore[attr-defined]
        self.actors: list[Character] = []
        self.player: Character | None = None


@dataclass
class DummyMessageLog:
    def add_message(self, *_args, **_kwargs) -> None:
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


def make_world() -> tuple[DummyController, Character, Character, AttackAction]:
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
    gw.actors.extend([attacker, defender])
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
        AttackAction(
            cast(Controller, controller),
            attacker,
            defender,
            weapon,
        ),
    )


def make_world_ranged() -> tuple[DummyController, Character, Character, AttackAction]:
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
    gw.actors.extend([attacker, defender])
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
        AttackAction(
            cast(Controller, controller),
            attacker,
            defender,
            weapon,
        ),
    )


def test_screen_shake_uses_damage_once() -> None:
    controller, attacker, defender, action = make_world()
    weapon = action.weapon
    assert weapon and weapon.melee_attack
    attack = weapon.melee_attack
    attack.damage_dice.roll = MagicMock(return_value=4)
    check = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
    outcome = determine_outcome(check, attacker, defender, weapon)
    damage = action._apply_combat_outcome(check, outcome, attack, weapon)
    action._handle_post_attack_effects(check, attack, weapon, damage)

    # Damage dice should have been rolled only once
    assert attack.damage_dice.roll.call_count == 1
    intensity = controller.frame_manager.trigger_screen_shake.call_args[0][0]
    assert intensity == 0.6


def test_screen_shake_ranged_attack_intensity() -> None:
    controller, attacker, defender, action = make_world_ranged()
    weapon = action.weapon
    assert weapon and weapon.ranged_attack
    attack = weapon.ranged_attack
    attack.damage_dice.roll = MagicMock(return_value=4)
    check = D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS)
    outcome = determine_outcome(check, attacker, defender, weapon)
    damage = action._apply_combat_outcome(check, outcome, attack, weapon)
    action._handle_post_attack_effects(check, attack, weapon, damage)

    intensity = controller.frame_manager.trigger_screen_shake.call_args[0][0]
    assert intensity == 0.32
