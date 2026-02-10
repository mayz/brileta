from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

from brileta import colors
from brileta.controller import Controller
from brileta.events import MessageEvent, reset_event_bus_for_testing, subscribe_to_event
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.actors import NPC, Character
from brileta.game.actors.status_effects import OffBalanceEffect
from brileta.game.enums import OutcomeTier
from brileta.game.game_world import GameWorld
from brileta.game.items.item_core import Item
from brileta.game.items.item_types import PISTOL_TYPE
from brileta.game.resolution.d20_system import D20ResolutionResult
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager


@dataclass
class DummyMessageLog:
    messages: list[str]

    def __init__(self) -> None:
        self.messages = []
        subscribe_to_event(MessageEvent, lambda e: self.add_message(e.text))

    def add_message(self, text: str, *_args, **_kwargs) -> None:
        self.messages.append(text)


@dataclass
class DummyFrameManager:
    def get_visible_bounds(self) -> None:
        return None

    def create_effect(self, *args, **kwargs) -> None:
        pass

    def trigger_screen_shake(self, *_args, **_kwargs) -> None:
        pass


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def make_world() -> tuple[DummyController, Character, Character, NPC, AttackIntent]:
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    bystander = NPC(3, 1, "B", colors.WHITE, "By", game_world=cast(GameWorld, gw))
    gw.add_actor(attacker)
    gw.add_actor(defender)
    gw.add_actor(bystander)
    gw.player = attacker

    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    weapon = PISTOL_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)
    intent = AttackIntent(cast(Controller, controller), attacker, defender, weapon)
    return controller, attacker, defender, bystander, intent


def test_weapon_drop_and_noise_alert(
    patch_consequences_rng: Callable[[list[float]], AbstractContextManager[Any]],
) -> None:
    controller, attacker, _defender, bystander, intent = make_world()
    executor = AttackExecutor()

    # Force random to select weapon_drop (roll < 0.40)
    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackExecutor,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackExecutor,
            "_handle_post_attack_effects",
        ),
        patch_consequences_rng([0.20]),  # Selects weapon_drop
    ):
        result = executor.execute(intent)

    assert result is not None
    assert any(c.type == "weapon_drop" for c in result.consequences)
    assert attacker.inventory.get_active_item() is None
    ground_actor = next(
        a
        for a in controller.gw.actors
        if a is not attacker and a.x == attacker.x and a.y == attacker.y
    )
    assert ground_actor.inventory is not None
    assert any(
        it.name == cast(Item, intent.weapon).name for it in ground_actor.inventory
    )
    assert bystander.ai.disposition_toward(attacker) == -75


def test_self_injury_consequence(
    patch_consequences_rng: Callable[[list[float]], AbstractContextManager[Any]],
) -> None:
    """Test that self_injury consequence deals damage to attacker."""
    _controller, attacker, _defender, _bystander, intent = make_world()
    executor = AttackExecutor()

    initial_hp = attacker.health.hp if attacker.health else 0

    # Force random to select self_injury (0.40 <= roll < 0.75)
    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackExecutor,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackExecutor,
            "_handle_post_attack_effects",
        ),
        patch_consequences_rng([0.50]),  # Selects self_injury
    ):
        result = executor.execute(intent)

    assert result is not None
    assert any(c.type == "self_injury" for c in result.consequences)
    # Attacker should have taken some damage (at least 1, since dice min is 1)
    assert attacker.health is not None
    assert attacker.health.hp < initial_hp


def test_self_injury_death_handling(
    patch_consequences_rng: Callable[[list[float]], AbstractContextManager[Any]],
) -> None:
    """Test that self_injury consequence properly handles death visuals.

    Regression test: When an attacker fumbles and kills themselves with
    self-injury damage, the death handling must run (glyph -> 'x', color -> DEAD).
    Previously, direct HP manipulation bypassed Actor.take_damage() and left
    corpses with their original glyph.
    """
    _controller, attacker, _defender, _bystander, intent = make_world()
    executor = AttackExecutor()

    # Set attacker HP low enough that self-injury will kill them
    assert attacker.health is not None
    attacker.health._hp = 1

    # Force random to select self_injury (0.40 <= roll < 0.75)
    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackExecutor,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackExecutor,
            "_handle_post_attack_effects",
        ),
        patch_consequences_rng([0.50]),  # Selects self_injury
    ):
        executor.execute(intent)

    # Attacker should be dead
    assert not attacker.health.is_alive()

    # Death visuals should have been applied (this was the bug)
    assert attacker.ch == "x", "Death handling should change glyph to 'x'"
    assert attacker.color == colors.DEAD, "Death handling should change color to DEAD"
    assert not attacker.blocks_movement, "Dead actors should not block movement"


def test_off_balance_consequence(
    patch_consequences_rng: Callable[[list[float]], AbstractContextManager[Any]],
) -> None:
    """Test that off_balance consequence applies OffBalanceEffect to attacker."""
    _controller, attacker, _defender, _bystander, intent = make_world()
    executor = AttackExecutor()

    # Ensure attacker doesn't have OffBalanceEffect initially
    assert not attacker.status_effects.has_status_effect(OffBalanceEffect)

    # Force random to select off_balance (roll >= 0.75)
    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackExecutor,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackExecutor,
            "_handle_post_attack_effects",
        ),
        patch_consequences_rng([0.80]),  # Selects off_balance
    ):
        result = executor.execute(intent)

    assert result is not None
    assert any(c.type == "off_balance" for c in result.consequences)
    # Attacker should now have OffBalanceEffect
    assert attacker.status_effects.has_status_effect(OffBalanceEffect)


def test_weapon_drop_consequence_still_works(
    patch_consequences_rng: Callable[[list[float]], AbstractContextManager[Any]],
) -> None:
    """Test that weapon_drop consequence still works with new random selection."""
    _controller, attacker, _defender, _bystander, intent = make_world()
    executor = AttackExecutor()

    # Force random to select weapon_drop (roll < 0.40)
    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackExecutor,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackExecutor,
            "_handle_post_attack_effects",
        ),
        patch_consequences_rng([0.20]),  # Selects weapon_drop
    ):
        result = executor.execute(intent)

    assert result is not None
    assert any(c.type == "weapon_drop" for c in result.consequences)
    assert attacker.inventory.get_active_item() is None
