from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import patch

from catley import colors
from catley.controller import Controller
from catley.events import MessageEvent, reset_event_bus_for_testing, subscribe_to_event
from catley.game.actions.combat import AttackAction
from catley.game.actors import Character
from catley.game.ai import DispositionBasedAI
from catley.game.enums import Disposition, OutcomeTier
from catley.game.items.item_core import Item
from catley.game.items.item_types import PISTOL_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from catley.world.game_state import GameWorld
from catley.world.map import GameMap


class DummyGameWorld(GameWorld):
    def __init__(self) -> None:
        # Do not call GameWorld.__init__ to avoid full setup.
        self.game_map = GameMap(5, 5)
        self.game_map.transparent[:] = True
        self.actors: list[Character] = []
        self.player: Character | None = None


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
    def create_effect(self, *args, **kwargs) -> None:
        pass

    def trigger_screen_shake(self, *_args, **_kwargs) -> None:
        pass


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld
    message_log: DummyMessageLog
    frame_manager: DummyFrameManager


def make_world() -> tuple[
    DummyController, Character, Character, Character, AttackAction
]:
    reset_event_bus_for_testing()
    gw = DummyGameWorld()
    attacker = Character(1, 1, "A", colors.WHITE, "Att", game_world=cast(GameWorld, gw))
    defender = Character(2, 1, "D", colors.WHITE, "Def", game_world=cast(GameWorld, gw))
    bystander = Character(3, 1, "B", colors.WHITE, "By", game_world=cast(GameWorld, gw))
    bystander.ai = DispositionBasedAI(disposition=Disposition.WARY)
    gw.actors.extend([attacker, defender, bystander])
    gw.player = attacker

    controller = DummyController(
        gw=gw, message_log=DummyMessageLog(), frame_manager=DummyFrameManager()
    )
    weapon = PISTOL_TYPE.create()
    attacker.inventory.equip_to_slot(weapon, 0)
    action = AttackAction(cast(Controller, controller), attacker, defender, weapon)
    return controller, attacker, defender, bystander, action


def test_weapon_drop_and_noise_alert() -> None:
    controller, attacker, defender, bystander, action = make_world()

    with (
        patch.object(AttackAction, "_validate_attack", return_value={}),
        patch.object(
            AttackAction,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.CRITICAL_FAILURE),
        ),
        patch.object(
            AttackAction,
            "_apply_combat_outcome",
            return_value=0,
        ),
        patch.object(
            AttackAction,
            "_handle_post_attack_effects",
        ),
    ):
        result = action.execute()

    assert result is not None
    assert any(c.type == "weapon_drop" for c in result.consequences)
    assert attacker.inventory.get_active_weapon() is None
    ground_items = controller.gw.get_pickable_items_at_location(attacker.x, attacker.y)
    assert ground_items
    assert len(ground_items) > 0
    first_item = cast(Item, ground_items[0])
    assert first_item.name == cast(Item, action.weapon).name
    assert isinstance(bystander.ai, DispositionBasedAI)
    b_ai = cast(DispositionBasedAI, bystander.ai)
    assert b_ai.disposition == Disposition.HOSTILE
