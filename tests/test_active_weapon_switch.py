from __future__ import annotations

from typing import cast
from unittest.mock import patch

from catley.controller import Controller
from catley.game.actions.combat import AttackAction, AttackConsequenceGenerator
from catley.game.enums import OutcomeTier
from catley.game.items.item_types import HUNTING_RIFLE_TYPE
from catley.game.resolution.d20_system import D20ResolutionResult
from tests.test_action_discovery import _make_combat_world


def test_attack_switches_active_weapon() -> None:
    controller, player, target, _, _ = _make_combat_world()
    rifle = HUNTING_RIFLE_TYPE.create()
    player.inventory.equip_to_slot(rifle, 1)
    player.inventory.switch_to_weapon_slot(0)

    action = AttackAction(cast(Controller, controller), player, target, rifle)

    with (
        patch.object(AttackAction, "_validate_attack", return_value={}),
        patch.object(
            AttackAction,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS),
        ),
        patch.object(AttackAction, "_apply_combat_outcome", return_value=0),
        patch.object(AttackAction, "_handle_post_attack_effects"),
        patch.object(AttackConsequenceGenerator, "generate", return_value=[]),
    ):
        result = action.execute()

    assert result is not None
    assert player.inventory.active_weapon_slot == 1
