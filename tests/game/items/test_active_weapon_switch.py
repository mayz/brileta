from __future__ import annotations

from typing import cast
from unittest.mock import patch

from brileta.controller import Controller
from brileta.game.actions.combat import AttackIntent
from brileta.game.actions.executors.combat import AttackExecutor
from brileta.game.consequences import AttackConsequenceGenerator
from brileta.game.enums import OutcomeTier
from brileta.game.items.item_types import HUNTING_RIFLE_TYPE
from brileta.game.resolution.d20_system import D20ResolutionResult
from tests.game.actions.test_action_discovery import _make_combat_world


def test_attack_switches_active_weapon() -> None:
    controller, player, target, _, _ = _make_combat_world()
    rifle = HUNTING_RIFLE_TYPE.create()
    player.inventory.equip_to_slot(rifle, 1)
    player.inventory.switch_to_slot(0)

    intent = AttackIntent(cast(Controller, controller), player, target, rifle)
    executor = AttackExecutor()

    with (
        patch.object(AttackExecutor, "_validate_attack", return_value={}),
        patch.object(
            AttackExecutor,
            "_execute_attack_roll",
            return_value=D20ResolutionResult(outcome_tier=OutcomeTier.SUCCESS),
        ),
        patch.object(AttackExecutor, "_apply_combat_outcome", return_value=0),
        patch.object(AttackExecutor, "_handle_post_attack_effects"),
        patch.object(AttackConsequenceGenerator, "generate", return_value=[]),
    ):
        result = executor.execute(intent)

    assert result is not None
    assert player.inventory.active_slot == 1
