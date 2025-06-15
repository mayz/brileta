from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from catley.game.actions.misc import PickupIntent, SwitchWeaponIntent


class PickupExecutor(ActionExecutor):
    """Executes pickup intents by adding items to actor inventory."""

    def __init__(self) -> None:
        """Create a PickupExecutor without requiring a controller."""
        pass

    def execute(self, intent: PickupIntent) -> GameActionResult | None:  # type: ignore[override]
        # Move logic from old PickupAction.execute()
        for item in list(intent.items):
            actor_inv = getattr(intent.actor, "inventory", None)
            if actor_inv and actor_inv.can_add_voluntary_item(item):
                actor_inv.add_to_inventory(item)
        return GameActionResult()


class SwitchWeaponExecutor(ActionExecutor):
    """Executes weapon switching intents."""

    def __init__(self) -> None:
        """Create a SwitchWeaponExecutor without requiring a controller."""
        pass

    def execute(self, intent: SwitchWeaponIntent) -> GameActionResult | None:  # type: ignore[override]
        # Move logic from old SwitchWeaponAction.execute()
        inventory = getattr(intent.actor, "inventory", None)
        if inventory:
            inventory.switch_to_weapon_slot(intent.slot)
        return None
