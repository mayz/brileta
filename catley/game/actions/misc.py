from __future__ import annotations

from typing import TYPE_CHECKING

from .base import GameAction, GameActionResult

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class PickupAction(GameAction):
    """Simple placeholder action for picking up items."""

    def __init__(self, controller: Controller, actor: Character, items: list) -> None:
        super().__init__(controller, actor)
        self.items = items

    def execute(
        self,
    ) -> GameActionResult | None:  # pragma: no cover - simple placeholder
        for item in list(self.items):
            actor_inv = getattr(self.actor, "inventory", None)
            if actor_inv and actor_inv.can_add_voluntary_item(item):
                actor_inv.add_to_inventory(item)
        return GameActionResult()


class SwitchWeaponAction(GameAction):
    """Switch the actor's active weapon slot."""

    def __init__(self, controller: Controller, actor: Character, slot: int) -> None:
        super().__init__(controller, actor)
        self.slot = slot

    def execute(
        self,
    ) -> GameActionResult | None:  # pragma: no cover - simple state change
        inventory = getattr(self.actor, "inventory", None)
        if inventory:
            inventory.switch_to_weapon_slot(self.slot)
        return None
