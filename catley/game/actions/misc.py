from __future__ import annotations

from typing import TYPE_CHECKING

from .base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class PickupIntent(GameIntent):
    """Intent for picking up items from the ground."""

    def __init__(self, controller: Controller, actor: Character, items: list) -> None:
        super().__init__(controller, actor)
        self.items = items


class PickupItemsAtLocationIntent(GameIntent):
    """Intent for picking up all items at the actor's current location."""

    def __init__(self, controller: Controller, actor: Character) -> None:
        super().__init__(controller, actor)


class SwitchWeaponIntent(GameIntent):
    """Intent for switching the actor's active weapon slot."""

    def __init__(self, controller: Controller, actor: Character, slot: int) -> None:
        super().__init__(controller, actor)
        self.slot = slot
