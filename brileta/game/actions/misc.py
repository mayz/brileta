from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.game.action_plan import ActionPlan, ApproachStep, IntentStep
from brileta.game.actors.components import InventoryComponent
from brileta.game.countables import CountableType

from .base import GameIntent

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actors import Character
    from brileta.game.items.item_core import Item


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


class DropItemIntent(GameIntent):
    """Intent for dropping an item from inventory to the ground."""

    def __init__(self, controller: Controller, actor: Character, item: Item) -> None:
        super().__init__(controller, actor)
        self.item = item


class PickupCountableIntent(GameIntent):
    """Intent for picking up countables from an inventory source.

    Use amount=-1 to pick up all of the specified countable type.
    """

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        source: InventoryComponent,
        countable_type: CountableType,
        amount: int,
    ) -> None:
        super().__init__(controller, actor)
        self.source = source
        self.countable_type = countable_type
        self.amount = amount  # -1 for "all"


class DropCountableIntent(GameIntent):
    """Intent for dropping countables on the ground."""

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        countable_type: CountableType,
        amount: int,
    ) -> None:
        super().__init__(controller, actor)
        self.countable_type = countable_type
        self.amount = amount


# =============================================================================
# Action Plans for Miscellaneous Actions
# =============================================================================

PickupItemsPlan = ActionPlan(
    name="Pick Up Items",
    requires_target=False,
    requires_adjacency=False,
    steps=[
        ApproachStep(stop_distance=0),
        IntentStep(
            intent_class=PickupItemsAtLocationIntent,
            params=lambda ctx: {"actor": ctx.actor},
        ),
    ],
)
