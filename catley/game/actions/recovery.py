"""Recovery and rest-related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.events import MessageEvent, publish_event
from catley.game.actions.base import GameAction, GameActionResult
from catley.game.actors import Character, conditions
from catley.game.enums import Disposition
from catley.game.items.item_core import Item

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from catley.controller import Controller

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    pass


def is_safe_location(actor: Character, radius: int = 10) -> tuple[bool, str | None]:
    """Return ``True`` if no hostile actors are within ``radius`` tiles."""

    gw = actor.gw
    if gw is None:
        return True, None

    nearby = gw.actor_spatial_index.get_in_radius(actor.x, actor.y, radius)
    for other in nearby:
        if other is actor:
            continue
        if not isinstance(other, Character):
            continue
        disposition = getattr(getattr(other, "ai", None), "disposition", None)
        if disposition == Disposition.HOSTILE:
            return False, f"Hostile {other.name} nearby"

    # Future: check environmental hazards here
    return True, None


class UseConsumableAction(GameAction):
    """Action for using a consumable item."""

    def __init__(self, controller: Controller, actor: Character, item: Item) -> None:
        super().__init__(controller, actor)
        self.item = item

    def execute(self) -> GameActionResult | None:
        if not self.item.consumable_effect:
            publish_event(MessageEvent(f"{self.item.name} cannot be used", colors.RED))
            return None

        inventory = self.actor.inventory
        if not inventory:
            return None

        if not inventory.remove_from_inventory(self.item):
            publish_event(MessageEvent("Item not found in inventory", colors.RED))
            return None

        success = self.item.consumable_effect.consume(self.actor, self.controller)

        if not success:
            inventory.add_to_inventory(self.item)
            publish_event(MessageEvent(f"Cannot use {self.item.name}", colors.RED))

        return GameActionResult()


class RestAction(GameAction):
    """Short rest to recover armor points."""

    name = "Rest"
    description = "Take a brief rest to recover AP"

    def execute(self) -> GameActionResult | None:
        assert isinstance(self.actor, Character)
        safe, reason = is_safe_location(self.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot rest: {reason}", colors.RED))
            return None

        if not self.actor.health:
            return None

        if self.actor.health.ap == 0:
            publish_event(
                MessageEvent(
                    f"{self.actor.name}'s armor is broken and cannot be restored.",
                    colors.YELLOW,
                )
            )
            return None

        if self.actor.health.ap < self.actor.health.max_ap:
            self.actor.health.ap = self.actor.health.max_ap
            publish_event(MessageEvent("Recovered all AP", colors.GREEN))

        return GameActionResult()


class SleepAction(GameAction):
    """Sleep to restore HP and remove one exhaustion."""

    name = "Sleep"
    description = "Sleep to fully restore HP and reduce exhaustion"

    def execute(self) -> GameActionResult | None:
        assert isinstance(self.actor, Character)
        safe, reason = is_safe_location(self.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot sleep: {reason}", colors.RED))
            return None

        if self.actor.health:
            self.actor.health.hp = self.actor.health.max_hp
            publish_event(MessageEvent("Restored all HP", colors.GREEN))

        if self.actor.conditions:
            exhaustion = self.actor.conditions.get_conditions_by_type(
                conditions.Exhaustion
            )
            if exhaustion:
                self.actor.conditions.remove_condition(exhaustion[0])
                publish_event(MessageEvent("Removed 1 Exhaustion", colors.GREEN))

        return GameActionResult()


class ComfortableSleepAction(GameAction):
    """Sleep comfortably to remove all exhaustion."""

    name = "Comfortable Sleep"
    description = "Sleep soundly to remove all exhaustion and restore HP"

    def execute(self) -> GameActionResult | None:
        assert isinstance(self.actor, Character)
        safe, reason = is_safe_location(self.actor)
        if not safe:
            publish_event(MessageEvent(f"Cannot sleep: {reason}", colors.RED))
            return None

        if self.actor.health:
            self.actor.health.hp = self.actor.health.max_hp
            publish_event(MessageEvent("Restored all HP", colors.GREEN))

        if self.actor.conditions:
            removed = self.actor.conditions.remove_conditions_by_type(
                {conditions.Exhaustion}
            )
            if removed:
                publish_event(
                    MessageEvent(
                        f"Removed all {len(removed)} Exhaustion conditions",
                        colors.GREEN,
                    )
                )
            else:
                publish_event(MessageEvent("No Exhaustion to remove", colors.YELLOW))

        return GameActionResult()
