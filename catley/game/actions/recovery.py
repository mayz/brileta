"""Recovery and rest-related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actors import Character
from catley.game.enums import Disposition
from catley.game.items.item_core import Item

from .base import GameIntent

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from catley.controller import Controller


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


class UseConsumableIntent(GameIntent):
    """Intent for using a consumable item."""

    def __init__(self, controller: Controller, actor: Character, item: Item) -> None:
        super().__init__(controller, actor)
        self.item = item


class RestIntent(GameIntent):
    """Intent for short rest to recover armor points."""

    def __init__(self, controller: Controller, actor: Character) -> None:
        super().__init__(controller, actor)


class SleepIntent(GameIntent):
    """Intent for sleep to restore HP and remove one exhaustion."""

    def __init__(self, controller: Controller, actor: Character) -> None:
        super().__init__(controller, actor)


class ComfortableSleepIntent(GameIntent):
    """Intent for comfortable sleep to remove all exhaustion."""

    def __init__(self, controller: Controller, actor: Character) -> None:
        super().__init__(controller, actor)
