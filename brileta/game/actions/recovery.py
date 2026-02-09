"""Recovery and rest-related actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.game.action_plan import ActionPlan, ApproachStep, IntentStep
from brileta.game.actors import Character
from brileta.game.enums import Disposition
from brileta.game.items.item_core import Item

from .base import GameIntent

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from brileta.controller import Controller


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
    """Intent for using a consumable item on self."""

    def __init__(self, controller: Controller, actor: Character, item: Item) -> None:
        super().__init__(controller, actor)
        self.item = item


class UseConsumableOnTargetIntent(GameIntent):
    """Intent for using a consumable item on another character.

    This is used when the player targets another character with a consumable,
    such as using a healing item on an ally.
    """

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        item: Item,
        target: Character,
    ) -> None:
        super().__init__(controller, actor)
        self.item = item
        self.target = target


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


# =============================================================================
# Action Plans for Recovery Actions
# =============================================================================

UseConsumableOnTargetPlan = ActionPlan(
    name="Use Consumable",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=UseConsumableOnTargetIntent,
            params=lambda ctx: {
                "actor": ctx.actor,
                "item": ctx.item,
                "target": ctx.target_actor,
            },
        ),
    ],
)
