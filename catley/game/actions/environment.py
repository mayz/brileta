from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.action_plan import ActionPlan, ApproachStep, IntentStep

from .base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Actor, Character


class SearchContainerIntent(GameIntent):
    """Intent to search a container or corpse for items.

    When executed, this opens the loot UI (DualPaneMenu) with the target
    actor as the source, allowing the player to transfer items between
    their inventory and the container/corpse.
    """

    def __init__(
        self,
        controller: Controller,
        actor: Character,
        target: Actor,
    ) -> None:
        """Create a search container intent.

        Args:
            controller: The game controller
            actor: The character performing the search (usually the player)
            target: The container or dead character being searched
        """
        super().__init__(controller, actor)
        self.target = target


class OpenDoorIntent(GameIntent):
    """Intent for opening a closed door tile."""

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.x = x
        self.y = y


class CloseDoorIntent(GameIntent):
    """Intent for closing an open door tile."""

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.x = x
        self.y = y


# =============================================================================
# Action Plans for Environment Interactions
# =============================================================================

SearchContainerPlan = ActionPlan(
    name="Search",
    requires_target=True,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=SearchContainerIntent,
            params=lambda ctx: {
                "actor": ctx.actor,
                "target": ctx.target_actor,
            },
        ),
    ],
)

OpenDoorPlan = ActionPlan(
    name="Open Door",
    requires_target=False,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=OpenDoorIntent,
            params=lambda ctx: {
                "actor": ctx.actor,
                "x": ctx.target_position[0] if ctx.target_position else 0,
                "y": ctx.target_position[1] if ctx.target_position else 0,
            },
        ),
    ],
)

CloseDoorPlan = ActionPlan(
    name="Close Door",
    requires_target=False,
    requires_adjacency=True,
    steps=[
        ApproachStep(stop_distance=1),
        IntentStep(
            intent_class=CloseDoorIntent,
            params=lambda ctx: {
                "actor": ctx.actor,
                "x": ctx.target_position[0] if ctx.target_position else 0,
                "y": ctx.target_position[1] if ctx.target_position else 0,
            },
        ),
    ],
)
