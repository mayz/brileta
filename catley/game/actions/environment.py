from __future__ import annotations

from typing import TYPE_CHECKING

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
