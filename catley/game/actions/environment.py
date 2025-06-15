from __future__ import annotations

from typing import TYPE_CHECKING

from .base import GameIntent

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


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
