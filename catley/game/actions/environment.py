from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameAction
from catley.world import tile_types

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class OpenDoorAction(GameAction):
    """Action for opening a closed door tile."""

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.game_map = controller.gw.game_map
        self.x = x
        self.y = y

    def execute(self) -> None:  # pragma: no cover - simple state change
        if (
            self.game_map.tiles[self.x, self.y] == tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
        ):
            self.game_map.tiles[self.x, self.y] = tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
            self.game_map.invalidate_property_caches()


class CloseDoorAction(GameAction):
    """Action for closing an open door tile."""

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.game_map = controller.gw.game_map
        self.x = x
        self.y = y

    def execute(self) -> None:  # pragma: no cover - simple state change
        if (
            self.game_map.tiles[self.x, self.y] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
        ):
            self.game_map.tiles[self.x, self.y] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
            self.game_map.invalidate_property_caches()
