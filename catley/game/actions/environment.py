from __future__ import annotations

from typing import TYPE_CHECKING

from catley.game.actions.base import GameAction, GameActionResult
from catley.world import tile_types

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character


class OpenDoorAction(GameAction):
    """Action for opening a closed door tile."""

    name: str = "Open Door"
    description: str = "Open the door"

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.game_map = controller.gw.game_map
        self.x = x
        self.y = y

    def execute(
        self,
    ) -> GameActionResult | None:  # pragma: no cover - simple state change
        if (
            self.game_map.tiles[self.x, self.y] == tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
        ):
            self.game_map.tiles[self.x, self.y] = tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
            self.game_map.invalidate_property_caches()
            return GameActionResult(should_update_fov=True)
        return None


class CloseDoorAction(GameAction):
    """Action for closing an open door tile."""

    name: str = "Close Door"
    description: str = "Close the door"

    def __init__(
        self, controller: Controller, actor: Character, x: int, y: int
    ) -> None:
        super().__init__(controller, actor)
        self.game_map = controller.gw.game_map
        self.x = x
        self.y = y

    def execute(
        self,
    ) -> GameActionResult | None:  # pragma: no cover - simple state change
        if (
            self.game_map.tiles[self.x, self.y] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
        ):
            self.game_map.tiles[self.x, self.y] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
            self.game_map.invalidate_property_caches()
            return GameActionResult(should_update_fov=True)
        return None
