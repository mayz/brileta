from __future__ import annotations

from typing import TYPE_CHECKING

from catley.environment import tile_types
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent


class OpenDoorExecutor(ActionExecutor):
    """Executes door opening intents."""

    def __init__(self) -> None:
        """Create an OpenDoorExecutor without requiring a controller."""
        pass

    def execute(self, intent: OpenDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if (
            game_map.tiles[intent.x, intent.y] == tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
        ):
            game_map.tiles[intent.x, intent.y] = tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
            game_map.invalidate_property_caches()
            return GameActionResult(should_update_fov=True)
        return GameActionResult()


class CloseDoorExecutor(ActionExecutor):
    """Executes door closing intents."""

    def __init__(self) -> None:
        """Create a CloseDoorExecutor without requiring a controller."""
        pass

    def execute(self, intent: CloseDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if (
            game_map.tiles[intent.x, intent.y] == tile_types.TILE_TYPE_ID_DOOR_OPEN  # type: ignore[attr-defined]
        ):
            game_map.tiles[intent.x, intent.y] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
            game_map.invalidate_property_caches()
            return GameActionResult(should_update_fov=True)
        return GameActionResult()
