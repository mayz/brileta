from __future__ import annotations

from typing import TYPE_CHECKING

from catley.environment.tile_types import TileTypeID
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from catley.game.actions.environment import CloseDoorIntent, OpenDoorIntent


class OpenDoorExecutor(ActionExecutor):
    """Executes door opening intents."""

    def execute(self, intent: OpenDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if game_map.tiles[intent.x, intent.y] == TileTypeID.DOOR_CLOSED:
            game_map.tiles[intent.x, intent.y] = TileTypeID.DOOR_OPEN
            game_map.invalidate_property_caches()

            # Notify action panel that door action completed
            frame_manager = intent.controller.frame_manager
            if frame_manager and hasattr(frame_manager, "action_panel_view"):
                frame_manager.action_panel_view.on_door_action_completed(
                    intent.x, intent.y
                )

            return GameActionResult(should_update_fov=True)
        return GameActionResult()


class CloseDoorExecutor(ActionExecutor):
    """Executes door closing intents."""

    def execute(self, intent: CloseDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if game_map.tiles[intent.x, intent.y] == TileTypeID.DOOR_OPEN:
            game_map.tiles[intent.x, intent.y] = TileTypeID.DOOR_CLOSED
            game_map.invalidate_property_caches()

            # Notify action panel that door action completed
            frame_manager = intent.controller.frame_manager
            if frame_manager and hasattr(frame_manager, "action_panel_view"):
                frame_manager.action_panel_view.on_door_action_completed(
                    intent.x, intent.y
                )

            return GameActionResult(should_update_fov=True)
        return GameActionResult()
