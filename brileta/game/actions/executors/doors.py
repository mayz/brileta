from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.environment.tile_types import TileTypeID
from brileta.game.actions.base import GameActionResult
from brileta.game.actions.executors.base import ActionExecutor

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.environment import CloseDoorIntent, OpenDoorIntent


def _notify_door_action(controller: Controller, x: int, y: int) -> None:
    """Tell the action panel a door was opened/closed, if the UI is present."""
    fm = controller.frame_manager
    if fm is None:
        return
    apv = getattr(fm, "action_panel_view", None)
    if apv is None:
        return
    callback = getattr(apv, "on_door_action_completed", None)
    if callback is not None:
        callback(x, y)


class OpenDoorExecutor(ActionExecutor):
    """Executes door opening intents."""

    def execute(self, intent: OpenDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if game_map.tiles[intent.x, intent.y] == TileTypeID.DOOR_CLOSED:
            game_map.tiles[intent.x, intent.y] = TileTypeID.DOOR_OPEN
            game_map.invalidate_property_caches()

            _notify_door_action(intent.controller, intent.x, intent.y)

            return GameActionResult(should_update_fov=True)
        return GameActionResult()


class CloseDoorExecutor(ActionExecutor):
    """Executes door closing intents."""

    def execute(self, intent: CloseDoorIntent) -> GameActionResult | None:  # type: ignore[override]
        game_map = intent.controller.gw.game_map
        if game_map.tiles[intent.x, intent.y] == TileTypeID.DOOR_OPEN:
            game_map.tiles[intent.x, intent.y] = TileTypeID.DOOR_CLOSED
            game_map.invalidate_property_caches()

            _notify_door_action(intent.controller, intent.x, intent.y)

            return GameActionResult(should_update_fov=True)
        return GameActionResult()
