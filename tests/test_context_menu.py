from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import tcod.event

from catley import colors
from catley.controller import Controller
from catley.environment import tile_types
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from catley.util.coordinates import RootConsoleTilePos
from catley.view.render.backends.tcod.renderer import TCODRenderer
from catley.view.ui.context_menu import ContextMenu
from tests.helpers import DummyGameWorld
from tests.test_canvases import _make_renderer


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.renderer = _make_renderer()
        tcod_renderer = cast(TCODRenderer, self.renderer)
        cast(Any, tcod_renderer.root_console).width = 80
        cast(Any, tcod_renderer.root_console).height = 50
        self.coordinate_converter = SimpleNamespace(pixel_to_tile=lambda x, y: (x, y))


def _make_controller() -> Controller:
    gw = DummyGameWorld()
    player = Character(
        0, 0, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    return DummyController(gw)


def _make_controller_with_door(reachable: bool) -> tuple[Controller, tuple[int, int]]:
    controller = _make_controller()
    gm = controller.gw.game_map
    door_pos = (3, 0)
    gm.tiles[door_pos] = tile_types.TILE_TYPE_ID_DOOR_CLOSED  # type: ignore[attr-defined]
    if not reachable:
        gm.tiles[2, 0] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
        gm.tiles[3, 1] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
        gm.tiles[4, 0] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
        gm.tiles[2, 1] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
        gm.tiles[4, 1] = tile_types.TILE_TYPE_ID_WALL  # type: ignore[attr-defined]
    gm.invalidate_property_caches()
    return controller, door_pos


def test_context_menu_closes_on_click_outside() -> None:
    controller = _make_controller()
    menu = ContextMenu(controller, None, (5, 5))
    menu.show()
    menu._calculate_dimensions()

    outside: RootConsoleTilePos = (
        (menu.x_tiles + menu.width + 1) * controller.renderer.tile_dimensions[0],
        (menu.y_tiles + menu.height + 1) * controller.renderer.tile_dimensions[1],
    )
    event = tcod.event.MouseButtonDown(outside, outside, tcod.event.MouseButton.LEFT)

    consumed = menu.handle_input(event)

    assert consumed
    assert not menu.is_active


def test_context_menu_stays_open_on_click_inside() -> None:
    controller = _make_controller()
    menu = ContextMenu(controller, None, (5, 5))
    menu.show()
    menu._calculate_dimensions()

    inside: RootConsoleTilePos = (
        (menu.x_tiles + 1) * controller.renderer.tile_dimensions[0],
        (menu.y_tiles + 1) * controller.renderer.tile_dimensions[1],
    )
    event = tcod.event.MouseButtonDown(inside, inside, tcod.event.MouseButton.LEFT)

    consumed = menu.handle_input(event)

    assert consumed
    assert menu.is_active


def test_menu_hides_unreachable_door_action() -> None:
    controller, door_pos = _make_controller_with_door(reachable=False)
    menu = ContextMenu(controller, door_pos, (0, 0))
    menu.populate_options()
    texts = [o.text for o in menu.options]
    assert "Go to and Open Door" not in texts


def test_menu_shows_reachable_door_action() -> None:
    controller, door_pos = _make_controller_with_door(reachable=True)
    menu = ContextMenu(controller, door_pos, (0, 0))
    menu.populate_options()
    texts = [o.text for o in menu.options]
    assert "Go to and Open Door" in texts
