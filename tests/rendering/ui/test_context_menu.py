from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import tcod.event

from catley import colors
from catley.backends.tcod.graphics import TCODGraphicsContext
from catley.controller import Controller
from catley.environment.tile_types import TileTypeID
from catley.game.actors import Character
from catley.game.game_world import GameWorld
from catley.game.turn_manager import TurnManager
from catley.util.coordinates import RootConsoleTilePos
from catley.view.ui.context_menu import ContextMenu
from tests.helpers import DummyGameWorld
from tests.rendering.backends.test_canvases import _make_renderer


@dataclass
class DummyController(Controller):
    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.graphics = _make_renderer()
        tcod_renderer = cast(TCODGraphicsContext, self.graphics)
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
    gm.tiles[door_pos] = TileTypeID.DOOR_CLOSED
    if not reachable:
        gm.tiles[2, 0] = TileTypeID.WALL
        gm.tiles[3, 1] = TileTypeID.WALL
        gm.tiles[4, 0] = TileTypeID.WALL
        gm.tiles[2, 1] = TileTypeID.WALL
        gm.tiles[4, 1] = TileTypeID.WALL
    gm.invalidate_property_caches()
    return controller, door_pos


def test_context_menu_closes_on_click_outside() -> None:
    controller = _make_controller()
    menu = ContextMenu(controller, None, (5, 5))
    menu.show()
    menu._calculate_dimensions()

    # Calculate outside position using menu's actual pixel dimensions
    tile_w, tile_h = controller.graphics.tile_dimensions
    menu_pixel_x = menu.x_tiles * tile_w
    menu_pixel_y = menu.y_tiles * tile_h
    # Click outside the menu's pixel bounds (to the right and below)
    outside: RootConsoleTilePos = (
        menu_pixel_x + menu.pixel_width + 10,
        menu_pixel_y + menu.pixel_height + 10,
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

    # Calculate inside position using menu's actual pixel dimensions
    tile_w, tile_h = controller.graphics.tile_dimensions
    menu_pixel_x = menu.x_tiles * tile_w
    menu_pixel_y = menu.y_tiles * tile_h
    # Click inside the menu's pixel bounds (with some padding from edges)
    inside: RootConsoleTilePos = (
        menu_pixel_x + menu._char_width,  # One character in from left edge
        menu_pixel_y + menu._line_height,  # One line down from top edge
    )
    event = tcod.event.MouseButtonDown(inside, inside, tcod.event.MouseButton.LEFT)

    consumed = menu.handle_input(event)

    assert consumed
    assert menu.is_active


def test_menu_shows_door_action_regardless_of_reachability() -> None:
    """Door actions are always shown. ActionPlans handle pathfinding lazily.

    Previously, the menu would check reachability before showing options.
    With ActionPlans, we always show the option and let the plan handle
    path calculation (and graceful cancellation if unreachable).
    """
    # Test unreachable door - option should still be shown
    controller_unreachable, door_pos_unreachable = _make_controller_with_door(
        reachable=False
    )
    menu = ContextMenu(controller_unreachable, door_pos_unreachable, (0, 0))
    menu.populate_options()
    texts = [o.text for o in menu.options]
    assert "Go to and Open Door" in texts

    # Test reachable door - option should be shown
    controller_reachable, door_pos_reachable = _make_controller_with_door(
        reachable=True
    )
    menu = ContextMenu(controller_reachable, door_pos_reachable, (0, 0))
    menu.populate_options()
    texts = [o.text for o in menu.options]
    assert "Go to and Open Door" in texts
