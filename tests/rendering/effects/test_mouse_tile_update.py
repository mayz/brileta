from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from catley.controller import Controller
from catley.game.actors import Actor
from catley.util.coordinates import RootConsoleTilePos
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.views.world_view import WorldView


class DummyActor:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.ch = "@"
        self.color = (255, 255, 255)


class DummyGameMap:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visible = [[True] * height for _ in range(width)]


class DummyGW:
    def __init__(self) -> None:
        self.player = DummyActor(0, 0)
        self.actors = [self.player]
        self.game_map = DummyGameMap(20, 20)
        self.mouse_tile_location_on_map = None


class DummyFrameManager:
    def __init__(self) -> None:
        self.cursor_manager = SimpleNamespace(
            mouse_pixel_x=0,
            mouse_pixel_y=0,
        )
        self.world_view: WorldView | None = None

    def get_world_coords_from_root_tile_coords(
        self, coords: RootConsoleTilePos
    ) -> tuple[int, int] | None:
        assert self.world_view is not None
        view = self.world_view
        vp_x = coords[0] - view.x
        vp_y = coords[1] - view.y
        if not (0 <= vp_x < view.width and 0 <= vp_y < view.height):
            return None
        return view.viewport_system.screen_to_world(vp_x, vp_y)


@dataclass
class DummyController:
    gw: DummyGW
    graphics: object
    clock: object
    active_mode: object | None
    is_combat_mode: object
    coordinate_converter: object
    frame_manager: DummyFrameManager


def make_controller(root_tile: RootConsoleTilePos = (5, 5)) -> DummyController:
    gw = DummyGW()
    renderer = SimpleNamespace(
        clear_console=lambda *a, **k: None,
        root_console=None,
        blit_console=lambda *a, **k: None,
        pixel_to_tile=lambda _x, _y: root_tile,
    )
    clock = SimpleNamespace(last_delta_time=0.016)
    coordinate_converter = SimpleNamespace(pixel_to_tile=lambda _x, _y: root_tile)
    frame_manager = DummyFrameManager()
    return DummyController(
        gw=gw,
        graphics=renderer,
        clock=clock,
        active_mode=None,
        is_combat_mode=lambda: False,
        coordinate_converter=coordinate_converter,
        frame_manager=frame_manager,
    )


def test_mouse_tile_location_updates_with_camera() -> None:
    controller = make_controller()
    view = WorldView(cast(Controller, controller), ScreenShake())
    controller.frame_manager.world_view = view
    view.set_bounds(0, 0, 10, 10)

    player = controller.gw.player
    player.x = 5
    player.y = 5
    view.viewport_system.update_camera(
        cast(Actor, player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    view.viewport_system.camera.set_position(player.x, player.y)
    view._update_mouse_tile_location()
    assert controller.gw.mouse_tile_location_on_map == (5, 5)

    player.x = 6
    view.viewport_system.update_camera(
        cast(Actor, player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    view.viewport_system.camera.set_position(player.x, player.y)
    view._update_mouse_tile_location()
    assert controller.gw.mouse_tile_location_on_map == (6, 5)
