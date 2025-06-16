from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from catley.controller import Controller
from catley.game.actors import Actor
from catley.util.coordinates import RootConsoleTilePos
from catley.view.panels.world_panel import WorldPanel
from catley.view.render.effects.screen_shake import ScreenShake


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
        self.world_panel: WorldPanel | None = None

    def get_world_coords_from_root_tile_coords(
        self, coords: RootConsoleTilePos
    ) -> tuple[int, int] | None:
        assert self.world_panel is not None
        panel = self.world_panel
        vp_x = coords[0] - panel.x
        vp_y = coords[1] - panel.y
        if not (0 <= vp_x < panel.width and 0 <= vp_y < panel.height):
            return None
        return panel.viewport_system.screen_to_world(vp_x, vp_y)


@dataclass
class DummyController:
    gw: DummyGW
    renderer: object
    clock: object
    active_mode: object | None
    is_targeting_mode: object
    coordinate_converter: object
    frame_manager: DummyFrameManager


def make_controller(root_tile: RootConsoleTilePos = (5, 5)) -> DummyController:
    gw = DummyGW()
    renderer = SimpleNamespace(
        clear_console=lambda *a, **k: None,
        root_console=None,
        blit_console=lambda *a, **k: None,
    )
    clock = SimpleNamespace(last_delta_time=0.016)
    coordinate_converter = SimpleNamespace(pixel_to_tile=lambda _x, _y: root_tile)
    frame_manager = DummyFrameManager()
    return DummyController(
        gw=gw,
        renderer=renderer,
        clock=clock,
        active_mode=None,
        is_targeting_mode=lambda: False,
        coordinate_converter=coordinate_converter,
        frame_manager=frame_manager,
    )


def test_mouse_tile_location_updates_with_camera() -> None:
    controller = make_controller()
    panel = WorldPanel(cast(Controller, controller), ScreenShake())
    controller.frame_manager.world_panel = panel
    panel.resize(0, 0, 10, 10)

    player = controller.gw.player
    player.x = 5
    player.y = 5
    panel.viewport_system.update_camera(
        cast(Actor, player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    panel.viewport_system.camera.set_position(player.x, player.y)
    panel._update_mouse_tile_location()
    assert controller.gw.mouse_tile_location_on_map == (5, 5)

    player.x = 6
    panel.viewport_system.update_camera(
        cast(Actor, player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    panel.viewport_system.camera.set_position(player.x, player.y)
    panel._update_mouse_tile_location()
    assert controller.gw.mouse_tile_location_on_map == (6, 5)
