from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

from brileta.controller import Controller
from brileta.game.actors import Actor
from brileta.util.coordinates import Rect, RootConsoleTilePos
from brileta.view.render.effects.screen_shake import ScreenShake
from brileta.view.views.world_view import WorldView


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
    def __init__(self, graphics: SimpleNamespace | None = None) -> None:
        self.cursor_manager = SimpleNamespace(
            mouse_pixel_x=0,
            mouse_pixel_y=0,
        )
        self.world_view: WorldView | None = None
        self.graphics: SimpleNamespace | None = graphics

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

    def pixel_to_world_tile(
        self, pixel_x: float, pixel_y: float
    ) -> tuple[int, int] | None:
        """Scroll-compensated pixel-to-world conversion (mirrors FrameManager)."""
        assert self.world_view is not None and self.graphics is not None
        cam_frac_x, cam_frac_y = (
            self.world_view.viewport_system.get_camera_fractional_offset()
        )
        base_px = self.graphics.console_to_screen_coords(0.0, 0.0)
        frac_px = self.graphics.console_to_screen_coords(cam_frac_x, cam_frac_y)
        adjusted_x = pixel_x + (frac_px[0] - base_px[0])
        adjusted_y = pixel_y + (frac_px[1] - base_px[1])
        root_tile = self.graphics.pixel_to_tile(adjusted_x, adjusted_y)
        return self.get_world_coords_from_root_tile_coords(root_tile)

    def get_visible_bounds(self) -> Rect | None:
        if self.world_view is None:
            return None
        return self.world_view.viewport_system.get_visible_bounds()


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
        console_to_screen_coords=lambda x, y: (float(x), float(y)),
    )
    clock = SimpleNamespace(last_delta_time=0.016)
    coordinate_converter = SimpleNamespace(pixel_to_tile=lambda _x, _y: root_tile)
    frame_manager = DummyFrameManager(graphics=renderer)
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


def test_pixel_to_world_tile_compensates_for_fractional_camera() -> None:
    """pixel_to_world_tile should compensate for the camera's fractional offset.

    When the camera sits between integer tile positions, smooth scrolling shifts
    all visual content by the fractional amount. Click detection must account for
    this so the tile under the cursor matches what the user sees, not the
    unshifted grid. Without the compensation, clicks near tile boundaries would
    map to the wrong world tile.
    """
    # Use tile_size=10 for easy pixel math. Each tile is 10 px wide.
    tile_size = 10

    # Create a controller with a realistic pixel_to_tile that respects pixel positions.
    gw = DummyGW()
    renderer = SimpleNamespace(
        pixel_to_tile=lambda px_x, px_y: (
            int(px_x // tile_size),
            int(px_y // tile_size),
        ),
        console_to_screen_coords=lambda x, y: (x * tile_size, y * tile_size),
    )
    frame_manager = DummyFrameManager(graphics=renderer)
    controller = DummyController(
        gw=gw,
        graphics=renderer,
        clock=SimpleNamespace(last_delta_time=0.016),
        active_mode=None,
        is_combat_mode=lambda: False,
        coordinate_converter=None,
        frame_manager=frame_manager,
    )

    view = WorldView(cast(Controller, controller), ScreenShake())
    frame_manager.world_view = view
    view.set_bounds(0, 0, 10, 10)

    # Place camera at fractional position 5.3 (cam_frac = 0.3).
    # This shifts visual content LEFT by 0.3 tiles = 3 pixels.
    player = controller.gw.player
    player.x = 5
    player.y = 5
    view.viewport_system.update_camera(
        cast(Actor, player), gw.game_map.width, gw.game_map.height
    )
    view.viewport_system.camera.set_position(5.3, 5.0)

    cam_frac_x, _ = view.viewport_system.get_camera_fractional_offset()
    assert abs(cam_frac_x - 0.3) < 0.01, f"Expected cam_frac ~0.3, got {cam_frac_x}"

    # World tile 5 is at viewport position 5 in the rounded grid. After the
    # 0.3-tile visual shift, its left edge appears at pixel (5 - 0.3) * 10 = 47.
    # Clicking at pixel 48 (just inside tile 5's visual boundary) should yield
    # world tile 5. Without fractional compensation, pixel 48 maps to root tile
    # floor(48/10) = 4, which would incorrectly resolve to a different world tile.
    result = frame_manager.pixel_to_world_tile(48.0, 50.0)
    assert result is not None
    world_x, _ = result
    assert world_x == 5, (
        f"Expected world tile x=5 for pixel just inside the visual boundary, got {world_x}"
    )
