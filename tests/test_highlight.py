from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import numpy as np

from catley.controller import Controller
from catley.game.actors import Actor
from catley.types import Opacity
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
        from catley.environment import tile_types

        self.width = width
        self.height = height
        self.dark_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.light_appearance_map = np.zeros(
            (width, height), dtype=tile_types.TileTypeAppearance
        )
        self.explored = np.zeros((width, height), dtype=bool)
        self.visible = np.zeros((width, height), dtype=bool)


class DummyGW:
    def __init__(self) -> None:
        self.player = DummyActor(0, 0)
        self.actors = [self.player]
        self.game_map = DummyGameMap(20, 20)
        self.game_map.visible[:] = True
        self.selected_actor = None
        self.mouse_tile_location_on_map = None
        self.lighting = MagicMock()
        self.lighting.compute_lighting_with_shadows.side_effect = (
            lambda w, h, *_args, **_kwargs: 1.0 * np.ones((w, h, 3))
        )

    def add_actor(self, actor: DummyActor) -> None:
        self.actors.append(actor)


@dataclass
class DummyController:
    gw: DummyGW
    graphics: MagicMock
    clock: object
    active_mode: object | None
    is_targeting_mode: object


def make_controller() -> DummyController:
    from types import SimpleNamespace

    gw = DummyGW()
    graphics = MagicMock()
    graphics.clear_console = lambda *args, **kwargs: None
    graphics.root_console = None
    graphics.blit_console = lambda *args, **kwargs: None
    clock = SimpleNamespace(last_delta_time=0.016)
    return DummyController(
        gw=gw,
        graphics=graphics,
        clock=clock,
        active_mode=None,
        is_targeting_mode=lambda: False,
    )


def test_highlight_actor_converts_world_to_screen() -> None:
    controller = make_controller()
    controller.gw.player.x = 10
    controller.gw.player.y = 10
    view = WorldView(cast(Controller, controller), ScreenShake())
    view.set_bounds(0, 0, 10, 10)
    view.viewport_system.update_camera(
        cast(Actor, controller.gw.player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    view.viewport_system.camera.set_position(10.0, 10.0)

    actor = DummyActor(12, 10)
    controller.gw.add_actor(actor)

    view.highlight_actor(cast(Actor, actor), (1, 2, 3))

    # Verify the renderer method was called with correct coordinates
    vp_x, vp_y = view.viewport_system.world_to_screen(actor.x, actor.y)
    expected_root_x = view.x + vp_x
    expected_root_y = view.y + vp_y
    controller.graphics.draw_tile_highlight.assert_called_once_with(
        expected_root_x, expected_root_y, (1, 2, 3), Opacity(0.4)
    )


def test_highlight_actor_offscreen_is_ignored() -> None:
    controller = make_controller()
    controller.gw.player.x = 10
    controller.gw.player.y = 10
    view = WorldView(cast(Controller, controller), ScreenShake())
    view.set_bounds(0, 0, 10, 10)
    view.viewport_system.update_camera(
        cast(Actor, controller.gw.player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    view.viewport_system.camera.set_position(10.0, 10.0)

    offscreen_actor = DummyActor(0, 0)
    controller.gw.add_actor(offscreen_actor)

    view.highlight_actor(cast(Actor, offscreen_actor), (4, 5, 6))

    # Verify the renderer method was not called for offscreen actors
    controller.graphics.draw_tile_highlight.assert_not_called()
