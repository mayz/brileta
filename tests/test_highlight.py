from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import numpy as np
from view.panels.world_panel import WorldPanel

from catley.controller import Controller
from catley.game.actors import Actor
from catley.view.effects.screen_shake import ScreenShake


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


@dataclass
class DummyController:
    gw: DummyGW
    renderer: object
    clock: object
    active_mode: object | None
    is_targeting_mode: object


def make_controller() -> DummyController:
    from types import SimpleNamespace

    gw = DummyGW()
    renderer = SimpleNamespace(
        clear_console=lambda *args, **kwargs: None,
        root_console=None,
        blit_console=lambda *args, **kwargs: None,
    )
    clock = SimpleNamespace(last_delta_time=0.016)
    return DummyController(
        gw=gw,
        renderer=renderer,
        clock=clock,
        active_mode=None,
        is_targeting_mode=lambda: False,
    )


def test_highlight_actor_converts_world_to_screen() -> None:
    controller = make_controller()
    controller.gw.player.x = 10
    controller.gw.player.y = 10
    panel = WorldPanel(cast(Controller, controller), ScreenShake())
    panel.resize(0, 0, 10, 10)
    panel.viewport_system.update_camera(
        cast(Actor, controller.gw.player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    panel.viewport_system.camera.set_position(10.0, 10.0)

    actor = DummyActor(12, 10)
    controller.gw.actors.append(actor)

    panel.highlight_actor(cast(Actor, actor), (1, 2, 3))

    px, py = panel.viewport_system.world_to_screen(actor.x, actor.y)
    assert np.array_equal(panel.game_map_console.rgb["bg"][px, py], [1, 2, 3])


def test_highlight_actor_offscreen_is_ignored() -> None:
    controller = make_controller()
    controller.gw.player.x = 10
    controller.gw.player.y = 10
    panel = WorldPanel(cast(Controller, controller), ScreenShake())
    panel.resize(0, 0, 10, 10)
    panel.viewport_system.update_camera(
        cast(Actor, controller.gw.player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    panel.viewport_system.camera.set_position(10.0, 10.0)

    offscreen_actor = DummyActor(0, 0)
    controller.gw.actors.append(offscreen_actor)
    before = panel.game_map_console.rgb["bg"].copy()

    panel.highlight_actor(cast(Actor, offscreen_actor), (4, 5, 6))

    assert np.array_equal(panel.game_map_console.rgb["bg"], before)
