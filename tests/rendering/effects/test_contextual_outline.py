from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from catley import colors
from catley.config import CONTEXTUAL_OUTLINE_ALPHA
from catley.controller import Controller
from catley.game.actors import Actor
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.views.world_view import WorldView


class DummyActor:
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.ch = "@"
        self.color = (255, 255, 255)
        self.character_layers: list[object] = []


class DummyGameMap:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visible = np.zeros((width, height), dtype=bool)


class DummyGW:
    def __init__(self) -> None:
        self.player = DummyActor(0, 0)
        self.actors = [self.player]
        self.game_map = DummyGameMap(20, 20)


@dataclass
class DummyController:
    gw: DummyGW
    graphics: MagicMock
    clock: object
    active_mode: object | None
    is_combat_mode: Callable[[], bool]
    contextual_target: DummyActor | None


def make_controller(*, is_combat: bool) -> DummyController:
    gw = DummyGW()
    gw.game_map.visible[:] = True
    graphics = MagicMock()
    graphics.console_to_screen_coords = lambda x, y: (x, y)
    clock = SimpleNamespace(last_delta_time=0.016, last_time=0.0)
    return DummyController(
        gw=gw,
        graphics=graphics,
        clock=clock,
        active_mode=None,
        is_combat_mode=lambda: is_combat,
        contextual_target=None,
    )


def _setup_view(controller: DummyController) -> WorldView:
    view = WorldView(cast(Controller, controller), ScreenShake())
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
    return view


def test_contextual_outline_renders_for_visible_target_out_of_combat() -> None:
    controller = make_controller(is_combat=False)
    view = _setup_view(controller)

    target = DummyActor(6, 5)
    controller.gw.actors.append(target)
    controller.contextual_target = target

    view._render_contextual_target_outline()

    vp_x, vp_y = view.viewport_system.world_to_screen(target.x, target.y)
    expected_root_x = view.x + vp_x
    expected_root_y = view.y + vp_y
    controller.graphics.draw_actor_outline.assert_called_once_with(
        target.ch,
        expected_root_x,
        expected_root_y,
        colors.CONTEXTUAL_OUTLINE,
        float(CONTEXTUAL_OUTLINE_ALPHA),
        scale_x=1.0,
        scale_y=1.0,
    )


@pytest.mark.parametrize(
    ("is_combat", "is_visible"),
    [
        (True, True),
        (False, False),
    ],
)
def test_contextual_outline_skips_when_combat_or_not_visible(
    is_combat: bool, is_visible: bool
) -> None:
    controller = make_controller(is_combat=is_combat)
    view = _setup_view(controller)

    target = DummyActor(6, 5)
    controller.gw.actors.append(target)
    controller.contextual_target = target
    controller.gw.game_map.visible[target.x, target.y] = is_visible

    view._render_contextual_target_outline()

    controller.graphics.draw_actor_outline.assert_not_called()
