"""Unit tests for persistent NPC presence indicators.

Covers the FOV/on-screen gates that decide whether an indicator draws. Actual
GPU rendering is mocked (backend-specific), so these assert on draw calls.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

from brileta import colors
from brileta.game.actors import NPC
from brileta.game.actors.indicators import IndicatorKind
from brileta.game.game_world import GameWorld
from brileta.types import ViewOffset
from brileta.view.render.effects.indicators import IndicatorRenderer
from tests.helpers import DummyGameWorld


def _make_graphics() -> MagicMock:
    graphics = MagicMock()
    graphics.tile_dimensions = (10, 10)
    graphics.console_to_screen_coords.return_value = (0.0, 0.0)
    graphics.texture_from_numpy.return_value = object()
    return graphics


def _make_viewport(is_visible: bool = True) -> MagicMock:
    viewport = MagicMock()
    viewport.is_visible.return_value = is_visible
    viewport.world_to_screen_float.return_value = (0.0, 0.0)
    viewport.get_display_scale_factors.return_value = (1.0, 1.0)
    return viewport


def _world_with_indicator_npc(x: int = 5, y: int = 5) -> tuple[DummyGameWorld, NPC]:
    gw = DummyGameWorld()
    npc = NPC(x, y, "N", colors.WHITE, "NPC", game_world=cast(GameWorld, gw))
    npc.indicator = IndicatorKind.ATTACK
    gw.add_actor(npc)
    return gw, npc


def test_draws_indicator_for_visible_npc() -> None:
    gw, _npc = _world_with_indicator_npc()
    graphics = _make_graphics()
    renderer = IndicatorRenderer()

    renderer.render(
        graphics, _make_viewport(), cast(ViewOffset, (0, 0)), cast(GameWorld, gw)
    )

    assert graphics.draw_texture_alpha.call_count == 1


def test_skips_npc_outside_fov() -> None:
    gw, npc = _world_with_indicator_npc()
    gw.game_map.visible[npc.x, npc.y] = False  # Hidden from the player.
    graphics = _make_graphics()
    renderer = IndicatorRenderer()

    renderer.render(
        graphics, _make_viewport(), cast(ViewOffset, (0, 0)), cast(GameWorld, gw)
    )

    assert graphics.draw_texture_alpha.call_count == 0


def test_skips_npc_offscreen() -> None:
    gw, _npc = _world_with_indicator_npc()
    graphics = _make_graphics()
    renderer = IndicatorRenderer()

    renderer.render(
        graphics,
        _make_viewport(is_visible=False),
        cast(ViewOffset, (0, 0)),
        cast(GameWorld, gw),
    )

    assert graphics.draw_texture_alpha.call_count == 0


def test_skips_npc_with_no_indicator() -> None:
    gw, npc = _world_with_indicator_npc()
    npc.indicator = None
    graphics = _make_graphics()
    renderer = IndicatorRenderer()

    renderer.render(
        graphics, _make_viewport(), cast(ViewOffset, (0, 0)), cast(GameWorld, gw)
    )

    assert graphics.draw_texture_alpha.call_count == 0
