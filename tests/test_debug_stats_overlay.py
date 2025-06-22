from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from catley.controller import Controller
from catley.util.live_vars import live_variable_registry
from catley.view.ui.debug_stats_overlay import DebugStatsOverlay


class DummyController:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
        )


def test_calculate_dimensions() -> None:
    live_variable_registry.register("ds.var1", getter=lambda: 1)
    live_variable_registry.register(
        "ds.longer",
        getter=lambda: 22.5,
        formatter=lambda v: f"{v:.1f}",
    )
    live_variable_registry._watched_variables.clear()
    live_variable_registry.watch("ds.var1")
    live_variable_registry.watch("ds.longer")

    overlay = DebugStatsOverlay(cast("Controller", DummyController()))
    overlay._calculate_dimensions()

    expected_width = max(len("ds.var1: 1"), len("ds.longer: 22.5")) + 2
    assert overlay.width == expected_width
    assert overlay.height == 2
    assert overlay.x_tiles == 80 - overlay.width
    assert overlay.y_tiles == 1
    assert overlay.pixel_width == overlay.width * 8
    assert overlay.pixel_height == overlay.height * 16

    live_variable_registry._watched_variables.clear()
    overlay._calculate_dimensions()
    assert overlay.width == 0
    assert overlay.height == 0


def test_draw_content() -> None:
    live_variable_registry.register(
        "ds.b",
        getter=lambda: 2.0,
        formatter=lambda v: f"{v:.1f}",
    )
    live_variable_registry.register("ds.a", getter=lambda: 1)
    live_variable_registry._watched_variables.clear()
    live_variable_registry.watch("ds.b")
    live_variable_registry.watch("ds.a")

    overlay = DebugStatsOverlay(cast("Controller", DummyController()))
    overlay.canvas = MagicMock()
    overlay.tile_dimensions = (8, 16)

    overlay.draw_content()

    calls = overlay.canvas.draw_text.call_args_list
    assert len(calls) == 2
    assert calls[0].args[2] == "ds.a: 1"
    assert calls[1].args[2] == "ds.b: 2.0"
    assert calls[0].args[0] == 0
    assert calls[0].args[1] == 0
    assert calls[1].args[1] == 16
