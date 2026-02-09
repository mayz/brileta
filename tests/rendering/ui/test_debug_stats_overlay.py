from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

from brileta.controller import Controller
from brileta.util.live_vars import live_variable_registry
from brileta.view.ui.debug_stats_overlay import DebugStatsOverlay


class DummyController:
    def __init__(self) -> None:
        self.graphics = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
            console_width_tiles=80,
            console_height_tiles=50,
            create_canvas=lambda transparent=True: MagicMock(),
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
    overlay._refresh_display_lines()
    overlay._calculate_dimensions()

    expected_width = max(len("ds.var1: 1"), len("ds.longer: 22.5")) + 2
    assert overlay.width == expected_width
    assert overlay.height == 2
    assert overlay.x_tiles == 80 - 60  # Right-aligned at fixed column offset
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
    overlay._refresh_display_lines()

    overlay.draw_content()

    calls = overlay.canvas.draw_text.call_args_list
    assert len(calls) == 2
    assert calls[0].args[2] == "ds.a: 1"
    assert calls[1].args[2] == "ds.b: 2.0"
    assert calls[0].args[0] == 0
    assert calls[0].args[1] == 0
    assert calls[1].args[1] == 16


def test_draw_uses_single_getter_call_per_refresh(monkeypatch) -> None:
    getter = MagicMock(return_value=123)
    live_variable_registry.register("ds.once", getter=getter)
    live_variable_registry.watch("ds.once")

    overlay = DebugStatsOverlay(cast("Controller", DummyController()))
    overlay.canvas = MagicMock()
    overlay.canvas.end_frame.return_value = object()
    overlay.show()

    monkeypatch.setattr(
        "brileta.view.ui.debug_stats_overlay.perf_counter",
        lambda: 1.0,
    )
    overlay.draw()

    assert getter.call_count == 1


def test_draw_throttles_refresh_to_2hz(monkeypatch) -> None:
    getter = MagicMock(return_value=1)
    live_variable_registry.register("ds.throttle", getter=getter)
    live_variable_registry.watch("ds.throttle")

    overlay = DebugStatsOverlay(cast("Controller", DummyController()))
    overlay.canvas = MagicMock()
    overlay.canvas.end_frame.return_value = object()
    overlay.show()

    times = iter([1.0, 1.1, 1.4, 1.6])
    monkeypatch.setattr(
        "brileta.view.ui.debug_stats_overlay.perf_counter",
        lambda: next(times),
    )

    overlay.draw()  # refresh
    overlay.draw()  # skip
    overlay.draw()  # skip
    overlay.draw()  # refresh at +0.6s

    assert getter.call_count == 2
