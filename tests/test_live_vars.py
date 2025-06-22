from __future__ import annotations

from types import SimpleNamespace

import pytest

from catley.util.caching import ResourceCache
from catley.util.clock import Clock
from catley.util.live_vars import LiveVariable, live_variable_registry
from catley.view.ui.fps_overlay import FPSOverlay


class DummyController:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
        )


class DummyClock(Clock):
    def __init__(self) -> None:
        self.last_time = 0.0


def test_register_and_get_variable() -> None:
    live_variable_registry.register("test.var", getter=lambda: 1, setter=lambda v: None)
    var = live_variable_registry.get_variable("test.var")
    assert var is not None
    assert var.get_value() == 1


def test_register_duplicate_raises() -> None:
    live_variable_registry.register("dup.var", getter=lambda: 0)
    with pytest.raises(ValueError):
        live_variable_registry.register("dup.var", getter=lambda: 1)


def test_get_all_variables_sorted() -> None:
    live_variable_registry.register("b.var", getter=lambda: 0)
    live_variable_registry.register("a.var", getter=lambda: 0)
    names = [v.name for v in live_variable_registry.get_all_variables()]
    assert names == ["a.var", "b.var"]


def test_live_variable_read_write() -> None:
    store: dict[str, int] = {"value": 0}
    var = LiveVariable(
        name="rw",
        description="",
        getter=lambda: store["value"],
        setter=lambda v: store.__setitem__("value", v),
    )
    assert var.get_value() == 0
    assert var.set_value(5) is True
    assert var.get_value() == 5


def test_live_variable_read_only() -> None:
    store: dict[str, int] = {"value": 3}
    var = LiveVariable(
        name="ro",
        description="",
        getter=lambda: store["value"],
        setter=None,
    )
    assert var.get_value() == 3
    assert var.set_value(10) is False
    assert var.get_value() == 3


def test_fps_overlay_registers_variable() -> None:
    FPSOverlay(DummyController(), DummyClock())
    assert live_variable_registry.get_variable("dev.show_fps") is not None


def test_resource_cache_registers_stats_variable() -> None:
    ResourceCache("TestCache")
    var = live_variable_registry.get_variable("cache.TestCache.stats")
    assert var is not None
    assert "TestCache" in var.description
