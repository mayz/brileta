from __future__ import annotations

from types import SimpleNamespace

import pytest

from catley.util.caching import ResourceCache
from catley.util.live_vars import (
    LiveVariable,
    live_variable_registry,
    record_time_live_variable,
)


class DummyController:
    def __init__(self) -> None:
        self.renderer = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
        )


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


def test_resource_cache_registers_stats_variable() -> None:
    ResourceCache("TestCache")
    var = live_variable_registry.get_variable("cache.TestCache.stats")
    assert var is not None
    assert "TestCache" in var.description


def test_watch_and_unwatch() -> None:
    live_variable_registry.register("watch.var1", getter=lambda: 1)
    live_variable_registry._watched_variables.clear()
    live_variable_registry.watch("watch.var1")
    assert [v.name for v in live_variable_registry.get_watched_variables()] == [
        "watch.var1"
    ]
    live_variable_registry.unwatch("watch.var1")
    assert live_variable_registry.get_watched_variables() == []


def test_toggle_watch() -> None:
    live_variable_registry.register("watch.var2", getter=lambda: 2)
    live_variable_registry._watched_variables.clear()
    live_variable_registry.toggle_watch("watch.var2")
    assert [v.name for v in live_variable_registry.get_watched_variables()] == [
        "watch.var2"
    ]
    live_variable_registry.toggle_watch("watch.var2")
    assert live_variable_registry.get_watched_variables() == []


def test_record_metric_raises_on_unknown_name() -> None:
    """record_metric raises KeyError when the metric is not registered."""
    with pytest.raises(KeyError, match="not registered"):
        live_variable_registry.record_metric("nonexistent.metric", 1.0)


def test_record_metric_raises_on_non_metric_variable() -> None:
    """record_metric raises KeyError when the variable has no stats tracker."""
    live_variable_registry.register("plain.var", getter=lambda: 0)
    with pytest.raises(KeyError, match="not a metric"):
        live_variable_registry.record_metric("plain.var", 1.0)


def test_record_metric_succeeds_for_registered_metric() -> None:
    """record_metric works for properly registered metrics."""
    live_variable_registry.register_metric("timing.test", description="Test metric")
    # Should not raise.
    live_variable_registry.record_metric("timing.test", 42.0)
    var = live_variable_registry.get_variable("timing.test")
    assert var is not None
    assert var.stats_var is not None
    assert var.stats_var.sample_count == 1


def test_get_watched_variables_sorted() -> None:
    live_variable_registry.register("watch.b", getter=lambda: 1)
    live_variable_registry.register("watch.a", getter=lambda: 1)
    live_variable_registry._watched_variables.clear()
    live_variable_registry.watch("watch.b")
    live_variable_registry.watch("watch.a")
    names = [v.name for v in live_variable_registry.get_watched_variables()]
    assert names == ["watch.a", "watch.b"]


def test_record_time_raises_in_strict_mode_for_unregistered_metric() -> None:
    """record_time_live_variable raises KeyError when strict and metric missing."""
    live_variable_registry.strict = True
    with (
        pytest.raises(KeyError, match="not registered"),
        record_time_live_variable("missing.metric"),
    ):
        pass  # body executes, error raised in finally


def test_record_time_skips_silently_in_non_strict_mode() -> None:
    """record_time_live_variable silently skips when not strict."""
    live_variable_registry.strict = False
    # Should not raise.
    with record_time_live_variable("missing.metric"):
        pass


def test_record_time_records_in_strict_mode_for_registered_metric() -> None:
    """record_time_live_variable records successfully in strict mode."""
    live_variable_registry.strict = True
    live_variable_registry.register_metric("strict.timing", description="Test")
    with record_time_live_variable("strict.timing"):
        pass
    var = live_variable_registry.get_variable("strict.timing")
    assert var is not None
    assert var.stats_var is not None
    assert var.stats_var.sample_count == 1
