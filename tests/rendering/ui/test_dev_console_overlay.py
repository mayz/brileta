import time
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from brileta import input_events
from brileta.app import App, AppConfig
from brileta.controller import Controller
from brileta.util.live_vars import live_variable_registry
from brileta.view.ui.dev_console_overlay import DevConsoleOverlay
from brileta.view.ui.overlays import TextOverlay


class DummyApp(App):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(AppConfig(width=80, height=50, title="test", vsync=False))

    def run(self) -> None:
        pass

    def prepare_for_new_frame(self) -> None:
        pass

    def present_frame(self) -> None:
        pass

    def toggle_fullscreen(self) -> None:
        pass

    def _exit_backend(self) -> None:
        pass

    def quit(self) -> None:
        """Simulate exit for testing."""
        raise SystemExit()


class DummyController:
    def __init__(self) -> None:
        self.graphics = SimpleNamespace(
            root_console=SimpleNamespace(width=80, height=50),
            tile_dimensions=(8, 16),
            sdl_renderer=None,
        )
        self.app = DummyApp()


def make_overlay() -> DevConsoleOverlay:
    return DevConsoleOverlay(cast("Controller", DummyController()))


def test_execute_command_set_get_toggle_watch() -> None:
    store: dict[str, object] = {"i": 0, "f": 0.0, "b": False}
    live_variable_registry.register(
        "test.int",
        getter=lambda: store["i"],
        setter=lambda v: store.__setitem__("i", v),
    )
    live_variable_registry.register(
        "test.float",
        getter=lambda: store["f"],
        setter=lambda v: store.__setitem__("f", v),
    )
    live_variable_registry.register(
        "test.bool",
        getter=lambda: store["b"],
        setter=lambda v: store.__setitem__("b", v),
    )

    ov = make_overlay()

    ov.input_buffer = "set test.int 5"
    ov._execute_command()
    assert store["i"] == 5

    ov.input_buffer = "set test.float 3.5"
    ov._execute_command()
    assert store["f"] == 3.5

    ov.input_buffer = "set test.bool true"
    ov._execute_command()
    assert store["b"] is True

    ov.input_buffer = "get test.int"
    ov._execute_command()
    assert "test.int =" in ov.history[-1]

    ov.input_buffer = "toggle test.bool"
    ov._execute_command()
    assert store["b"] is False

    ov.input_buffer = "show test.int"
    ov._execute_command()
    assert live_variable_registry.is_watched("test.int") is True

    prev_len = len(ov.history)
    ov.input_buffer = "help"
    ov._execute_command()
    added = list(ov.history)[prev_len + 1 :]
    assert added[0] == "Available Commands:"
    assert any(line.startswith("  help") for line in added[1:])


def test_execute_command_errors() -> None:
    ov = make_overlay()

    ov.input_buffer = "set missing 1"
    ov._execute_command()
    assert "not found" in ov.history[-1]

    ov.input_buffer = "unknowncmd"
    ov._execute_command()
    assert any("Unknown command" in line for line in ov.history)
    assert any("Available Commands:" in line for line in ov.history)


def test_show_command_invalid_var() -> None:
    ov = make_overlay()
    ov.input_buffer = "show missing"
    ov._execute_command()
    # With pattern matching, unknown vars are treated as prefix patterns
    assert ov.history[-1] == "No variables matching 'missing'."


@pytest.mark.parametrize("cmd", ["quit", "exit"])
def test_quit_and_exit_commands(cmd: str) -> None:
    ov = make_overlay()
    ov.input_buffer = cmd
    with pytest.raises(SystemExit):
        ov._execute_command()


def test_list_no_args_enters_filter_mode() -> None:
    """'list' with no arguments enters interactive filter mode."""
    live_variable_registry.register("filt.a", getter=lambda: 1)

    ov = make_overlay()
    ov.input_buffer = "list"
    ov._execute_command()

    assert ov._filter_mode is True
    assert len(ov._filter_results) > 0


def test_list_with_pattern_shows_tagged_output() -> None:
    """'list <pattern>' uses substring matching and includes [var]/[metric] tags."""
    live_variable_registry.register(
        "cache.size", getter=lambda: 1, description="Size of cache"
    )
    live_variable_registry.register(
        "cache.enabled", getter=lambda: True, description="Cache enabled"
    )
    live_variable_registry.register(
        "misc.value", getter=lambda: 0, description="Misc value"
    )

    ov = make_overlay()

    # Substring filter
    prev_len = len(ov.history)
    ov.input_buffer = "list cache"
    ov._execute_command()
    filtered = list(ov.history)[prev_len + 1 :]
    assert len(filtered) == 2
    assert all("[var]" in line for line in filtered)
    assert all("cache." in line for line in filtered)


def test_list_vars_and_metrics_categories() -> None:
    """'list vars' and 'list metrics' filter by kind."""
    live_variable_registry.register("cat.var", getter=lambda: 1)
    live_variable_registry.register_metric("cat.metric", description="A metric")

    ov = make_overlay()

    prev_len = len(ov.history)
    ov.input_buffer = "list vars"
    ov._execute_command()
    var_lines = list(ov.history)[prev_len + 1 :]
    assert any("cat.var" in line for line in var_lines)
    assert not any("cat.metric" in line for line in var_lines)

    prev_len = len(ov.history)
    ov.input_buffer = "list metrics"
    ov._execute_command()
    metric_lines = list(ov.history)[prev_len + 1 :]
    assert any("cat.metric" in line for line in metric_lines)
    assert not any("cat.var" in line for line in metric_lines)


def test_natural_syntax_get() -> None:
    """Typing a bare variable name prints its value."""
    store = {"v": 42}
    live_variable_registry.register(
        "nat.get",
        getter=lambda: store["v"],
    )

    ov = make_overlay()
    ov.input_buffer = "nat.get"
    ov._execute_command()
    assert "nat.get = 42" in ov.history[-1]


def test_get_uses_live_variable_formatter() -> None:
    """Value output in command mode uses the variable formatter when available."""
    store = {"v": 135.0}
    live_variable_registry.register(
        "fmt.get",
        getter=lambda: store["v"],
        formatter=lambda v: f"{float(v):.1f} deg",
    )

    ov = make_overlay()
    ov.input_buffer = "get fmt.get"
    ov._execute_command()
    assert ov.history[-1] == "fmt.get = 135.0 deg"


def test_get_uses_default_float_format_when_no_formatter() -> None:
    """Float values use the built-in compact formatter when unset."""
    store = {"v": 0.85}
    live_variable_registry.register(
        "fmt.default.float",
        getter=lambda: store["v"],
    )

    ov = make_overlay()
    ov.input_buffer = "get fmt.default.float"
    ov._execute_command()
    assert ov.history[-1] == "fmt.default.float = 0.85"


def test_get_uses_display_decimals_when_configured() -> None:
    """display_decimals overrides default float precision for console output."""
    store = {"v": 0.8567}
    live_variable_registry.register(
        "fmt.decimals.get",
        getter=lambda: store["v"],
        display_decimals=3,
    )

    ov = make_overlay()
    ov.input_buffer = "get fmt.decimals.get"
    ov._execute_command()
    assert ov.history[-1] == "fmt.decimals.get = 0.857"


def test_get_display_decimals_preserves_trailing_zeros_when_explicitly_set() -> None:
    """Explicit display_decimals should keep fixed-width decimal formatting."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "fmt.decimals.fixed",
        getter=lambda: store["v"],
        display_decimals=1,
    )

    ov = make_overlay()
    ov.input_buffer = "get fmt.decimals.fixed"
    ov._execute_command()
    assert ov.history[-1] == "fmt.decimals.fixed = 10.0"


def test_get_display_decimals_zero_preserves_significant_integer_zeros() -> None:
    """display_decimals=0 should not collapse values like 400 into 4."""
    store = {"v": 400.0}
    live_variable_registry.register(
        "fmt.decimals.zero",
        getter=lambda: store["v"],
        display_decimals=0,
    )

    ov = make_overlay()
    ov.input_buffer = "get fmt.decimals.zero"
    ov._execute_command()
    assert ov.history[-1] == "fmt.decimals.zero = 400"


def test_get_uses_default_bool_int_and_repr_formatting() -> None:
    """Default formatting path should handle bool, int, and repr fallback values."""
    live_variable_registry.register("fmt.default.bool", getter=lambda: True)
    live_variable_registry.register("fmt.default.int", getter=lambda: 42)
    live_variable_registry.register("fmt.default.repr", getter=lambda: [1, 2])

    ov = make_overlay()

    ov.input_buffer = "get fmt.default.bool"
    ov._execute_command()
    assert ov.history[-1] == "fmt.default.bool = True"

    ov.input_buffer = "get fmt.default.int"
    ov._execute_command()
    assert ov.history[-1] == "fmt.default.int = 42"

    ov.input_buffer = "get fmt.default.repr"
    ov._execute_command()
    assert ov.history[-1] == "fmt.default.repr = [1, 2]"


def test_slider_activates_on_ranged_var_get() -> None:
    """Bare variable get activates slider for ranged writable variables."""
    store = {"v": 42.0}
    live_variable_registry.register(
        "slider.ranged",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    ov.input_buffer = "slider.ranged"
    ov._execute_command()

    assert ov._slider_var is not None
    assert ov._slider_var.name == "slider.ranged"


def test_get_for_slider_var_does_not_duplicate_static_value_output() -> None:
    """Ranged vars should show slider UI without adding a duplicate value line."""
    store = {"v": 42.0}
    live_variable_registry.register(
        "slider.get.nodup",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    prev_len = len(ov.history)
    ov.input_buffer = "get slider.get.nodup"
    ov._execute_command()

    appended = list(ov.history)[prev_len:]
    assert appended == ["> get slider.get.nodup"]
    assert ov._slider_var is not None
    assert ov._slider_var.name == "slider.get.nodup"


def test_slider_activates_on_ranged_var_set() -> None:
    """Explicit set command should activate slider for ranged writable vars."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.setcmd",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    ov.input_buffer = "set slider.setcmd 45"
    ov._execute_command()

    assert store["v"] == 45.0
    assert ov._slider_var is not None
    assert ov._slider_var.name == "slider.setcmd"


def test_set_for_slider_var_does_not_duplicate_static_value_output() -> None:
    """Set command should keep history concise when slider UI is active."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.set.nodup",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    prev_len = len(ov.history)
    ov.input_buffer = "set slider.set.nodup 45"
    ov._execute_command()

    appended = list(ov.history)[prev_len:]
    assert appended == ["> set slider.set.nodup 45"]
    assert store["v"] == 45.0
    assert ov._slider_var is not None
    assert ov._slider_var.name == "slider.set.nodup"


def test_slider_not_activated_without_range() -> None:
    """Writable vars without value_range should not get a slider."""
    store = {"v": 1.0}
    live_variable_registry.register(
        "slider.norange",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
    )

    ov = make_overlay()
    ov.input_buffer = "slider.norange"
    ov._execute_command()

    assert ov._slider_var is None


def test_slider_not_activated_for_readonly_var() -> None:
    """Read-only vars should not get a slider, even with a range."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.readonly",
        getter=lambda: store["v"],
        value_range=(0.0, 20.0),
    )

    ov = make_overlay()
    ov.input_buffer = "slider.readonly"
    ov._execute_command()

    assert ov._slider_var is None


def test_slider_not_activated_for_metric() -> None:
    """Metric vars are excluded from slider controls."""
    store = {"v": 0.0}
    live_variable_registry.register(
        "slider.metric",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        metric=True,
        value_range=(0.0, 1.0),
    )

    ov = make_overlay()
    ov.input_buffer = "slider.metric"
    ov._execute_command()

    assert ov._slider_var is None


def test_slider_cleared_on_new_command() -> None:
    """Submitting a new command clears an active slider unless reactivated."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.clear",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    ov.input_buffer = "slider.clear"
    ov._execute_command()
    assert ov._slider_var is not None

    ov.input_buffer = "help"
    ov._execute_command()
    assert ov._slider_var is None


def test_slider_persists_across_hide_show() -> None:
    """Active slider variable persists when console is hidden and shown again."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.persist",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "slider.persist"
    ov._execute_command()
    assert ov._slider_var is not None

    ov.hide()
    ov.show()

    assert ov._slider_var is not None
    assert ov._slider_var.name == "slider.persist"


def test_hide_clears_slider_dragging_state() -> None:
    """Hiding the console should end an in-progress slider drag."""
    ov = make_overlay()
    ov.show()
    ov._slider_dragging = True

    ov.hide()

    assert ov._slider_dragging is False


def test_get_slider_numeric_value_returns_none_for_bool_and_non_numeric() -> None:
    """_get_slider_numeric_value should reject non-numeric slider values."""
    live_variable_registry.register("slider.bool", getter=lambda: True)
    live_variable_registry.register("slider.text", getter=lambda: "abc")

    ov = make_overlay()

    bool_var = live_variable_registry.get_variable("slider.bool")
    assert bool_var is not None
    ov._slider_var = bool_var
    assert ov._get_slider_numeric_value() is None

    text_var = live_variable_registry.get_variable("slider.text")
    assert text_var is not None
    ov._slider_var = text_var
    assert ov._get_slider_numeric_value() is None


def test_slider_hit_test() -> None:
    """Slider row hit testing uses geometry captured during draw."""
    store = {"v": 20.0}
    live_variable_registry.register(
        "slider.hit",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )
    var = live_variable_registry.get_variable("slider.hit")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (120, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    assert ov._hit_test_slider(10, 72) is True
    assert ov._hit_test_slider(10, 20) is False


def test_slider_label_uses_live_variable_formatter() -> None:
    """Slider label uses the same formatter path as command output."""
    store = {"v": 135.0}
    live_variable_registry.register(
        "fmt.slider",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        formatter=lambda v: f"{float(v):.1f} deg",
        value_range=(0.0, 360.0),
    )
    var = live_variable_registry.get_variable("fmt.slider")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (140, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    first_text_call = ov.canvas.draw_text.call_args_list[0]
    assert first_text_call.args[2] == "fmt.slider = 135.0 deg"


def test_slider_label_uses_default_float_format_when_no_formatter() -> None:
    """Slider label falls back to compact float formatting by default."""
    store = {"v": 0.85}
    live_variable_registry.register(
        "fmt.slider.default",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 360.0),
    )
    var = live_variable_registry.get_variable("fmt.slider.default")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (140, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    first_text_call = ov.canvas.draw_text.call_args_list[0]
    assert first_text_call.args[2] == "fmt.slider.default = 0.85"


def test_slider_label_uses_display_decimals_when_configured() -> None:
    """display_decimals overrides default float precision for slider labels."""
    store = {"v": 0.8567}
    live_variable_registry.register(
        "fmt.slider.decimals",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        display_decimals=3,
        value_range=(0.0, 1.0),
    )
    var = live_variable_registry.get_variable("fmt.slider.decimals")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (140, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    first_text_call = ov.canvas.draw_text.call_args_list[0]
    assert first_text_call.args[2] == "fmt.slider.decimals = 0.857"


def test_draw_slider_reads_live_value_once_per_frame() -> None:
    """Slider draw should call the getter once and reuse the value."""
    store = {"v": 0.5}
    calls = {"count": 0}

    def getter() -> float:
        calls["count"] += 1
        return store["v"]

    live_variable_registry.register(
        "slider.once",
        getter=getter,
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 1.0),
    )
    var = live_variable_registry.get_variable("slider.once")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (140, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    assert calls["count"] == 1


def test_draw_slider_track_is_left_aligned_and_width_capped() -> None:
    """Track should stay left-aligned while respecting the width cap."""
    store = {"v": 0.5}
    live_variable_registry.register(
        "slider.widthcap",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 1.0),
    )
    var = live_variable_registry.get_variable("slider.widthcap")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 2000
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (100, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    assert ov._slider_track_rect is not None
    track_x, _track_y, track_w, _track_h = ov._slider_track_rect
    assert track_x == 8
    assert track_w == 520


def test_draw_slider_track_is_rendered_on_row_below_label() -> None:
    """Track geometry should be on the row beneath the label text row."""
    store = {"v": 135.0}
    live_variable_registry.register(
        "sun.azimuth",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        display_decimals=1,
        value_range=(0.0, 360.0),
    )
    var = live_variable_registry.get_variable("sun.azimuth")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.side_effect = lambda text: (len(str(text)) * 8, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    assert ov._slider_track_rect is not None
    _track_x, track_y, _track_w, _track_h = ov._slider_track_rect
    assert track_y >= 64


def test_draw_slider_track_geometry_is_stable_when_value_text_width_changes() -> None:
    """Track geometry should not shift as formatted value text length changes."""
    store = {"v": 99.9}
    live_variable_registry.register(
        "slider.stable",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 200.0),
    )
    var = live_variable_registry.get_variable("slider.stable")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.side_effect = lambda text: (len(str(text)) * 8, 16, 16)
    ov._slider_var = var

    ov._draw_slider(row_y=48, tile_h=16)
    first_rect = ov._slider_track_rect

    store["v"] = 100.0  # One more digit in the formatted value.
    ov._draw_slider(row_y=48, tile_h=16)
    second_rect = ov._slider_track_rect

    assert first_rect == second_rect


def test_draw_slider_thumb_is_centered_on_value_ratio() -> None:
    """Thumb should be centered on the same ratio used for fill/value mapping."""
    store = {"v": 50.0}
    live_variable_registry.register(
        "slider.thumb.center",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )
    var = live_variable_registry.get_variable("slider.thumb.center")
    assert var is not None

    ov = make_overlay()
    ov.pixel_width = 640
    ov.canvas = MagicMock()
    ov.canvas.get_text_metrics.return_value = (120, 16, 16)
    ov._slider_var = var
    ov._draw_slider(row_y=48, tile_h=16)

    assert ov._slider_track_rect is not None
    track_x, _track_y, track_w, _track_h = ov._slider_track_rect
    thumb_call = ov.canvas.draw_rect.call_args_list[2]
    thumb_x, _thumb_y, thumb_w, _thumb_h = thumb_call.args[:4]

    thumb_center_x = thumb_x + (thumb_w / 2)
    expected_center_x = track_x + ((track_w - 1) / 2)
    assert thumb_center_x == pytest.approx(expected_center_x, abs=1.0)


def test_slider_apply_from_pixel() -> None:
    """Slider pixel mapping converts x position into ranged variable values."""
    store = {"v": 0.0}
    live_variable_registry.register(
        "slider.apply",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )
    var = live_variable_registry.get_variable("slider.apply")
    assert var is not None

    ov = make_overlay()
    ov._slider_var = var
    ov._slider_track_rect = (100, 0, 200, 8)

    ov._apply_slider_from_pixel(100)
    assert store["v"] == pytest.approx(0.0)

    ov._apply_slider_from_pixel(200)
    assert store["v"] == pytest.approx(50.0)

    ov._apply_slider_from_pixel(300)
    assert store["v"] == pytest.approx(100.0)


def test_slider_mouse_drag_sequence_applies_values() -> None:
    """Mouse down/move/up should drag slider and update values continuously."""
    store = {"v": 0.0}
    live_variable_registry.register(
        "slider.drag",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )
    var = live_variable_registry.get_variable("slider.drag")
    assert var is not None

    ov = make_overlay()
    ov.show()
    ov.pixel_width = 640
    ov._slider_var = var
    ov._slider_track_rect = (100, 46, 200, 8)
    ov._slider_track_y_center = 50
    ov._slider_tile_h = 16

    consumed_down = ov.handle_input(
        input_events.MouseButtonDown(
            position=input_events.Point(120, 50), button=input_events.MouseButton.LEFT
        )
    )
    assert consumed_down is True
    assert ov._slider_dragging is True
    assert store["v"] == pytest.approx(10.0)

    consumed_move = ov.handle_input(
        input_events.MouseMotion(
            position=input_events.Point(250, 50), motion=input_events.Point(0, 0)
        )
    )
    assert consumed_move is True
    assert store["v"] == pytest.approx(75.0)

    consumed_up = ov.handle_input(
        input_events.MouseButtonUp(
            position=input_events.Point(250, 50), button=input_events.MouseButton.LEFT
        )
    )
    assert consumed_up is True
    assert ov._slider_dragging is False


def test_natural_syntax_set() -> None:
    """Typing 'varname = value' sets the variable."""
    store = {"v": 0}
    live_variable_registry.register(
        "nat.set",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", v),
    )

    ov = make_overlay()

    # With spaces around =
    ov.input_buffer = "nat.set = 7"
    ov._execute_command()
    assert store["v"] == 7
    assert "nat.set = 7" in ov.history[-1]

    # Without spaces
    ov.input_buffer = "nat.set=99"
    ov._execute_command()
    assert store["v"] == 99


def test_natural_syntax_prefix_single_match_activates_slider() -> None:
    """A single prefix-pattern match should activate slider for ranged vars."""
    store = {"v": 42.0}
    live_variable_registry.register(
        "prefix.single",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )

    ov = make_overlay()
    ov.input_buffer = "prefix.*"
    ov._execute_command()

    assert ov._slider_var is not None
    assert ov._slider_var.name == "prefix.single"


def test_draw_invalidates_when_slider_value_changes_externally() -> None:
    """draw() should invalidate when active slider value changes outside drag."""
    store = {"v": 10.0}
    live_variable_registry.register(
        "slider.external.change",
        getter=lambda: store["v"],
        setter=lambda v: store.__setitem__("v", float(v)),
        value_range=(0.0, 100.0),
    )
    var = live_variable_registry.get_variable("slider.external.change")
    assert var is not None

    ov = make_overlay()
    ov.show()
    ov._slider_var = var
    ov._slider_dragging = False
    ov._slider_last_value = 10.0
    ov._filter_mode = False
    ov._last_blink_time = time.perf_counter()
    invalidate_mock = MagicMock()
    ov.invalidate = invalidate_mock

    # Simulate an external update to the controlled variable.
    store["v"] = 20.0
    with patch.object(TextOverlay, "draw", autospec=True, return_value=None):
        ov.draw()

    assert ov._slider_last_value == 20.0
    invalidate_mock.assert_called()


def test_natural_syntax_set_missing_value() -> None:
    """'varname =' without a value shows usage."""
    live_variable_registry.register("nat.empty", getter=lambda: 0)

    ov = make_overlay()
    ov.input_buffer = "nat.empty ="
    ov._execute_command()
    assert "Usage:" in ov.history[-1]


def test_handle_input_typing_and_submit(monkeypatch) -> None:
    ov = make_overlay()
    ov.show()
    exec_mock = MagicMock()
    ov._execute_command = exec_mock

    for ch in "set":
        e = input_events.TextInput(ch)
        ov.handle_input(e)

    assert ov.input_buffer == "set"

    e = input_events.TextInput("`")
    ov.handle_input(e)
    assert ov.input_buffer == "set"  # backtick ignored

    e = input_events.KeyDown(sym=input_events.KeySym.BACKSPACE)
    ov.handle_input(e)
    assert ov.input_buffer == "se"

    e = input_events.KeyDown(sym=input_events.KeySym.RETURN)
    ov.handle_input(e)
    assert exec_mock.called
    assert ov.input_buffer == ""


@pytest.mark.parametrize(
    "key_sym", [input_events.KeySym.ESCAPE, input_events.KeySym.GRAVE]
)
def test_handle_input_escape_and_backquote_hide_console(
    key_sym: input_events.KeySym,
) -> None:
    ov = make_overlay()
    ov.show()

    assert ov.is_active

    e = input_events.KeyDown(sym=key_sym)
    ov.handle_input(e)

    assert not ov.is_active


def test_draw_content_uses_tile_height() -> None:
    ov = make_overlay()
    ov.canvas = MagicMock()
    ov.tile_dimensions = (8, 16)
    ov.height = 3
    ov.pixel_width = 10
    ov.pixel_height = 52  # 3 * 16 + 4 descender padding
    ov.history.clear()
    ov.history.append("line")
    ov._cursor_visible = False
    ov.canvas.get_effective_line_height.return_value = 12
    ov.canvas.get_text_metrics.return_value = (4, 0, 12)
    ov.canvas.get_font_metrics.return_value = (8, 4)

    ov.draw_content()

    calls = ov.canvas.draw_text.call_args_list
    assert calls[0].args[1] == 16  # history line y position
    assert calls[1].args[1] == 32  # prompt y position


def test_tab_completion_shows_candidates_and_common_prefix() -> None:
    """First Tab completes to common prefix and lists candidates in history."""
    live_variable_registry.register("tab.alpha", getter=lambda: 0)
    live_variable_registry.register("tab.beta", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get tab."

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))

    # Common prefix of "tab.alpha" and "tab.beta" is "tab.".
    # Input should stay at common prefix (no further common chars).
    assert ov.input_buffer == "get tab."

    # Both candidates should be listed in history.
    assert any("tab.alpha" in line for line in ov.history)
    assert any("tab.beta" in line for line in ov.history)


def test_tab_completion_unique_match() -> None:
    """Single candidate completes immediately without listing."""
    live_variable_registry.register("uniq.only", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    prev_history_len = len(ov.history)
    ov.input_buffer = "get uniq."

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer == "get uniq.only"
    # No candidates should be printed for a unique match.
    assert len(ov.history) == prev_history_len


def test_tab_completion_cycles_after_listing() -> None:
    """Subsequent Tabs cycle through candidates after the initial listing."""
    live_variable_registry.register("cyc.a", getter=lambda: 0)
    live_variable_registry.register("cyc.b", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get cyc."

    # First Tab: show candidates, complete to common prefix.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer == "get cyc."

    # Second Tab: cycle to first candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    first = ov.input_buffer
    assert first in {"get cyc.a", "get cyc.b"}

    # Third Tab: cycle to second candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    assert ov.input_buffer != first


def test_tab_completion_reverse_with_shift_tab() -> None:
    """Shift+Tab cycles candidates in reverse order."""
    live_variable_registry.register("stab.a", getter=lambda: 0)
    live_variable_registry.register("stab.b", getter=lambda: 0)
    live_variable_registry.register("stab.c", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "get stab."

    # First Tab: show candidates + common prefix.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))

    # Forward to first candidate.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    first = ov.input_buffer

    # Forward to second.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.TAB))
    second = ov.input_buffer
    assert second != first

    # Shift+Tab back to first.
    ov.handle_input(
        input_events.KeyDown(
            sym=input_events.KeySym.TAB, mod=input_events.Modifier.SHIFT
        )
    )
    assert ov.input_buffer == first


def test_history_navigation_up_down() -> None:
    ov = make_overlay()
    ov.input_buffer = "cmd1"
    ov._execute_command()
    ov.input_buffer = "cmd2"
    ov._execute_command()
    ov.show()
    ov.input_buffer = ""

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov.input_buffer == "cmd1"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov.input_buffer == "cmd2"
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov.input_buffer == ""


def test_console_consumes_all_keydown_events() -> None:
    """Console should consume all KeyDown events to prevent mode interference.

    This tests the fix for the bug where pressing I while console is open
    would open the inventory menu instead of just typing in the console.
    """
    ov = make_overlay()
    ov.show()

    # KeyDown for 'I' should be consumed (returns True) even though
    # the console doesn't explicitly handle this key
    key_i = input_events.KeySym(ord("i"))
    event = input_events.KeyDown(sym=key_i)
    result = ov.handle_input(event)

    assert result is True  # Event was consumed, won't leak to mode

    # Same for other arbitrary keys like 'T' (targeting mode trigger)
    key_t = input_events.KeySym(ord("t"))
    event = input_events.KeyDown(sym=key_t)
    result = ov.handle_input(event)

    assert result is True


def test_scrollback_shift_up_down() -> None:
    """Shift+Up/Down scrolls through command history."""
    ov = make_overlay()
    ov.show()
    shift = input_events.Modifier.SHIFT

    # Fill history with some lines.
    for i in range(20):
        ov.history.append(f"line {i}")

    # Shift+Up increases scroll offset.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP, mod=shift))
    assert ov._scroll_offset == 3

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP, mod=shift))
    assert ov._scroll_offset == 6

    # Shift+Down decreases scroll offset.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN, mod=shift))
    assert ov._scroll_offset == 3

    # Can't go below zero.
    for _ in range(5):
        ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN, mod=shift))
    assert ov._scroll_offset == 0


def test_filter_mode_navigation() -> None:
    """Filter mode supports typing, navigation, and selection."""
    live_variable_registry.register("flt.alpha", getter=lambda: 1)
    live_variable_registry.register("flt.beta", getter=lambda: 2)
    live_variable_registry.register("flt.gamma", getter=lambda: 3)

    ov = make_overlay()
    ov.show()

    # Enter filter mode.
    ov._enter_filter_mode()
    assert ov._filter_mode is True
    assert len(ov._filter_results) >= 3  # at least our three vars

    # Type to filter.
    ov.handle_input(input_events.TextInput("f"))
    ov.handle_input(input_events.TextInput("l"))
    ov.handle_input(input_events.TextInput("t"))
    assert all("flt." in v.name for v in ov._filter_results)

    # Navigate with Down.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.DOWN))
    assert ov._filter_selected == 1

    # Navigate with Up.
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.UP))
    assert ov._filter_selected == 0

    # Select with Enter.
    selected_name = ov._filter_results[0].name
    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.RETURN))
    assert ov._filter_mode is False
    assert ov.input_buffer == selected_name


def test_filter_mode_escape_cancels() -> None:
    """Escape in filter mode exits without selecting."""
    live_variable_registry.register("fesc.a", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov.input_buffer = "original"

    ov._enter_filter_mode()
    assert ov._filter_mode is True

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.ESCAPE))
    assert ov._filter_mode is False
    # Input buffer should NOT have been changed (no selection made).
    assert ov.input_buffer == "original"


def test_filter_mode_backspace() -> None:
    """Backspace in filter mode removes characters from the filter text."""
    live_variable_registry.register("fbsp.x", getter=lambda: 0)

    ov = make_overlay()
    ov.show()
    ov._enter_filter_mode("fbsp")
    assert ov._filter_text == "fbsp"

    ov.handle_input(input_events.KeyDown(sym=input_events.KeySym.BACKSPACE))
    assert ov._filter_text == "fbs"


def test_filter_mode_consumes_keydown() -> None:
    """Filter mode consumes all KeyDown events."""
    ov = make_overlay()
    ov.show()
    ov._enter_filter_mode()

    key_i = input_events.KeySym(ord("i"))
    event = input_events.KeyDown(sym=key_i)
    result = ov.handle_input(event)
    assert result is True
