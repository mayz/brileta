"""Tests for GLFW window resize constraints and aspect clamping."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brileta import config
from brileta.backends.glfw.app import GlfwApp

glfw = pytest.importorskip("glfw")


def test_clamp_resize_to_aspect_range_for_tall_window(monkeypatch) -> None:
    """Tall portrait requests should be widened to the configured min aspect."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)

    assert GlfwApp._clamp_resize_to_aspect_range(300, 800) == (960, 800)


def test_clamp_resize_to_aspect_range_for_ultrawide_window(monkeypatch) -> None:
    """Overly wide requests should be raised to the configured max aspect."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)

    assert GlfwApp._clamp_resize_to_aspect_range(2200, 600) == (2200, 917)


def test_load_native_tile_size_from_tileset(monkeypatch) -> None:
    """Native tile dimensions should come from atlas dimensions/rows/cols."""
    mock_tileset = Mock()
    mock_tileset.size = (80, 160)
    mock_tileset_cm = Mock()
    mock_tileset_cm.__enter__ = Mock(return_value=mock_tileset)
    mock_tileset_cm.__exit__ = Mock(return_value=False)

    monkeypatch.setattr(config, "TILESET_COLUMNS", 4)
    monkeypatch.setattr(config, "TILESET_ROWS", 8)
    monkeypatch.setattr(
        "brileta.backends.glfw.app.PILImage.open",
        lambda _path: mock_tileset_cm,
    )

    app = GlfwApp.__new__(GlfwApp)
    assert app._load_native_tile_size() == (20, 20)


def test_on_resize_clamps_before_layout_update(monkeypatch) -> None:
    """Out-of-range resize should be corrected, then relayout should run once."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)
    set_window_size = Mock()
    monkeypatch.setattr(glfw, "set_window_size", set_window_size)
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (960, 800))

    app = GlfwApp.__new__(GlfwApp)
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._applying_resize_constraints = False

    fake_window = object()
    GlfwApp._on_resize(app, fake_window, 300, 800)

    set_window_size.assert_called_once_with(fake_window, 960, 800)
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()


def test_on_resize_updates_layout_when_shape_is_valid(monkeypatch) -> None:
    """In-range resize should proceed through graphics and frame-manager updates."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)
    set_window_size = Mock()
    monkeypatch.setattr(glfw, "set_window_size", set_window_size)
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (1200, 800))

    app = GlfwApp.__new__(GlfwApp)
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._applying_resize_constraints = False

    GlfwApp._on_resize(app, object(), 1200, 800)

    set_window_size.assert_not_called()
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()


def test_on_resize_ignores_minimized_dimensions(monkeypatch) -> None:
    """Minimized (zero-sized) resize events should not force clamp/resync."""
    set_window_size = Mock()
    monkeypatch.setattr(glfw, "set_window_size", set_window_size)

    app = GlfwApp.__new__(GlfwApp)
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._applying_resize_constraints = False

    GlfwApp._on_resize(app, object(), 0, 0)

    set_window_size.assert_not_called()
    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()


def test_on_resize_defers_when_dpi_is_transient(monkeypatch) -> None:
    """Resize callback should defer relayout during transient 1x fullscreen hops."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)
    set_window_size = Mock()
    monkeypatch.setattr(glfw, "set_window_size", set_window_size)
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (1920, 1200))

    app = GlfwApp.__new__(GlfwApp)
    app.graphics = Mock()
    app.graphics.locked_content_scale = 2
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._applying_resize_constraints = False
    app._pending_out_of_band_size = None
    app._pending_out_of_band_frames = 0

    GlfwApp._on_resize(app, object(), 1920, 1200)

    set_window_size.assert_not_called()
    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()


def test_sync_dimensions_outside_resize_callback_updates_layout(monkeypatch) -> None:
    """Out-of-band size changes should trigger graphics/layout resync."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._last_window_size = (800, 600)
    app._last_framebuffer_size = (1600, 1200)

    monkeypatch.setattr(glfw, "get_window_size", lambda _window: (1200, 700))
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (2400, 1400))
    app._pending_out_of_band_size = None
    app._pending_out_of_band_frames = 0

    GlfwApp._sync_dimensions_outside_resize_callback(app)
    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()

    GlfwApp._sync_dimensions_outside_resize_callback(app)

    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()


def test_sync_dimensions_outside_resize_callback_skips_minimized(monkeypatch) -> None:
    """Out-of-band minimized events should not force graphics recreation."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._last_window_size = (800, 600)
    app._last_framebuffer_size = (1600, 1200)

    monkeypatch.setattr(glfw, "get_window_size", lambda _window: (800, 600))
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (0, 0))
    app._pending_out_of_band_size = None
    app._pending_out_of_band_frames = 0

    GlfwApp._sync_dimensions_outside_resize_callback(app)
    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()

    GlfwApp._sync_dimensions_outside_resize_callback(app)

    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()


def test_sync_dimensions_outside_resize_callback_defers_scale_mismatch(
    monkeypatch,
) -> None:
    """Transient DPI mismatch should defer relayout until stable or timeout."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app.graphics = Mock()
    app.graphics.locked_content_scale = 2
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._last_window_size = (1512, 916)
    app._last_framebuffer_size = (3024, 1832)
    app._pending_out_of_band_size = None
    app._pending_out_of_band_frames = 0

    monkeypatch.setattr(GlfwApp, "_OUT_OF_BAND_SCALE_MISMATCH_MAX_DEFER_FRAMES", 3)
    monkeypatch.setattr(glfw, "get_window_size", lambda _window: (1920, 1200))
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (1920, 1200))

    GlfwApp._sync_dimensions_outside_resize_callback(app)
    GlfwApp._sync_dimensions_outside_resize_callback(app)
    app.graphics.update_dimensions.assert_not_called()
    frame_manager.on_window_resized.assert_not_called()

    # Third observation reaches defer timeout and applies the best-known size.
    GlfwApp._sync_dimensions_outside_resize_callback(app)
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()


def test_on_resize_records_framebuffer_size_to_avoid_double_sync(monkeypatch) -> None:
    """Successful resize callback should not force an immediate out-of-band rerun."""
    monkeypatch.setattr(config, "WINDOW_MIN_ASPECT_RATIO", 1.2)
    monkeypatch.setattr(config, "WINDOW_MAX_ASPECT_RATIO", 2.4)
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (1200, 800))

    app = GlfwApp.__new__(GlfwApp)
    app.graphics = Mock()
    app.graphics.locked_content_scale = 1
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._applying_resize_constraints = False
    app._pending_out_of_band_size = None
    app._pending_out_of_band_frames = 0

    GlfwApp._on_resize(app, object(), 1200, 800)

    assert app._last_framebuffer_size == (1200, 800)
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()


def test_toggle_fullscreen_enters_monitor_mode_when_windowed(monkeypatch) -> None:
    """Windowed mode should toggle into monitor fullscreen."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app._enter_monitor_fullscreen = Mock()
    app._exit_monitor_fullscreen = Mock()
    monkeypatch.setattr(glfw, "get_window_monitor", lambda _window: None)

    GlfwApp.toggle_fullscreen(app)

    app._enter_monitor_fullscreen.assert_called_once()
    app._exit_monitor_fullscreen.assert_not_called()


def test_toggle_fullscreen_exits_monitor_mode_when_fullscreen(monkeypatch) -> None:
    """Monitor fullscreen should toggle back to windowed mode."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app._enter_monitor_fullscreen = Mock()
    app._exit_monitor_fullscreen = Mock()
    monkeypatch.setattr(glfw, "get_window_monitor", lambda _window: object())

    GlfwApp.toggle_fullscreen(app)

    app._exit_monitor_fullscreen.assert_called_once()
    app._enter_monitor_fullscreen.assert_not_called()


def test_pick_fullscreen_video_mode_prefers_closest_resolution_then_refresh(
    monkeypatch,
) -> None:
    """Mode picker should choose nearest size and highest refresh among ties."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()

    mode_a = SimpleNamespace(
        size=SimpleNamespace(width=3000, height=1800), refresh_rate=60
    )
    mode_b = SimpleNamespace(
        size=SimpleNamespace(width=3000, height=1800), refresh_rate=120
    )
    mode_c = SimpleNamespace(
        size=SimpleNamespace(width=3024, height=1890), refresh_rate=120
    )
    current_mode = SimpleNamespace(
        size=SimpleNamespace(width=3024, height=1832), refresh_rate=60
    )

    monkeypatch.setattr(glfw, "get_primary_monitor", lambda: object())
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (3024, 1832))
    monkeypatch.setattr(glfw, "get_video_mode", lambda _monitor: current_mode)
    monkeypatch.setattr(
        glfw, "get_video_modes", lambda _monitor: [mode_a, mode_b, mode_c]
    )

    picked = GlfwApp._pick_fullscreen_video_mode(app)

    assert picked is mode_b


def test_pick_fullscreen_video_mode_falls_back_to_current_when_modes_missing(
    monkeypatch,
) -> None:
    """If GLFW returns no mode list, picker should keep current video mode."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    current_mode = SimpleNamespace(
        size=SimpleNamespace(width=1920, height=1080), refresh_rate=60
    )

    monkeypatch.setattr(glfw, "get_primary_monitor", lambda: object())
    monkeypatch.setattr(glfw, "get_framebuffer_size", lambda _window: (1920, 1080))
    monkeypatch.setattr(glfw, "get_video_mode", lambda _monitor: current_mode)
    monkeypatch.setattr(glfw, "get_video_modes", lambda _monitor: [])

    picked = GlfwApp._pick_fullscreen_video_mode(app)

    assert picked is current_mode


def test_enter_monitor_fullscreen_uses_single_monitor_lookup(monkeypatch) -> None:
    """Monitor selection should be reused by mode picker during fullscreen entry."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app.windowed_x = 0
    app.windowed_y = 0
    app.windowed_width = 800
    app.windowed_height = 600
    app._arm_forced_dimension_resync = Mock()

    monitor = object()
    mode = SimpleNamespace(
        size=SimpleNamespace(width=1920, height=1080), refresh_rate=120
    )
    monitor_calls = {"count": 0}

    def _get_primary_monitor():
        monitor_calls["count"] += 1
        return monitor

    monkeypatch.setattr(glfw, "get_primary_monitor", _get_primary_monitor)
    monkeypatch.setattr(glfw, "get_window_pos", lambda _window: (11, 22))
    monkeypatch.setattr(glfw, "get_window_size", lambda _window: (900, 700))
    monkeypatch.setattr(
        GlfwApp,
        "_pick_fullscreen_video_mode",
        lambda _self, passed_monitor=None: mode if passed_monitor is monitor else None,
    )
    set_window_monitor = Mock()
    monkeypatch.setattr(glfw, "set_window_monitor", set_window_monitor)

    GlfwApp._enter_monitor_fullscreen(app)

    assert monitor_calls["count"] == 1
    set_window_monitor.assert_called_once_with(
        app.window,
        monitor,
        0,
        0,
        1920,
        1080,
        120,
    )
    app._arm_forced_dimension_resync.assert_called_once_with("enter-monitor-fullscreen")


def test_forced_dimension_resync_counts_down_and_skips_duplicate_signatures(
    monkeypatch,
) -> None:
    """Forced resync should update once per unique size signature while counting down."""
    app = GlfwApp.__new__(GlfwApp)
    app.window = object()
    app.graphics = Mock()
    frame_manager = Mock()
    app.controller = SimpleNamespace(frame_manager=frame_manager)
    app._pending_out_of_band_size = (1, 1, 1, 1)
    app._pending_out_of_band_frames = 9
    app._last_window_size = (0, 0)
    app._last_framebuffer_size = (0, 0)
    app._force_dimension_resync_frames = 3
    app._forced_resync_last_signature = None

    window_size = [1200, 800]
    framebuffer_size = [2400, 1600]
    monkeypatch.setattr(glfw, "get_window_size", lambda _window: tuple(window_size))
    monkeypatch.setattr(
        glfw, "get_framebuffer_size", lambda _window: tuple(framebuffer_size)
    )

    GlfwApp._maybe_force_dimension_resync(app)
    assert app._force_dimension_resync_frames == 2
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()

    # Same signature should not trigger another relayout.
    GlfwApp._maybe_force_dimension_resync(app)
    assert app._force_dimension_resync_frames == 1
    app.graphics.update_dimensions.assert_called_once()
    frame_manager.on_window_resized.assert_called_once()

    window_size[:] = [1280, 800]
    framebuffer_size[:] = [2560, 1600]
    GlfwApp._maybe_force_dimension_resync(app)
    assert app._force_dimension_resync_frames == 0
    assert app._pending_out_of_band_size is None
    assert app._pending_out_of_band_frames == 0
    assert app._last_window_size == (1280, 800)
    assert app._last_framebuffer_size == (2560, 1600)
    assert app.graphics.update_dimensions.call_count == 2
    assert frame_manager.on_window_resized.call_count == 2


def test_arm_forced_dimension_resync_sets_timer_and_clears_signature() -> None:
    """Arming resync should reset dedup signature and extend timer."""
    app = GlfwApp.__new__(GlfwApp)
    app._force_dimension_resync_frames = 7
    app._forced_resync_last_signature = (1, 2, 3, 4)

    GlfwApp._arm_forced_dimension_resync(app, "test-trigger")

    assert (
        app._force_dimension_resync_frames
        == GlfwApp._FULLSCREEN_TRANSITION_RESYNC_FRAMES
    )
    assert app._forced_resync_last_signature is None
