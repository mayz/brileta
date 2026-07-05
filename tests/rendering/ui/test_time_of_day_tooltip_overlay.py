"""Tests for the HUD time-of-day hover tooltip."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from brileta import config
from brileta.game.clock import format_clock_time
from brileta.view.ui.time_of_day_tooltip_overlay import TimeOfDayTooltipOverlay
from brileta.view.views.time_of_day_dial_view import TimeOfDayDialView


def _make_overlay(mouse_x: int, mouse_y: int) -> TimeOfDayTooltipOverlay:
    graphics = SimpleNamespace(
        tile_dimensions=(10, 10),
        console_width_tiles=80,
        console_height_tiles=50,
        console_to_screen_coords=lambda x, y: (x * 10, y * 10),
    )
    dial_view = SimpleNamespace(
        visible=True,
        x=76,
        y=3,
        width=4,
        height=2,
    )
    controller = SimpleNamespace(
        graphics=graphics,
        use_24_hour_clock=False,
        gw=SimpleNamespace(clock=SimpleNamespace(time_of_day=0.5)),
        frame_manager=SimpleNamespace(
            cursor_manager=SimpleNamespace(
                mouse_pixel_x=mouse_x,
                mouse_pixel_y=mouse_y,
            )
        ),
    )
    return TimeOfDayTooltipOverlay(
        cast(Any, controller),
        cast(TimeOfDayDialView, dial_view),
    )


def test_time_of_day_tooltip_hides_when_not_hovering_dial() -> None:
    overlay = _make_overlay(mouse_x=200, mouse_y=200)

    overlay._calculate_dimensions()

    assert overlay.pixel_width == 0
    assert overlay.pixel_height == 0


def test_time_of_day_tooltip_shows_formatted_time_when_hovering_dial() -> None:
    overlay = _make_overlay(mouse_x=780, mouse_y=40)

    overlay._calculate_dimensions()

    assert overlay._FONT_SIZE_PX == config.PLAYER_STATUS_FONT_SIZE
    # Matches the 12-hour clock string for noon (0.5).
    assert overlay._label_text == format_clock_time(0.5, use_24_hour=False)
    assert overlay.pixel_width > 0
    assert overlay.pixel_height > 0
    assert overlay.y_tiles >= 5.0


def test_time_of_day_tooltip_honors_24_hour_preference() -> None:
    overlay = _make_overlay(mouse_x=780, mouse_y=40)
    overlay.controller.use_24_hour_clock = True

    overlay._calculate_dimensions()

    assert overlay._label_text == "12:00"
