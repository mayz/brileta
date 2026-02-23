"""Tests for responsive FrameManager tile-space layout calculations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from brileta.types import TileDimensions
from brileta.view.frame_manager import FrameManager


class _ViewRecorder:
    """Small view stub that records set_bounds calls."""

    def __init__(self) -> None:
        self.tile_dimensions: TileDimensions = (0, 0)
        self.bounds: tuple[int, int, int, int] = (0, 0, 0, 0)

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.bounds = (x1, y1, x2, y2)


class _EquipmentViewRecorder(_ViewRecorder):
    """View recorder with equipment-specific width calculation."""

    def __init__(self, min_width: int) -> None:
        super().__init__()
        self._min_width = min_width

    def calculate_min_width(self) -> int:
        return self._min_width


class _MiniMapViewRecorder(_ViewRecorder):
    """View recorder with a visible toggle flag."""

    def __init__(self) -> None:
        super().__init__()
        self.visible = True


def _build_layout_stub(
    console_width_tiles: int,
    console_height_tiles: int,
    equipment_min_width: int,
    *,
    tile_dimensions: TileDimensions = (64, 64),
    native_tile_size: TileDimensions | None = None,
) -> SimpleNamespace:
    """Construct a minimal object compatible with FrameManager._layout_views."""
    world_view = _ViewRecorder()
    player_status_view = _ViewRecorder()
    action_panel_view = _ViewRecorder()
    mini_map_view = _MiniMapViewRecorder()
    message_log_view = _ViewRecorder()
    equipment_view = _EquipmentViewRecorder(min_width=equipment_min_width)

    return SimpleNamespace(
        graphics=SimpleNamespace(
            console_width_tiles=console_width_tiles,
            console_height_tiles=console_height_tiles,
            tile_dimensions=tile_dimensions,
            native_tile_size=native_tile_size or tile_dimensions,
        ),
        world_view=world_view,
        player_status_view=player_status_view,
        action_panel_view=action_panel_view,
        mini_map_view=mini_map_view,
        message_log_view=message_log_view,
        equipment_view=equipment_view,
        controller=SimpleNamespace(
            gw=SimpleNamespace(game_map=SimpleNamespace(width=120, height=80))
        ),
        _clamp_tiles=FrameManager._clamp_tiles,
        _compute_ui_scale=FrameManager._compute_ui_scale,
    )


def test_layout_matches_legacy_shape_on_80x50() -> None:
    """80x50 keeps familiar panel sizes while using responsive calculations."""
    fm_stub = _build_layout_stub(80, 50, equipment_min_width=20)

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    assert fm_stub.world_view.bounds == (20, 0, 80, 39)
    assert fm_stub.action_panel_view.bounds == (0, 0, 20, 26)
    assert fm_stub.mini_map_view.bounds == (0, 26, 20, 39)
    assert fm_stub.message_log_view.bounds == (0, 39, 60, 50)
    assert fm_stub.equipment_view.bounds == (60, 39, 80, 50)
    assert fm_stub.player_status_view.bounds == (40, 0, 80, 20)


def test_layout_scales_for_zoomed_small_console() -> None:
    """Small consoles keep world area usable instead of collapsing to a corner."""
    fm_stub = _build_layout_stub(47, 28, equipment_min_width=20)

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    assert fm_stub.world_view.bounds == (10, 0, 47, 23)
    assert fm_stub.action_panel_view.bounds == (0, 0, 10, 16)
    assert fm_stub.mini_map_view.bounds == (0, 16, 10, 23)
    assert fm_stub.message_log_view.bounds == (0, 23, 27, 28)
    assert fm_stub.equipment_view.bounds == (27, 23, 47, 28)
    assert fm_stub.player_status_view.bounds == (23, 0, 47, 20)


def test_layout_falls_back_to_world_when_console_is_tiny() -> None:
    """Extremely small consoles avoid negative or inverted bounds."""
    fm_stub = _build_layout_stub(20, 12, equipment_min_width=20)

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    assert fm_stub.world_view.bounds == (0, 0, 20, 12)
    assert fm_stub.action_panel_view.bounds == (0, 0, 0, 12)
    assert fm_stub.mini_map_view.bounds == (0, 0, 0, 0)
    assert fm_stub.message_log_view.bounds == (0, 12, 8, 12)
    assert fm_stub.equipment_view.bounds == (8, 12, 20, 12)
    assert fm_stub.player_status_view.bounds == (2, 0, 20, 12)


def test_layout_hidden_minimap_gives_sidebar_back_to_action_panel() -> None:
    """When the mini-map is hidden, the action panel should reclaim all space."""
    fm_stub = _build_layout_stub(80, 50, equipment_min_width=20)
    fm_stub.mini_map_view.visible = False

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    assert fm_stub.action_panel_view.bounds == (0, 0, 20, 39)
    assert fm_stub.mini_map_view.bounds == (0, 0, 0, 0)


def test_layout_caps_bottom_bar_height_when_tiles_are_zoomed() -> None:
    """Bottom UI should not balloon in pixels when display tiles are oversized."""
    fm_stub = _build_layout_stub(
        80,
        50,
        equipment_min_width=20,
        tile_dimensions=(64, 64),
        native_tile_size=(20, 20),
    )

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    # Pixel cap shrinks bottom UI from 10 rows to ~3 rows at this zoom level.
    assert fm_stub.world_view.bounds == (20, 0, 80, 46)
    assert fm_stub.action_panel_view.bounds == (0, 0, 20, 33)
    assert fm_stub.mini_map_view.bounds == (0, 33, 20, 46)
    assert fm_stub.message_log_view.bounds == (0, 46, 60, 50)
    assert fm_stub.equipment_view.bounds == (60, 46, 80, 50)


@pytest.mark.parametrize(
    ("layout_scale", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.375),
        (0.6, 0.48),
        (1.0, 1.0),
    ],
)
def test_compute_ui_scale_matches_non_linear_curve(
    layout_scale: float, expected: float
) -> None:
    """UI scale curve should stay stable at representative control points."""
    assert FrameManager._compute_ui_scale(layout_scale) == pytest.approx(expected)


def test_layout_is_not_overly_aggressive_at_intermediate_sizes() -> None:
    """Intermediate consoles should keep HUD compact without over-collapsing."""
    fm_stub = _build_layout_stub(60, 38, equipment_min_width=20)

    FrameManager._layout_views(fm_stub)  # type: ignore[arg-type]

    assert fm_stub.world_view.bounds == (13, 0, 60, 30)
    assert fm_stub.action_panel_view.bounds == (0, 0, 13, 22)
    assert fm_stub.mini_map_view.bounds == (0, 22, 13, 30)
    assert fm_stub.message_log_view.bounds == (0, 30, 40, 38)
    assert fm_stub.equipment_view.bounds == (40, 30, 60, 38)
    assert fm_stub.player_status_view.bounds == (30, 0, 60, 20)
