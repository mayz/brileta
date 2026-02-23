from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from brileta.controller import Controller
from brileta.util.coordinates import PixelCoord
from brileta.view.render.canvas import Canvas, DrawOperation
from brileta.view.ui.zoom_indicator_overlay import ZoomIndicatorOverlay


class FakeCanvas(Canvas):
    def __init__(self) -> None:
        super().__init__(transparent=True)
        self._artifact: object = object()

    @property
    def artifact_type(self) -> str:
        return "fake"

    def create_texture(self, renderer: Any, artifact: Any) -> object:
        _ = renderer
        _ = artifact
        return object()

    def get_text_metrics(
        self, text: str, font_size: int | None = None
    ) -> tuple[int, int, int]:
        _ = font_size
        # Fixed metrics so positioning math is deterministic.
        return (8 * len(text), 12, 12)

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        _ = max_width
        _ = font_size
        return [text]

    def _update_scaling_internal(self, tile_height: int) -> None:
        _ = tile_height

    def get_effective_line_height(self) -> int:
        return 12

    def _prepare_for_rendering(self) -> bool:
        return True

    def _render_text_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        text: str,
        color: Any,
        font_size: int | None = None,
    ) -> None:
        _ = pixel_x, pixel_y, text, color, font_size

    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: Any,
        fill: bool,
    ) -> None:
        _ = pixel_x, pixel_y, width, height, color, fill

    def _render_frame_op(
        self,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        fg: Any,
        bg: Any,
    ) -> None:
        _ = tile_x, tile_y, width, height, fg, bg

    def _create_artifact_from_rendered_content(self) -> object:
        return self._artifact


class DummyController:
    def __init__(self) -> None:
        self.graphics = SimpleNamespace(
            tile_dimensions=(10, 20),
            draw_texture_alpha=MagicMock(),
            console_to_screen_coords=lambda x, y: (x * 10.0, y * 20.0),
        )
        self.clock = SimpleNamespace(last_time=0.0)
        self.frame_manager = None


class DummyWorldView:
    def __init__(self) -> None:
        self.x = 2
        self.y = 3
        self.width = 20
        self.height = 12
        self.visible = True
        self.viewport_zoom = 1.0
        self.is_default_viewport_zoom = True


def _make_overlay() -> tuple[ZoomIndicatorOverlay, DummyController, DummyWorldView]:
    controller = DummyController()
    world_view = DummyWorldView()
    with patch.object(ZoomIndicatorOverlay, "_get_backend", return_value=FakeCanvas()):
        overlay = ZoomIndicatorOverlay(
            cast("Controller", controller),
            cast(Any, world_view),
        )
    overlay.show()
    return overlay, controller, world_view


def test_zoom_indicator_hidden_at_default_without_recent_change() -> None:
    overlay, controller, _world_view = _make_overlay()

    overlay.draw()
    overlay.present()

    assert overlay.pixel_width == 0
    controller.graphics.draw_texture_alpha.assert_not_called()


def test_zoom_indicator_wakes_then_persists_when_non_default() -> None:
    overlay, controller, world_view = _make_overlay()
    world_view.viewport_zoom = 2.0
    world_view.is_default_viewport_zoom = False
    controller.clock.last_time = 10.0

    overlay.notify_zoom_changed()
    overlay.draw()
    overlay.present()

    first_call = controller.graphics.draw_texture_alpha.call_args
    assert first_call is not None
    assert first_call.args[1] >= 0  # pixel x
    assert first_call.args[2] >= 0  # pixel y
    assert float(first_call.args[3]) == pytest.approx(overlay._WAKE_ALPHA)
    assert overlay._reset_hint_text == "Press 0 to reset"

    controller.clock.last_time = 12.0
    controller.graphics.draw_texture_alpha.reset_mock()
    overlay.invalidate()
    overlay.draw()
    overlay.present()
    second_call = controller.graphics.draw_texture_alpha.call_args
    assert second_call is not None
    assert float(second_call.args[3]) == pytest.approx(overlay._PERSISTENT_ALPHA)
    assert overlay._reset_hint_text == ""


def test_zoom_indicator_default_zoom_fades_out_after_pulse() -> None:
    overlay, controller, world_view = _make_overlay()
    world_view.viewport_zoom = 1.0
    world_view.is_default_viewport_zoom = True
    controller.clock.last_time = 5.0

    overlay.notify_zoom_changed()
    overlay.draw()
    overlay.present()
    assert controller.graphics.draw_texture_alpha.called

    controller.clock.last_time = 7.0
    controller.graphics.draw_texture_alpha.reset_mock()
    overlay.present()
    controller.graphics.draw_texture_alpha.assert_not_called()


def test_zoom_indicator_avoids_player_status_view_overlap() -> None:
    overlay, controller, world_view = _make_overlay()
    world_view.viewport_zoom = 2.0
    world_view.is_default_viewport_zoom = False
    controller.clock.last_time = 1.0
    # Occupy the top-right of the world viewport where the zoom chip prefers to sit.
    controller.frame_manager = SimpleNamespace(
        player_status_view=SimpleNamespace(x=14, y=3, width=8, height=4, visible=True),
        equipment_view=SimpleNamespace(x=0, y=0, width=0, height=0, visible=False),
    )

    overlay.notify_zoom_changed()
    controller.clock.last_time = 3.0
    overlay.draw()

    # Falls back to top-left inside the world viewport.
    assert overlay._present_pixel_x == 28
    assert overlay._present_pixel_y == 68


def test_zoom_indicator_uses_text_bbox_offsets_to_avoid_clipping() -> None:
    class OffsetCanvas(FakeCanvas):
        def get_text_bbox(
            self, text: str, font_size: int | None = None
        ) -> tuple[int, int, int, int]:
            _ = text, font_size
            # Simulate a font whose glyph pixels start 3px below the anchor.
            return (0, 3, 80, 12)

    controller = DummyController()
    world_view = DummyWorldView()
    world_view.viewport_zoom = 0.5
    world_view.is_default_viewport_zoom = False
    controller.clock.last_time = 1.0

    with patch.object(
        ZoomIndicatorOverlay, "_get_backend", return_value=OffsetCanvas()
    ):
        overlay = ZoomIndicatorOverlay(
            cast("Controller", controller), cast(Any, world_view)
        )
    overlay.show()
    overlay.notify_zoom_changed()
    overlay.draw()

    text_ops = [op for op in overlay.canvas._frame_ops if op[0] is DrawOperation.TEXT]
    label_text_ops = [op for op in text_ops if op[3] == overlay._label_text]
    assert len(label_text_ops) >= 2
    # Main text op (second draw_text) should be shifted upward by the bbox y offset.
    main_text_op = label_text_ops[-1]
    assert main_text_op[2] == 1


def test_zoom_indicator_reset_hint_is_transient_during_wake_phase_only() -> None:
    overlay, controller, world_view = _make_overlay()
    world_view.viewport_zoom = 1.5
    world_view.is_default_viewport_zoom = False
    controller.clock.last_time = 4.0

    overlay.notify_zoom_changed()
    overlay.draw()
    wake_height = overlay.pixel_height
    wake_text_ops = [
        op for op in overlay.canvas._frame_ops if op[0] is DrawOperation.TEXT
    ]
    assert any(op[3] == "Press 0 to reset" for op in wake_text_ops)

    controller.clock.last_time = 6.0
    overlay.invalidate()
    overlay.draw()
    persistent_height = overlay.pixel_height
    persistent_text_ops = [
        op for op in overlay.canvas._frame_ops if op[0] is DrawOperation.TEXT
    ]
    assert not any(op[3] == "Press 0 to reset" for op in persistent_text_ops)
    assert persistent_height < wake_height
