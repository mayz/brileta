from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from brileta.controller import Controller
from brileta.util.coordinates import PixelCoord
from brileta.view.render.canvas import Canvas
from brileta.view.ui.dev_console_overlay import DevConsoleOverlay
from brileta.view.ui.overlays import OverlaySystem, TextOverlay


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
        _ = text
        _ = font_size
        return (0, 0, 0)

    def wrap_text(
        self, text: str, max_width: int, font_size: int | None = None
    ) -> list[str]:
        _ = max_width
        _ = font_size
        return [text]

    def _update_scaling_internal(self, tile_height: int) -> None:
        _ = tile_height

    def get_effective_line_height(self) -> int:
        return 0

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
        _ = pixel_x
        _ = pixel_y
        _ = text
        _ = color
        _ = font_size

    def _render_rect_op(
        self,
        pixel_x: PixelCoord,
        pixel_y: PixelCoord,
        width: PixelCoord,
        height: PixelCoord,
        color: Any,
        fill: bool,
    ) -> None:
        _ = pixel_x
        _ = pixel_y
        _ = width
        _ = height
        _ = color
        _ = fill

    def _render_frame_op(
        self,
        tile_x: int,
        tile_y: int,
        width: int,
        height: int,
        fg: Any,
        bg: Any,
    ) -> None:
        _ = tile_x
        _ = tile_y
        _ = width
        _ = height
        _ = fg
        _ = bg

    def _create_artifact_from_rendered_content(self) -> object:
        return self._artifact


class DummyController:
    def __init__(self) -> None:
        self.graphics = SimpleNamespace(
            tile_dimensions=(8, 16),
            console_width_tiles=80,
            console_height_tiles=50,
            release_texture=lambda tex: None,
        )


class CountingOverlay(TextOverlay):
    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.draw_calls = 0

    def _get_backend(self) -> Canvas:
        return FakeCanvas()

    def _calculate_dimensions(self) -> None:
        tile_w, tile_h = self.tile_dimensions
        self.x_tiles = 0
        self.y_tiles = 0
        self.width = 1
        self.height = 1
        self.pixel_width = tile_w
        self.pixel_height = tile_h

    def draw_content(self) -> None:
        self.draw_calls += 1

    def handle_input(self, event: Any) -> bool:
        _ = event
        return False


class ConsumingOverlay(CountingOverlay):
    def handle_input(self, event: Any) -> bool:
        _ = event
        return True


def test_text_overlay_caches_until_invalidated() -> None:
    overlay = CountingOverlay(cast("Controller", DummyController()))
    overlay.show()

    overlay.draw()
    overlay.draw()

    assert overlay.draw_calls == 1

    overlay.invalidate()
    overlay.draw()

    assert overlay.draw_calls == 2


def test_text_overlay_always_dirty() -> None:
    overlay = CountingOverlay(cast("Controller", DummyController()))
    overlay.always_dirty = True
    overlay.show()

    overlay.draw()
    overlay.draw()

    assert overlay.draw_calls == 2


def test_overlay_system_invalidates_on_consumed_input() -> None:
    controller = cast("Controller", DummyController())
    overlay = ConsumingOverlay(controller)
    overlay.show()
    overlay.draw()
    assert overlay._dirty is False

    system = OverlaySystem(controller)
    system.show_overlay(overlay)
    system.handle_input(MagicMock())

    assert overlay._dirty is True


def test_overlay_system_invalidate_all() -> None:
    controller = cast("Controller", DummyController())
    overlay = CountingOverlay(controller)
    overlay.show()
    overlay.draw()
    assert overlay._dirty is False

    system = OverlaySystem(controller)
    system.show_overlay(overlay)
    system.invalidate_all()

    assert overlay._dirty is True


def test_dev_console_blink_invalidates(monkeypatch: Any) -> None:
    class CountingConsole(DevConsoleOverlay):
        def __init__(self, controller: Controller) -> None:
            super().__init__(controller)
            self.draw_calls = 0

        def _get_backend(self) -> Canvas:
            return FakeCanvas()

        def draw_content(self) -> None:
            self.draw_calls += 1

    times = iter([0.0, 0.0, 0.6])
    monkeypatch.setattr("time.perf_counter", lambda: next(times))

    overlay = CountingConsole(cast("Controller", DummyController()))
    overlay.show()

    overlay.draw()
    overlay.draw()

    assert overlay.draw_calls == 1

    overlay.draw()

    assert overlay.draw_calls == 2
