import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

from typing import Any, cast
from unittest.mock import MagicMock

import tcod.sdl.render
import tcod.sdl.video

from catley import colors
from catley.util.message_log import MessageLog
from catley.view.panels.message_log_panel import MessageLogPanel
from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import PillowTextBackend


class DummyTexture:
    def __init__(self) -> None:
        self.blend_mode = None


class DummyRenderer:
    def clear(self) -> None:
        pass

    def copy(self, *args, **kwargs) -> None:
        pass

    def present(self) -> None:
        pass

    def read_pixels(self, format: str = "RGBA"):
        import numpy as np

        return np.ones((10, 10, 4), dtype=np.uint8)

    def upload_texture(self, pixels) -> DummyTexture:
        return DummyTexture()


def test_message_log_panel_ttf_rendering_visible(monkeypatch: Any) -> None:
    monkeypatch.setattr(tcod.sdl.video, "new_window", lambda *a, **kw: object())
    monkeypatch.setattr(
        tcod.sdl.render, "new_renderer", lambda *a, **kw: DummyRenderer()
    )

    window = tcod.sdl.video.new_window(160, 80)
    renderer = tcod.sdl.render.new_renderer(
        window,
        software=True,  # pyright: ignore[reportCallIssue]
        target_textures=True,  # pyright: ignore[reportCallIssue]
    )

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock(spec=Renderer)
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    panel = MessageLogPanel(log, renderer=renderer_stub)
    panel.tile_dimensions = (8, 16)
    panel.resize(0, 0, 20, 5)

    panel.draw(renderer_stub)
    renderer.clear()
    panel.present(renderer_stub)

    pixels = renderer.read_pixels(format="RGBA")

    assert pixels.max() > 0, "Expected rendered texture to have non-black pixels"


def test_message_log_panel_font_scales_on_resize(monkeypatch: Any) -> None:
    monkeypatch.setattr(tcod.sdl.video, "new_window", lambda *a, **kw: object())
    monkeypatch.setattr(
        tcod.sdl.render, "new_renderer", lambda *a, **kw: DummyRenderer()
    )

    window = tcod.sdl.video.new_window(160, 80)
    renderer = tcod.sdl.render.new_renderer(
        window,
        software=True,  # pyright: ignore[reportCallIssue]
        target_textures=True,  # pyright: ignore[reportCallIssue]
    )

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock(spec=Renderer)
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    panel = MessageLogPanel(log, renderer=renderer_stub)
    panel.tile_dimensions = (8, 16)
    panel.resize(0, 0, 20, 5)

    panel.draw(renderer_stub)
    assert panel.text_backend is not None
    tb = cast(PillowTextBackend, panel.text_backend)
    ascent, descent = tb.get_font_metrics()
    initial_line_height = ascent + descent

    renderer_stub.tile_dimensions = (16, 32)
    panel.tile_dimensions = (16, 32)
    panel.resize(0, 0, 20, 5)
    panel.draw(renderer_stub)

    assert panel.text_backend is not None
    tb = cast(PillowTextBackend, panel.text_backend)
    ascent, descent = tb.get_font_metrics()
    assert (ascent + descent) > initial_line_height
