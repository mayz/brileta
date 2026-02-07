import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

from typing import cast
from unittest.mock import MagicMock

from catley import colors
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.types import InterpolationAlpha
from catley.util.message_log import MessageLog
from catley.view.render.graphics import GraphicsContext
from catley.view.views.message_log_view import MessageLogView


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

    def read_pixels(self, format: str = "RGBA"):  # noqa: A002
        import numpy as np

        return np.ones((10, 10, 4), dtype=np.uint8)

    def upload_texture(self, pixels) -> DummyTexture:
        return DummyTexture()


def test_message_log_view_ttf_rendering_visible() -> None:
    renderer = DummyRenderer()

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock(spec=GraphicsContext)
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    view = MessageLogView(log, graphics=renderer_stub)
    view.tile_dimensions = (8, 16)
    view.set_bounds(0, 0, 20, 5)

    view.draw(renderer_stub, InterpolationAlpha(0.0))
    renderer.clear()
    view.present(renderer_stub, InterpolationAlpha(0.0))

    pixels = renderer.read_pixels(format="RGBA")

    assert pixels.max() > 0, "Expected rendered texture to have non-black pixels"


def test_message_log_view_font_stays_fixed_on_resize() -> None:
    """Font size stays fixed when MessageLogView uses explicit font_size."""
    renderer = DummyRenderer()

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock(spec=GraphicsContext)
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    view = MessageLogView(log, graphics=renderer_stub)
    view.tile_dimensions = (8, 16)
    view.set_bounds(0, 0, 20, 5)

    view.draw(renderer_stub, InterpolationAlpha(0.0))
    assert view.canvas is not None
    tb = cast(PillowImageCanvas, view.canvas)
    ascent, descent = tb.get_font_metrics()
    initial_line_height = ascent + descent

    renderer_stub.tile_dimensions = (16, 32)
    view.tile_dimensions = (16, 32)
    view.set_bounds(0, 0, 20, 5)
    view.draw(renderer_stub, InterpolationAlpha(0.0))

    assert view.canvas is not None
    tb = cast(PillowImageCanvas, view.canvas)
    ascent, descent = tb.get_font_metrics()
    # With explicit font_size, font should stay the same regardless of tile size
    assert (ascent + descent) == initial_line_height
