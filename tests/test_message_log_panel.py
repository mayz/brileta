import os
from typing import cast
from unittest.mock import MagicMock

import tcod.sdl.render
import tcod.sdl.video

from catley import colors, config
from catley.util.message_log import MessageLog
from catley.view.panels.message_log_panel import MessageLogPanel
from catley.view.text_backend import PillowTextBackend, TextBackend


def test_message_log_panel_ttf_rendering_visible() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_RENDER_DRIVER"] = "software"

    window = tcod.sdl.video.new_window(160, 80)
    renderer = tcod.sdl.render.new_renderer(window, software=True, target_textures=True)

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock()
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    backend = PillowTextBackend(
        config.MESSAGE_LOG_FONT_PATH,
        16,
        renderer,
    )
    panel = MessageLogPanel(
        log,
        text_backend=backend,
    )
    panel.tile_dimensions = (8, 16)
    panel.resize(0, 0, 20, 5)

    panel.draw(renderer_stub)
    renderer.clear()
    panel.present(renderer_stub)

    pixels = renderer.read_pixels(format="RGBA")

    assert pixels.max() > 0, "Expected rendered texture to have non-black pixels"


def test_message_log_panel_font_scales_on_resize() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["SDL_RENDER_DRIVER"] = "software"

    window = tcod.sdl.video.new_window(160, 80)
    renderer = tcod.sdl.render.new_renderer(window, software=True, target_textures=True)

    log = MessageLog()
    log.add_message("Hello", fg=colors.WHITE)

    renderer_stub = MagicMock()
    renderer_stub.sdl_renderer = renderer
    renderer_stub.tile_dimensions = (8, 16)

    backend = PillowTextBackend(
        config.MESSAGE_LOG_FONT_PATH,
        16,
        renderer,
    )
    panel = MessageLogPanel(
        log,
        text_backend=backend,
    )
    panel.tile_dimensions = (8, 16)
    panel.resize(0, 0, 20, 5)

    panel.draw(renderer_stub)
    assert panel.text_backend is not None
    tb = cast(TextBackend, panel.text_backend)
    ascent, descent = tb.get_font_metrics()
    initial_line_height = ascent + descent

    renderer_stub.tile_dimensions = (16, 32)
    panel.draw(renderer_stub)

    assert panel.text_backend is not None
    tb = cast(TextBackend, panel.text_backend)
    ascent, descent = tb.get_font_metrics()
    assert (ascent + descent) > initial_line_height
