import os
from unittest.mock import MagicMock

import tcod.sdl.render
import tcod.sdl.video

from catley import colors
from catley.ui.panels.message_log_panel import MessageLogPanel
from catley.util.message_log import MessageLog


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

    panel = MessageLogPanel(
        log,
        x=0,
        y=0,
        width=20,
        height=5,
        tile_dimensions=(8, 16),
    )

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

    panel = MessageLogPanel(
        log,
        x=0,
        y=0,
        width=20,
        height=5,
        tile_dimensions=(8, 16),
    )

    panel.draw(renderer_stub)
    initial_line_height = panel.line_height_px

    renderer_stub.tile_dimensions = (16, 32)
    panel.draw(renderer_stub)

    assert panel.line_height_px > initial_line_height
