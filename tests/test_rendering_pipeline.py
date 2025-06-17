import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

import tcod
from tcod.console import Console

from catley import config


def test_sdl_tile_rendering() -> None:
    """Render a single tile using SDL to ensure pipeline stability."""
    tileset = tcod.tileset.load_tilesheet(
        config.TILESET_PATH,
        columns=config.TILESET_COLUMNS,
        rows=config.TILESET_ROWS,
        charmap=tcod.tileset.CHARMAP_CP437,
    )

    console = Console(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, order="F")

    with tcod.context.new(console=console, tileset=tileset, vsync=False) as context:
        sdl_renderer = context.sdl_renderer
        sdl_atlas = context.sdl_atlas
        assert sdl_renderer is not None
        assert sdl_atlas is not None

        texture = tcod.sdl.render.Texture(sdl_atlas.p.texture)
        texture.color_mod = (255, 0, 0)

        tile_index = tcod.tileset.CHARMAP_CP437.index(ord("@"))
        tile_w, tile_h = tileset.tile_shape
        src_x = (tile_index % config.TILESET_COLUMNS) * tile_w
        src_y = (tile_index // config.TILESET_COLUMNS) * tile_h
        source_rect = (src_x, src_y, tile_w, tile_h)
        dest_rect = (50, 50, tile_w, tile_h)

        for _ in range(1):
            sdl_renderer.clear()
            sdl_renderer.copy(texture, source_rect, dest_rect)
            sdl_renderer.present()

    assert True
