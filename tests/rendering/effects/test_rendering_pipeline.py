import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_RENDER_DRIVER"] = "software"

from types import SimpleNamespace
from typing import Self

from brileta import config
from brileta.util.tilesets import CHARMAP_CP437


class DummyTexture:
    def __init__(self, *_args: object) -> None:
        self.color_mod = None


class DummyRenderer:
    def __init__(self) -> None:
        self.copy_calls: list[
            tuple[DummyTexture, tuple[int, int, int, int], tuple[int, int, int, int]]
        ] = []
        self.present_calls = 0

    def clear(self) -> None:
        pass

    def copy(
        self,
        texture: DummyTexture,
        source_rect: tuple[int, int, int, int],
        dest_rect: tuple[int, int, int, int],
    ) -> None:
        self.copy_calls.append((texture, source_rect, dest_rect))

    def present(self) -> None:
        self.present_calls += 1


class DummyAtlas:
    def __init__(self) -> None:
        self.p = SimpleNamespace(texture=None)


class DummyTileset:
    def __init__(self, tile_shape: tuple[int, int]) -> None:
        self.tile_shape = tile_shape


class DummyContext:
    def __init__(self) -> None:
        self.sdl_renderer = DummyRenderer()
        self.sdl_atlas = DummyAtlas()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def test_sdl_tile_rendering() -> None:
    """Render a single tile using SDL to ensure pipeline stability."""
    tileset = DummyTileset(tile_shape=(20, 20))
    console = object()

    with DummyContext() as context:
        sdl_renderer = context.sdl_renderer
        sdl_atlas = context.sdl_atlas
        assert sdl_renderer is not None
        assert sdl_atlas is not None

        texture = DummyTexture(sdl_atlas.p.texture)
        texture.color_mod = (255, 0, 0)

        tile_index = CHARMAP_CP437.index(ord("@"))
        tile_w, tile_h = tileset.tile_shape
        src_x = (tile_index % config.TILESET_COLUMNS) * tile_w
        src_y = (tile_index // config.TILESET_COLUMNS) * tile_h
        source_rect = (src_x, src_y, tile_w, tile_h)
        dest_rect = (50, 50, tile_w, tile_h)

        sdl_renderer.clear()
        sdl_renderer.copy(texture, source_rect, dest_rect)
        sdl_renderer.present()

    assert console is not None
    assert sdl_renderer.present_calls == 1
    assert sdl_renderer.copy_calls == [(texture, source_rect, dest_rect)]
