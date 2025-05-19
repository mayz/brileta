import numpy as np
import tcod
from colors import DARK_GROUND, DARK_WALL, LIGHT_GROUND, LIGHT_WALL
from fov import FieldOfView
from model import Model
from tcod.console import Console


class Renderer:
    def __init__(
        self, screen_width: int, screen_height: int, model: Model, fov: FieldOfView
    ):
        self.colors: dict[str, tcod.Color] = {
            "dark_wall": DARK_WALL,
            "dark_ground": DARK_GROUND,
            "light_wall": LIGHT_WALL,
            "light_ground": LIGHT_GROUND,
        }

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model = model
        self.fov = fov

        tileset = tcod.tileset.load_tilesheet(
            "Taffer_20x20.png", 16, 16, tcod.tileset.CHARMAP_CP437
        )

        self.root_console = Console(screen_width, screen_height, order="F")
        self.context = tcod.context.new(
            columns=screen_width,
            rows=screen_height,
            tileset=tileset,
            title="Catley McCat",
            vsync=True,
        )

        # In the future, re-run whenever the map composition changes.
        self._optimize_map_info()

    def _optimize_map_info(self):
        """In the future, re-run whenever the map composition changes."""
        m = self.model.game_map

        self.con: Console = Console(m.width, m.height, order="F")

        # The getter properties in the Python tcod library call transpose()
        # on the underlying numpy arrays, so it's important to get a reference to
        # the arrays once, outside of a loop.
        self.bg = self.con.bg
        self.fg = self.con.fg
        self.ch = self.con.ch

        is_wall = np.full((m.width, m.height), False, dtype=np.bool_, order="F")
        for x in range(m.width):
            for y in range(m.height):
                if m.tiles[x][y].blocks_sight:
                    is_wall[(x, y)] = True
        wall_idx = np.where(is_wall)

        self.dark_map_bg = np.full_like(self.bg, self.colors.get("dark_ground"))
        self.dark_map_bg[wall_idx] = self.colors.get("dark_wall")

        self.light_map_bg = np.full_like(self.bg, self.colors.get("light_ground"))
        self.light_map_bg[wall_idx] = self.colors.get("light_wall")

        # Right now, which tiles we've explored only concerns the renderer.
        self.map_tiles_explored = np.full(
            (m.width, m.height), False, dtype=np.bool_, order="F"
        )

    def render_all(self):
        self.con.clear()

        self._render_map()
        self._render_entities()

        self.con.blit(self.root_console)
        self.context.present(self.root_console)  # Show the console.

    def _render_map(self):
        explored_idx = np.where(self.map_tiles_explored)
        self.bg[explored_idx] = self.dark_map_bg[explored_idx]

        visible_idx = np.where(self.fov.fov_map.fov)
        self.bg[visible_idx] = self.light_map_bg[visible_idx]
        self.map_tiles_explored[visible_idx] = True

        # FIXME(mark): Handle torch lighting.
        # See FOVSample in https://github.com/libtcod/python-tcod/blob/master/examples/samples_tcod.py
        # lit_color = 'light_wall' if wall else 'light_ground'
        # unlit_color = 'dark_wall' if wall else 'dark_ground'

        # dx = x - p.x
        # dy = y - p.y
        # distance = sqrt((dx * dx) + (dy * dy))

        # fov_radius = self.fov.fov_radius
        # lit_pct = 1.0 - (min(fov_radius, distance) / fov_radius)
        # self.bg[x, y] = (self.colors.get(lit_color) * lit_pct) + (
        # self.colors.get(unlit_color) * (1.0 - lit_pct))

    def _render_entities(self):
        for e in self.model.entities:
            if self.fov.contains(e.x, e.y):
                self.ch[e.x, e.y] = e.ch
                self.fg[e.x, e.y] = e.color

    def _render_bar(self, x, y, total_width, name, value, maximum, color, bg_color):
        pass
