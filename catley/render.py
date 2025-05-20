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

        # Make sure FOV is up to date before rendering
        self.fov.recompute_if_needed()

        self._render_map()
        self._render_entities()

        self.con.blit(self.root_console)
        self.context.present(self.root_console)  # Show the console.

    def _render_map(self):
        explored_idx = np.where(self.map_tiles_explored)
        self.bg[explored_idx] = self.dark_map_bg[explored_idx]

        # Compute lighting once and store it
        self.current_light_intensity = self.model.lighting.compute_lighting(
            self.model.game_map.width, self.model.game_map.height
        )

        # Apply lighting to visible areas
        light_mask = self.fov.fov_map.fov
        visible_y, visible_x = np.where(light_mask)  # Get coordinates of visible cells

        if len(visible_y) > 0:  # Only process if we have visible cells
            # Get RGB light intensities for visible cells
            cell_light = self.current_light_intensity[visible_y, visible_x]

            # Blend colors based on light intensity for each RGB channel
            for i in range(3):  # For each RGB channel
                light_intensity = cell_light[..., i]
                self.bg[visible_y, visible_x, i] = (
                    self.light_map_bg[visible_y, visible_x, i] * light_intensity +
                    self.dark_map_bg[visible_y, visible_x, i] * (1.0 - light_intensity)
                )

            self.map_tiles_explored[visible_y, visible_x] = True

    def _render_entities(self):
        for e in self.model.entities:
            if self.fov.contains(e.x, e.y):
                self.ch[e.x, e.y] = e.ch

                # Use the already computed RGB light intensity
                light_rgb = self.current_light_intensity[e.x, e.y]

                # Apply RGB lighting to each color channel
                lit_color = tuple(
                    int(min(255, color * light))  # Clamp to valid color range
                    for color, light in zip(e.color, light_rgb, strict=True)
                )
                self.fg[e.x, e.y] = lit_color

    def _render_bar(self, x, y, total_width, name, value, maximum, color, bg_color):
        pass
