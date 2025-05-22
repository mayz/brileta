import colors
import numpy as np
import tcod
from clock import Clock
from fov import FieldOfView
from model import Entity, Model
from tcod.console import Console


class FPSDisplay:
    SHOW_FPS = False

    def __init__(self, clock: Clock, update_interval: float = 0.5) -> None:
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def update(self) -> None:
        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            self.display_string = f"FPS: {self.clock.last_fps:.1f}"
            self.last_update = current_time

    def render(self, renderer: "Renderer", x: int, y: int) -> None:
        self.update()
        renderer._render_text(x, y, self.display_string, fg=colors.YELLOW)


class Renderer:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        model: Model,
        fov: FieldOfView,
        clock: Clock,
    ) -> None:
        self.colors: dict[str, colors.Color] = {
            "dark_wall": colors.DARK_WALL,
            "dark_ground": colors.DARK_GROUND,
            "light_wall": colors.LIGHT_WALL,
            "light_ground": colors.LIGHT_GROUND,
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

        self.fps_display = FPSDisplay(clock)

    def _optimize_map_info(self) -> None:
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

    def render_all(self) -> None:
        self.con.clear()

        # Make sure FOV is up to date before rendering
        self.fov.recompute_if_needed()

        # Clear the root console first
        self.root_console.clear()

        # Render map and entities to game console
        self._render_map()
        self._render_entities()

        # Blit game console to root console
        self.con.blit(
            dest=self.root_console,
            dest_x=0,
            dest_y=0,
            width=self.con.width,
            height=self.con.height,
        )

        if self.fps_display.SHOW_FPS:
            self.fps_display.render(self, self.screen_width - 12, 0)

        # Present the final console and handle vsync timing
        self.context.present(self.root_console, keep_aspect=True, integer_scaling=True)

    def _render_map(self) -> None:
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
                self.bg[visible_y, visible_x, i] = self.light_map_bg[
                    visible_y, visible_x, i
                ] * light_intensity + self.dark_map_bg[visible_y, visible_x, i] * (
                    1.0 - light_intensity
                )

            self.map_tiles_explored[visible_y, visible_x] = True

    def _render_entities(self) -> None:
        for e in self.model.entities:
            if e == self.model.player:
                continue

            if self.fov.contains(e.x, e.y):
                self._render_entity(e)

        # Always draw the player last.
        self._render_entity(self.model.player)

    def _render_entity(self, e: Entity) -> None:
        self.ch[e.x, e.y] = ord(e.ch)

        # Use the already computed RGB light intensity
        light_rgb = self.current_light_intensity[e.x, e.y]

        # Apply RGB lighting to each color channel
        lit_color = tuple(
            int(min(255, color * light))  # Clamp to valid color range
            for color, light in zip(e.color, light_rgb, strict=True)
        )
        self.fg[e.x, e.y] = lit_color

    def _render_text(
        self, x: int, y: int, text: str, fg: colors.Color = colors.WHITE
    ) -> None:
        """Render text at a specific position with a given color"""
        self.root_console.print(x=x, y=y, string=text, fg=fg)
