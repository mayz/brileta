from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.context
from tcod.console import Console

from catley import colors
from catley.config import PERFORMANCE_TESTING
from catley.ui.menu_core import MenuSystem
from catley.util.clock import Clock
from catley.util.message_log import MessageLog
from catley.world.game_state import GameWorld

from .renderer import Renderer as LowLevelRenderer

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.render.frame_manager import FrameManager


class FPSDisplay:
    def __init__(self, clock: Clock, update_interval: float = 0.5) -> None:
        self.clock = clock
        self.update_interval = update_interval
        self.last_update = clock.last_time
        self.display_string = "FPS: 0.0"

    def update(self) -> None:
        current_time = self.clock.last_time
        if current_time - self.last_update >= self.update_interval:
            if PERFORMANCE_TESTING:
                self.display_string = (
                    f"FPS: mean {self.clock.mean_fps:.0f}, "
                    f"last {self.clock.last_fps:.0f}"
                )
            else:
                self.display_string = f"FPS: {self.clock.mean_fps:.1f}"

            self.last_update = current_time

    def render(self, frame_manager: FrameManager) -> None:
        self.update()

        fps_width = len(self.display_string)
        # Access renderer and root_console through the frame_manager
        x_position = max(0, frame_manager.renderer.root_console.width - fps_width - 1)
        frame_manager.renderer.draw_text(
            x_position, 0, self.display_string, fg=colors.YELLOW
        )


class Renderer:
    def __init__(
        self,
        controller: Controller,
        screen_width: int,
        screen_height: int,
        game_world: GameWorld,
        clock: Clock,
        message_log: MessageLog,
        menu_system: MenuSystem,
        context: tcod.context.Context,
        root_console: Console,
        tile_dimensions: tuple[int, int],
    ) -> None:
        self.low_level_renderer = LowLevelRenderer(
            context, root_console, tile_dimensions
        )

        self.controller = controller
        self.fps_display = FPSDisplay(clock)
