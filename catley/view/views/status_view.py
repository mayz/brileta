"""UI view for displaying active status effects and conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actors import Character, Condition, StatusEffect
from catley.view.render.canvas import TCODConsoleCanvas
from catley.view.render.renderer import Renderer

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class StatusView(TextView):
    """View that displays active conditions and status effects."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = TCODConsoleCanvas(renderer)

    def get_cache_key(self) -> int:
        """The key is the modifiers' revision number."""
        return self.controller.gw.player.modifiers.revision

    def draw_content(self, renderer: Renderer) -> None:
        """Render the status view if player has active effects."""

        tile_w, tile_h = self.tile_dimensions
        pixel_width = self.width * tile_w
        pixel_height = self.height * tile_h
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player
        all_effects = player.modifiers.get_all_active_effects()
        if not all_effects:
            return

        current_x_tile = 0
        conditions = self._get_condition_lines(player)
        status_effects = self._get_status_effect_lines(player)

        for text, color in conditions:
            token = f"[{text}]"
            if current_x_tile + len(token) > self.width:
                break
            self.canvas.draw_text(current_x_tile * tile_w, 0, token, color)
            current_x_tile += len(token) + 1

        for text in status_effects:
            token = f"[{text}]"
            if current_x_tile + len(token) > self.width:
                break
            self.canvas.draw_text(current_x_tile * tile_w, 0, token, colors.LIGHT_GREY)
            current_x_tile += len(token) + 1

    def _get_condition_lines(self, player: Character) -> list[tuple[str, colors.Color]]:
        """Get formatted display lines for conditions."""
        conditions = [
            effect
            for effect in player.modifiers.get_all_active_effects()
            if isinstance(effect, Condition)
        ]
        if not conditions:
            return []

        condition_counts: dict[str, list[Condition]] = {}
        for condition in conditions:
            condition_counts.setdefault(condition.name, []).append(condition)

        lines: list[tuple[str, colors.Color]] = []
        for name, condition_list in condition_counts.items():
            count = len(condition_list)
            color = condition_list[0].display_color
            text = f"{name} x{count}" if count > 1 else name
            lines.append((text, color))

        return lines

    def _get_status_effect_lines(self, player: Character) -> list[str]:
        """Get formatted display lines for status effects."""
        effects = [
            effect
            for effect in player.modifiers.get_all_active_effects()
            if isinstance(effect, StatusEffect)
        ]
        if not effects:
            return []

        lines = []
        for effect in effects:
            if effect.duration > 0:
                suffix = "turn" if effect.duration == 1 else "turns"
                lines.append(f"{effect.name} ({effect.duration} {suffix})")
            else:
                lines.append(effect.name)
        return lines
