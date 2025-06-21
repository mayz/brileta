"""UI view for displaying active status effects and conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actors import Character, Condition, StatusEffect
from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import TCODTextBackend

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class StatusView(TextView):
    """View that displays active conditions and status effects."""

    def __init__(self, controller: Controller, renderer: Renderer) -> None:
        super().__init__()
        self.controller = controller
        self.text_backend = TCODTextBackend(renderer)

    def needs_redraw(self, renderer: Renderer) -> bool:
        _ = renderer
        return True

    def draw_content(self, renderer: Renderer) -> None:
        """Render the status view if player has active effects."""

        player = self.controller.gw.player
        all_effects = player.modifiers.get_all_active_effects()
        if not all_effects:
            return

        tile_width, tile_height = self.tile_dimensions
        conditions = self._get_condition_lines(player)
        status_effects = self._get_status_effect_lines(player)

        current_y = 0

        if conditions:
            self.text_backend.draw_text(
                0, current_y * tile_height, "CONDITIONS:", colors.YELLOW
            )
            current_y += 1
            for text, color in conditions:
                self.text_backend.draw_text(0, current_y * tile_height, text, color)
                current_y += 1
            current_y += 1

        if status_effects:
            self.text_backend.draw_text(
                0, current_y * tile_height, "STATUS EFFECTS:", colors.CYAN
            )
            current_y += 1
            for text in status_effects:
                self.text_backend.draw_text(
                    0, current_y * tile_height, text, colors.LIGHT_GREY
                )
                current_y += 1

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
