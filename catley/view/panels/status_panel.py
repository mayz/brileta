"""UI panel for displaying active status effects and conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actors import Character
from catley.game.conditions import Condition
from catley.view.render.renderer import Renderer
from catley.view.render.text_backend import TextBackend

from .panel import Panel

if TYPE_CHECKING:
    from catley.controller import Controller


class StatusPanel(Panel):
    """Panel that displays active conditions and status effects."""

    def __init__(
        self, controller: Controller, *, text_backend: TextBackend | None = None
    ) -> None:
        super().__init__(text_backend)
        self.controller = controller

    def draw_content(self, renderer: Renderer) -> None:
        """Render the status panel if player has active effects."""

        assert self.text_backend is not None

        player = self.controller.gw.player
        conditions = self._get_condition_lines(player)
        status_effects = self._get_status_effect_lines(player)

        if not conditions and not status_effects:
            return

        current_y = 0

        if conditions:
            self.text_backend.draw_text(0, current_y, "CONDITIONS:", colors.YELLOW)
            current_y += 1
            for text, color in conditions:
                self.text_backend.draw_text(0, current_y, text, color)
                current_y += 1
            current_y += 1

        if status_effects:
            self.text_backend.draw_text(0, current_y, "STATUS EFFECTS:", colors.CYAN)
            current_y += 1
            for text in status_effects:
                self.text_backend.draw_text(0, current_y, text, colors.LIGHT_GREY)
                current_y += 1

    def _get_condition_lines(self, player: Character) -> list[tuple[str, colors.Color]]:
        """Get formatted display lines for conditions."""
        if not player.inventory:
            return []

        conditions = player.get_conditions()
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
        if not player.status_effects:
            return []

        lines = []
        for effect in player.status_effects:
            if effect.duration > 0:
                suffix = "turn" if effect.duration == 1 else "turns"
                lines.append(f"{effect.name} ({effect.duration} {suffix})")
            else:
                lines.append(effect.name)
        return lines
