"""UI view for displaying active status effects and conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actors import Character, Condition, StatusEffect
from catley.types import InterpolationAlpha
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class StatusView(TextView):
    """View that displays active conditions and status effects.

    Layout: Vertical, one item per line.
    Order: Status effects first (temporary), then conditions (persistent).
    Click behavior: Only condition rows are clickable (opens inventory).
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = self.controller.graphics.create_canvas()
        # Track displayed counts for click detection
        self._num_status_effects_displayed = 0
        self._num_conditions_displayed = 0

    def get_cache_key(self) -> int:
        """The key is the modifiers' revision number."""
        return self.controller.gw.player.modifiers.revision

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        """Render status effects and conditions vertically, one per line."""
        tile_w, tile_h = self.tile_dimensions
        pixel_width = self.width * tile_w
        pixel_height = self.height * tile_h
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player

        # Gather items: status effects first, then conditions
        status_effects = self._get_status_effect_lines(player)
        conditions = self._get_condition_lines(player)

        if not status_effects and not conditions:
            self._num_status_effects_displayed = 0
            self._num_conditions_displayed = 0
            return

        # Build display list: (text, color, is_condition)
        # Status effects first (no brackets), then conditions (with brackets)
        display_items: list[tuple[str, colors.Color, bool]] = [
            (text, colors.LIGHT_GREY, False) for text in status_effects
        ]
        display_items.extend((f"[{text}]", color, True) for text, color in conditions)

        # Reset counts
        self._num_status_effects_displayed = 0
        self._num_conditions_displayed = 0

        # Draw vertically, one per line
        max_lines = self.height
        for i, (text, color, is_condition) in enumerate(display_items[:max_lines]):
            # Show overflow indicator on last line if needed
            if i == max_lines - 1 and len(display_items) > max_lines:
                remaining = len(display_items) - max_lines + 1
                overflow_text = f"+{remaining} more"
                self.canvas.draw_text(0, i * tile_h, overflow_text, colors.YELLOW)
                break

            self.canvas.draw_text(0, i * tile_h, text, color)
            if is_condition:
                self._num_conditions_displayed += 1
            else:
                self._num_status_effects_displayed += 1

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
