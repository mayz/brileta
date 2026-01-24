"""View that displays the player's status in the upper-right corner.

Displays four lines ordered by permanence:
1. HP - core vitality
2. Armor - equipment protection
3. Injuries/Conditions - semi-permanent wounds (bracketed)
4. Temporary Effects - combat states (pipe-separated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.game.actors import Condition, StatusEffect
from catley.game.actors.status_effects import EncumberedEffect
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class PlayerStatusView(TextView):
    """View that displays the player's HP, armor, conditions, and effects.

    Display format (4 lines):
    - Line 1: HP: current/max
    - Line 2: Armor: [PR:X] AP: current/max (or BROKEN)
    - Line 3: [Condition1] [Condition2] ... (bracketed, semi-permanent)
    - Line 4: Effect1 | Effect2 | ... (pipe-separated, temporary)
    """

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        """Initialize the view.

        Position and size will be set by FrameManager.resize().
        """
        super().__init__()
        self.controller = controller
        self._graphics = graphics
        # Use PillowImageCanvas with VGA font for crisp pixel rendering
        self.canvas = PillowImageCanvas(
            graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.PLAYER_STATUS_FONT_SIZE,
            line_spacing=1.0,
        )
        # Override cache to release textures on eviction
        self._texture_cache = ResourceCache[Any, Any](
            name=f"{self.__class__.__name__}Render",
            max_size=1,
            on_evict=lambda tex: self._graphics.release_texture(tex),
        )

    def get_cache_key(
        self,
    ) -> tuple[int, int, int, int, int, bool, int, tuple[int, int]]:
        """Cache key includes HP, outfit state, and modifier revision."""
        player = self.controller.gw.player

        # Get outfit capability if equipped
        outfit_cap = player.inventory.outfit_capability
        if outfit_cap is not None and outfit_cap.has_protection:
            return (
                player.health.hp,
                player.health.max_hp,
                outfit_cap.protection,
                outfit_cap.ap,
                outfit_cap.max_ap,
                outfit_cap.is_broken,
                player.modifiers.revision,
                self.tile_dimensions,
            )

        # No protective outfit - just HP and modifiers
        return (
            player.health.hp,
            player.health.max_hp,
            0,
            0,
            0,
            False,
            player.modifiers.revision,
            self.tile_dimensions,
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update canvas scaling when view is resized."""
        super().set_bounds(x1, y1, x2, y2)
        # Reconfigure canvas scaling when tile dimensions change
        self.canvas.configure_scaling(self.tile_dimensions[1])

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        tile_width, _tile_height = self.tile_dimensions
        pixel_width = self.width * tile_width
        # No background fill - transparent overlay lets world show through

        player = self.controller.gw.player

        # Get font metrics for line positioning
        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent
        x_padding = 8

        # Track current y position (right-align text to the view)
        # Add top padding to buffer from the edge
        y = ascent // 2

        # Line 1: HP
        hp_text = f"HP: {player.health.hp}/{player.health.max_hp}"
        hp_width, _, _ = self.canvas.get_text_metrics(hp_text)
        hp_x = pixel_width - hp_width - x_padding
        self.canvas.draw_text(hp_x, y, hp_text, colors.WHITE)
        y += line_height

        # Line 2: Armor (if equipped with protective outfit)
        outfit_cap = player.inventory.outfit_capability
        if outfit_cap is not None and outfit_cap.has_protection:
            if outfit_cap.is_broken:
                armor_text = f"Armor: [PR:{outfit_cap.protection}] AP: BROKEN"
                armor_color = colors.RED
            else:
                armor_text = (
                    f"Armor: [PR:{outfit_cap.protection}] "
                    f"AP: {outfit_cap.ap}/{outfit_cap.max_ap}"
                )
                # Color based on AP percentage
                ap_ratio = outfit_cap.ap / outfit_cap.max_ap
                if ap_ratio <= 0.25:
                    armor_color = colors.ORANGE
                elif ap_ratio <= 0.5:
                    armor_color = colors.YELLOW
                else:
                    armor_color = colors.WHITE

            armor_width, _, _ = self.canvas.get_text_metrics(armor_text)
            armor_x = pixel_width - armor_width - x_padding
            self.canvas.draw_text(armor_x, y, armor_text, armor_color)
        y += line_height

        # Line 3: Conditions (semi-permanent, bracketed)
        conditions = self._get_conditions()
        if conditions:
            # Build bracketed condition string: [Cond1] [Cond2]
            condition_parts = [f"[{text}]" for text, _ in conditions]
            condition_text = " ".join(condition_parts)
            cond_width, _, _ = self.canvas.get_text_metrics(condition_text)
            cond_x = pixel_width - cond_width - x_padding

            # Draw each condition in its own color
            current_x = cond_x
            for text, color in conditions:
                bracketed = f"[{text}]"
                self.canvas.draw_text(current_x, y, bracketed, color)
                text_width, _, _ = self.canvas.get_text_metrics(bracketed + " ")
                current_x += text_width
        y += line_height

        # Line 4: Status effects (temporary, pipe-separated)
        effects = self._get_status_effects()
        if effects:
            # Build pipe-separated effect string: Effect1 | Effect2
            effect_parts = [text for text, _ in effects]
            effect_text = " | ".join(effect_parts)
            eff_width, _, _ = self.canvas.get_text_metrics(effect_text)
            eff_x = pixel_width - eff_width - x_padding

            # Draw each effect with appropriate coloring
            current_x = eff_x
            for i, (text, color) in enumerate(effects):
                self.canvas.draw_text(current_x, y, text, color)
                text_width, _, _ = self.canvas.get_text_metrics(text)
                current_x += text_width
                # Draw separator if not last
                if i < len(effects) - 1:
                    sep = " | "
                    self.canvas.draw_text(current_x, y, sep, colors.GREY)
                    sep_width, _, _ = self.canvas.get_text_metrics(sep)
                    current_x += sep_width

    def _get_status_effects(self) -> list[tuple[str, colors.Color]]:
        """Get active status effects with display text and color."""
        player = self.controller.gw.player
        effects = [
            effect
            for effect in player.modifiers.get_all_active_effects()
            if isinstance(effect, StatusEffect)
        ]
        if not effects:
            return []

        lines: list[tuple[str, colors.Color]] = []
        for effect in effects:
            # Use severity-based coloring for encumbered effect
            if isinstance(effect, EncumberedEffect):
                over_capacity = player.inventory.get_slots_over_capacity()
                if over_capacity >= 3:
                    color = colors.RED
                elif over_capacity >= 1:
                    color = colors.ORANGE
                else:
                    color = colors.YELLOW
            else:
                color = effect.display_color
            lines.append((effect.name, color))
        return lines

    def _get_conditions(self) -> list[tuple[str, colors.Color]]:
        """Get active conditions with display text and color."""
        player = self.controller.gw.player
        conditions = [
            effect
            for effect in player.modifiers.get_all_active_effects()
            if isinstance(effect, Condition)
        ]
        if not conditions:
            return []

        # Group conditions by name and count
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
