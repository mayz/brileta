"""
Combat indicator overlay for displaying combat mode status.

Shows the current weapon, attack mode, and hit probability for the selected target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tcod.event

from catley import colors
from catley.game import ranges
from catley.game.actions.discovery.action_context import ActionContextBuilder
from catley.view.ui.overlays import TextOverlay

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actors import Character
    from catley.view.render.canvas import Canvas


class CombatIndicatorOverlay(TextOverlay):
    """Non-interactive overlay that displays combat mode status.

    Shows weapon name, attack mode (melee/ranged), and hit probability
    for the currently selected target.
    """

    def __init__(self, controller: Controller) -> None:
        super().__init__(controller)
        self.is_interactive = False
        self._context_builder = ActionContextBuilder()
        # Cache the last displayed text to detect when dimensions need updating
        self._last_text: str = ""

    def _get_backend(self) -> Canvas:
        """Return a backend-appropriate canvas."""
        return self.controller.graphics.create_canvas(transparent=True)

    def _calculate_dimensions(self) -> None:
        """Calculate dimensions for the combat indicator."""
        text = self._build_display_text()
        self._last_text = text

        # Calculate overlay dimensions
        self.width = len(text) + 2  # Add 2 for padding
        self.height = 1

        # Center horizontally, place just below top bar
        self.x_tiles = (self.controller.graphics.console_width_tiles - self.width) // 2
        self.y_tiles = 1

        # Calculate pixel dimensions
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        self.pixel_width = self.width * self.tile_dimensions[0]
        self.pixel_height = self.height * self.tile_dimensions[1]

    def _build_display_text(self) -> str:
        """Build the display text based on current targeting state."""
        target = self._get_current_target()
        if target is None:
            return "[ COMBAT ]"

        player = self.controller.gw.player
        weapon = player.inventory.get_active_weapon()

        # Fallback to fists if no weapon equipped
        if weapon is None:
            from catley.game.items.item_types import FISTS_TYPE

            weapon = FISTS_TYPE.create()

        # Determine attack mode based on distance
        distance = ranges.calculate_distance(player.x, player.y, target.x, target.y)

        # Check for melee attack (distance == 1)
        if distance == 1 and weapon.melee_attack:
            attack_mode = "Melee"
            prob = self._context_builder.calculate_combat_probability(
                self.controller, player, target, "strength"
            )
            prob_str = f"{int(prob * 100)}%"
            return f"[ {weapon.name} ({attack_mode}) - {prob_str} ]"

        # Check for ranged attack
        if weapon.ranged_attack:
            # Check ammo
            if weapon.ranged_attack.current_ammo <= 0:
                return f"[ {weapon.name} (NO AMMO) ]"

            # Check range
            range_cat = ranges.get_range_category(distance, weapon)
            range_mods = ranges.get_range_modifier(weapon, range_cat)

            if range_mods is None:
                return f"[ {weapon.name} (OUT OF RANGE) ]"

            attack_mode = "Ranged"
            prob = self._context_builder.calculate_combat_probability(
                self.controller, player, target, "observation", range_mods
            )
            prob_str = f"{int(prob * 100)}%"
            return f"[ {weapon.name} ({attack_mode}) - {prob_str} ]"

        # Melee weapon but target not adjacent
        if weapon.melee_attack and distance > 1:
            return f"[ {weapon.name} (TOO FAR) ]"

        return "[ COMBAT ]"

    def _get_current_target(self) -> Character | None:
        """Get the currently selected target from combat mode."""
        combat_mode = self.controller.combat_mode
        if combat_mode.active:
            return combat_mode._get_current_target()
        return None

    def draw_content(self) -> None:
        """Draw the combat indicator text."""
        assert self.canvas is not None

        text = self._build_display_text()

        # If text changed, we need to recalculate dimensions and recreate canvas
        if text != self._last_text:
            self._last_text = text
            self._calculate_dimensions()
            self._create_canvas()
            if self.canvas is None:
                return

        # Center the text within the overlay
        text_x_offset = (self.width - len(text)) // 2

        self.canvas.draw_text(
            pixel_x=text_x_offset * self.tile_dimensions[0],
            pixel_y=0,
            text=text,
            color=colors.RED,
        )

    def handle_input(self, event: tcod.event.Event) -> bool:
        """Handle input events. Since this is non-interactive, always return False."""
        return False
