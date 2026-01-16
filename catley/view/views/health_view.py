from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.types import InterpolationAlpha
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class HealthView(TextView):
    """View that displays the player's HP and outfit status.

    Display format:
    - HP: current/max  Armor: [PR:X] AP: current/max (if outfit has protection)
    - HP: current/max (if no protective outfit)
    - Shows "AP: BROKEN" when AP is 0
    """

    def __init__(self, controller: Controller, graphics: GraphicsContext) -> None:
        """Initialize the view.

        Position and size will be set by FrameManager.resize().
        """

        super().__init__()
        self.controller = controller
        self.canvas = graphics.create_canvas()

    def get_cache_key(self) -> tuple[int, int, int, int, int, bool]:
        """Cache key includes HP and outfit state."""
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
            )

        # No protective outfit - just HP
        return (player.health.hp, player.health.max_hp, 0, 0, 0, False)

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        tile_width, tile_height = self.tile_dimensions
        pixel_width = self.width * tile_width
        pixel_height = self.height * tile_height
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        player = self.controller.gw.player

        # Build the status text
        hp_text = f"HP: {player.health.hp}/{player.health.max_hp}"

        # Get outfit capability if equipped
        outfit_cap = player.inventory.outfit_capability

        armor_text: str | None = None
        armor_color = colors.WHITE

        if outfit_cap is not None and outfit_cap.has_protection:
            # Show armor stats: Armor: [PR:X] AP: current/max or BROKEN
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

        full_text = f"{hp_text}  {armor_text}" if armor_text is not None else hp_text

        # Draw HP in white (ensure it's always visible, minimum x=0)
        x_pos_tiles = max(0, self.width - len(full_text) - 1)
        self.canvas.draw_text(x_pos_tiles * tile_width, 0, hp_text, colors.WHITE)

        # Draw armor stats in appropriate color (if present)
        if armor_text is not None:
            armor_x = x_pos_tiles + len(hp_text) + 2  # +2 for the "  " separator
            self.canvas.draw_text(armor_x * tile_width, 0, armor_text, armor_color)
