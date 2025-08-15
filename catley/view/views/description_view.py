"""UI view for displaying context about hovered tiles, actors, and items."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.environment import tile_types
from catley.types import InterpolationAlpha
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class DescriptionView(TextView):
    """View that displays information about what the mouse is hovering over."""

    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = self.controller.graphics.create_canvas()

    def get_cache_key(self) -> int:
        """Cache key that includes view identity to prevent cross-view pollution."""
        mouse_pos = str(self.controller.gw.mouse_tile_location_on_map)
        return hash((id(self), mouse_pos))

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        """Render the description view - copied from StatusView pattern."""

        tile_w, tile_h = self.tile_dimensions
        pixel_width = self.width * tile_w
        pixel_height = self.height * tile_h
        self.canvas.draw_rect(0, 0, pixel_width, pixel_height, colors.BLACK, fill=True)

        # Get description text
        description_text = self._get_description_text()
        if not description_text:
            return

        # Draw like StatusView does
        self.canvas.draw_text(0, 0, description_text, colors.WHITE)

    def _get_description_text(self) -> str | None:
        """Get simple description text."""
        gw = self.controller.gw
        mouse_pos = gw.mouse_tile_location_on_map

        if mouse_pos is None:
            return None

        x, y = mouse_pos
        if not (0 <= x < gw.game_map.width and 0 <= y < gw.game_map.height):
            return None

        non_blocking_actor = None

        # Check for visible actors first (blocking actors have priority)
        if gw.game_map.visible[x, y]:
            actor = gw.get_actor_at_location(x, y)
            if actor:
                if actor.blocks_movement:
                    return actor.name
                # Store non-blocking actor for fallback
                non_blocking_actor = actor

        # Check for items
        items = gw.get_pickable_items_at_location(x, y)
        if items and gw.game_map.visible[x, y]:
            if len(items) == 1:
                return items[0].name
            return f"{len(items)} items"

        # Show non-blocking actor if no items
        if non_blocking_actor:
            return non_blocking_actor.name

        # Get tile name
        tile_id = gw.game_map.tiles[x, y]
        tile_name = tile_types.get_tile_type_name_by_id(tile_id)

        if gw.game_map.visible[x, y]:
            return tile_name
        if gw.game_map.explored[x, y]:
            return f"{tile_name} (remembered)"
        return None
