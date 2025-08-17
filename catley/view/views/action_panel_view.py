"""Action panel view that displays target info and available actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.environment import tile_types
from catley.game.actions.discovery import ActionCategory, ActionDiscovery, ActionOption
from catley.game.actors import Character
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionPanelView(TextView):
    """Left sidebar panel showing target info and available actions."""

    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = PillowImageCanvas(controller.graphics)
        self.discovery = ActionDiscovery()
        self._cached_actions: list[ActionOption] = []
        self._cached_target_name: str | None = None
        self._cached_target_description: str | None = None

        # View pixel dimensions will be calculated when resize() is called
        self.view_width_px = 0
        self.view_height_px = 0

        # Override the cache from the base class for pixel-based rendering
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render", max_size=1
        )

    def get_cache_key(self) -> tuple:
        """Cache key based on mouse position and view dimensions."""
        mouse_pos = str(self.controller.gw.mouse_tile_location_on_map)
        player_pos = f"{self.controller.gw.player.x},{self.controller.gw.player.y}"
        current_tile_dimensions = self.controller.graphics.tile_dimensions
        return (
            mouse_pos,
            player_pos,
            current_tile_dimensions,
            self.view_width_px,
            self.view_height_px,
        )

    def set_bounds(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Override set_bounds to update pixel dimensions when view is resized."""
        super().set_bounds(x1, y1, x2, y2)
        # Update pixel dimensions based on new size
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]
        self.canvas.configure_scaling(self.tile_dimensions[1])

    def draw_content(
        self, graphics: GraphicsContext, alpha: InterpolationAlpha
    ) -> None:
        """Render the action panel content."""
        # Update cached tile dimensions and recalculate pixel dimensions
        self.tile_dimensions = graphics.tile_dimensions
        self.view_width_px = self.width * self.tile_dimensions[0]
        self.view_height_px = self.height * self.tile_dimensions[1]

        # Clear background
        self.canvas.draw_rect(
            0, 0, self.view_width_px, self.view_height_px, colors.BLACK, fill=True
        )

        # Update cached data
        self._update_cached_data()

        # Get font metrics for proper line spacing
        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent

        # Start rendering from top with some padding
        y_pixel = ascent + 5  # 5px top padding
        x_padding = 5  # 5px left padding

        # Target name section
        if self._cached_target_name:
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text=self._cached_target_name,
                color=colors.YELLOW,
            )
            y_pixel += line_height

            # Target description (if available)
            if self._cached_target_description:
                # Word wrap description to fit panel width (accounting for padding)
                wrapped_lines = self.canvas.wrap_text(
                    self._cached_target_description,
                    self.view_width_px - (x_padding * 2),
                )
                for line in wrapped_lines[:3]:  # Limit to 3 lines
                    self.canvas.draw_text(
                        pixel_x=x_padding,
                        pixel_y=y_pixel - ascent,
                        text=line,
                        color=colors.GREY,
                    )
                    y_pixel += line_height

            y_pixel += line_height // 2  # Add spacing

        # Actions section
        if self._cached_actions:
            # Assign hotkeys to actions that don't have them
            hotkey_chars = "abcdefghijklmnopqrstuvwxyz"
            hotkey_index = 0
            for action in self._cached_actions:
                if not action.hotkey and hotkey_index < len(hotkey_chars):
                    action.hotkey = hotkey_chars[hotkey_index]
                    hotkey_index += 1

            # Group actions by category
            actions_by_category: dict[ActionCategory, list[ActionOption]] = {}
            for action in self._cached_actions:
                if action.category not in actions_by_category:
                    actions_by_category[action.category] = []
                actions_by_category[action.category].append(action)

            # Display actions by category
            category_names = {
                ActionCategory.COMBAT: "Combat",
                ActionCategory.ENVIRONMENT: "Environment",
                ActionCategory.ITEMS: "Items",
                ActionCategory.SOCIAL: "Social",
            }

            for category, actions in actions_by_category.items():
                # Category header
                category_name = category_names.get(category, "Other")
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text=f"{category_name}",
                    color=colors.DARK_GREY,
                )
                y_pixel += line_height

                # Action items
                for action in actions[:8]:  # More actions can fit with smaller font
                    # Clean up action name - remove redundant target name
                    action_name = action.name
                    if self._cached_target_name and isinstance(
                        self._cached_target_name, str
                    ):
                        # Remove patterns like "Verb TargetName with Weapon"
                        # to just "Verb with Weapon"
                        action_name = action_name.replace(
                            f" {self._cached_target_name} ", " "
                        )

                    hotkey_str = f"[{action.hotkey}]" if action.hotkey else "[ ]"
                    action_text = f"{hotkey_str} {action_name}"

                    # Add success probability if available
                    if action.success_probability is not None:
                        prob_percent = int(action.success_probability * 100)
                        action_text += f" ({prob_percent}%)"

                    self.canvas.draw_text(
                        pixel_x=x_padding,
                        pixel_y=y_pixel - ascent,
                        text=action_text,
                        color=colors.WHITE,
                    )
                    y_pixel += line_height

                y_pixel += line_height // 2  # Small spacing between categories

        elif self._cached_target_name is None:
            # Empty state - show helpful hints
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="Hover over targets",
                color=colors.DARK_GREY,
            )
            y_pixel += line_height
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="to see actions",
                color=colors.DARK_GREY,
            )
            y_pixel += line_height * 2

            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="Controls:",
                color=colors.GREY,
            )
            y_pixel += line_height
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="[Space] Action menu",
                color=colors.DARK_GREY,
            )
            y_pixel += line_height
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="[Right-click] Context",
                color=colors.DARK_GREY,
            )
            y_pixel += line_height
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="[?] Help",
                color=colors.DARK_GREY,
            )

    def _update_cached_data(self) -> None:
        """Update cached target information and available actions."""
        gw = self.controller.gw
        mouse_pos = gw.mouse_tile_location_on_map

        # Clear cache if no mouse position
        if mouse_pos is None:
            self._cached_target_name = None
            self._cached_target_description = None
            self._cached_actions = []
            return

        x, y = mouse_pos
        if not (0 <= x < gw.game_map.width and 0 <= y < gw.game_map.height):
            self._cached_target_name = None
            self._cached_target_description = None
            self._cached_actions = []
            return

        # Get target at mouse position
        target_actor = None
        non_blocking_actor = None

        # Check for visible actors first
        if gw.game_map.visible[x, y]:
            actor = gw.get_actor_at_location(x, y)
            if actor:
                if actor.blocks_movement:
                    target_actor = actor
                else:
                    non_blocking_actor = actor

        # Check for items if no blocking actor
        if not target_actor:
            items = gw.get_pickable_items_at_location(x, y)
            if items and gw.game_map.visible[x, y]:
                if len(items) == 1:
                    self._cached_target_name = items[0].name
                    self._cached_target_description = "An item on the ground"
                else:
                    self._cached_target_name = f"{len(items)} items"
                    self._cached_target_description = "Multiple items here"
                # No actions for items yet (would need item-specific actions)
                self._cached_actions = []
                return

        # Use non-blocking actor if no items
        if not target_actor and non_blocking_actor:
            target_actor = non_blocking_actor

        # Handle actor target
        if target_actor:
            self._cached_target_name = target_actor.name

            # Get description based on actor type
            if isinstance(target_actor, Character):
                # Characters always have health component
                if target_actor.health.is_alive():
                    self._cached_target_description = (
                        f"HP: {target_actor.health.hp}/{target_actor.health.max_hp}"
                    )
                else:
                    self._cached_target_description = "Deceased"

                # Get available actions for this target
                if target_actor is not gw.player:
                    self._cached_actions = self.discovery.get_options_for_target(
                        self.controller, gw.player, target_actor
                    )
                else:
                    self._cached_actions = []
            else:
                self._cached_target_description = None
                self._cached_actions = []
        else:
            # Show tile information
            tile_id = gw.game_map.tiles[x, y]
            tile_name = tile_types.get_tile_type_name_by_id(tile_id)

            if gw.game_map.visible[x, y]:
                self._cached_target_name = tile_name
            elif gw.game_map.explored[x, y]:
                self._cached_target_name = f"{tile_name} (remembered)"
            else:
                self._cached_target_name = None

            self._cached_target_description = None

            # Get environment actions for this specific tile
            if gw.game_map.visible[x, y]:
                # Build context for action discovery
                context = self.discovery.context_builder.build_context(
                    self.controller, gw.player
                )
                # Get tile-specific environment actions (e.g., door actions)
                env_discovery = self.discovery.environment_discovery
                self._cached_actions = (
                    env_discovery.discover_environment_actions_for_tile(
                        self.controller, gw.player, context, x, y
                    )
                )
            else:
                self._cached_actions = []

    def get_hotkeys(self) -> dict[str, ActionOption]:
        """Get current hotkey mappings for direct execution."""
        hotkeys = {}
        for action in self._cached_actions:
            if action.hotkey:
                hotkeys[action.hotkey.lower()] = action
        return hotkeys

    def invalidate_cache(self) -> None:
        """Clear the action panel cache to force refresh on next draw."""
        self._texture_cache.clear()

    def on_door_action_completed(self, door_x: int, door_y: int) -> None:
        """Called when a door action completes to refresh cache if hovering."""
        mouse_pos = self.controller.gw.mouse_tile_location_on_map
        if mouse_pos is not None:
            mouse_x, mouse_y = mouse_pos
            if mouse_x == door_x and mouse_y == door_y:
                self.invalidate_cache()
