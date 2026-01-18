"""Action panel view that displays target info and available actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.environment import tile_types
from catley.game.actions.discovery import ActionCategory, ActionDiscovery, ActionOption
from catley.game.actors import Character
from catley.game.actors.container import Container
from catley.game.items.properties import WeaponProperty
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.drawing_utils import draw_keycap

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

        # Sticky hotkeys: track action id -> hotkey for continuity
        self._previous_hotkeys: dict[str, str] = {}

        # View pixel dimensions will be calculated when resize() is called
        self.view_width_px = 0
        self.view_height_px = 0

        # Override the cache from the base class for pixel-based rendering
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render", max_size=1
        )

    def _draw_keycap_with_label(self, x: int, y: int, key: str, label: str) -> int:
        """Draw a keycap with label text. Returns the total width consumed."""
        current_x = x

        # Draw keycap
        keycap_width = draw_keycap(
            canvas=self.canvas,
            pixel_x=current_x,
            pixel_y=y,
            key=key,
            bg_color=colors.DARK_GREY,
            border_color=colors.GREY,
            text_color=colors.WHITE,
        )
        current_x += keycap_width

        # Draw label
        self.canvas.draw_text(
            pixel_x=current_x,
            pixel_y=y,
            text=label,
            color=colors.WHITE,
        )

        # Calculate actual text width to return total consumed width
        label_width, _, _ = self.canvas.get_text_metrics(label)

        return keycap_width + label_width  # Total width consumed

    def _get_action_priority(self, action: ActionOption) -> int:
        """Get sort priority for an action. Lower values sort first.

        Priority order:
        - 0: PREFERRED attacks (weapon's intended use)
        - 1: Regular attacks (no special property)
        - 2: IMPROVISED attacks (not designed as weapon)
        """
        weapon = action.static_params.get("weapon")
        attack_mode = action.static_params.get("attack_mode")

        if weapon is None or attack_mode is None:
            return 1  # Non-combat actions get middle priority

        # Get the attack spec based on mode
        attack = None
        if attack_mode == "melee" and weapon.melee_attack:
            attack = weapon.melee_attack
        elif attack_mode == "ranged" and weapon.ranged_attack:
            attack = weapon.ranged_attack

        if attack is None:
            return 1

        # Check properties
        properties = attack.properties
        if WeaponProperty.PREFERRED in properties:
            return 0  # Highest priority
        if WeaponProperty.IMPROVISED in properties:
            return 2  # Lowest priority
        return 1  # Middle priority

    def _assign_hotkeys(self, actions: list[ActionOption]) -> None:
        """Assign hotkeys to actions with priority sorting and sticky persistence.

        Actions are first sorted by priority (PREFERRED first, IMPROVISED last),
        then hotkeys are assigned with preference for previous assignments.
        """
        if not actions:
            self._previous_hotkeys.clear()
            return

        # Sort actions by priority within each category
        actions.sort(key=lambda a: (a.category.value, self._get_action_priority(a)))

        # Build new hotkey assignments
        hotkey_chars = "abcdefghijklmnopqrstuvwxyz"
        used_hotkeys: set[str] = set()
        new_hotkeys: dict[str, str] = {}

        # First pass: try to preserve previous hotkey assignments
        for action in actions:
            if action.id in self._previous_hotkeys:
                prev_key = self._previous_hotkeys[action.id]
                if prev_key not in used_hotkeys and prev_key in hotkey_chars:
                    action.hotkey = prev_key
                    used_hotkeys.add(prev_key)
                    new_hotkeys[action.id] = prev_key

        # Second pass: assign new hotkeys to actions that don't have one
        hotkey_index = 0
        for action in actions:
            if action.hotkey is None:
                # Find next available hotkey
                while hotkey_index < len(hotkey_chars):
                    candidate = hotkey_chars[hotkey_index]
                    hotkey_index += 1
                    if candidate not in used_hotkeys:
                        action.hotkey = candidate
                        used_hotkeys.add(candidate)
                        new_hotkeys[action.id] = candidate
                        break

        # Update previous hotkeys for next frame
        self._previous_hotkeys = new_hotkeys

    def get_cache_key(self) -> tuple:
        """Cache key for rendering invalidation."""
        gw = self.controller.gw
        mouse_pos = str(gw.mouse_tile_location_on_map)

        # Include target actor position - if they move, cache invalidates
        target_actor_key = ""
        if gw.mouse_tile_location_on_map:
            mx, my = gw.mouse_tile_location_on_map
            target = gw.get_actor_at_location(mx, my)
            if target:
                target_actor_key = f"{target.x},{target.y}"

        player = gw.player
        player_pos = f"{player.x},{player.y}"
        has_items_at_feet = gw.has_pickable_items_at_location(player.x, player.y)
        current_tile_dimensions = self.controller.graphics.tile_dimensions
        # Include inventory revision to detect weapon switches and inventory changes,
        # ensuring the panel re-renders immediately when the player switches weapons.
        inventory_revision = player.inventory.revision
        # Include modifier revision to detect status effect changes (off-balance, etc.),
        # ensuring probabilities update immediately when player state changes.
        modifier_revision = player.modifiers.revision
        return (
            mouse_pos,
            target_actor_key,
            player_pos,
            has_items_at_feet,
            current_tile_dimensions,
            self.view_width_px,
            self.view_height_px,
            inventory_revision,
            modifier_revision,
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
        y_pixel = ascent + 5
        x_padding = 8

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

        # ALWAYS check for items at player's feet (regardless of mouse hover)
        player = self.controller.gw.player
        items_here = self.controller.gw.get_pickable_items_at_location(
            player.x, player.y
        )

        if items_here:
            # Show items at player's feet prominently
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="Items at your feet:",
                color=colors.YELLOW,
            )
            y_pixel += line_height

            # List items (up to 3)
            for item in items_here[:3]:
                self.canvas.draw_text(
                    pixel_x=x_padding + 10,
                    pixel_y=y_pixel - ascent,
                    text=f"• {item.name}",
                    color=colors.WHITE,
                )
                y_pixel += line_height

            if len(items_here) > 3:
                self.canvas.draw_text(
                    pixel_x=x_padding + 10,
                    pixel_y=y_pixel - ascent,
                    text=f"• ...and {len(items_here) - 3} more",
                    color=colors.GREY,
                )
                y_pixel += line_height

            y_pixel += line_height // 2

            # Show pickup prompt
            self._draw_keycap_with_label(
                x=x_padding,
                y=y_pixel - ascent,
                key="I",
                label="Pick up items",
            )
            y_pixel += line_height * 2

        # Actions section (contextual actions based on mouse hover)
        if self._cached_actions:
            # Sort by priority and assign hotkeys with sticky persistence
            self._assign_hotkeys(self._cached_actions)

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
                ActionCategory.STUNT: "Stunts",
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
                        # Handle thrown weapons first: "Throw Weapon at TargetName"
                        # to just "Throw Weapon" (must be before the generic replace)
                        action_name = action_name.replace(
                            f" at {self._cached_target_name}", ""
                        )
                        # Remove patterns like "Verb TargetName with Weapon"
                        # to just "Verb with Weapon"
                        action_name = action_name.replace(
                            f" {self._cached_target_name} ", " "
                        )

                    # Use helper method to draw keycap with action name
                    if action.hotkey:
                        action_width = self._draw_keycap_with_label(
                            x=x_padding + 20,
                            y=y_pixel - ascent,
                            key=action.hotkey,
                            label=action_name,
                        )
                    else:
                        # Draw empty keycap placeholder
                        current_x = x_padding + 20
                        keycap_width = draw_keycap(
                            canvas=self.canvas,
                            pixel_x=current_x,
                            pixel_y=y_pixel - ascent,
                            key=" ",
                            bg_color=colors.BLACK,
                            border_color=colors.DARK_GREY,
                            text_color=colors.DARK_GREY,
                        )
                        current_x += keycap_width
                        self.canvas.draw_text(
                            pixel_x=current_x,
                            pixel_y=y_pixel - ascent,
                            text=action_name,
                            color=colors.WHITE,
                        )
                        action_width = (
                            current_x + len(action_name) * 8
                        )  # Rough estimate

                    # Draw success probability in grey if available
                    if action.success_probability is not None:
                        prob_percent = int(action.success_probability * 100)
                        prob_text = f" ({prob_percent}%)"

                        # Position probability after the keycap + action text
                        prob_x = x_padding + 20 + action_width
                        self.canvas.draw_text(
                            pixel_x=prob_x,
                            pixel_y=y_pixel - ascent,
                            text=prob_text,
                            color=colors.GREY,
                        )

                    y_pixel += line_height

                y_pixel += line_height // 2  # Small spacing between categories

        elif self._cached_target_name is None:
            # Show helpful hints only when no items at feet and no mouse target
            if not items_here:
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
            # [Space] Action menu - use helper method
            self._draw_keycap_with_label(
                x=x_padding + 20,
                y=y_pixel - ascent,
                key="Space",
                label="Action menu",
            )
            y_pixel += line_height

            # [I] Inventory - use helper method
            self._draw_keycap_with_label(
                x=x_padding + 20,
                y=y_pixel - ascent,
                key="I",
                label="Inventory",
            )
            y_pixel += line_height

            # [Right-click] Context - use helper method
            self._draw_keycap_with_label(
                x=x_padding + 20,
                y=y_pixel - ascent,
                key="R-Click",
                label="Context",
            )
            y_pixel += line_height

            # [?] Help - use helper method
            self._draw_keycap_with_label(
                x=x_padding + 20,
                y=y_pixel - ascent,
                key="?",
                label="Help",
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

                # Get item-specific actions from discovery system
                context = self.discovery.context_builder.build_context(
                    self.controller, gw.player
                )
                # Create item pickup actions based on distance
                distance = max(abs(x - gw.player.x), abs(y - gw.player.y))

                if distance == 0:
                    # Player is standing on items - G key works
                    self._cached_actions = []
                else:
                    # Player needs to move - create "Walk to" action
                    from catley.game.actions.discovery import (
                        ActionCategory,
                        ActionOption,
                    )

                    def create_pathfind_and_pickup(item_x: int, item_y: int):
                        def pathfind_and_pickup():
                            from catley.game.actions.misc import (
                                PickupItemsAtLocationIntent,
                            )
                            from catley.util.pathfinding import find_local_path

                            gm = self.controller.gw.game_map
                            # Find path directly to the item location
                            path = find_local_path(
                                gm,
                                self.controller.gw.actor_spatial_index,
                                gw.player,
                                (gw.player.x, gw.player.y),
                                (item_x, item_y),
                            )
                            if path:
                                # Create intent to pickup items when we arrive
                                pickup_intent = PickupItemsAtLocationIntent(
                                    self.controller, gw.player
                                )
                                self.controller.start_actor_pathfinding(
                                    gw.player,
                                    (item_x, item_y),
                                    final_intent=pickup_intent,
                                )
                                return True
                            return False

                        return pathfind_and_pickup

                    pickup_action = ActionOption(
                        id="walk-and-pickup",
                        name="Walk to and pick up",
                        description="Move to the items and pick them up",
                        category=ActionCategory.ITEMS,
                        action_class=None,  # type: ignore[arg-type]
                        requirements=[],
                        static_params={},
                        execute=create_pathfind_and_pickup(x, y),
                    )
                    self._cached_actions = [pickup_action]
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
            elif isinstance(target_actor, Container):
                # Container - show search action via environment discovery
                self._cached_target_description = None
                context = self.discovery.context_builder.build_context(
                    self.controller, gw.player
                )
                env_discovery = self.discovery.environment_discovery
                self._cached_actions = (
                    env_discovery.discover_environment_actions_for_tile(
                        self.controller,
                        gw.player,
                        context,
                        target_actor.x,
                        target_actor.y,
                    )
                )
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
