"""Action panel view that displays target info and available actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.environment import tile_types
from catley.game.actions.discovery import ActionCategory, ActionDiscovery, ActionOption
from catley.game.actors import Actor, Character
from catley.game.actors.container import Container
from catley.game.items.properties import WeaponProperty
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.ui_utils import draw_keycap

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller


class ActionPanelView(TextView):
    """Left sidebar panel showing target info and available actions."""

    def __init__(self, controller: Controller) -> None:
        super().__init__()
        self.controller = controller
        self.canvas = PillowImageCanvas(
            controller.graphics,
            font_path=config.UI_FONT_PATH,
            font_size=config.ACTION_PANEL_FONT_SIZE,
            line_spacing=1.0,
        )
        self.discovery = ActionDiscovery()
        self._cached_actions: list[ActionOption] = []
        self._cached_target_name: str | None = None
        self._cached_target_description: str | None = None

        # Sticky hotkeys: track action id -> hotkey for continuity
        self._previous_hotkeys: dict[str, str] = {}

        # Hit areas for mouse click detection: (x1, y1, x2, y2, action)
        self._action_hit_areas: list[tuple[int, int, int, int, ActionOption]] = []

        # View pixel dimensions will be calculated when resize() is called
        self.view_width_px = 0
        self.view_height_px = 0

        # Override the cache from the base class for pixel-based rendering
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render", max_size=1
        )

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        """Wrap text to fit within max_width pixels.

        Args:
            text: The text to wrap
            max_width: Maximum width in pixels

        Returns:
            List of lines that fit within max_width
        """
        if max_width <= 0:
            return [text]

        words = text.split(" ")
        lines: list[str] = []
        current_line = ""

        for word in words:
            test_line = f"{current_line} {word}".strip()
            width, _, _ = self.canvas.get_text_metrics(test_line)

            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines if lines else [text]

    def _draw_keycap_with_label(
        self,
        x: int,
        y: int,
        key: str,
        label: str,
        label_x: int | None = None,
        max_width: int | None = None,
    ) -> tuple[int, int]:
        """Draw a keycap with label text, wrapping if needed.

        Args:
            x: X position for keycap
            y: Y position for keycap and label
            key: Key text to display in keycap
            label: Label text to display after keycap
            label_x: If provided, draw label at this fixed x position for alignment
            max_width: If provided, wrap label to fit within this width from label_x

        Returns:
            Tuple of (width consumed on first line, total height consumed in pixels)
        """
        # Draw keycap
        keycap_width = draw_keycap(
            canvas=self.canvas,
            pixel_x=x,
            pixel_y=y,
            key=key,
            bg_color=colors.DARK_GREY,
            border_color=colors.GREY,
            text_color=colors.WHITE,
        )

        # Draw label at fixed position if provided, otherwise right after keycap
        actual_label_x = label_x if label_x is not None else x + keycap_width

        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent

        # Calculate available width for label
        if max_width is not None:
            available_width = max_width - actual_label_x
        else:
            # 8px right padding
            available_width = self.view_width_px - actual_label_x - 8

        # Wrap text if needed
        lines = self._wrap_text(label, available_width)

        # Draw each line
        current_y = y
        for line in lines:
            self.canvas.draw_text(
                pixel_x=actual_label_x,
                pixel_y=current_y,
                text=line,
                color=colors.WHITE,
            )
            current_y += line_height

        # Calculate width of first line for return value
        if lines:
            first_line_width, _, _ = self.canvas.get_text_metrics(lines[0])
        else:
            first_line_width = 0
        total_height = line_height * len(lines)

        return keycap_width + first_line_width, total_height

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

        # Include contextual target position so movement/hover changes invalidate.
        contextual_target_key = ""
        contextual_target = self.controller.contextual_target
        if contextual_target is not None:
            contextual_target_key = f"{contextual_target.x},{contextual_target.y}"

        # Include mouse target actor position - if they move, cache invalidates
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

        # Include combat mode state for action-centric rendering
        is_combat = self.controller.is_combat_mode()
        selected_action_id = ""
        if is_combat and self.controller.combat_mode.selected_action:
            selected_action_id = self.controller.combat_mode.selected_action.id

        return (
            mouse_pos,
            contextual_target_key,
            target_actor_key,
            player_pos,
            has_items_at_feet,
            current_tile_dimensions,
            self.view_width_px,
            self.view_height_px,
            inventory_revision,
            modifier_revision,
            is_combat,
            selected_action_id,
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

        # Clear hit areas for fresh tracking
        self._action_hit_areas.clear()

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
            _, pickup_height = self._draw_keycap_with_label(
                x=x_padding,
                y=y_pixel - ascent,
                key="I",
                label="Pick up items",
            )
            y_pixel += pickup_height + line_height

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
                    color=colors.GREY,
                )
                y_pixel += line_height

                # Action items
                for action in actions[:8]:  # More actions can fit with smaller font
                    # Check if this action is selected (in combat mode)
                    is_selected = action.static_params.get("_is_selected", False)

                    # Selection indicator prefix: ▶ for selected, spaces for alignment
                    selection_indicator = "▶ " if is_selected else "  "
                    indicator_width, _, _ = self.canvas.get_text_metrics(
                        selection_indicator
                    )

                    # Track this action's hit area (full row from padding to edge)
                    hit_y_start = y_pixel - ascent - 2
                    hit_y_end = y_pixel + line_height + 2
                    self._action_hit_areas.append(
                        (
                            x_padding,
                            hit_y_start,
                            self.view_width_px - x_padding,
                            hit_y_end,
                            action,
                        )
                    )

                    # Clean up action name - remove redundant target name
                    action_name = action.name
                    # Don't strip "Combat Mode" header since that's not a target name
                    if (
                        self._cached_target_name
                        and isinstance(self._cached_target_name, str)
                        and self._cached_target_name != "Combat Mode"
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

                    # Build full label with probability if available
                    full_label = action_name
                    if action.success_probability is not None:
                        prob_percent = int(action.success_probability * 100)
                        full_label = f"{action_name} ({prob_percent}%)"

                    # Choose text color based on selection state
                    text_color = colors.YELLOW if is_selected else colors.WHITE

                    # Draw selection indicator (▶ for selected, spaces for alignment)
                    indicator_x = x_padding
                    indicator_color = colors.YELLOW if is_selected else colors.DARK_GREY
                    self.canvas.draw_text(
                        pixel_x=indicator_x,
                        pixel_y=y_pixel - ascent,
                        text=selection_indicator,
                        color=indicator_color,
                    )

                    # Draw keycap after the indicator
                    keycap_x = x_padding + indicator_width
                    if action.hotkey:
                        _, action_height = self._draw_keycap_with_label(
                            x=keycap_x,
                            y=y_pixel - ascent,
                            key=action.hotkey,
                            label=full_label,
                        )
                    else:
                        # Draw empty keycap placeholder
                        keycap_width = draw_keycap(
                            canvas=self.canvas,
                            pixel_x=keycap_x,
                            pixel_y=y_pixel - ascent,
                            key=" ",
                            bg_color=colors.BLACK,
                            border_color=colors.DARK_GREY,
                            text_color=colors.DARK_GREY,
                        )
                        self.canvas.draw_text(
                            pixel_x=keycap_x + keycap_width,
                            pixel_y=y_pixel - ascent,
                            text=full_label,
                            color=text_color,
                        )
                        action_height = line_height

                    y_pixel += action_height + line_height // 3  # Gap between actions

                y_pixel += line_height // 4  # Extra spacing between categories

        elif self._cached_target_name is None:
            # Show helpful hints only when no items at feet and no mouse target
            if not items_here:
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text="Hover over targets",
                    color=colors.GREY,
                )
                y_pixel += line_height
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text="to see actions",
                    color=colors.GREY,
                )
                y_pixel += line_height * 2

            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="Controls:",
                color=colors.GREY,
            )
            # Extra spacing after header before keycaps
            y_pixel += line_height + line_height // 3

            # Calculate keycap sizing for right-aligned keycaps
            keycap_size = int(line_height * 0.85)
            keycap_font_size = max(8, int(keycap_size * 0.65))
            keycap_internal_padding = 12
            keycap_gap = 12  # Gap after keycap (from draw_keycap)
            label_gap = 8  # Extra gap before label

            def get_keycap_width(key: str) -> int:
                """Calculate the width of a keycap (excluding gap after)."""
                text_width, _, _ = self.canvas.get_text_metrics(
                    key.upper(), font_size=keycap_font_size
                )
                return max(keycap_size, text_width + keycap_internal_padding)

            # Find widest keycap to set the right edge of keycap column
            control_keys = ["Space", "I", "R-Click", "?"]
            widest_keycap = max(get_keycap_width(k) for k in control_keys)
            # Right edge of keycap column (keycaps right-align to this)
            keycap_right_edge = x_padding + widest_keycap
            # Label column starts after gap
            label_column_x = keycap_right_edge + keycap_gap + label_gap

            # Helper to draw right-aligned keycap with label
            def draw_control(key: str, label: str) -> int:
                """Draw control and return height consumed."""
                nonlocal y_pixel
                kw = get_keycap_width(key)
                keycap_x = keycap_right_edge - kw
                _, height = self._draw_keycap_with_label(
                    x=keycap_x,
                    y=y_pixel - ascent,
                    key=key,
                    label=label,
                    label_x=label_column_x,
                )
                return height

            y_pixel += draw_control("Space", "Action menu")
            y_pixel += draw_control("I", "Inventory")
            y_pixel += draw_control("R-Click", "Context")
            draw_control("?", "Help")

    def _update_cached_data(self) -> None:
        """Update cached target information and available actions."""
        # In combat mode, show action-centric panel instead of target-centric
        if self.controller.is_combat_mode():
            self._update_combat_mode_data()
            return

        gw = self.controller.gw
        contextual_target = self.controller.contextual_target

        if (
            contextual_target is not None
            and contextual_target in gw.actors
            and gw.game_map.visible[contextual_target.x, contextual_target.y]
        ):
            self._populate_actor_target_data(contextual_target)
            return

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
            self._populate_actor_target_data(target_actor)
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

    def _populate_actor_target_data(self, target_actor: Actor) -> None:
        """Populate action panel data for a target actor."""
        gw = self.controller.gw
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
            self._cached_actions = env_discovery.discover_environment_actions_for_tile(
                self.controller,
                gw.player,
                context,
                target_actor.x,
                target_actor.y,
            )
        else:
            self._cached_target_description = None
            self._cached_actions = []

    def _update_combat_mode_data(self) -> None:
        """Update cached data for combat mode's action-centric display.

        In combat mode, the panel shows available combat actions (Attack, Push,
        etc.) rather than target-specific actions. The player selects an action
        first, then clicks a target to execute it.

        Probabilities are shown in the cursor tooltip overlay, not here.
        """
        combat_mode = self.controller.combat_mode

        # Get player's combat actions without target (no probabilities)
        # Probabilities are displayed in the cursor tooltip instead
        self._cached_actions = combat_mode.get_available_combat_actions()

        # Simple header for combat mode
        self._cached_target_name = "Combat Mode"
        self._cached_target_description = None

        # Mark the selected action for highlighting during render
        selected = combat_mode.selected_action
        for action in self._cached_actions:
            # Store selection state in a way the render can check
            action.static_params["_is_selected"] = (
                selected is not None and action.id == selected.id
            )

    def get_hotkeys(self) -> dict[str, ActionOption]:
        """Get current hotkey mappings for direct execution."""
        hotkeys = {}
        for action in self._cached_actions:
            if action.hotkey:
                hotkeys[action.hotkey.lower()] = action
        return hotkeys

    def get_action_at_pixel(self, px: int, py: int) -> ActionOption | None:
        """Return the action at the given pixel coordinates, or None.

        Args:
            px: X pixel coordinate relative to the action panel.
            py: Y pixel coordinate relative to the action panel.

        Returns:
            The ActionOption at the given position, or None if no action.
        """
        for x1, y1, x2, y2, action in self._action_hit_areas:
            if x1 <= px < x2 and y1 <= py < y2:
                return action
        return None

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
