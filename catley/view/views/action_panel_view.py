"""Action panel view that displays target info and available actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catley import colors, config
from catley.backends.pillow.canvas import PillowImageCanvas
from catley.environment import tile_types
from catley.game.actions.discovery import ActionDiscovery, ActionOption
from catley.game.actors import Actor, Character
from catley.game.actors.container import Container, ItemPile
from catley.game.countables import CountableType
from catley.game.items.properties import WeaponProperty
from catley.types import InterpolationAlpha
from catley.util.caching import ResourceCache
from catley.view.render.graphics import GraphicsContext
from catley.view.ui.selectable_list import (
    LayoutMode,
    SelectableListRenderer,
    SelectableRow,
)

from .base import TextView

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.discovery import TargetType


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
        self._cached_default_action_id: str | None = None  # The right-click default
        self._cached_is_selected: bool = False  # True if target is selected (not hover)

        # Sticky hotkeys: track action id -> hotkey for continuity
        self._previous_hotkeys: dict[str, str] = {}

        # Separate renderers for each section to avoid shared hovered_index state.
        # Using one renderer for multiple sections causes cross-section highlighting
        # bugs where hovered_index from one section affects rendering of another.
        self._pickup_renderer = SelectableListRenderer(self.canvas, LayoutMode.KEYCAP)
        self._actions_renderer = SelectableListRenderer(self.canvas, LayoutMode.KEYCAP)
        self._controls_renderer = SelectableListRenderer(self.canvas, LayoutMode.KEYCAP)

        # View pixel dimensions will be calculated when resize() is called
        self.view_width_px = 0
        self.view_height_px = 0

        # Override the cache from the base class for pixel-based rendering
        self._texture_cache = ResourceCache[tuple, Any](
            name=f"{self.__class__.__name__}Render",
            max_size=1,
            on_evict=lambda tex: self.controller.graphics.release_texture(tex),
        )

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

    def _assign_hotkeys(
        self, actions: list[ActionOption], default_action_id: str | None = None
    ) -> None:
        """Assign hotkeys to actions with priority sorting and sticky persistence.

        Args:
            actions: The actions to assign hotkeys to.
            default_action_id: If provided, this action gets "a" hotkey first.

        Actions are first sorted by priority (PREFERRED first, IMPROVISED last),
        then hotkeys are assigned with preference for previous assignments.
        The default action always gets "a" to ensure it's the primary hotkey.
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

        # First: assign "a" to the default action (if it exists in the list)
        if default_action_id:
            for action in actions:
                if action.id == default_action_id or action.id.startswith(
                    default_action_id + "-"
                ):
                    action.hotkey = "a"
                    used_hotkeys.add("a")
                    new_hotkeys[action.id] = "a"
                    break

        # Second pass: try to preserve previous hotkey assignments
        for action in actions:
            if action.hotkey is not None:
                # Already assigned (default action)
                continue
            if action.id in self._previous_hotkeys:
                prev_key = self._previous_hotkeys[action.id]
                if prev_key not in used_hotkeys and prev_key in hotkey_chars:
                    action.hotkey = prev_key
                    used_hotkeys.add(prev_key)
                    new_hotkeys[action.id] = prev_key

        # Third pass: assign new hotkeys to actions that don't have one
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

        # Include selected target - this is the primary source when set
        selected_target_key = ""
        selected_target = self.controller.selected_target
        if selected_target is not None:
            selected_target_key = f"sel:{selected_target.x},{selected_target.y}"

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
            selected_target_key,
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

        # Update cached data
        self._update_cached_data()

        # Get font metrics for proper line spacing
        ascent, descent = self.canvas.get_font_metrics()
        line_height = ascent + descent

        # Start rendering from top with some padding
        y_pixel = ascent + 5
        x_padding = 8

        # Check for items/countables at player's feet (rendered later)
        player = self.controller.gw.player
        items_here = self.controller.gw.get_pickable_items_at_location(
            player.x, player.y
        )
        # Also check for countables in any ItemPile at player's location
        # Use spatial index since get_actor_at_location prioritizes blocking actors
        countables_here: dict[CountableType, int] = {}
        for actor in self.controller.gw.actor_spatial_index.get_at_point(
            player.x, player.y
        ):
            if isinstance(actor, ItemPile):
                countables_here = actor.inventory.countables
                break

        # Target name section (selected or hovered target)
        if self._cached_target_name:
            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text=self._cached_target_name,
                color=colors.YELLOW,
            )
            y_pixel += line_height

            # Show "Selected" indicator when target is selected (not just hovered)
            if self._cached_is_selected:
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text="Selected",
                    color=colors.GREY,
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

        # Actions section - flat list with default action first
        is_combat = self.controller.is_combat_mode()
        if self._cached_actions:
            # Prefix match: "search" matches "search-container-adjacent"
            default_action: ActionOption | None = None
            if self._cached_default_action_id:
                for action in self._cached_actions:
                    if (
                        action.id == self._cached_default_action_id
                        or action.id.startswith(self._cached_default_action_id + "-")
                    ):
                        default_action = action
                        break

            # Assign hotkeys to all actions (default action gets "a")
            self._assign_hotkeys(
                self._cached_actions,
                default_action.id if default_action else None,
            )

            # Sort actions: default first, then by priority
            sorted_actions: list[ActionOption] = []
            if default_action:
                sorted_actions.append(default_action)
            sorted_actions.extend(
                a for a in self._cached_actions if a is not default_action
            )

            # Convert actions to SelectableRows for unified rendering
            rows: list[SelectableRow] = []
            for action in sorted_actions[:10]:  # Limit to 10 actions
                # Clean up action name - remove redundant target name
                action_name = action.name
                if (
                    self._cached_target_name
                    and isinstance(self._cached_target_name, str)
                    and self._cached_target_name != "Combat Mode"
                ):
                    action_name = action_name.replace(
                        f" at {self._cached_target_name}", ""
                    )
                    action_name = action_name.replace(
                        f" {self._cached_target_name} ", " "
                    )

                # Build full label with probability if available
                full_label = action_name
                if action.success_probability is not None:
                    prob_percent = int(action.success_probability * 100)
                    full_label = f"{action_name} ({prob_percent}%)"

                # Check for selection state (combat mode)
                is_selected = action.static_params.get("_is_selected", False)

                # Selection indicator prefix: only in combat mode where it matters
                # In explore mode, no prefix needed - labels follow keycaps directly
                prefix_segments: list[tuple[str, colors.Color]] | None = None
                if is_combat:
                    if is_selected:
                        prefix_segments = [("▶ ", colors.YELLOW)]
                    else:
                        prefix_segments = [("  ", colors.WHITE)]

                # Choose text color based on selection state
                text_color = colors.YELLOW if is_selected else colors.WHITE

                rows.append(
                    SelectableRow(
                        text=full_label,
                        key=action.hotkey,
                        enabled=True,
                        color=text_color,
                        data=action,
                        prefix_segments=prefix_segments,
                    )
                )

            y_pixel = self._render_selectable_list(
                self._actions_renderer, rows, y_pixel, x_padding, line_height, ascent
            )
        else:
            # Actions section not rendered - clear stale hit areas to prevent
            # get_action_at_pixel from returning stale actions.
            self._actions_renderer.clear_hit_areas()

        # Items at player's feet section (after target+actions, as secondary context)
        if items_here or countables_here:
            from catley.game.countables import get_countable_display_name

            y_pixel += line_height  # Spacing before items section

            self.canvas.draw_text(
                pixel_x=x_padding,
                pixel_y=y_pixel - ascent,
                text="At your feet:",
                color=colors.YELLOW,
            )
            y_pixel += line_height

            # Build list of display strings (items + countables)
            display_items: list[str] = [item.name for item in items_here]
            for countable_type, quantity in countables_here.items():
                display_items.append(
                    get_countable_display_name(countable_type, quantity)
                )

            # List items (up to 3)
            for name in display_items[:3]:
                self.canvas.draw_text(
                    pixel_x=x_padding + 10,
                    pixel_y=y_pixel - ascent,
                    text=f"• {name}",
                    color=colors.WHITE,
                )
                y_pixel += line_height

            if len(display_items) > 3:
                self.canvas.draw_text(
                    pixel_x=x_padding + 10,
                    pixel_y=y_pixel - ascent,
                    text=f"• ...and {len(display_items) - 3} more",
                    color=colors.GREY,
                )
                y_pixel += line_height

            y_pixel += line_height // 2

            # Show pickup prompt
            pickup_row = [
                SelectableRow(
                    key="I",
                    text="Pick up items",
                    execute=self._on_inventory_click,
                )
            ]
            y_pixel = self._render_selectable_list(
                self._pickup_renderer,
                pickup_row,
                y_pixel,
                x_padding,
                line_height,
                ascent,
            )
            y_pixel += line_height  # Spacing after items section
        else:
            # Pickup section not rendered - clear stale hit areas to prevent
            # clicks from triggering the pickup callback.
            self._pickup_renderer.clear_hit_areas()

        # Show hints and controls only when no target is selected/hovered
        if self._cached_target_name is None and not self._cached_actions:
            # Show helpful hints only when no items at feet and no mouse target
            if not items_here:
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text="Click to select a",
                    color=colors.GREY,
                )
                y_pixel += line_height
                self.canvas.draw_text(
                    pixel_x=x_padding,
                    pixel_y=y_pixel - ascent,
                    text="target for actions",
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

            # Use SelectableListRenderer for controls section with execute callbacks
            control_rows: list[SelectableRow] = [
                SelectableRow(
                    key="Space",
                    text="Action menu",
                    execute=self._on_action_menu_click,
                ),
                SelectableRow(
                    key="I",
                    text="Inventory",
                    execute=self._on_inventory_click,
                ),
                SelectableRow(
                    key="R-Click",
                    text="Quick action",
                    execute=self._on_quick_action_click,
                ),
                SelectableRow(
                    key="?",
                    text="Help",
                    execute=self._on_help_click,
                ),
            ]
            y_pixel = self._render_selectable_list(
                self._controls_renderer,
                control_rows,
                y_pixel,
                x_padding,
                line_height,
                ascent,
            )
        else:
            # Controls section not rendered - clear stale hit areas to prevent
            # clicks in the actions area from accidentally triggering controls.
            self._controls_renderer.clear_hit_areas()

    def _update_cached_data(self) -> None:
        """Update cached target information and available actions.

        Target priority (outside combat mode):
        1. selected_target - sticky selection from left-click
        2. Mouse hover position - for tile/item inspection
        """
        # In combat mode, show action-centric panel instead of target-centric
        if self.controller.is_combat_mode():
            self._update_combat_mode_data()
            return

        gw = self.controller.gw

        # Priority 1: Use selected_target if set (sticky selection)
        selected_target = self.controller.selected_target
        if (
            selected_target is not None
            and selected_target in gw.actors
            and gw.game_map.visible[selected_target.x, selected_target.y]
        ):
            self._cached_is_selected = True
            self._populate_actor_target_data(selected_target)
            return

        # Priority 2: Mouse hover position (not selected)
        self._cached_is_selected = False
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
                    from catley.game.actions.misc import PickupItemsPlan

                    def create_pathfind_and_pickup(item_x: int, item_y: int):
                        def pathfind_and_pickup():
                            return self.controller.start_plan(
                                gw.player,
                                PickupItemsPlan,
                                target_position=(item_x, item_y),
                            )

                        return pathfind_and_pickup

                    pickup_action = ActionOption(
                        id="pickup-walk",  # Must start with "pickup" for default action
                        name="Walk to and pick up",
                        description="Move to the items and pick them up",
                        category=ActionCategory.ITEMS,
                        action_class=None,
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
        from catley.game.actions.discovery import classify_target

        gw = self.controller.gw
        # Use display_name for ItemPiles (handles countables correctly)
        if isinstance(target_actor, ItemPile):
            self._cached_target_name = target_actor.display_name
        else:
            self._cached_target_name = target_actor.name

        # Determine the default action for this target (for right-click indicator)
        target_type = classify_target(self.controller, target_actor)
        self._cached_default_action_id = self._get_default_action_id_for_type(
            target_type
        )

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
        elif isinstance(target_actor, ItemPile):
            # ItemPile - show "Walk to and pick up" action
            items = list(target_actor.inventory)
            has_countables = bool(target_actor.inventory.countables)
            if len(items) == 1 and not has_countables:
                self._cached_target_description = "An item on the ground"
            else:
                self._cached_target_description = "On the ground"

            # Create pickup action based on distance
            distance = max(
                abs(target_actor.x - gw.player.x),
                abs(target_actor.y - gw.player.y),
            )

            if distance == 0:
                # Player is standing on items - G key works, no action needed
                self._cached_actions = []
            else:
                # Player needs to move - create "Walk to and pick up" action
                from catley.game.actions.discovery import ActionCategory
                from catley.game.actions.misc import PickupItemsPlan

                item_x, item_y = target_actor.x, target_actor.y

                def create_pathfind_and_pickup(x: int, y: int):
                    def pathfind_and_pickup():
                        return self.controller.start_plan(
                            gw.player,
                            PickupItemsPlan,
                            target_position=(x, y),
                        )

                    return pathfind_and_pickup

                pickup_action = ActionOption(
                    id="pickup-walk",  # Must start with "pickup" for default action
                    name="Walk to and pick up",
                    description="Move to the items and pick them up",
                    category=ActionCategory.ITEMS,
                    action_class=None,
                    requirements=[],
                    static_params={},
                    execute=create_pathfind_and_pickup(item_x, item_y),
                )
                self._cached_actions = [pickup_action]
        else:
            self._cached_target_description = None
            self._cached_actions = []

    def _get_default_action_id_for_type(
        self, target_type: TargetType | None
    ) -> str | None:
        """Map target type to the action ID that matches the default action."""
        from catley.game.actions.discovery.types import TargetType as TT

        if target_type is None:
            return None

        # Map target types to action ID patterns used by the discovery system
        match target_type:
            case TT.NPC:
                return "talk"  # TalkIntent
            case TT.CONTAINER:
                return "search"  # SearchContainerIntent
            case TT.DOOR_CLOSED:
                return "open"  # OpenDoorIntent
            case TT.DOOR_OPEN:
                return "close"  # CloseDoorIntent
            case TT.ITEM_PILE:
                return "pickup"  # PickupItemsAtLocationIntent
            case TT.FLOOR:
                return "walk"  # Pathfinding
            case _:
                return None

    def _update_combat_mode_data(self) -> None:
        """Update cached data for combat mode's action-centric display.

        In combat mode, the panel shows available combat actions (Attack, Push,
        etc.) rather than target-specific actions. The player selects an action
        first, then clicks a target to execute it.

        Probabilities are shown in the cursor tooltip overlay, not here.
        """
        # Combat actions are different from explore actions - don't carry over
        # hotkeys. This prevents the explore-mode hotkey (e.g., "B" for Talk)
        # from incorrectly selecting a combat action.
        self._previous_hotkeys.clear()

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

    def _render_selectable_list(
        self,
        renderer: SelectableListRenderer,
        rows: list[SelectableRow],
        y_pixel: int,
        x_padding: int,
        line_height: int,
        ascent: int,
    ) -> int:
        """Render a list of selectable rows with standard parameters.

        Args:
            renderer: The renderer to use for this section.
            rows: The rows to render.
            y_pixel: Starting y position.
            x_padding: Horizontal padding from panel edge.
            line_height: Height of each line in pixels.
            ascent: Font ascent in pixels.

        Returns:
            The y position after rendering.
        """
        renderer.rows = rows
        return renderer.render(
            x_start=x_padding,
            y_start=y_pixel,
            max_width=self.view_width_px - x_padding * 2,
            line_height=line_height,
            ascent=ascent,
            row_gap=0,
        )

    # -------------------------------------------------------------------------
    # Control Row Callbacks
    # -------------------------------------------------------------------------

    def _on_action_menu_click(self) -> None:
        """Open the action browser menu when 'Action menu' control is clicked."""
        from catley.view.ui.action_browser_menu import ActionBrowserMenu
        from catley.view.ui.commands import OpenMenuUICommand

        OpenMenuUICommand(self.controller, ActionBrowserMenu).execute()

    def _on_inventory_click(self) -> None:
        """Open inventory when 'Inventory' control is clicked."""
        from catley.view.ui.commands import open_inventory_or_loot

        open_inventory_or_loot(self.controller)

    def _on_quick_action_click(self) -> None:
        """Show a message when 'Quick action' control is clicked.

        R-Click is context-dependent, so clicking on it just shows a hint.
        """
        from catley.events import MessageEvent, publish_event

        publish_event(
            MessageEvent("Right-click on a target for quick action", colors.GREY)
        )

    def _on_help_click(self) -> None:
        """Open the help menu when 'Help' control is clicked."""
        from catley.view.ui.commands import OpenMenuUICommand
        from catley.view.ui.help_menu import HelpMenu

        OpenMenuUICommand(self.controller, HelpMenu).execute()

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
        # Only the actions renderer contains ActionOptions
        row = self._actions_renderer.get_row_at_pixel(px, py)
        if row is not None and isinstance(row.data, ActionOption):
            return row.data
        return None

    def execute_at_pixel(self, px: int, py: int) -> bool:
        """Execute the callback for the row at the given pixel coordinates.

        This handles both action rows (via ActionOption) and control rows
        (via their execute callbacks).

        Args:
            px: X pixel coordinate relative to the action panel.
            py: Y pixel coordinate relative to the action panel.

        Returns:
            True if a callback was executed, False otherwise.
        """
        # Check all renderers for executable rows
        for renderer in (
            self._pickup_renderer,
            self._actions_renderer,
            self._controls_renderer,
        ):
            if renderer.execute_at_pixel(px, py):
                return True
        return False

    def update_hover_from_pixel(self, px: int, py: int) -> bool:
        """Update hover state from pixel coordinates.

        Args:
            px: X pixel coordinate relative to the action panel.
            py: Y pixel coordinate relative to the action panel.

        Returns:
            True if hover state changed (needs redraw).
        """
        # Update all renderers and return True if any changed.
        # Each renderer maintains its own hovered_index for its section.
        changed = False
        for renderer in (
            self._pickup_renderer,
            self._actions_renderer,
            self._controls_renderer,
        ):
            if renderer.update_hover_from_pixel(px, py):
                changed = True
        return changed

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
