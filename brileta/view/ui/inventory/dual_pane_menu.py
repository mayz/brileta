"""Dual-pane inventory/loot interface.

A unified Fallout-style inventory screen that shows player inventory on the left,
container/ground items on the right (when looting), and item details below.
Replaces the separate InventoryMenu and PickupMenu.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Final

from brileta import colors, input_events
from brileta.backends.pillow.canvas import PillowImageCanvas
from brileta.game.actors import Condition
from brileta.game.countables import CountableType
from brileta.game.enums import ItemSize
from brileta.game.items.item_core import Item
from brileta.view.ui.inventory.display_utils import (
    group_items_for_display,  # noqa: F401 (re-exported)
)
from brileta.view.ui.inventory.item_detail_panel import ItemDetailPanel
from brileta.view.ui.inventory.item_pane import InventoryPane, LootPane
from brileta.view.ui.inventory.item_transfer import (
    ActorInventorySource,  # noqa: F401 (re-exported for external consumers)
    ExternalInventory,
    ExternalSource,
    ItemTransferHandler,
)
from brileta.view.ui.overlays import Menu

if TYPE_CHECKING:
    from brileta.controller import Controller


class _MenuKeys:
    """KeySym constants for menu navigation keys.

    Using a class with class attributes ensures pattern matching treats
    these as value patterns (comparison) rather than capture patterns.
    """

    KEY_A: Final = input_events.KeySym(ord("a"))
    KEY_D: Final = input_events.KeySym(ord("d"))
    KEY_I: Final = input_events.KeySym(ord("i"))
    KEY_J: Final = input_events.KeySym(ord("j"))
    KEY_K: Final = input_events.KeySym(ord("k"))
    KEY_P: Final = input_events.KeySym(ord("p"))
    KEY_Q: Final = input_events.KeySym(ord("q"))
    KEY_T: Final = input_events.KeySym(ord("t"))
    KEY_U: Final = input_events.KeySym(ord("u"))


class PaneId(Enum):
    """Identifies which pane is active."""

    LEFT = "left"
    RIGHT = "right"


class DualPaneMenu(Menu):
    """Unified inventory/loot menu with dual panes and detail panel.

    When source is None, operates in inventory-only mode (right pane hidden).
    When source is an ExternalInventory, shows items at that location for looting.
    """

    # Layout constants (in tiles)
    TOTAL_WIDTH = 80
    SINGLE_PANE_WIDTH = 40  # inventory-only mode
    TOTAL_HEIGHT = 30
    LEFT_PANE_WIDTH = 35
    DIVIDER_WIDTH = 7
    RIGHT_PANE_WIDTH = 35
    DETAIL_HEIGHT = 7  # 2 header + 2 description + 1 gap + 1 stats + 1 gap
    HINT_HEIGHT = 2  # Extra row for whitespace above keycaps
    HEADER_HEIGHT = 1
    # Calculated: item list height = TOTAL_HEIGHT - HEADER - DETAIL - HINT - borders
    ITEM_LIST_HEIGHT = TOTAL_HEIGHT - HEADER_HEIGHT - DETAIL_HEIGHT - HINT_HEIGHT - 3

    def __init__(
        self,
        controller: Controller,
        source: ExternalSource = None,
    ) -> None:
        """Initialize the dual-pane menu.

        Args:
            controller: Game controller.
            source: External source to interact with - can be:
                - ExternalInventory: Position-based ground items
                - ActorInventorySource: Items from an actor (container/corpse)
                - None: Inventory-only mode (no right pane)
        """
        title = "Inventory" if source is None else "Loot"
        super().__init__(
            title, controller, width=self.TOTAL_WIDTH, max_height=self.TOTAL_HEIGHT
        )

        # Transfer handler for game logic (also owns the source reference)
        self.transfer_handler = ItemTransferHandler(
            controller, controller.gw.player, source
        )

        # Get inventory reference
        self.inventory = controller.gw.player.inventory

        # Pane components
        self.active_pane: PaneId = PaneId.LEFT
        self.left_pane = InventoryPane(self.inventory)
        self.right_pane = LootPane()

        # Detail panel component (manages header, description, stats, scrolling)
        self.detail = ItemDetailPanel()

    @property
    def source(self) -> ExternalSource:
        """External source for the right pane, stored in the transfer handler."""
        return self.transfer_handler.source

    @source.setter
    def source(self, value: ExternalSource) -> None:
        self.transfer_handler.source = value

    @property
    def _active_pane_obj(self) -> InventoryPane | LootPane:
        """Return the pane object for the currently active pane."""
        return self.left_pane if self.active_pane == PaneId.LEFT else self.right_pane

    def populate_options(self) -> None:
        """Populate both panes from their respective sources."""
        self._populate_left_pane()
        self._populate_right_pane()

        # Set initial detail item from active pane
        self._update_detail_from_cursor()

    def _populate_left_pane(self) -> None:
        """Delegate to left pane component."""
        self.left_pane.populate(self._equip_item)

    def _populate_right_pane(self) -> None:
        """Delegate to right pane component."""
        self.right_pane.populate(
            self.transfer_handler,
            self._transfer_to_inventory,
            self._transfer_countable_to_inventory,
        )

    def _update_detail_from_cursor(self) -> None:
        """Update detail_item based on current cursor position."""
        self.detail.update_from_cursor(self._active_pane_obj.current_option)
        # Update the generation context so the detail panel knows which pane is active
        self.detail.set_generate_context(
            is_left_pane=self.active_pane == PaneId.LEFT,
            inventory=self.inventory,
            transfer_handler=self.transfer_handler,
        )

    # -----------------------------------------------------------------------
    # Item transfer wrappers - delegate game logic to ItemTransferHandler,
    # then update UI state (pane contents, cursor position, detail panel).
    # -----------------------------------------------------------------------

    def _equip_item(self, item: Item) -> bool:
        """Equip or unequip an item, then refresh the left pane."""
        result = self.transfer_handler.equip_item(item)

        self._populate_left_pane()
        new_index = self.left_pane.find_item_index(item)
        if new_index is not None:
            self.left_pane.cursor = new_index
        self.left_pane.ensure_valid_cursor()
        self._update_detail_from_cursor()
        return result

    def _transfer_to_inventory(self, item: Item) -> bool:
        """Transfer one item from source to player inventory."""
        self.transfer_handler.transfer_to_inventory(item)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_right_transfer(item)
        return False

    def _transfer_all_to_inventory(self, item: Item) -> bool:
        """Transfer all items of the same type from source to inventory."""
        self.transfer_handler.transfer_all_to_inventory(item)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_right_transfer()
        return False

    def _transfer_countable_to_inventory(self, countable_type: CountableType) -> bool:
        """Transfer all countables of a type from source to inventory."""
        self.transfer_handler.transfer_countable_to_inventory(countable_type)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_right_transfer()
        return False

    def _transfer_to_container(self, item: Item) -> bool:
        """Transfer one item from player inventory to container/ground."""
        self.transfer_handler.transfer_to_container(item)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer(item)
        return False

    def _transfer_all_to_container(self, item: Item) -> bool:
        """Transfer all items of the same type from inventory to container."""
        self.transfer_handler.transfer_all_to_container(item)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer()
        return False

    def _transfer_countable_to_ground(self, countable_type: CountableType) -> bool:
        """Transfer all countables of a type from inventory to container/ground."""
        self.transfer_handler.transfer_countable_to_ground(countable_type)

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer()
        return False

    def _drop_item(self, item: Item) -> bool:
        """Drop an item to the ground, transitioning to dual-pane view."""
        self.transfer_handler.drop_item(item)

        player = self.controller.gw.player
        self.source = ExternalInventory(
            position=(player.x, player.y), label="On the ground"
        )
        self._calculate_dimensions()

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer(item)
        return False

    def _drop_all_of_type(self, item: Item) -> bool:
        """Drop all items of the same type to the ground."""
        # Check precondition so we don't transition to dual-pane on no-op
        if not any(
            i.item_type is item.item_type
            for i in self.controller.gw.player.inventory.get_items()
        ):
            return False

        self.transfer_handler.drop_all_of_type(item)

        player = self.controller.gw.player
        self.source = ExternalInventory(
            position=(player.x, player.y), label="On the ground"
        )
        self._calculate_dimensions()

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer()
        return False

    def _drop_countable(self, countable_type: CountableType) -> bool:
        """Drop all countables of a type to the ground."""
        # Check precondition so we don't transition to dual-pane on no-op
        if self.controller.gw.player.inventory.get_countable(countable_type) <= 0:
            return False

        self.transfer_handler.drop_countable(countable_type)

        player = self.controller.gw.player
        self.source = ExternalInventory(
            position=(player.x, player.y), label="On the ground"
        )
        self._calculate_dimensions()

        self._populate_left_pane()
        self._populate_right_pane()
        self._refresh_cursor_after_left_transfer()
        return False

    # -----------------------------------------------------------------------
    # Cursor fixup helpers for after transfers
    # -----------------------------------------------------------------------

    def _refresh_cursor_after_right_transfer(
        self, transferred_item: Item | None = None
    ) -> None:
        """Fix cursor after transferring from right pane to inventory."""
        if self.right_pane.has_selectable():
            self.right_pane.ensure_valid_cursor()
        else:
            self.active_pane = PaneId.LEFT
            if transferred_item is not None:
                new_index = self.left_pane.find_item_index(transferred_item)
                if new_index is not None:
                    self.left_pane.cursor = new_index
            self.left_pane.ensure_valid_cursor()
        self._update_detail_from_cursor()

    def _refresh_cursor_after_left_transfer(
        self, transferred_item: Item | None = None
    ) -> None:
        """Fix cursor after transferring from left pane to container/ground."""
        if self.left_pane.has_selectable():
            self.left_pane.ensure_valid_cursor()
        else:
            self.active_pane = PaneId.RIGHT
            if transferred_item is not None:
                new_index = self.right_pane.find_item_index(transferred_item)
                if new_index is not None:
                    self.right_pane.cursor = new_index
            self.right_pane.ensure_valid_cursor()
        self._update_detail_from_cursor()

    def _handle_drop_or_transfer(self) -> None:
        """Handle D/T/P key press: drop (single-pane) or transfer (dual-pane).

        In single-pane mode (inventory only):
            - Drops the item at player's feet and transitions to dual-pane

        In dual-pane mode:
            - Left pane: transfers item to container/ground
            - Right pane: transfers item to player inventory
        """
        option = self._active_pane_obj.current_option
        if option is None or not option.enabled:
            return

        if self.source is None:
            # Single-pane mode: drop item or countable
            if self.active_pane == PaneId.LEFT:
                if isinstance(option.data, Item):
                    self._drop_item(option.data)
                elif isinstance(option.data, CountableType):
                    self._drop_countable(option.data)
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane: transfer from inventory to container/ground
            if isinstance(option.data, Item):
                self._transfer_to_container(option.data)
            elif isinstance(option.data, CountableType):
                self._transfer_countable_to_ground(option.data)
        else:
            # Dual-pane: transfer from container/ground to inventory
            if isinstance(option.data, Item):
                self._transfer_to_inventory(option.data)
            elif isinstance(option.data, CountableType):
                self._transfer_countable_to_inventory(option.data)

    def _handle_drop_or_transfer_all(self) -> None:
        """Handle A key press: drop/transfer all items of the same type.

        For stacked items (Tiny size with same ItemType), drops or transfers
        all of them at once. For countables, behaves the same as D/T (already
        drops all). For non-stackable items, drops/transfers just the one.
        """
        option = self._active_pane_obj.current_option
        if option is None or not option.enabled:
            return

        if self.source is None:
            # Single-pane mode: drop all
            if self.active_pane == PaneId.LEFT:
                if isinstance(option.data, Item):
                    self._drop_all_of_type(option.data)
                elif isinstance(option.data, CountableType):
                    self._drop_countable(option.data)
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane: transfer all from inventory to container
            if isinstance(option.data, Item):
                self._transfer_all_to_container(option.data)
            elif isinstance(option.data, CountableType):
                self._transfer_countable_to_ground(option.data)
        else:
            # Dual-pane: transfer all from container to inventory
            if isinstance(option.data, Item):
                self._transfer_all_to_inventory(option.data)
            elif isinstance(option.data, CountableType):
                self._transfer_countable_to_inventory(option.data)

    def handle_input(self, event: input_events.InputEvent) -> bool:
        """Handle dual-pane navigation and actions.

        Key mappings:
        - Enter: Equip/unequip (left pane only, both modes)
        - U: Use consumable (left pane only, both modes)
        - D/T/P: Drop (single-pane) or Transfer (dual-pane) - all three keys
                 do the same action
        - Tab: Switch panes (dual-pane only)
        - Up/Down/J/K: Navigate item list
        - Left/Right: Scroll detail panel description (when overflow)
        - Esc/I/Q/Space: Close menu

        Mouse: The "..." scroll indicators turn white when hovered over,
        signaling interactivity. Click on them to scroll one line.
        """
        if not self.is_active:
            return False

        # Delegate to scrollable detail panel for Page Up/Down scrolling
        if (
            self.detail.scroll_panel is not None
            and self.detail.scroll_panel.handle_input(event)
        ):
            return True

        match event:
            # Tab switches panes (only if right pane exists)
            case input_events.KeyDown(sym=input_events.KeySym.TAB):
                if (
                    self.source is not None
                    and self.right_pane.options
                    and (
                        self.right_pane.has_selectable()
                        or self.active_pane == PaneId.RIGHT
                    )
                ):
                    self.active_pane = (
                        PaneId.RIGHT if self.active_pane == PaneId.LEFT else PaneId.LEFT
                    )
                    self._update_detail_from_cursor()
                return True

            # Up/Down arrow keys navigate within active pane
            case input_events.KeyDown(sym=input_events.KeySym.UP):
                self._move_cursor(-1)
                return True
            case input_events.KeyDown(sym=input_events.KeySym.DOWN):
                self._move_cursor(1)
                return True

            # Left/Right arrow keys scroll detail panel description
            case input_events.KeyDown(sym=input_events.KeySym.LEFT):
                if self.detail.scroll_panel and self.detail.scroll_panel.has_overflow():
                    self.detail.scroll_panel.scroll_up()
                return True
            case input_events.KeyDown(sym=input_events.KeySym.RIGHT):
                if self.detail.scroll_panel and self.detail.scroll_panel.has_overflow():
                    self.detail.scroll_panel.scroll_down()
                return True

            # Vim keys for navigation (j=down, k=up)
            case input_events.KeyDown(sym=_MenuKeys.KEY_J):
                self._move_cursor(1)
                return True
            case input_events.KeyDown(sym=_MenuKeys.KEY_K):
                self._move_cursor(-1)
                return True

            # D/T/P keys: Drop (single-pane) or Transfer (dual-pane)
            # All three keys do the same action
            case input_events.KeyDown(
                sym=_MenuKeys.KEY_D | _MenuKeys.KEY_T | _MenuKeys.KEY_P
            ):
                self._handle_drop_or_transfer()
                return True

            # A key: Drop All / Transfer All for stacked items
            case input_events.KeyDown(sym=_MenuKeys.KEY_A):
                self._handle_drop_or_transfer_all()
                return True

            # U key: Use consumable (left pane only)
            case input_events.KeyDown(sym=_MenuKeys.KEY_U):
                if self.active_pane == PaneId.LEFT:
                    option = self.left_pane.current_option
                    if (
                        option is not None
                        and option.enabled
                        and isinstance(option.data, Item)
                        and self.transfer_handler.use_consumable(option.data)
                    ):
                        self.hide()
                return True

            # Enter: Equip/unequip (left pane only)
            case input_events.KeyDown(sym=input_events.KeySym.RETURN):
                if self.active_pane == PaneId.LEFT:
                    option = self.left_pane.current_option
                    if (
                        option is not None
                        and option.enabled
                        and isinstance(option.data, Item)
                    ):
                        self._equip_item(option.data)
                # Enter does nothing on right pane
                return True

            # Escape closes menu
            case input_events.KeyDown(sym=input_events.KeySym.ESCAPE):
                self.hide()
                return True

            # I key closes menu (toggle behavior)
            case input_events.KeyDown(sym=_MenuKeys.KEY_I):
                self.hide()
                return True

            # Q key closes menu (instead of quitting)
            case input_events.KeyDown(sym=_MenuKeys.KEY_Q):
                self.hide()
                return True

            # Space closes menu
            case input_events.KeyDown(sym=input_events.KeySym.SPACE):
                self.hide()
                return True

            # Mouse motion updates hover state
            case input_events.MouseMotion():
                return self._handle_mouse_motion(event)

            # Mouse click
            case input_events.MouseButtonDown(button=input_events.MouseButton.LEFT):
                return self._handle_mouse_click(event)

            case input_events.KeyUp():
                return False  # Let KeyUp events pass through to modes

        return True  # Consume all other input while menu is active

    def _move_cursor(self, delta: int) -> None:
        """Move cursor within active pane."""
        self._active_pane_obj.move_cursor(delta)
        self._update_detail_from_cursor()

    def _activate_current_item(self) -> bool:
        """Activate the currently selected item (used for mouse clicks).

        Left pane: equip the item
        Right pane: transfer to inventory (pick up)

        Returns True if menu should close.
        """
        option = self._active_pane_obj.current_option
        if option is not None and option.enabled and option.action:
            result = option.action()
            return result is True
        return False

    def _handle_mouse_motion(self, event: input_events.MouseMotion) -> bool:
        """Handle mouse motion to update hover state and detect scroll zones."""
        mouse_px_x, mouse_px_y = event.position

        # Convert to menu-relative coordinates
        tile_w, tile_h = self.tile_dimensions
        menu_px_x = self.x_tiles * tile_w
        menu_px_y = self.y_tiles * tile_h
        rel_px_x = mouse_px_x - menu_px_x
        rel_px_y = mouse_px_y - menu_px_y

        char_w = self._char_width
        line_h = self._line_height
        if char_w == 0 or line_h == 0:
            return True

        # Convert to character/line position
        rel_char_x = int(rel_px_x // char_w)
        rel_line_y = int(rel_px_y // line_h)

        # Track hover zone for visual feedback (ellipsis color change)
        self.detail.hover_scroll_zone = self._get_description_scroll_zone(rel_line_y)

        # Determine which pane the mouse is over
        # Left pane: chars 1 to LEFT_PANE_WIDTH
        # Right pane: chars LEFT_PANE_WIDTH + DIVIDER_WIDTH to end
        # Items start at line 2: border (0) + header (1) + items start (2)
        item_start_y = self.HEADER_HEIGHT + 2

        if 1 <= rel_char_x < self.LEFT_PANE_WIDTH:
            # Over left pane - map visual line to option index
            # Layout: stored items at top, separator, equipment pinned at bottom
            option_line = rel_line_y - item_start_y
            option_index = self.left_pane.map_line_to_index(
                option_line, self.ITEM_LIST_HEIGHT
            )
            if option_index is not None:
                self.active_pane = PaneId.LEFT
                self.left_pane.cursor = option_index
                self._update_detail_from_cursor()
        elif (
            self.source is not None
            and self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            <= rel_char_x
            < self.TOTAL_WIDTH - 1
        ):
            # Over right pane - only select if in visible item list area
            option_line = rel_line_y - item_start_y
            if (
                0
                <= option_line
                < min(self.ITEM_LIST_HEIGHT, len(self.right_pane.options))
            ):
                self.active_pane = PaneId.RIGHT
                self.right_pane.cursor = option_line
                self._update_detail_from_cursor()

        return True

    def _get_description_scroll_zone(self, rel_line_y: int) -> int:
        """Determine scroll zone from a line position in the description area."""
        return self.detail.get_scroll_zone(
            rel_line_y, self.TOTAL_HEIGHT, self.DETAIL_HEIGHT, self.HINT_HEIGHT
        )

    def _handle_mouse_click(self, event: input_events.MouseButtonDown) -> bool:
        """Handle mouse click to select/transfer items or scroll description.

        Clicking on the "..." scroll indicators scrolls one line in that
        direction. The indicators turn white when hovered to signal
        interactivity.

        Clicks in the item list area activate the selected item.
        Clicks in the hint bar area are ignored.
        """
        mouse_px_x, mouse_px_y = event.position

        # Convert to menu-relative coordinates
        tile_w, tile_h = self.tile_dimensions
        menu_px_x = self.x_tiles * tile_w
        menu_px_y = self.y_tiles * tile_h
        rel_px_x = mouse_px_x - menu_px_x
        rel_px_y = mouse_px_y - menu_px_y

        if not (0 <= rel_px_x < self.pixel_width and 0 <= rel_px_y < self.pixel_height):
            self.hide()
            return True

        # Check which region was clicked
        char_w = self._char_width
        line_h = self._line_height
        if char_w == 0 or line_h == 0:
            return True

        rel_line_y = int(rel_px_y // line_h)

        # Detail panel region starts at separator line
        sep_y = self.TOTAL_HEIGHT - self.DETAIL_HEIGHT - self.HINT_HEIGHT - 1
        detail_start_y_px = (sep_y + 1) * line_h
        hint_start_y_px = (self.TOTAL_HEIGHT - self.HINT_HEIGHT) * line_h

        # Handle clicks in description area for scrolling
        if detail_start_y_px <= rel_px_y < hint_start_y_px:
            scroll_zone = self._get_description_scroll_zone(rel_line_y)
            # Helper returns non-zero only when _detail_panel exists
            if self.detail.scroll_panel is not None:
                if scroll_zone == -1:
                    self.detail.scroll_panel.scroll_up()
                elif scroll_zone == 1:
                    self.detail.scroll_panel.scroll_down()
            return True

        # If click is in hint bar area, ignore
        if rel_px_y >= hint_start_y_px:
            return True

        # Otherwise, click is in item list area - activate item
        result = self._activate_current_item()
        if result:
            self.hide()
        return True

    def _generate_item_detail(
        self, entity: Item | Condition | CountableType
    ) -> tuple[list[str], list[str], str | None]:
        """Generate detail data for an entity using the current menu context."""
        canvas = self.canvas if isinstance(self.canvas, PillowImageCanvas) else None
        return self.detail.generate_detail(
            entity,
            is_left_pane=self.active_pane == PaneId.LEFT,
            inventory=self.inventory,
            transfer_handler=self.transfer_handler,
            canvas=canvas,
            char_width=self._char_width,
            menu_width=self.width,
        )

    def _calculate_dimensions(self) -> None:
        """Calculate menu dimensions in pixels using font metrics."""
        # Get font metrics from the PillowImageCanvas
        assert isinstance(self.canvas, PillowImageCanvas)
        self._line_height = self.canvas.get_effective_line_height()
        self._char_width, _, _ = self.canvas.get_text_metrics(" ")

        if self._line_height == 0 or self._char_width == 0:
            return

        # Use narrower width for inventory-only mode (in characters)
        self.width = self.SINGLE_PANE_WIDTH if self.source is None else self.TOTAL_WIDTH
        self.height = self.TOTAL_HEIGHT

        # Calculate pixel dimensions from font metrics
        self.pixel_width = self.width * self._char_width
        self.pixel_height = self.height * self._line_height

        # Store tile dimensions for screen positioning
        self.tile_dimensions = self.controller.graphics.tile_dimensions
        tile_w, tile_h = self.tile_dimensions

        # Center on screen
        screen_pixel_w = self.controller.graphics.console_width_tiles * tile_w
        screen_pixel_h = self.controller.graphics.console_height_tiles * tile_h

        # Calculate centered pixel position, then convert to tiles
        center_px_x = (screen_pixel_w - self.pixel_width) // 2
        center_px_y = (screen_pixel_h - self.pixel_height) // 2

        self.x_tiles = center_px_x // tile_w if tile_w > 0 else 0
        self.y_tiles = center_px_y // tile_h if tile_h > 0 else 0

        # Configure the detail panel for description scrolling
        # Layout: 2 header + 2 description + 1 gap + 1 stats + 1 gap = 7 lines
        panel_width = self.width - 4  # 2 chars margin on each side
        panel_max_lines = 2  # 2 description lines visible (scroll for more)
        self.detail.configure(panel_max_lines, panel_width)

        # Create list renderers for item panes (lazily, uses INLINE mode)
        self.left_pane.ensure_renderer(self.canvas)
        self.right_pane.ensure_renderer(self.canvas)

    # Visual constants for menu styling
    MENU_BG_COLOR: colors.Color = (15, 15, 15)  # subtle dark grey, not pure black

    def draw_content(self) -> None:
        """Render the complete dual-pane interface using pixel-based layout."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)

        char_w = self._char_width
        line_h = self._line_height

        # Fill entire background with black
        self.canvas.draw_rect(
            pixel_x=0,
            pixel_y=0,
            width=self.pixel_width,
            height=self.pixel_height,
            color=colors.BLACK,
            fill=True,
        )

        # Draw double-line box frame using inherited method
        self._draw_box_frame()

        # Fill interior with dark grey (slightly lighter than pure black)
        interior_px_x = char_w
        interior_px_y = line_h
        interior_width = (self.width - 2) * char_w
        interior_height = (self.height - 2) * line_h
        self.canvas.draw_rect(
            pixel_x=interior_px_x,
            pixel_y=interior_px_y,
            width=interior_width,
            height=interior_height,
            color=self.MENU_BG_COLOR,
            fill=True,
        )

        # Draw pane headers
        self._draw_pane_headers()

        # Draw left pane items
        self._draw_left_pane_items()

        # Draw divider and right pane (only in loot mode)
        if self.source is not None:
            self._draw_divider()
            self._draw_right_pane_items()

        # Draw detail panel
        self._draw_detail_panel()

        # Draw hint bar
        self._draw_hint_bar()

    def _draw_pane_headers(self) -> None:
        """Draw the capacity bar headers for both panes."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height

        player = self.controller.gw.player
        used_space = player.inventory.get_used_inventory_slots()
        total_slots = player.inventory.total_inventory_slots
        slots_over = player.inventory.get_slots_over_capacity()

        # Determine capacity text color based on encumbrance severity
        # Yellow = at/under capacity, Orange = 1-2 over, Red = 3+ over
        if slots_over >= 3:
            capacity_color = colors.RED
        elif slots_over >= 1:
            capacity_color = colors.ORANGE
        else:
            capacity_color = colors.YELLOW

        # Left pane header (line 1, inside the border - line 0 is the top border)
        current_x = 1
        header_y = 1

        # Title
        prefix_text = "Inventory:"
        title_color = colors.YELLOW

        self.canvas.draw_text(
            pixel_x=current_x * char_w,
            pixel_y=header_y * line_h,
            text=prefix_text,
            color=title_color,
        )
        prefix_width, _, _ = self.canvas.get_text_metrics(prefix_text + " ")
        current_px_x = current_x * char_w + prefix_width

        # Capacity bar - draw slots as colored rectangles
        slot_colors = player.inventory.get_inventory_slot_colors()
        bar_max_chars = self.LEFT_PANE_WIDTH - current_x - 8
        slot_width = char_w  # Each slot is one character wide
        for i in range(min(total_slots, bar_max_chars)):
            color = slot_colors[i] if i < len(slot_colors) else colors.GREY
            self.canvas.draw_rect(
                current_px_x, header_y * line_h, slot_width, line_h, color, fill=True
            )
            current_px_x += slot_width

        # Add a small gap before usage text
        current_px_x += char_w // 2

        # Usage text with severity-based color
        usage_text = f"{used_space}/{total_slots}"
        self.canvas.draw_text(
            current_px_x, header_y * line_h, usage_text, capacity_color
        )

        # Right pane header (only in loot mode)
        if self.source is not None:
            right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
            self.canvas.draw_text(
                right_start * char_w,
                header_y * line_h,
                f"{self.source.label}:",
                colors.YELLOW,
            )

    def _draw_left_pane_items(self) -> None:
        """Delegate to left pane component."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height
        start_y = self.HEADER_HEIGHT + 2
        self.left_pane.draw(
            canvas=self.canvas,
            char_w=char_w,
            line_h=line_h,
            start_y=start_y,
            max_width=(self.LEFT_PANE_WIDTH - 2) * char_w,
            item_list_height=self.ITEM_LIST_HEIGHT,
            is_active=self.active_pane == PaneId.LEFT,
            has_source=self.source is not None,
            menu_width=self.width,
            left_pane_width=self.LEFT_PANE_WIDTH,
        )

    def _draw_divider(self) -> None:
        """Draw the center divider with transfer arrow.

        Uses Unicode box-drawing double vertical line (║) to match the outer
        frame, with proper T-junctions at the top border (╦).
        Triangle arrows (▶/◀) indicate transfer direction.
        """
        assert self.canvas is not None
        char_w = self._char_width
        line_h = self._line_height

        divider_x = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH // 2
        # Arrow centered in item list area (which starts at line 2)
        mid_y = self.HEADER_HEIGHT + 2 + self.ITEM_LIST_HEIGHT // 2

        # Column positions for the two divider lines
        left_divider_col = self.LEFT_PANE_WIDTH + 1
        right_divider_col = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH - 2

        # Draw T-junctions at top border (row 0) where dividers meet horizontal line
        self.canvas.draw_text(left_divider_col * char_w, 0, "╦", colors.WHITE)
        self.canvas.draw_text(right_divider_col * char_w, 0, "╦", colors.WHITE)

        # Draw vertical lines using double box-drawing character (║)
        # Lines start at row 1 (below the top border T-junction)
        for y in range(1, self.HEADER_HEIGHT + 2 + self.ITEM_LIST_HEIGHT):
            self.canvas.draw_text(
                left_divider_col * char_w,
                y * line_h,
                "║",
                colors.WHITE,
            )
            self.canvas.draw_text(
                right_divider_col * char_w,
                y * line_h,
                "║",
                colors.WHITE,
            )

        # Draw transfer arrow using black triangles (U+25B6, U+25C0)
        arrow = "▶" if self.active_pane == PaneId.LEFT else "◀"
        arrow_color = colors.WHITE
        self.canvas.draw_text(divider_x * char_w, mid_y * line_h, arrow, arrow_color)

    def _draw_right_pane_items(self) -> None:
        """Delegate to right pane component."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        char_w = self._char_width
        line_h = self._line_height
        start_y = self.HEADER_HEIGHT + 2
        right_start = self.LEFT_PANE_WIDTH + self.DIVIDER_WIDTH
        self.right_pane.draw(
            canvas=self.canvas,
            char_w=char_w,
            line_h=line_h,
            start_y=start_y,
            right_start_col=right_start,
            max_width=(self.TOTAL_WIDTH - right_start - 1) * char_w,
            item_list_height=self.ITEM_LIST_HEIGHT,
            is_active=self.active_pane == PaneId.RIGHT,
        )

    def _draw_detail_panel(self) -> None:
        """Draw the item detail panel at the bottom."""
        assert self.canvas is not None
        assert isinstance(self.canvas, PillowImageCanvas)
        self.detail.draw(
            canvas=self.canvas,
            char_w=self._char_width,
            line_h=self._line_height,
            total_height=self.TOTAL_HEIGHT,
            detail_height=self.DETAIL_HEIGHT,
            hint_height=self.HINT_HEIGHT,
            menu_width=self.width,
            has_source=self.source is not None,
            left_pane_width=self.LEFT_PANE_WIDTH,
            divider_width=self.DIVIDER_WIDTH,
        )

    def _get_hint_lines(self) -> list[str]:
        """Get hint text wrapped into lines that fit the current width.

        Footer commands vary by mode, active pane, and selected item:
        - [Enter] Equip: Always shown (any item can be held)
        - [U] Use: Only if selected item has consumable_effect
        - [D] Drop / [T] Transfer: Always shown
        - [Tab] Switch: Dual-pane mode only
        """
        # Get selected item to determine which hints to show
        selected_item: Item | None = None
        current_data = self._active_pane_obj.current_data
        if isinstance(current_data, Item):
            selected_item = current_data

        # Build hint parts based on context
        parts: list[str] = []

        if self.source is None:
            # Single-pane (inventory only)
            parts.append("[Enter] Equip")
            if selected_item and selected_item.consumable_effect:
                parts.append("[U] Use")
            parts.append("[D] Drop")
            parts.append("[Esc] Close")
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane, left pane focused
            parts.append("[Enter] Equip")
            if selected_item and selected_item.consumable_effect:
                parts.append("[U] Use")
            parts.append("[T] Transfer")
            parts.append("[Tab] Switch")
            parts.append("[Esc]")
        else:
            # Dual-pane, right pane focused
            parts.append("[T] Transfer")
            parts.append("[Tab] Switch")
            parts.append("[Esc] Close")

        hint_text = "  ".join(parts)

        # Use pixel-based width checking if canvas is available
        if isinstance(self.canvas, PillowImageCanvas) and self._char_width > 0:
            max_width = (self.width - 4) * self._char_width
            hint_width, _, _ = self.canvas.get_text_metrics(hint_text)
            if hint_width <= max_width:
                return [hint_text]

            # Wrap using pixel widths
            lines: list[str] = []
            remaining = hint_text
            while remaining:
                remaining_width, _, _ = self.canvas.get_text_metrics(remaining)
                if remaining_width <= max_width:
                    lines.append(remaining)
                    break
                # Find last space before max_width
                split_at = -1
                for j in range(len(remaining) - 1, 0, -1):
                    test_text = remaining[:j]
                    test_width, _, _ = self.canvas.get_text_metrics(test_text)
                    if test_width <= max_width:
                        # Found a position that fits, now find a space
                        space_pos = test_text.rfind("  ")
                        if space_pos == -1:
                            space_pos = test_text.rfind(" ")
                        split_at = space_pos if space_pos > 0 else j
                        break
                if split_at <= 0:
                    split_at = len(remaining) // 2
                lines.append(remaining[:split_at].rstrip())
                remaining = remaining[split_at:].lstrip()
            return lines

        # Fallback to character-based wrapping
        max_len = self.width - 4
        if len(hint_text) <= max_len:
            return [hint_text]

        lines = []
        remaining = hint_text
        while remaining:
            if len(remaining) <= max_len:
                lines.append(remaining)
                break
            split_at = remaining.rfind("  ", 0, max_len)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            lines.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return lines

    def _get_hint_items(self) -> list[tuple[str, str]]:
        """Get hint items as (key, label) pairs for keycap rendering.

        Returns a list of (key_text, label_text) tuples. The key_text is
        displayed inside a keycap, and label_text follows it.
        """
        # Get selected data to determine which hints to show
        selected_item: Item | None = None
        selected_countable: CountableType | None = None
        current_data = self._active_pane_obj.current_data
        if isinstance(current_data, Item):
            selected_item = current_data
        elif isinstance(current_data, CountableType):
            selected_countable = current_data

        items: list[tuple[str, str]] = []

        # Check if description has overflow (scroll hint only shown when needed)
        has_scroll = (
            self.detail.scroll_panel is not None
            and self.detail.scroll_panel.has_overflow()
        )

        # Check if selected item is stackable (Tiny size) AND has multiple in stack
        # We only show "All" hint when there's more than one item to transfer
        stack_count = 0
        if selected_item and selected_item.size == ItemSize.TINY:
            if self.active_pane == PaneId.LEFT:
                stack_count = sum(
                    1
                    for i in self.inventory.get_items()
                    if i.item_type is selected_item.item_type
                )
            else:
                source_items = self.transfer_handler.get_items_from_source()
                stack_count = sum(
                    1 for i in source_items if i.item_type is selected_item.item_type
                )
        show_all_hint = stack_count > 1

        if self.source is None:
            # Single-pane (inventory only)
            if selected_countable:
                # Countables can only be dropped (all at once)
                items.append(("D", "Drop All"))
            else:
                items.append(("ENTER", "Equip"))
                if selected_item and selected_item.consumable_effect:
                    items.append(("U", "Use"))
                items.append(("D", "Drop"))
                if show_all_hint:
                    items.append(("A", "Drop All"))
            if has_scroll:
                items.append(("←→", "Scroll"))
        elif self.active_pane == PaneId.LEFT:
            # Dual-pane, left pane focused
            if selected_countable:
                items.append(("T", "Transfer All"))
            else:
                items.append(("ENTER", "Equip"))
                if selected_item and selected_item.consumable_effect:
                    items.append(("U", "Use"))
                items.append(("T", "Transfer"))
                if show_all_hint:
                    items.append(("A", "Transfer All"))
            items.append(("TAB", "Switch"))
            if has_scroll:
                items.append(("←→", "Scroll"))
        else:
            # Dual-pane, right pane focused
            if selected_countable:
                items.append(("T", "Transfer All"))
            else:
                items.append(("T", "Transfer"))
                if show_all_hint:
                    items.append(("A", "Transfer All"))
            items.append(("TAB", "Switch"))
            if has_scroll:
                items.append(("←→", "Scroll"))

        return items

    def _draw_hint_bar(self) -> None:
        """Draw the keyboard hints at the bottom using keycap-style rendering."""
        from brileta.view.ui.ui_utils import draw_keycap

        assert self.canvas is not None
        char_w = self._char_width
        line_h = self._line_height

        hint_items = self._get_hint_items()

        # Starting position: 2 chars from left edge, on second-to-last row
        current_x = 2 * char_w
        y = (self.TOTAL_HEIGHT - 2) * line_h
        spacing = 16  # Pixels between hint groups

        # Use larger keycaps than the default (which is 85% of line height)
        keycap_size = int(line_h * 1.15)

        for key, label in hint_items:
            # Draw keycap at larger size
            width = draw_keycap(self.canvas, current_x, y, key, keycap_size=keycap_size)
            current_x += width

            # Draw label if present
            if label:
                self.canvas.draw_text(current_x, y, label, colors.GREY)
                label_width, _, _ = self.canvas.get_text_metrics(label)
                current_x += label_width + spacing
            else:
                current_x += spacing
