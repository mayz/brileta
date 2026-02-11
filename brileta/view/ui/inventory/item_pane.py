"""Item pane components for the dual-pane inventory menu.

Each pane owns its own options list, cursor, and renderer. The base class
provides cursor management and rendering helpers. Subclasses handle
data-specific population and drawing.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from brileta import colors
from brileta.game.actors import Condition
from brileta.game.countables import CountableType, get_countable_definition
from brileta.game.items.item_core import Item
from brileta.view.ui.inventory.display_utils import (
    get_category_prefix,
    group_items_for_display,
)
from brileta.view.ui.overlays import MenuOption
from brileta.view.ui.selectable_list import (
    LayoutMode,
    SelectableListRenderer,
    SelectableRow,
)

if TYPE_CHECKING:
    from brileta.backends.pillow.canvas import PillowImageCanvas
    from brileta.game.actors.components import CharacterInventory
    from brileta.view.ui.inventory.item_transfer import ItemTransferHandler


class ItemPane:
    """Base class for a pane in the dual-pane inventory menu.

    Owns an options list, cursor position, and a SelectableListRenderer.
    Provides cursor management and item lookup.
    """

    def __init__(self) -> None:
        self.options: list[MenuOption] = []
        self.cursor: int = 0
        self._renderer: SelectableListRenderer | None = None

    @property
    def current_option(self) -> MenuOption | None:
        """Return the MenuOption at the current cursor, or None."""
        if 0 <= self.cursor < len(self.options):
            return self.options[self.cursor]
        return None

    @property
    def current_data(self) -> Item | Condition | CountableType | None:
        """Return the data of the option at the current cursor."""
        opt = self.current_option
        return opt.data if opt is not None else None

    def has_selectable(self) -> bool:
        """Return True if any option in this pane is enabled/selectable."""
        return any(opt.enabled for opt in self.options)

    def ensure_valid_cursor(self) -> None:
        """Ensure cursor is on an enabled option.

        If the current position is valid and enabled, keeps it.
        Otherwise finds the nearest enabled option (preferring forward,
        then backward). Falls back to position 0 if no enabled options exist.
        """
        options_len = len(self.options)

        # If current position is valid and enabled, keep it
        if 0 <= self.cursor < options_len and self.options[self.cursor].enabled:
            return

        if options_len == 0:
            self.cursor = 0
            return

        # Find nearest enabled option (prefer forward, then backward)
        forward_start = max(0, self.cursor)
        for i in range(forward_start, options_len):
            if self.options[i].enabled:
                self.cursor = i
                return
        backward_start = min(self.cursor - 1, options_len - 1)
        for i in range(backward_start, -1, -1):
            if self.options[i].enabled:
                self.cursor = i
                return

        # No enabled options - set to 0
        self.cursor = 0

    def move_cursor(self, delta: int) -> None:
        """Move cursor by delta, skipping disabled options."""
        new_cursor = self.cursor + delta
        while 0 <= new_cursor < len(self.options):
            if self.options[new_cursor].enabled:
                self.cursor = new_cursor
                break
            new_cursor += delta

    def find_item_index(self, item: Item) -> int | None:
        """Find the index of an item in this pane's options."""
        for i, opt in enumerate(self.options):
            if opt.data is item:
                return i
        return None

    def ensure_renderer(self, canvas: PillowImageCanvas) -> None:
        """Lazily create the SelectableListRenderer if needed."""
        if self._renderer is None:
            self._renderer = SelectableListRenderer(canvas, LayoutMode.INLINE)


class InventoryPane(ItemPane):
    """Left pane: player inventory with pinned equipment slots at the bottom.

    The visual layout splits stored items (flowing from top) from equipment
    slots (pinned at bottom), separated by a horizontal line.
    """

    def __init__(self, inventory: CharacterInventory) -> None:
        super().__init__()
        self._inventory = inventory

    @property
    def equipment_slot_count(self) -> int:
        """Number of equipment slots (weapon slots + outfit slot)."""
        return len(self._inventory.ready_slots) + 1

    def populate(self, equip_callback: Callable[[Item], bool]) -> None:
        """Populate with player inventory items.

        Uses item grouping for Tiny stored items to display stacked with
        "x N" suffix. Equipment slots are always shown individually (even
        when empty).

        Args:
            equip_callback: Function to call when an item's equip action is
                triggered. Receives the item and returns True if menu should close.
        """
        self.options.clear()

        # Collect equipped items separately from stored items
        # Equipment slots are always shown (even when empty/None)
        equipped_items: list[tuple[Item | Condition | None, str, int]] = []

        # Add all ready slots (even if empty)
        for slot_index in range(len(self._inventory.ready_slots)):
            equipped_item = self._inventory.ready_slots[slot_index]
            equipped_items.append((equipped_item, f"slot:{slot_index}", 1))

        # Add outfit slot (even if empty)
        if self._inventory.equipped_outfit is not None:
            outfit_item, _ = self._inventory.equipped_outfit
            equipped_items.append((outfit_item, "outfit", 1))
        else:
            equipped_items.append((None, "outfit", 1))

        # Group stored items for stacked display (Tiny items only)
        # First separate items from conditions since conditions don't stack
        stored_items_only = [
            item for item in self._inventory._stored_items if isinstance(item, Item)
        ]
        stored_conditions = [
            cond
            for cond in self._inventory._stored_items
            if isinstance(cond, Condition)
        ]

        # Group items (tiny items will be combined)
        grouped_items = group_items_for_display(stored_items_only)

        # Build stored items list with count: (entity, status, count)
        stored_items: list[tuple[Item | Condition, str, int]] = [
            (item, "stored", count) for item, count in grouped_items
        ]
        # Conditions always have count 1
        stored_items.extend([(cond, "stored", 1) for cond in stored_conditions])

        # Add countables (coins, etc.) at the top of stored section
        for countable_type in CountableType:
            quantity = self._inventory.get_countable(countable_type)
            if quantity > 0:
                defn = get_countable_definition(countable_type)
                display_name = defn.plural_name if quantity > 1 else defn.name
                self.options.append(
                    MenuOption(
                        key=None,
                        text=f"{quantity} {display_name}",
                        action=None,  # Drop via D key handler
                        enabled=True,
                        color=colors.WHITE,
                        data=countable_type,
                        prefix_segments=[
                            ("      ", colors.WHITE),  # Alignment with equipped slots
                            (f"{defn.glyph} ", defn.color),
                        ],
                    )
                )

        # Stored items first, then equipped items at the bottom
        # (Visual separator is drawn in draw(), not as a list item)
        all_items_to_display = stored_items + equipped_items

        for entity, item_status, count in all_items_to_display:
            # Handle empty equipment slots (None)
            if entity is None:
                prefix_segments: list[tuple[str, colors.Color]] = []
                if item_status.startswith("slot:"):
                    slot_num = int(item_status.split(":")[1]) + 1
                    prefix_segments.append((f" [{slot_num}]  ", colors.GREY))
                elif item_status == "outfit":
                    prefix_segments.append((" [\U0000f007]  ", colors.GREY))
                self.options.append(
                    MenuOption(
                        key=None,
                        text="(empty)",
                        enabled=False,
                        color=colors.GREY,
                        is_primary_action=False,
                        prefix_segments=prefix_segments,
                    )
                )

            elif isinstance(entity, Item):
                item = entity

                # Build prefix segments: slot indicator (if equipped) + category
                prefix_segments = []

                # Parse equipped slot number for prefix
                if item_status.startswith("slot:"):
                    slot_num = int(item_status.split(":")[1]) + 1
                    prefix_segments.append((f" [{slot_num}]  ", colors.YELLOW))
                elif item_status == "outfit":
                    # Show glyph for equipped outfit/armor
                    prefix_segments.append((" [\U0000f007]  ", colors.CATEGORY_OUTFIT))
                else:
                    # Add alignment spaces to match equipped items
                    prefix_segments.append(("      ", colors.WHITE))

                # Add category prefix
                cat_prefix, cat_color = get_category_prefix(item)
                if cat_prefix and cat_color:
                    prefix_segments.append((cat_prefix, cat_color))

                # Add stack count suffix for grouped items
                display_text = item.name
                if count > 1:
                    display_text = f"{item.name} x {count}"

                self.options.append(
                    MenuOption(
                        key=None,
                        text=display_text,
                        action=functools.partial(equip_callback, item),
                        enabled=True,
                        color=colors.WHITE,
                        data=item,
                        prefix_segments=prefix_segments,
                    )
                )

            elif isinstance(entity, Condition):
                condition = entity
                # Conditions show red block in icon position to indicate slot usage
                self.options.append(
                    MenuOption(
                        key=None,
                        text=condition.name,
                        action=None,
                        enabled=False,
                        color=condition.display_color,
                        force_color=True,
                        data=condition,
                        prefix_segments=[
                            ("      ", colors.WHITE),  # Alignment with equipped slots
                            ("\u2588 ", colors.RED),  # Red block as slot indicator
                        ],
                    )
                )

    def draw(
        self,
        canvas: PillowImageCanvas,
        char_w: int,
        line_h: int,
        start_y: int,
        max_width: int,
        item_list_height: int,
        is_active: bool,
        has_source: bool,
        menu_width: int,
        left_pane_width: int,
    ) -> None:
        """Draw stored items and pinned equipment slots with separator.

        Args:
            canvas: Canvas to draw on.
            char_w: Character width in pixels.
            line_h: Line height in pixels.
            start_y: First line (in lines, not pixels) of the item area.
            max_width: Maximum width in pixels for item text.
            item_list_height: Total height of the item list area in lines.
            is_active: Whether this pane is the active (focused) pane.
            has_source: Whether dual-pane mode is active (affects separator style).
            menu_width: Total menu width in characters (for full-width separator).
            left_pane_width: Left pane width in characters.
        """
        assert self._renderer is not None

        # Split options: stored items (+ separator) vs equipment slots
        # Equipment slots are always the last N items (weapon slots + outfit)
        num_equipment = self.equipment_slot_count
        stored_options = self.options[:-num_equipment]
        equipment_options = self.options[-num_equipment:]

        # Calculate where equipment slots should be pinned (bottom of list area)
        # Equipment gets the last EQUIPMENT_SLOT_COUNT lines, plus 1 for separator
        equipment_start_line = start_y + item_list_height - num_equipment

        # Render stored items from the top (limited to available space above equipment)
        # Reserve space for: separator line (1) + equipment slots (3)
        max_stored_lines = item_list_height - num_equipment - 1
        visible_stored = stored_options[:max_stored_lines]

        self._renderer.rows = [
            SelectableRow(
                text=option.text,
                key=option.key,
                enabled=option.enabled,
                color=option.color,
                data=option,
                prefix_segments=option.prefix_segments,
                force_color=option.force_color,
            )
            for option in visible_stored
        ]
        # Set hover for stored items (cursor in range of stored section)
        if is_active and self.cursor < len(visible_stored):
            self._renderer.hovered_index = self.cursor
        else:
            self._renderer.hovered_index = None

        self._renderer.render(
            x_start=char_w,
            y_start=start_y * line_h,
            max_width=max_width,
            line_height=line_h,
            ascent=0,
            row_gap=0,
        )

        # Draw separator line above equipment section (left pane only)
        # This line only spans the left pane since equipment is a left-pane concept
        separator_line_y = equipment_start_line - 1
        separator_px_y = separator_line_y * line_h

        # Left junction with border
        canvas.draw_text(0, separator_px_y, "\u255f", colors.WHITE)

        # Determine right edge based on mode
        if not has_source:
            # Inventory-only mode: line spans full width
            right_edge = menu_width - 1
            right_junction = "\u2562"
            # Horizontal line across
            for x in range(1, right_edge):
                canvas.draw_text(x * char_w, separator_px_y, "\u2500", colors.WHITE)
            # Right junction with border
            canvas.draw_text(
                right_edge * char_w, separator_px_y, right_junction, colors.WHITE
            )
        else:
            # Dual-pane mode: line spans only left pane, connects to divider
            left_divider_col = left_pane_width + 1
            # Horizontal line up to the divider
            for x in range(1, left_divider_col):
                canvas.draw_text(x * char_w, separator_px_y, "\u2500", colors.WHITE)
            # Junction where line meets the divider (mirror of left side's \u255f)
            canvas.draw_text(
                left_divider_col * char_w, separator_px_y, "\u2562", colors.WHITE
            )

        # Render equipment slots at fixed bottom position
        equipment_rows = [
            SelectableRow(
                text=option.text,
                key=option.key,
                enabled=option.enabled,
                color=option.color,
                data=option,
                prefix_segments=option.prefix_segments,
                force_color=option.force_color,
            )
            for option in equipment_options
        ]

        # Calculate hover index for equipment section
        equipment_start_index = len(stored_options)
        if is_active and self.cursor >= equipment_start_index:
            self._renderer.hovered_index = self.cursor - equipment_start_index
        else:
            self._renderer.hovered_index = None

        self._renderer.rows = equipment_rows
        self._renderer.render(
            x_start=char_w,
            y_start=equipment_start_line * line_h,
            max_width=max_width,
            line_height=line_h,
            ascent=0,
            row_gap=0,
        )

    def map_line_to_index(self, option_line: int, item_list_height: int) -> int | None:
        """Map a visual line offset in the left pane to an options index.

        The left pane layout is:
        - Stored items at top (lines 0 to max_stored_lines-1)
        - Separator line (not selectable)
        - Equipment slots pinned at bottom (lines item_list_height-num_equipment
          to item_list_height-1)

        Args:
            option_line: Line offset from the start of the item list area (0-indexed).
            item_list_height: Total height of the item list area in lines.

        Returns:
            Index into options, or None if the line is not selectable
            (out of bounds, separator, or empty area).
        """
        if option_line < 0 or option_line >= item_list_height:
            return None

        num_equipment = self.equipment_slot_count
        num_stored = len(self.options) - num_equipment
        max_stored_lines = item_list_height - num_equipment - 1
        visible_stored = min(num_stored, max_stored_lines)

        # Stored items section: lines 0 to visible_stored-1
        if option_line < visible_stored:
            return option_line

        # Equipment section: last num_equipment lines of the list area
        equipment_start_line = item_list_height - num_equipment
        if option_line >= equipment_start_line:
            equipment_offset = option_line - equipment_start_line
            equipment_index = num_stored + equipment_offset
            if equipment_index < len(self.options):
                return equipment_index

        # Separator or empty area between stored and equipment
        return None


class LootPane(ItemPane):
    """Right pane: items from an external source (container/ground)."""

    def populate(
        self,
        transfer_handler: ItemTransferHandler,
        transfer_item_fn: Callable[[Item], bool],
        transfer_countable_fn: Callable[[CountableType], bool],
    ) -> None:
        """Populate from the external source.

        Uses item grouping for Tiny items to display stacked with "x N" suffix.
        Also displays countables from the source.

        Args:
            transfer_handler: Handler providing access to source items/countables.
            transfer_item_fn: Callback for transferring an item to inventory.
            transfer_countable_fn: Callback for transferring a countable to inventory.
        """
        self.options.clear()

        if transfer_handler.source is None:
            # Inventory-only mode - no right pane content
            return

        items = transfer_handler.get_items_from_source()
        countables = transfer_handler.get_countables_from_source()

        if not items and not countables:
            self.options.append(
                MenuOption(
                    key=None,
                    text="(no items)",
                    enabled=False,
                    color=colors.GREY,
                    is_primary_action=False,
                )
            )
            return

        # Add countables first
        for countable_type, quantity in countables.items():
            if quantity > 0:
                defn = get_countable_definition(countable_type)
                display_name = defn.plural_name if quantity > 1 else defn.name
                self.options.append(
                    MenuOption(
                        key=None,
                        text=f"{quantity} {display_name}",
                        action=functools.partial(transfer_countable_fn, countable_type),
                        enabled=True,
                        color=colors.WHITE,
                        data=countable_type,
                        prefix_segments=[
                            (f"{defn.glyph} ", defn.color),
                        ],
                    )
                )

        # Group tiny items for stacked display
        grouped_items = group_items_for_display(items)

        for item, count in grouped_items:
            # Build prefix segments with category prefix
            prefix_segments: list[tuple[str, colors.Color]] = []
            cat_prefix, cat_color = get_category_prefix(item)
            if cat_prefix and cat_color:
                prefix_segments.append((cat_prefix, cat_color))

            # Add stack count suffix for grouped items
            display_text = item.name
            if count > 1:
                display_text = f"{item.name} x {count}"

            self.options.append(
                MenuOption(
                    key=None,
                    text=display_text,
                    action=functools.partial(transfer_item_fn, item),
                    enabled=True,
                    color=colors.WHITE,
                    data=item,
                    prefix_segments=prefix_segments if prefix_segments else None,
                )
            )

    def draw(
        self,
        canvas: PillowImageCanvas,
        char_w: int,
        line_h: int,
        start_y: int,
        right_start_col: int,
        max_width: int,
        item_list_height: int,
        is_active: bool,
    ) -> None:
        """Draw items in the right pane.

        Args:
            canvas: Canvas to draw on.
            char_w: Character width in pixels.
            line_h: Line height in pixels.
            start_y: First line (in lines, not pixels) of the item area.
            right_start_col: Starting column of the right pane in characters.
            max_width: Maximum width in pixels for item text.
            item_list_height: Total height of the item list area in lines.
            is_active: Whether this pane is the active (focused) pane.
        """
        assert self._renderer is not None

        # Convert MenuOptions to SelectableRows (limit to visible area)
        self._renderer.rows = [
            SelectableRow(
                text=option.text,
                key=option.key,
                enabled=option.enabled,
                color=option.color,
                data=option,
                prefix_segments=option.prefix_segments,
                force_color=option.force_color,
            )
            for option in self.options[:item_list_height]
        ]
        # Set hover state based on active pane and cursor
        self._renderer.hovered_index = self.cursor if is_active else None

        # Render with proper coordinates
        self._renderer.render(
            x_start=right_start_col * char_w,
            y_start=start_y * line_h,
            max_width=max_width,
            line_height=line_h,
            ascent=0,
            row_gap=0,
        )
