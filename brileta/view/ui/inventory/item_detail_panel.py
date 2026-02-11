"""Item detail panel for the dual-pane inventory menu.

Generates and renders the detail view (header, description, stats) for the
currently selected item, condition, or countable. The description area is
scrollable via a ScrollableTextPanel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.game.actors import Condition
from brileta.game.countables import CountableType, get_countable_definition
from brileta.game.enums import ItemSize
from brileta.game.items.item_core import Item
from brileta.view.ui.inventory.display_utils import (
    get_category_name,
    get_category_prefix,
)
from brileta.view.ui.scrollable_text_panel import ScrollableTextPanel

if TYPE_CHECKING:
    from brileta.backends.pillow.canvas import PillowImageCanvas
    from brileta.game.actors.components import CharacterInventory
    from brileta.view.ui.inventory.item_transfer import ItemTransferHandler
    from brileta.view.ui.overlays import MenuOption


class ItemDetailPanel:
    """Item detail display with fixed header/stats and scrollable description.

    Manages detail data generation, cached rendering state, and drawing for
    the bottom section of the DualPaneMenu.

    Layout (7 lines total):
    - Line 0: Category (fixed, colored)
    - Line 1: Name + Size (fixed)
    - Line 2: Description line 1 (scrollable)
    - Line 3: Description line 2 (scrollable)
    - Line 4: (gap)
    - Line 5: Stats (fixed, always visible)
    - Line 6: (gap before hints)
    """

    def __init__(self) -> None:
        self.detail_item: Item | Condition | CountableType | None = None

        # Scrollable panel for description text (created in configure())
        self._panel: ScrollableTextPanel | None = None
        # Track last rendered item to avoid resetting scroll unnecessarily
        self._last_rendered: Item | Condition | CountableType | None = None
        # Cached detail data: fixed header lines, stats line
        self._header: list[str] = []
        self._stats: str | None = None

        # Hover scroll zone for visual feedback
        # -1=top zone (scroll up), 0=none, +1=bottom zone (scroll down)
        self.hover_scroll_zone: int = 0

    @property
    def scroll_panel(self) -> ScrollableTextPanel | None:
        """Expose the scroll panel for input delegation and hint bar queries."""
        return self._panel

    def configure(self, max_visible_lines: int, width_chars: int) -> None:
        """Create or update the ScrollableTextPanel dimensions.

        Called when menu dimensions change (e.g., switching between
        single-pane and dual-pane mode).
        """
        if (
            self._panel is None
            or self._panel.max_visible_lines != max_visible_lines
            or self._panel.width_chars != width_chars
        ):
            self._panel = ScrollableTextPanel(
                max_visible_lines=max_visible_lines,
                width_chars=width_chars,
            )

    def update_from_cursor(self, option: MenuOption | None) -> None:
        """Update detail_item based on the currently selected menu option."""
        if option is not None and option.data is not None:
            self.detail_item = option.data
        else:
            self.detail_item = None

    def generate_detail(
        self,
        entity: Item | Condition | CountableType,
        *,
        is_left_pane: bool,
        inventory: CharacterInventory,
        transfer_handler: ItemTransferHandler,
        canvas: PillowImageCanvas | None,
        char_width: int,
        menu_width: int,
    ) -> tuple[list[str], list[str], str | None]:
        """Generate header, description, and stats for an entity.

        Returns a tuple of (header_lines, description_lines, stats_line):
        - header_lines: Category + name/size (always visible, max 2 lines)
        - description_lines: Item description (scrollable)
        - stats_line: Protection/attack stats (always visible, or None)

        Args:
            entity: The item, condition, or countable to describe.
            is_left_pane: True if the entity is from the left (inventory) pane.
                Used to determine where to look up countable quantities.
            inventory: Player inventory (for countable quantities when is_left_pane).
            transfer_handler: Transfer handler (for source countable quantities).
            canvas: Canvas for pixel-based text wrapping. May be None in tests.
            char_width: Character width in pixels for wrap calculation.
            menu_width: Menu width in characters for wrap calculation.
        """
        header: list[str] = []
        description: list[str] = []
        stats: str | None = None

        def wrap_description(text: str) -> list[str]:
            """Wrap description text to fit the detail panel width."""
            if canvas is not None and char_width > 0:
                # Calculate max width for description, accounting for:
                # - 4 chars margin (2 each side)
                # - 3 chars for "..." ellipsis that may be prepended/appended
                max_desc_width = (menu_width - 7) * char_width
                return canvas.wrap_text(text, max_desc_width)
            return [text]

        if isinstance(entity, CountableType):
            defn = get_countable_definition(entity)
            # Get quantity from appropriate source based on active pane
            if is_left_pane:
                quantity = inventory.get_countable(entity)
            else:
                countables = transfer_handler.get_countables_from_source()
                quantity = countables.get(entity, 0)
            header.append("Currency")
            name = defn.plural_name if quantity != 1 else defn.name
            header.append(f"{quantity} {name} - Tiny (shares slot)")
            description.extend(wrap_description("Money. You know how it works."))
            return header, description, stats

        if isinstance(entity, Condition):
            header.append(f"Condition: {entity.name}")
            if hasattr(entity, "description") and entity.description:
                description.extend(wrap_description(entity.description))
            return header, description, stats

        item: Item = entity

        # Header line 1: category name (will be rendered in category color)
        category_name = get_category_name(item.category)
        if category_name:
            header.append(category_name)

        # Header line 2: name + size
        size_text = {
            ItemSize.TINY: "Tiny (shares slot)",
            ItemSize.NORMAL: "Normal (1 slot)",
            ItemSize.BIG: "Big (2 slots)",
            ItemSize.HUGE: "Huge (4 slots)",
        }
        header.append(f"{item.name} - {size_text.get(item.size, 'Unknown size')}")

        # Description - wrap long descriptions using word boundaries
        if item.description:
            description.extend(wrap_description(item.description))

        # Check if this is an outfit item - capability is now stored on the item
        from brileta.game.outfit import get_outfit_spec

        outfit_spec = get_outfit_spec(item.item_type)
        if outfit_spec is not None:
            # Get capability from item (persists damage state across equip/unequip)
            outfit_cap = item.outfit_capability

            if outfit_spec.protection > 0:
                if outfit_cap is not None:
                    if outfit_cap.is_broken:
                        stats = f"Protection: {outfit_spec.protection}  AP: BROKEN"
                    else:
                        stats = (
                            f"Protection: {outfit_spec.protection}  "
                            f"AP: {outfit_cap.ap}/{outfit_cap.max_ap}"
                        )
                else:
                    # Fallback (shouldn't happen for outfit items)
                    stats = (
                        f"Protection: {outfit_spec.protection}  "
                        f"AP: {outfit_spec.max_ap}"
                    )
            else:
                stats = "No protection (clothing only)"
            return header, description, stats

        # Find the attack to display: prefer one with PREFERRED, else first available
        from brileta.game.items.properties import (
            WEAPON_PROPERTY_DESCRIPTIONS,
            TacticalProperty,
            WeaponProperty,
        )

        attacks: list[tuple[str, object | None]] = [
            ("Melee", item.melee_attack),
            ("Ranged", item.ranged_attack),
            ("Area", item.area_effect),
        ]

        # Look for attack with PREFERRED
        chosen_attack: tuple[str, object] | None = None
        for label, attack in attacks:
            if attack and WeaponProperty.PREFERRED in attack.properties:  # type: ignore[unresolved-attribute]
                chosen_attack = (label, attack)
                break

        # Fallback to first available attack
        if not chosen_attack:
            for label, attack in attacks:
                if attack:
                    chosen_attack = (label, attack)
                    break

        # Build stats line for the chosen attack
        if chosen_attack:
            label, attack = chosen_attack
            if label == "Ranged":
                # Skip ammo display for THROWN weapons (single-use items)
                if WeaponProperty.THROWN in attack.properties:  # type: ignore[unresolved-attribute]
                    stats = f"Ranged: {attack.damage_dice}"  # type: ignore[unresolved-attribute]
                else:
                    ammo_str = (
                        f"{attack.current_ammo}/{attack.max_ammo}"  # type: ignore[unresolved-attribute]
                        if attack.max_ammo  # type: ignore[unresolved-attribute]
                        else "N/A"
                    )
                    stats = f"Ranged: {attack.damage_dice} [{ammo_str}]"  # type: ignore[unresolved-attribute]
            else:
                stats = f"{label}: {attack.damage_dice}"  # type: ignore[unresolved-attribute]

            # Append weapon properties to description (not stats line)
            description.extend(
                WEAPON_PROPERTY_DESCRIPTIONS[prop]
                for prop in attack.properties  # type: ignore[unresolved-attribute]
                if isinstance(prop, WeaponProperty | TacticalProperty)
                and prop in WEAPON_PROPERTY_DESCRIPTIONS
            )

        return header, description, stats

    def get_scroll_zone(
        self, rel_line_y: int, total_height: int, detail_height: int, hint_height: int
    ) -> int:
        """Determine scroll zone from a line position in the description area.

        Args:
            rel_line_y: The line position relative to the menu top (0-indexed).
            total_height: Total menu height in lines.
            detail_height: Detail panel height in lines.
            hint_height: Hint bar height in lines.

        Returns:
            -1 if should scroll up, +1 if should scroll down, 0 if not in scroll zone.
        """
        sep_y = total_height - detail_height - hint_height - 1
        desc_start_y = sep_y + 1 + 2  # detail_start_y + 2 (after header lines)
        desc_end_y = desc_start_y + 2  # 2 description lines

        if self._panel is None or not self._panel.has_overflow():
            return 0

        if not (desc_start_y <= rel_line_y < desc_end_y):
            return 0

        can_up = self._panel.can_scroll_up()
        can_down = self._panel.can_scroll_down()

        if can_up and can_down:
            # Both directions available - use top line for up, bottom for down
            return -1 if rel_line_y == desc_start_y else 1
        if can_up:
            return -1
        if can_down:
            return 1
        return 0

    def draw(
        self,
        canvas: PillowImageCanvas,
        char_w: int,
        line_h: int,
        total_height: int,
        detail_height: int,
        hint_height: int,
        menu_width: int,
        has_source: bool,
        left_pane_width: int,
        divider_width: int,
    ) -> None:
        """Draw the detail panel: separator line, header, description, and stats.

        Args:
            canvas: Canvas to draw on.
            char_w: Character width in pixels.
            line_h: Line height in pixels.
            total_height: Total menu height in lines.
            detail_height: Detail panel height in lines.
            hint_height: Hint bar height in lines.
            menu_width: Menu width in characters.
            has_source: Whether dual-pane mode is active (affects separator junctions).
            left_pane_width: Left pane width in chars (for divider junction positions).
            divider_width: Divider width in chars (for divider junction positions).
        """
        # Separator line position
        sep_y = total_height - detail_height - hint_height - 1
        separator_px_y = sep_y * line_h

        # Column positions for divider lines (used in loot mode)
        left_divider_col = left_pane_width + 1
        right_divider_col = left_pane_width + divider_width - 2

        # Left junction (double-vertical to single-horizontal)
        canvas.draw_text(0, separator_px_y, "\u255f", colors.WHITE)

        # Horizontal line with junction characters where dividers meet
        for x in range(1, menu_width - 1):
            # In loot mode, draw junctions where divider columns meet separator
            if has_source and x in (left_divider_col, right_divider_col):
                # \u2568 = double vertical up meeting single horizontal
                canvas.draw_text(x * char_w, separator_px_y, "\u2568", colors.WHITE)
            else:
                canvas.draw_text(x * char_w, separator_px_y, "\u2500", colors.WHITE)

        # Right junction
        canvas.draw_text(
            (menu_width - 1) * char_w, separator_px_y, "\u2562", colors.WHITE
        )

        detail_start_y = sep_y + 1
        panel_x = 2 * char_w

        if self.detail_item is None:
            canvas.draw_text(
                panel_x,
                detail_start_y * line_h,
                "(No item selected)",
                colors.GREY,
            )
            self._last_rendered = None
            return

        # Generate separated detail data (header, description, stats)
        assert self._panel is not None
        if self.detail_item is not self._last_rendered:
            header, description, stats = self.generate_detail(
                self.detail_item,
                is_left_pane=self._generate_is_left_pane,
                inventory=self._generate_inventory,
                transfer_handler=self._generate_transfer_handler,
                canvas=canvas,
                char_width=char_w,
                menu_width=menu_width,
            )
            # Store header and stats for fixed rendering
            self._header = header
            self._stats = stats
            # Only description goes into the scrollable panel
            self._panel.set_content(description)
            self._last_rendered = self.detail_item

        # Determine category color for first header line
        first_line_color: colors.Color = colors.WHITE
        if isinstance(self.detail_item, Item):
            _, cat_color = get_category_prefix(self.detail_item)
            if cat_color:
                first_line_color = cat_color

        # Draw fixed header lines (lines 0-1)
        for i, line in enumerate(self._header[:2]):  # Max 2 header lines
            color = first_line_color if i == 0 else colors.WHITE
            canvas.draw_text(panel_x, (detail_start_y + i) * line_h, line, color)

        # Draw scrollable description (lines 2-3)
        # Description panel starts at line 2 within the detail area
        desc_start_y = detail_start_y + 2
        self._panel.draw(
            canvas=canvas,
            x=panel_x,
            y=desc_start_y * line_h,
            line_height=line_h,
            char_width=char_w,
            text_color=colors.WHITE,
            hover_zone=self.hover_scroll_zone,
        )

        # Draw fixed stats line (line 5, always visible)
        if self._stats:
            stats_y = detail_start_y + 5  # Line 5 within detail area
            canvas.draw_text(panel_x, stats_y * line_h, self._stats, colors.WHITE)

    def set_generate_context(
        self,
        *,
        is_left_pane: bool,
        inventory: CharacterInventory,
        transfer_handler: ItemTransferHandler,
    ) -> None:
        """Set the context needed for lazy detail generation during draw().

        Called by the orchestrator before draw() whenever the active pane or
        data sources change. This avoids passing these through the draw call
        chain while keeping the panel decoupled from the menu.
        """
        self._generate_is_left_pane = is_left_pane
        self._generate_inventory = inventory
        self._generate_transfer_handler = transfer_handler
