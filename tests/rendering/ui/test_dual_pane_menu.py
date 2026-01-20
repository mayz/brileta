"""Unit tests for the DualPaneMenu class."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import tcod.event

from catley import colors
from catley.backends.tcod.graphics import TCODGraphicsContext
from catley.controller import Controller
from catley.game.actors import Character
from catley.game.enums import ItemCategory
from catley.game.game_world import GameWorld
from catley.game.items.item_types import (
    COMBAT_KNIFE_TYPE,
    PISTOL_MAGAZINE_TYPE,
    PISTOL_TYPE,
    STIM_TYPE,
)
from catley.game.items.junk_item_types import JUNK_ITEM_TYPES
from catley.game.turn_manager import TurnManager
from catley.view.ui.dual_pane_menu import DualPaneMenu, ExternalInventory, PaneId
from tests.helpers import DummyGameWorld
from tests.rendering.backends.test_canvases import _make_renderer


@dataclass
class DummyController(Controller):
    """Minimal controller for testing menus."""

    gw: DummyGameWorld

    def __post_init__(self) -> None:
        self.turn_manager = TurnManager(self)
        self.frame_manager = None
        self.message_log = None
        self.graphics = _make_renderer()
        tcod_renderer = cast(TCODGraphicsContext, self.graphics)
        cast(Any, tcod_renderer.root_console).width = 100
        cast(Any, tcod_renderer.root_console).height = 50
        self.coordinate_converter = SimpleNamespace(pixel_to_tile=lambda x, y: (x, y))
        self.overlay_system = None


def _make_controller() -> Controller:
    """Create a test controller with a player."""
    gw = DummyGameWorld()
    player = Character(
        5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
    )
    gw.player = player
    gw.add_actor(player)
    return DummyController(gw)


def _make_controller_with_inventory_items() -> Controller:
    """Create a controller with items in player's inventory."""
    controller = _make_controller()
    player = controller.gw.player

    # Add items to player inventory
    pistol = PISTOL_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.add_to_inventory(pistol)
    player.inventory.add_to_inventory(knife)

    return controller


def _make_controller_with_ground_items() -> Controller:
    """Create a controller with items on the ground at player's location."""
    controller = _make_controller()
    player = controller.gw.player

    # Add items directly to the DummyGameWorld's items dict
    # (DummyGameWorld.get_pickable_items_at_location reads from self.items)
    stim = STIM_TYPE.create()
    knife = COMBAT_KNIFE_TYPE.create()
    location = (player.x, player.y)
    controller.gw.items[location] = [stim, knife]

    return controller


# -----------------------------------------------------------------------------
# Basic Functionality Tests
# -----------------------------------------------------------------------------


def test_dual_pane_menu_inventory_only_mode() -> None:
    """Menu should work in inventory-only mode when source is None."""
    controller = _make_controller_with_inventory_items()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    # Left pane should have items
    assert len(menu.left_options) == 2

    # Right pane should be empty
    assert len(menu.right_options) == 0

    # Title should indicate inventory mode
    assert menu.title == "Inventory"


def test_dual_pane_menu_loot_mode() -> None:
    """Menu should work in loot mode when source is provided."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    # Right pane should have ground items
    assert len(menu.right_options) == 2

    # Title should indicate loot mode
    assert menu.title == "Loot"


def test_dual_pane_menu_shows_player_inventory() -> None:
    """Left pane should display all player inventory items."""
    controller = _make_controller_with_inventory_items()
    menu = DualPaneMenu(controller)
    menu.show()

    option_names = [opt.text for opt in menu.left_options]
    assert "Pistol" in option_names
    assert "Combat Knife" in option_names


def test_dual_pane_menu_shows_ground_items() -> None:
    """Right pane should display items at source location."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    option_names = [opt.text for opt in menu.right_options]
    assert "Stimpack" in option_names
    assert "Combat Knife" in option_names


def test_dual_pane_menu_empty_inventory() -> None:
    """Menu should handle empty inventory gracefully."""
    controller = _make_controller()
    menu = DualPaneMenu(controller)
    menu.show()

    # Should have a placeholder option
    assert len(menu.left_options) == 1
    assert menu.left_options[0].text == "(inventory empty)"
    assert not menu.left_options[0].enabled


# -----------------------------------------------------------------------------
# Navigation Tests
# -----------------------------------------------------------------------------


def test_tab_switches_panes() -> None:
    """Tab key should toggle between left and right panes."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player

    # Add item to player inventory so left pane has content
    pistol = PISTOL_TYPE.create()
    player.inventory.add_to_inventory(pistol)

    location = (player.x, player.y)
    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    # Start on left pane
    assert menu.active_pane == PaneId.LEFT

    # Tab should switch to right pane
    tab_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.TAB,
        sym=tcod.event.KeySym.TAB,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(tab_event)
    assert menu.active_pane == PaneId.RIGHT

    # Tab again should switch back to left pane
    menu.handle_input(tab_event)
    assert menu.active_pane == PaneId.LEFT


def test_tab_does_not_switch_in_inventory_only_mode() -> None:
    """Tab should do nothing when there's no right pane."""
    controller = _make_controller_with_inventory_items()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    assert menu.active_pane == PaneId.LEFT

    tab_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.TAB,
        sym=tcod.event.KeySym.TAB,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(tab_event)

    # Should still be on left pane
    assert menu.active_pane == PaneId.LEFT


def test_arrow_keys_move_cursor() -> None:
    """Arrow keys should move cursor within active pane."""
    controller = _make_controller_with_inventory_items()
    menu = DualPaneMenu(controller)
    menu.show()

    assert menu.left_cursor == 0

    # Press down arrow
    down_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.DOWN,
        sym=tcod.event.KeySym.DOWN,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(down_event)
    assert menu.left_cursor == 1

    # Press up arrow
    up_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.UP,
        sym=tcod.event.KeySym.UP,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(up_event)
    assert menu.left_cursor == 0


def test_escape_closes_menu() -> None:
    """Escape key should close the menu."""
    controller = _make_controller()
    menu = DualPaneMenu(controller)
    menu.show()
    assert menu.is_active

    esc_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.ESCAPE,
        sym=tcod.event.KeySym.ESCAPE,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(esc_event)

    assert not menu.is_active


# -----------------------------------------------------------------------------
# Transfer Tests
# -----------------------------------------------------------------------------


def test_transfer_item_to_inventory() -> None:
    """Selecting an item in right pane should transfer it to player."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    initial_ground_items = len(controller.gw.get_pickable_items_at_location(*location))
    initial_inventory_size = len(player.inventory)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    # Right pane should have items
    assert len(menu.right_options) == initial_ground_items

    # Transfer first item
    first_item = menu.right_options[0].data
    assert first_item is not None
    menu._transfer_to_inventory(first_item)

    # Inventory should have one more item
    assert len(player.inventory) == initial_inventory_size + 1


def test_transfer_removes_item_from_ground() -> None:
    """Transferred item should be removed from ground and menu should update."""
    from tests.helpers import get_controller_with_player_and_map

    controller = get_controller_with_player_and_map()
    player = controller.gw.player
    location = (player.x, player.y)

    # Spawn a single item using the real spawning system
    item = COMBAT_KNIFE_TYPE.create()
    controller.gw.spawn_ground_item(item, *location)

    # Verify item is on ground
    ground_items_before = controller.gw.get_pickable_items_at_location(*location)
    assert len(ground_items_before) == 1, "Should have 1 item on ground"
    assert ground_items_before[0] is item, "Should be the same item object"

    # Open menu
    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    assert len(menu.right_options) == 1, "Menu should show 1 item"
    menu_item = menu.right_options[0].data
    assert menu_item is not None, "Menu item data should not be None"
    assert menu_item is item, "Menu should reference the same item object"

    # Transfer
    menu._transfer_to_inventory(menu_item)

    # Verify item is removed from ground
    ground_items_after = controller.gw.get_pickable_items_at_location(*location)
    assert len(ground_items_after) == 0, "Item should be removed from ground"

    # Verify item is in player inventory
    assert item in player.inventory, "Item should be in player inventory"

    # Verify menu updated to show no items
    assert len(menu.right_options) == 1, "Menu should have 1 option (placeholder)"
    assert menu.right_options[0].text == "(no items)", (
        "Menu should show no items placeholder"
    )


def test_transfer_fails_when_inventory_full() -> None:
    """Transfer should fail gracefully when inventory is full."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    # Fill inventory to capacity
    for _ in range(player.inventory.total_inventory_slots):
        player.inventory.add_to_inventory(COMBAT_KNIFE_TYPE.create())

    initial_inventory_size = len(player.inventory)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    # Try to transfer (should fail)
    first_item = menu.right_options[0].data
    assert first_item is not None
    menu._transfer_to_inventory(first_item)

    # Inventory size should not change
    assert len(player.inventory) == initial_inventory_size


# -----------------------------------------------------------------------------
# Use/Equip Tests
# -----------------------------------------------------------------------------


def test_use_item_equips_weapon() -> None:
    """Using an equippable item should equip it to active slot."""
    controller = _make_controller()
    player = controller.gw.player

    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.add_to_inventory(knife)

    menu = DualPaneMenu(controller)
    menu.show()

    # Knife should be in inventory, not equipped
    assert knife in player.inventory
    assert knife not in player.inventory.attack_slots

    # Use the knife
    menu._equip_item(knife)

    # Knife should now be equipped
    assert knife not in player.inventory._stored_items
    assert knife in player.inventory.attack_slots


def test_use_item_unequips_equipped_weapon() -> None:
    """Using an already equipped item should unequip it."""
    controller = _make_controller()
    player = controller.gw.player

    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.equip_to_slot(knife, 0)

    # Knife should be equipped
    assert knife in player.inventory.attack_slots

    menu = DualPaneMenu(controller)
    menu.show()

    # Use the knife (should unequip)
    menu._equip_item(knife)

    # Knife should now be in stored items, not equipped
    assert knife in player.inventory._stored_items
    assert knife not in player.inventory.attack_slots


# -----------------------------------------------------------------------------
# Detail Panel Tests
# -----------------------------------------------------------------------------


def test_detail_panel_updates_on_cursor_move() -> None:
    """Detail item should update when cursor moves."""
    controller = _make_controller_with_inventory_items()
    menu = DualPaneMenu(controller)
    menu.show()

    # Initial detail item should be first item
    first_item = menu.left_options[0].data
    assert menu.detail_item == first_item

    # Move cursor down
    down_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.DOWN,
        sym=tcod.event.KeySym.DOWN,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(down_event)

    # Detail item should be second item
    second_item = menu.left_options[1].data
    assert menu.detail_item == second_item


def test_generate_item_detail_includes_name() -> None:
    """Item detail should include the item name."""
    controller = _make_controller()
    knife = COMBAT_KNIFE_TYPE.create()

    menu = DualPaneMenu(controller)
    header, _description, _stats = menu._generate_item_detail(knife)

    # Header lines should contain item name
    assert any("Combat Knife" in line for line in header)


def test_generate_item_detail_includes_size() -> None:
    """Item detail should include size information."""
    controller = _make_controller()
    knife = COMBAT_KNIFE_TYPE.create()

    menu = DualPaneMenu(controller)
    header, _description, _stats = menu._generate_item_detail(knife)

    # Header should mention size (Normal for combat knife)
    assert any("Normal" in line or "slot" in line for line in header)


# -----------------------------------------------------------------------------
# Equipped Item Display Tests
# -----------------------------------------------------------------------------


def test_equipped_items_show_slot_prefix() -> None:
    """Equipped items should display slot number in prefix_segments."""
    controller = _make_controller()
    player = controller.gw.player

    pistol = PISTOL_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the pistol option
    pistol_option = next((opt for opt in menu.left_options if opt.data == pistol), None)
    assert pistol_option is not None

    # Check prefix_segments contains slot indicator
    assert pistol_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in pistol_option.prefix_segments)
    assert "[1]" in combined_prefix


# -----------------------------------------------------------------------------
# Mouse Interaction Tests
# -----------------------------------------------------------------------------


def test_click_outside_closes_menu() -> None:
    """Clicking outside menu bounds should close it."""
    controller = _make_controller()
    menu = DualPaneMenu(controller)
    menu.show()
    menu._calculate_dimensions()

    assert menu.is_active

    # Click far outside the menu
    outside_pos = (1000, 1000)
    click_event = tcod.event.MouseButtonDown(
        outside_pos, outside_pos, tcod.event.MouseButton.LEFT
    )
    menu.handle_input(click_event)

    assert not menu.is_active


# -----------------------------------------------------------------------------
# Hint Line Wrapping Tests
# -----------------------------------------------------------------------------


def test_get_hint_lines_single_line_when_wide() -> None:
    """Hint text should fit on one line when menu is wide enough."""
    controller = _make_controller()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    # Full width (80 tiles) should fit hint on one line
    menu.width = 80
    lines = menu._get_hint_lines()

    assert len(lines) == 1
    # New hint format: "[Enter] Equip  [U] Use  [D] Drop  [Esc] Close"
    assert "[Enter] Equip" in lines[0]


def test_get_hint_lines_wraps_when_narrow() -> None:
    """Hint text should wrap to multiple lines when menu is narrow."""
    controller = _make_controller()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    # Very narrow width (25 tiles) should require wrapping even for short hints
    menu.width = 25
    lines = menu._get_hint_lines()

    assert len(lines) > 1


def test_get_hint_lines_preserves_all_hints() -> None:
    """All hint keywords should be preserved when wrapping."""
    controller = _make_controller()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    menu.width = 40
    lines = menu._get_hint_lines()
    combined = " ".join(lines)

    # Base hints always present (Use is context-sensitive, not always shown)
    # Single-pane: "[Enter] Equip  [D] Drop  [Esc] Close"
    assert "Enter" in combined
    assert "Equip" in combined
    assert "Drop" in combined
    assert "Esc" in combined


def test_get_hint_lines_splits_on_double_space() -> None:
    """Wrapping should prefer splitting on double-space boundaries."""
    controller = _make_controller()
    menu = DualPaneMenu(controller, source=None)
    menu.show()

    # Very narrow width to force wrapping
    menu.width = 25

    lines = menu._get_hint_lines()

    # First line should end cleanly at a double-space boundary, not mid-word.
    # Valid endings are: "Equip", "Drop", "Close", or a non-alpha character
    assert not lines[0].endswith("-")
    valid_word_endings = ("Equip", "Drop", "Close")
    assert not lines[0][-1].isalpha() or lines[0].endswith(valid_word_endings)


def test_get_hint_lines_hides_use_for_non_consumables() -> None:
    """[U] Use hint should only appear when a consumable is selected."""
    from unittest.mock import MagicMock

    from catley.game.items.item_core import Item

    controller = _make_controller()
    menu = DualPaneMenu(controller, source=None)
    menu.width = 80  # Set width to avoid wrapping issues

    # Create mock items - one with consumable_effect, one without
    weapon = MagicMock(spec=Item)
    weapon.consumable_effect = None

    consumable = MagicMock(spec=Item)
    consumable.consumable_effect = MagicMock()  # Has effect

    # Manually set up menu state without calling show()
    menu.left_options = [
        MagicMock(data=weapon, enabled=True),
        MagicMock(data=consumable, enabled=True),
    ]
    menu.active_pane = PaneId.LEFT

    # Select weapon - should NOT show Use
    menu.left_cursor = 0
    lines = menu._get_hint_lines()
    combined = " ".join(lines)
    assert "Use" not in combined

    # Select consumable - should show Use
    menu.left_cursor = 1
    lines = menu._get_hint_lines()
    combined = " ".join(lines)
    assert "Use" in combined


def test_get_hint_lines_loot_mode_different_text() -> None:
    """Loot mode should show different hint text including Tab."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On ground"))
    menu.show()

    menu.width = 80
    lines = menu._get_hint_lines()
    combined = " ".join(lines)

    # Loot mode should mention Tab for pane switching
    assert "Tab" in combined
    assert "Transfer" in combined


# -----------------------------------------------------------------------------
# Category Prefix Tests
# -----------------------------------------------------------------------------


def test_weapon_items_show_category_prefix() -> None:
    """Weapon items should display colored dot category indicator."""
    controller = _make_controller()
    player = controller.gw.player

    pistol = PISTOL_TYPE.create()
    player.inventory.add_to_inventory(pistol)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the pistol option
    pistol_option = next((opt for opt in menu.left_options if opt.data == pistol), None)
    assert pistol_option is not None

    # Check prefix_segments contains category dot indicator
    assert pistol_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in pistol_option.prefix_segments)
    assert "\u25cf" in combined_prefix


def test_consumable_items_show_category_prefix() -> None:
    """Consumable items should display colored dot category indicator."""
    controller = _make_controller()
    player = controller.gw.player

    stim = STIM_TYPE.create()
    player.inventory.add_to_inventory(stim)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the stim option
    stim_option = next((opt for opt in menu.left_options if opt.data == stim), None)
    assert stim_option is not None

    # Check prefix_segments contains category dot indicator
    assert stim_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in stim_option.prefix_segments)
    assert "\u25cf" in combined_prefix


def test_junk_items_show_category_prefix() -> None:
    """Junk items should display colored dot category indicator."""
    controller = _make_controller()
    player = controller.gw.player

    junk = JUNK_ITEM_TYPES[0].create()
    player.inventory.add_to_inventory(junk)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the junk option
    junk_option = next((opt for opt in menu.left_options if opt.data == junk), None)
    assert junk_option is not None

    # Check prefix_segments contains category dot indicator
    assert junk_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in junk_option.prefix_segments)
    assert "\u25cf" in combined_prefix


def test_munitions_items_show_category_prefix() -> None:
    """Munitions items should display colored dot category indicator."""
    controller = _make_controller()
    player = controller.gw.player

    ammo = PISTOL_MAGAZINE_TYPE.create()
    player.inventory.add_to_inventory(ammo)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the ammo option
    ammo_option = next((opt for opt in menu.left_options if opt.data == ammo), None)
    assert ammo_option is not None

    # Check prefix_segments contains category dot indicator
    assert ammo_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in ammo_option.prefix_segments)
    assert "\u25cf" in combined_prefix


def test_category_prefix_has_correct_color() -> None:
    """Category dot indicators should use the correct category color."""
    controller = _make_controller()
    player = controller.gw.player

    pistol = PISTOL_TYPE.create()
    player.inventory.add_to_inventory(pistol)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the pistol option
    pistol_option = next((opt for opt in menu.left_options if opt.data == pistol), None)
    assert pistol_option is not None
    assert pistol_option.prefix_segments is not None

    # Find the segment with the dot indicator
    category_segment = next(
        (seg for seg in pistol_option.prefix_segments if "\u25cf" in seg[0]), None
    )
    assert category_segment is not None
    assert category_segment[1] == colors.CATEGORY_WEAPON


def test_equipped_item_shows_both_slot_and_category() -> None:
    """Equipped items should show both slot number and category dot."""
    controller = _make_controller()
    player = controller.gw.player

    pistol = PISTOL_TYPE.create()
    player.inventory.equip_to_slot(pistol, 0)

    menu = DualPaneMenu(controller)
    menu.show()

    # Find the pistol option
    pistol_option = next((opt for opt in menu.left_options if opt.data == pistol), None)
    assert pistol_option is not None
    assert pistol_option.prefix_segments is not None

    # Check both slot and category dot are present
    combined_prefix = "".join(seg[0] for seg in pistol_option.prefix_segments)
    assert "[1]" in combined_prefix
    assert "\u25cf" in combined_prefix

    # Slot should come before category dot
    assert combined_prefix.index("[1]") < combined_prefix.index("\u25cf")


def test_right_pane_items_show_category_prefix() -> None:
    """Items in the right pane (loot) should show category dot indicator."""
    controller = _make_controller_with_ground_items()
    player = controller.gw.player
    location = (player.x, player.y)

    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On ground"))
    menu.show()

    # Find the stim option in right pane
    stim_option = next(
        (opt for opt in menu.right_options if opt.text == "Stimpack"), None
    )
    assert stim_option is not None

    # Check prefix_segments contains category dot indicator
    assert stim_option.prefix_segments is not None
    combined_prefix = "".join(seg[0] for seg in stim_option.prefix_segments)
    assert "\u25cf" in combined_prefix


def test_item_category_property() -> None:
    """Items should expose category through the category property."""
    pistol = PISTOL_TYPE.create()
    assert pistol.category == ItemCategory.WEAPON

    stim = STIM_TYPE.create()
    assert stim.category == ItemCategory.CONSUMABLE

    ammo = PISTOL_MAGAZINE_TYPE.create()
    assert ammo.category == ItemCategory.MUNITIONS

    junk = JUNK_ITEM_TYPES[0].create()
    assert junk.category == ItemCategory.JUNK


# -----------------------------------------------------------------------------
# Container Persistence Tests
# -----------------------------------------------------------------------------


def test_permanent_container_persists_when_emptied() -> None:
    """Permanent containers like bookcases should not be removed when emptied.

    Permanent containers have blocks_movement=True and should remain in the
    game world even after all items are transferred out, allowing players
    to put items back into them later.

    Containers are accessed via ActorInventorySource (not ExternalInventory),
    which is the path used when bumping into a container.
    """
    from catley.game.actors.container import create_bookcase
    from catley.view.ui.dual_pane_menu import ActorInventorySource
    from tests.helpers import get_controller_with_player_and_map

    controller = get_controller_with_player_and_map()
    player = controller.gw.player

    # Place bookcase adjacent to player (you bump into it to interact)
    bookcase_x, bookcase_y = player.x + 1, player.y
    item = COMBAT_KNIFE_TYPE.create()
    bookcase = create_bookcase(x=bookcase_x, y=bookcase_y, items=[item])
    controller.gw.add_actor(bookcase)

    # Verify bookcase is permanent (blocks movement)
    assert bookcase.blocks_movement is True
    assert len(bookcase.inventory) == 1

    # Open menu via ActorInventorySource (how containers are actually accessed)
    menu = DualPaneMenu(
        controller, source=ActorInventorySource(actor=bookcase, label="Bookcase")
    )
    menu.show()

    # Transfer the item
    menu._transfer_to_inventory(item)

    # Bookcase should still exist even though it's empty
    assert bookcase in controller.gw.actors, "Permanent container should persist"
    assert len(bookcase.inventory) == 0


def test_temporary_ground_pile_removed_when_emptied() -> None:
    """Temporary ground item piles should be removed when emptied.

    Temporary ground piles (created when items are dropped) have
    blocks_movement=False and should be removed from the game world
    once all items are picked up.
    """
    from tests.helpers import get_controller_with_player_and_map

    controller = get_controller_with_player_and_map()
    player = controller.gw.player
    location = (player.x, player.y)

    # Spawn a single item using the real spawning system (creates temp pile)
    item = COMBAT_KNIFE_TYPE.create()
    ground_pile = controller.gw.spawn_ground_item(item, *location)

    # Verify the pile is temporary (doesn't block movement)
    assert ground_pile.blocks_movement is False
    assert ground_pile in controller.gw.actors

    # Open menu at the pile location
    menu = DualPaneMenu(controller, source=ExternalInventory(location, "On the ground"))
    menu.show()

    # Transfer the item
    menu._transfer_to_inventory(item)

    # Ground pile should be removed since it's empty and temporary
    assert ground_pile not in controller.gw.actors, "Temporary pile should be removed"


def test_empty_permanent_container_can_be_searched() -> None:
    """Empty permanent containers should still be searchable to deposit items.

    When bumping into an empty bookcase, the menu should open so the player
    can transfer items INTO the container. The menu should be dual-pane (not
    single-pane) so the player can see both their inventory and the container.
    """
    from unittest.mock import MagicMock

    from catley.game.actions.environment import SearchContainerIntent
    from catley.game.actions.executors.containers import SearchContainerExecutor
    from catley.game.actors.container import create_bookcase
    from catley.view.ui.dual_pane_menu import ActorInventorySource
    from tests.helpers import get_controller_with_player_and_map

    controller = get_controller_with_player_and_map()
    player = controller.gw.player

    # Create an empty bookcase
    bookcase = create_bookcase(x=player.x + 1, y=player.y, items=[])
    controller.gw.add_actor(bookcase)

    # Verify it's empty and permanent
    assert len(bookcase.inventory) == 0
    assert bookcase.blocks_movement is True

    # Mock the overlay system to capture the menu
    captured_menu = None

    def capture_menu(menu: DualPaneMenu) -> None:
        nonlocal captured_menu
        captured_menu = menu

    controller.overlay_system = MagicMock()
    controller.overlay_system.show_overlay = capture_menu

    # Execute search action
    intent = SearchContainerIntent(controller, player, bookcase)
    executor = SearchContainerExecutor()
    result = executor.execute(intent)

    # Should succeed - empty permanent containers can be opened
    assert result.succeeded is True, "Empty permanent container should be searchable"

    # Verify the menu was created with a source (dual-pane, not single-pane)
    assert captured_menu is not None, "Menu should have been created"
    assert captured_menu.source is not None, "Menu should have a source (dual-pane)"
    assert isinstance(captured_menu.source, ActorInventorySource)
    assert captured_menu.source.actor is bookcase


# -----------------------------------------------------------------------------
# Arrow Key Scrolling Tests
# -----------------------------------------------------------------------------


def test_arrow_keys_scroll_detail_description() -> None:
    """Left/Right arrows should scroll detail panel when description overflows."""
    from catley.game.outfit import LEATHER_ARMOR_TYPE

    controller = _make_controller()
    player = controller.gw.player

    # Leather armor has a long description that should overflow
    armor = LEATHER_ARMOR_TYPE.create()
    player.inventory.add_item(armor)

    menu = DualPaneMenu(controller)
    menu.show()
    menu._calculate_dimensions()  # Initialize _detail_panel

    # Verify the panel exists and has overflow after showing the menu
    assert menu._detail_panel is not None, "Detail panel should be initialized"

    # The menu should show the armor as the only item, so it should be selected
    assert menu.detail_item is armor, "Armor should be selected"

    # Check if there's overflow (may depend on description length)
    if menu._detail_panel.has_overflow():
        initial_offset = menu._detail_panel.scroll_offset

        # Right arrow should scroll down
        right_event = tcod.event.KeyDown(
            scancode=tcod.event.Scancode.RIGHT,
            sym=tcod.event.KeySym.RIGHT,
            mod=tcod.event.Modifier.NONE,
        )
        menu.handle_input(right_event)
        assert menu._detail_panel.scroll_offset > initial_offset, (
            "Right arrow should scroll down"
        )

        # Left arrow should scroll back up
        left_event = tcod.event.KeyDown(
            scancode=tcod.event.Scancode.LEFT,
            sym=tcod.event.KeySym.LEFT,
            mod=tcod.event.Modifier.NONE,
        )
        menu.handle_input(left_event)
        assert menu._detail_panel.scroll_offset == initial_offset, (
            "Left arrow should scroll back up"
        )


def test_arrow_keys_no_effect_without_overflow() -> None:
    """Arrow keys should have no effect when description doesn't overflow."""
    controller = _make_controller()
    player = controller.gw.player

    # Combat knife has a short description that shouldn't overflow
    knife = COMBAT_KNIFE_TYPE.create()
    player.inventory.add_to_inventory(knife)

    menu = DualPaneMenu(controller)
    menu.show()
    menu._calculate_dimensions()  # Initialize _detail_panel

    assert menu._detail_panel is not None
    initial_offset = menu._detail_panel.scroll_offset

    # Right arrow should not crash and should not change offset
    right_event = tcod.event.KeyDown(
        scancode=tcod.event.Scancode.RIGHT,
        sym=tcod.event.KeySym.RIGHT,
        mod=tcod.event.Modifier.NONE,
    )
    menu.handle_input(right_event)
    assert menu._detail_panel.scroll_offset == initial_offset


# -----------------------------------------------------------------------------
# Condition Detail Generation Tests
# -----------------------------------------------------------------------------


def test_generate_item_detail_for_condition() -> None:
    """Condition detail should include name and description."""
    from catley.game.actors.conditions import Condition

    controller = _make_controller()
    menu = DualPaneMenu(controller)

    # Create a condition with description
    condition = Condition(
        name="Test Condition",
        description="This is a test condition description.",
    )

    header, description, stats = menu._generate_item_detail(condition)

    # Header should contain "Condition: <name>"
    assert len(header) == 1
    assert "Condition:" in header[0]
    assert "Test Condition" in header[0]

    # Description should contain the condition's description
    assert len(description) == 1
    assert description[0] == "This is a test condition description."

    # Stats should be None for conditions
    assert stats is None


def test_generate_item_detail_for_condition_without_description() -> None:
    """Condition without description should still work."""
    from catley.game.actors.conditions import Condition

    controller = _make_controller()
    menu = DualPaneMenu(controller)

    condition = Condition(name="Empty Condition")

    header, description, stats = menu._generate_item_detail(condition)

    assert "Empty Condition" in header[0]
    assert description == []
    assert stats is None


def test_generate_item_detail_before_show() -> None:
    """_generate_item_detail should work even before show() is called.

    This tests the guard against uninitialized canvas state.
    """
    controller = _make_controller()
    menu = DualPaneMenu(controller)

    # Don't call show() - canvas and _char_width will be uninitialized
    knife = COMBAT_KNIFE_TYPE.create()

    # Should not crash
    header, description, _stats = menu._generate_item_detail(knife)

    # Should still produce valid output
    assert any("Combat Knife" in line for line in header)
    # Description should be raw (not wrapped) since canvas isn't initialized
    if knife.description:
        assert knife.description in description
