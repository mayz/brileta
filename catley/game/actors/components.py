"""
Component system for game actor capabilities.

Implements the component pattern to break down complex actor behavior
into focused, reusable pieces. Instead of having monolithic "god classes" that try
to handle everything (stats, health, inventory, AI, rendering, etc.), actors are
composed of specialized components that each handle one specific capability.

This approach follows the principle of "composition over inheritance".

Components:
    StatsComponent: Core ability scores (strength, toughness, etc.) and derived values
    HealthComponent: Physical integrity - HP, armor, damage, healing, death
    InventoryComponent: Item storage and equipment management
    VisualEffectsComponent: Visual feedback effects like damage flashing

Benefits:
    - Each component has a single, clear responsibility
    - Components can be mixed and matched (not all actors need all components)
    - Easier to test, debug, and extend individual behaviors
    - Avoids the "everything in one class" problem that leads to unmaintainable code
    - Components can be data-driven and configured independently

Usage:
    Components are typically created and composed together in Actor.__init__():

    self.stats = StatsComponent(strength=5, toughness=10)
    self.health = HealthComponent(self.stats)
    self.inventory = InventoryComponent(self.stats)
    self.visual_effects = VisualEffectsComponent()

Note:
    Components may reference each other (e.g., HealthComponent needs StatsComponent
    for max_hp calculation), but dependencies are kept minimal and explicit.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catley.game.outfit import OutfitCapability
    from catley.view.render.effects.effects import ContinuousEffect

    from .core import Actor
    from .idle_animation import IdleAnimationProfile
    from .status_effects import StatusEffect

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED
from catley.constants.movement import MovementConstants
from catley.events import MessageEvent, publish_event
from catley.game.enums import ItemSize
from catley.game.items.item_core import Item

from .conditions import Condition, Exhaustion
from .status_effects import EncumberedEffect, StatusEffect


@dataclass(slots=True)
class StatsComponent:
    """Manages an Actor's core ability scores and derived stats."""

    strength: int = 0
    toughness: int = 0
    agility: int = 0
    observation: int = 0
    intelligence: int = 0
    demeanor: int = 0
    weirdness: int = 0

    @property
    def max_hp(self) -> int:
        """Derive max HP from toughness."""
        return self.toughness + 5

    @property
    def inventory_slots(self) -> int:
        """Derive inventory capacity from strength."""
        return self.strength + 5


@dataclass(slots=True)
class HealthComponent:
    """Handles physical integrity - HP, damage from any source, healing.

    Note: Armor protection is now handled by the outfit system (catley.game.outfit).
    This component only tracks HP. Damage reduction from armor happens in the
    combat system before calling take_damage().
    """

    stats: StatsComponent
    hp: int = field(init=False)

    def __post_init__(self) -> None:
        self.hp = self.stats.max_hp

    @property
    def max_hp(self) -> int:
        """Get max HP from stats component."""
        return self.stats.max_hp

    def take_damage(self, amount: int, damage_type: str = "normal") -> None:
        """Handle damage to the actor.

        Note: Armor protection is applied before this method is called.
        The outfit system (catley.game.outfit) handles damage reduction.
        This method only applies the final damage amount to HP.

        Args:
            amount: Amount of damage to take (after armor reduction)
            damage_type: "normal", "radiation", or "armor_piercing"
                (damage_type is kept for interface compatibility but no longer
                affects armor - armor-piercing is handled by the outfit system)
        """
        if amount > 0:
            self.hp = max(0, self.hp - amount)

    def heal(self, amount: int) -> None:
        """Heal the actor by the specified amount, up to max_hp.

        Args:
            amount: Amount to heal
        """
        self.hp = min(self.max_hp, self.hp + amount)

    def is_alive(self) -> bool:
        """Return True if the actor is alive (HP > 0)."""
        return self.hp > 0


class InventoryComponent:
    """Base class for item storage.

    Provides a common interface for item storage that both character inventories
    (with stats-based capacity, equipment slots, conditions) and simple container
    storage (fixed capacity, items only) can implement.

    Subclasses:
        CharacterInventory: Stats-backed inventory with equipment slots and conditions
        ContainerStorage: Fixed-capacity simple item storage for containers
    """

    def __init__(self, actor: Actor | None = None) -> None:
        self.actor = actor  # Back-reference for status effects
        self.revision = 0  # Revision counter for UI caching

    def _increment_revision(self) -> None:
        """Bump the revision to signal a state change."""
        self.revision += 1

    @property
    def capacity(self) -> int:
        """Get the maximum capacity of this inventory. Subclasses must implement."""
        raise NotImplementedError

    def get_items(self) -> list[Item]:
        """Get a copy of all items in this inventory. Subclasses must implement."""
        raise NotImplementedError

    def add_item(self, item: Item) -> tuple[bool, str]:
        """Add an item to this inventory. Subclasses must implement."""
        raise NotImplementedError

    def remove_item(self, item: Item) -> bool:
        """Remove an item from this inventory. Subclasses must implement."""
        raise NotImplementedError

    def has_item(self, item: Item) -> bool:
        """Check if this inventory contains the given item."""
        return item in self.get_items()

    def __iter__(self):
        """Allow iteration over items. Subclasses must implement."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return number of stored items. Subclasses must implement."""
        raise NotImplementedError

    def __contains__(self, item: object) -> bool:
        """Check if item is in this inventory. Subclasses must implement."""
        raise NotImplementedError


class ContainerStorage(InventoryComponent):
    """Fixed-capacity storage for containers.

    A simple item storage implementation for containers like crates, chests,
    lockers, etc. Unlike CharacterInventory, this class:
    - Has a fixed capacity (not stats-based)
    - Stores only items (not conditions)
    - Has no equipment slots
    - Has no encumbrance mechanics
    """

    def __init__(
        self,
        capacity: int = 10,
        actor: Actor | None = None,
    ) -> None:
        super().__init__(actor)
        self._capacity = capacity
        self._items: list[Item] = []

    @property
    def capacity(self) -> int:
        """Get the fixed capacity of this container."""
        return self._capacity

    def get_items(self) -> list[Item]:
        """Get a copy of all items in this container."""
        return self._items.copy()

    def add_item(self, item: Item) -> tuple[bool, str]:
        """Add an item to this container if space allows."""
        if len(self._items) >= self._capacity:
            return False, f"Cannot add {item.name} - container full!"
        self._items.append(item)
        self._increment_revision()
        return True, f"Added {item.name}."

    def remove_item(self, item: Item) -> bool:
        """Remove an item from this container."""
        if item in self._items:
            self._items.remove(item)
            self._increment_revision()
            return True
        return False

    def is_empty(self) -> bool:
        """Return True if no items are stored."""
        return len(self._items) == 0

    def has_items(self) -> bool:
        """Return True if there are stored items."""
        return len(self._items) > 0

    def __iter__(self):
        """Allow iteration over items."""
        return iter(self._items)

    def __len__(self) -> int:
        """Return number of items."""
        return len(self._items)

    def __contains__(self, item: object) -> bool:
        """Check if item is in this container."""
        return item in self._items


class CharacterInventory(InventoryComponent):
    """Stats-backed inventory with equipment slots for characters.

    This is the full-featured inventory used by player characters and NPCs.
    It includes:
    - Stats-based capacity (derived from strength)
    - Equipment/ready slots for weapons
    - Outfit slot for armor/clothing
    - Support for conditions (injuries, status effects that take inventory space)
    - Encumbrance mechanics
    """

    def __init__(
        self,
        stats_component: StatsComponent,
        num_ready_slots: int = 2,
        actor: Actor | None = None,
    ) -> None:
        super().__init__(actor)
        self.stats = stats_component
        self._stored_items: list[Item | Condition] = []

        self.ready_slots: list[Item | None] = [None] * num_ready_slots
        self.active_slot: int = 0

        # Outfit slot (armor/clothing) - stores tuple of (Item, OutfitCapability)
        self._equipped_outfit: tuple[Item, OutfitCapability] | None = None

    @property
    def capacity(self) -> int:
        """Get capacity from stats component."""
        return self.stats.inventory_slots

    def get_items(self) -> list[Item]:
        """Get a copy of all items (excluding conditions) in this inventory."""
        return [item for item in self._stored_items if isinstance(item, Item)]

    def add_item(self, item: Item) -> tuple[bool, str]:
        """Add an item to this inventory (alias for add_voluntary_item)."""
        return self.add_voluntary_item(item)

    def remove_item(self, item: Item) -> bool:
        """Remove an item from this inventory (alias for remove_from_inventory)."""
        return self.remove_from_inventory(item)

    def __iter__(self):
        """Allow iteration over stored items."""
        return iter(self._stored_items)

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self._stored_items)

    def __contains__(self, item: object) -> bool:
        """Check if item is in stored items."""
        return item in self._stored_items

    def is_empty(self) -> bool:
        """Return True if no items are stored."""
        return len(self._stored_items) == 0

    def has_items(self) -> bool:
        """Return True if there are stored items."""
        return len(self._stored_items) > 0

    @property
    def total_inventory_slots(self) -> int:
        """Get max inventory slots from stats."""
        return self.stats.inventory_slots

    # === Storage Management ===

    def get_used_inventory_slots(self) -> int:
        """Calculate inventory slots currently used (includes equipped items)."""
        used_space = 0
        has_tiny_items = False

        # Count stored items
        for entity_in_slot in self._stored_items:
            if isinstance(entity_in_slot, Item):
                item = entity_in_slot
                if item.size == ItemSize.TINY:
                    has_tiny_items = True
                elif item.size == ItemSize.NORMAL:
                    used_space += 1
                elif item.size == ItemSize.BIG:
                    used_space += 2
                elif item.size == ItemSize.HUGE:
                    used_space += 4
            elif isinstance(entity_in_slot, Condition):
                used_space += 1

        # Count equipped weapons
        for equipped_item in self.ready_slots:
            if equipped_item:
                if equipped_item.size == ItemSize.TINY:
                    has_tiny_items = True
                elif equipped_item.size == ItemSize.NORMAL:
                    used_space += 1
                elif equipped_item.size == ItemSize.BIG:
                    used_space += 2
                elif equipped_item.size == ItemSize.HUGE:
                    used_space += 4

        # Count equipped outfit
        if self._equipped_outfit is not None:
            outfit_item, _ = self._equipped_outfit
            if outfit_item.size == ItemSize.TINY:
                has_tiny_items = True
            elif outfit_item.size == ItemSize.NORMAL:
                used_space += 1
            elif outfit_item.size == ItemSize.BIG:
                used_space += 2
            elif outfit_item.size == ItemSize.HUGE:
                used_space += 4

        if has_tiny_items:
            used_space += 1

        return used_space

    def is_encumbered(self) -> bool:
        """Return True if carrying more than base capacity."""
        return self.get_used_inventory_slots() > self.stats.inventory_slots

    def get_slots_over_capacity(self) -> int:
        """Return the number of slots over base capacity, or 0 if not encumbered.

        Used for graduated encumbrance penalties - the more over capacity,
        the greater the speed penalty.
        """
        over = self.get_used_inventory_slots() - self.stats.inventory_slots
        return max(0, over)

    def can_add_voluntary_item(self, item: Item) -> bool:
        """Check if a voluntary item can be added within base capacity."""
        current_used = self.get_used_inventory_slots()
        additional_space = self._calculate_space_needed(item)
        return (current_used + additional_space) <= self.stats.inventory_slots

    def _calculate_space_needed(self, item: Item | Condition) -> int:
        """Calculate how many slots an item needs."""
        if isinstance(item, Item):
            if item.size == ItemSize.TINY:
                has_tiny = any(
                    isinstance(i, Item) and i.size == ItemSize.TINY
                    for i in self._stored_items
                ) or any(
                    eq_item is not None and eq_item.size == ItemSize.TINY
                    for eq_item in self.ready_slots
                )
                return 0 if has_tiny else 1
            if item.size == ItemSize.NORMAL:
                return 1
            if item.size == ItemSize.BIG:
                return 2
            if item.size == ItemSize.HUGE:
                return 4
        if isinstance(item, Condition):
            return 1
        return 1

    def add_voluntary_item(self, item: Item) -> tuple[bool, str]:
        """Add a voluntary item if space allows."""
        if self.can_add_voluntary_item(item):
            self._stored_items.append(item)
            self._update_encumbrance_status()
            self._increment_revision()
            return True, f"Added {item.name}."
        return False, f"Cannot carry {item.name} - inventory full!"

    def add_condition(self, condition: Condition) -> tuple[list[Item], str]:
        """Add a condition to inventory. May cause encumbrance if over capacity.

        Conditions (injuries, sickness, etc.) are involuntary - they're always added.
        If this pushes the player over capacity, they become encumbered and must
        manually decide what to drop. This creates meaningful risk/reward tension:
        carrying more loot means risking encumbrance if injured.

        Returns:
            A tuple of (dropped_items, message). dropped_items is always empty
            since conditions don't auto-drop items anymore.
        """
        self._stored_items.append(condition)
        self._update_encumbrance_status()
        self._increment_revision()

        message = f"Added {condition.name}."
        if self.is_encumbered():
            message += " You are now encumbered!"

        return [], message

    def _update_encumbrance_status(self) -> None:
        """Add or remove encumbrance status effect based on current load."""
        if not self.actor:
            return

        currently_encumbered = self.actor.status_effects.has_status_effect(
            EncumberedEffect
        )
        should_be_encumbered = self.is_encumbered()

        if should_be_encumbered and not currently_encumbered:
            self.actor.status_effects.apply_status_effect(EncumberedEffect())
            publish_event(
                MessageEvent(f"{self.actor.name} is encumbered!", colors.YELLOW)
            )
        elif not should_be_encumbered and currently_encumbered:
            self.actor.status_effects.remove_status_effect(EncumberedEffect)
            publish_event(
                MessageEvent(
                    f"{self.actor.name} is no longer encumbered.", colors.GREEN
                )
            )

    def can_add_to_inventory(self, item_to_add: Item | Condition) -> bool:
        """DEPRECATED. Use ``can_add_voluntary_item`` or ``add_condition``."""
        if isinstance(item_to_add, Item):
            return self.can_add_voluntary_item(item_to_add)
        return True

    def add_to_inventory(self, item: Item | Condition) -> tuple[bool, str, list[Item]]:
        """Add item to inventory. Returns (success, message, dropped_items)."""
        if isinstance(item, Condition):
            dropped_items, message = self.add_condition(item)
            return True, message, dropped_items
        success, message = self.add_voluntary_item(item)
        return success, message, []

    def remove_from_inventory(self, item: Item | Condition) -> bool:
        """Remove item from inventory."""
        if item in self._stored_items:
            self._stored_items.remove(item)
            self._update_encumbrance_status()
            self._increment_revision()
            return True
        return False

    def try_remove_item(self, item: Item) -> bool:
        """Remove an item from either equipped slots or stored inventory.

        Checks equipped ready slots first, then equipped outfit, then stored
        inventory. Stops after finding and removing the item from one location.

        Args:
            item: The exact item instance to remove.

        Returns:
            True if the item was found and removed, False otherwise.
        """
        # Check equipped ready slots first
        for slot_idx, equipped in enumerate(self.ready_slots):
            if equipped is item:
                self.unequip_slot(slot_idx)
                return True

        # Check equipped outfit
        if self._equipped_outfit is not None:
            outfit_item, _ = self._equipped_outfit
            if outfit_item is item:
                self._equipped_outfit = None
                self._update_encumbrance_status()
                self._increment_revision()
                return True

        # Check stored inventory
        return self.remove_from_inventory(item)

    def get_inventory_slot_colors(self) -> list[colors.Color]:
        """Get colors representing filled inventory slots."""
        slot_colors: list[colors.Color] = []
        has_processed_tiny_slot = False

        # Process stored items
        for entity_in_slot in self._stored_items:
            if isinstance(entity_in_slot, Item):
                item = entity_in_slot
                item_bar_color = colors.WHITE

                if item.size == ItemSize.TINY:
                    if not has_processed_tiny_slot:
                        slot_colors.append(item_bar_color)
                        has_processed_tiny_slot = True
                elif item.size == ItemSize.NORMAL:
                    slot_colors.append(item_bar_color)
                elif item.size == ItemSize.BIG:
                    slot_colors.extend([item_bar_color] * 2)
                elif item.size == ItemSize.HUGE:
                    slot_colors.extend([item_bar_color] * 4)
            elif isinstance(entity_in_slot, Condition):
                slot_colors.append(entity_in_slot.display_color)

        # Process equipped weapons
        for equipped_item in self.ready_slots:
            if equipped_item:
                item_bar_color = colors.WHITE

                if equipped_item.size == ItemSize.TINY:
                    if not has_processed_tiny_slot:
                        slot_colors.append(item_bar_color)
                        has_processed_tiny_slot = True
                elif equipped_item.size == ItemSize.NORMAL:
                    slot_colors.append(item_bar_color)
                elif equipped_item.size == ItemSize.BIG:
                    slot_colors.extend([item_bar_color] * 2)
                elif equipped_item.size == ItemSize.HUGE:
                    slot_colors.extend([item_bar_color] * 4)

        # Process equipped outfit
        if self._equipped_outfit is not None:
            outfit_item, _ = self._equipped_outfit
            armor_color = colors.WHITE

            if outfit_item.size == ItemSize.TINY:
                if not has_processed_tiny_slot:
                    slot_colors.append(armor_color)
                    has_processed_tiny_slot = True
            elif outfit_item.size == ItemSize.NORMAL:
                slot_colors.append(armor_color)
            elif outfit_item.size == ItemSize.BIG:
                slot_colors.extend([armor_color] * 2)
            elif outfit_item.size == ItemSize.HUGE:
                slot_colors.extend([armor_color] * 4)

        return slot_colors

    # === Equipment Management ===

    def switch_to_slot(self, slot_index: int) -> bool:
        """Switch to a specific ready slot. Returns True if successful."""
        if 0 <= slot_index < len(self.ready_slots):
            self.active_slot = slot_index
            self._increment_revision()
            return True
        return False

    def get_active_item(self) -> Item | None:
        """Get the currently active equipped item."""
        if 0 <= self.active_slot < len(self.ready_slots):
            return self.ready_slots[self.active_slot]
        return None

    @property
    def num_ready_slots(self) -> int:
        """Get number of ready slots this character has"""
        return len(self.ready_slots)

    def equip_to_slot(self, item: Item, slot_index: int = 0) -> Item | None:
        """Equip item to specified slot, return what was there before"""
        if not (0 <= slot_index < len(self.ready_slots)):
            raise ValueError(f"Invalid slot index: {slot_index}")

        old_item = self.ready_slots[slot_index]
        self.ready_slots[slot_index] = item
        self._increment_revision()
        return old_item

    def unequip_slot(self, slot_index: int) -> Item | None:
        """Unequip specified slot, return what was equipped"""
        if not (0 <= slot_index < len(self.ready_slots)):
            raise ValueError(f"Invalid slot index: {slot_index}")

        item = self.ready_slots[slot_index]
        self.ready_slots[slot_index] = None
        if item is not None:
            self._increment_revision()
        return item

    def equip_from_inventory(self, item: Item, slot_index: int) -> tuple[bool, str]:
        """Equip ``item`` from stored inventory into ``slot_index``.

        This handles removing the item from ``_stored_items`` and returning
        any previously equipped item back to storage. The operation either
        succeeds entirely or is reverted.

        Returns a tuple ``(success, message)`` describing the result.
        """

        if item not in self._stored_items:
            return False, f"{item.name} is not in inventory"
        if not (0 <= slot_index < len(self.ready_slots)):
            return False, f"Invalid slot index: {slot_index}"

        # Remove the item from stored inventory first
        self._stored_items.remove(item)

        old_item = self.ready_slots[slot_index]
        if old_item is not None:
            if not self.can_add_voluntary_item(old_item):
                # Revert and fail
                self._stored_items.append(item)
                return False, f"No room to unequip {old_item.name}"
            self._stored_items.append(old_item)

        self.ready_slots[slot_index] = item
        self._update_encumbrance_status()
        self._increment_revision()

        if old_item is not None:
            return True, f"Equipped {item.name}; unequipped {old_item.name}."
        return True, f"Equipped {item.name}."

    def unequip_to_inventory(self, slot_index: int) -> tuple[bool, str]:
        """Unequip the item in ``slot_index`` and store it."""
        if not (0 <= slot_index < len(self.ready_slots)):
            return False, f"Invalid slot index: {slot_index}"

        item = self.ready_slots[slot_index]
        if item is None:
            return False, "No item to unequip"

        # No capacity check needed - moving from equipped to stored doesn't
        # change total used space (same size calculation applies to both).
        self.ready_slots[slot_index] = None
        self._stored_items.append(item)
        self._update_encumbrance_status()
        self._increment_revision()
        return True, f"Unequipped {item.name}."

    def get_equipped_items(self) -> list[tuple[Item, int]]:
        """Get all equipped items with their slot index"""
        items = []
        for i, item in enumerate(self.ready_slots):
            if item is not None:
                items.append((item, i))
        return items

    def get_slot_display_name(self, slot_index: int) -> str:
        """Get display name for a slot (Primary, Secondary, etc.)"""
        names = ["Primary", "Secondary", "Tertiary", "Quaternary"]
        if slot_index < len(names):
            return names[slot_index]
        return f"Attack {slot_index + 1}"

    def get_available_attacks(self) -> list[tuple[Item | None, int, str]]:
        """Get all possible attacks including unarmed slots"""
        attacks = []
        for i, item in enumerate(self.ready_slots):
            display_name = self.get_slot_display_name(i)
            attacks.append((item, i, display_name))
        return attacks

    # === Outfit Management ===

    @property
    def equipped_outfit(self) -> tuple[Item, OutfitCapability] | None:
        """Get the currently equipped outfit (item and capability tuple).

        Returns:
            Tuple of (Item, OutfitCapability) if an outfit is equipped, None otherwise.
        """
        return self._equipped_outfit

    @property
    def outfit_capability(self) -> OutfitCapability | None:
        """Get the outfit capability for the equipped outfit.

        Returns:
            The OutfitCapability if an outfit is equipped, None otherwise.
        """
        if self._equipped_outfit is None:
            return None
        return self._equipped_outfit[1]

    def equip_outfit(self, item: Item) -> tuple[bool, str]:
        """Equip an outfit from stored inventory.

        The item must be in _stored_items and be a valid outfit type.
        If an outfit is already equipped, it will be returned to inventory first.

        Uses the item's existing outfit_capability (created when the item was
        instantiated) so damage state persists across equip/unequip cycles.

        Args:
            item: The outfit item to equip.

        Returns:
            Tuple of (success, message).
        """
        from catley.game.outfit import get_outfit_spec

        # Check if item is in inventory
        if item not in self._stored_items:
            return False, f"{item.name} is not in inventory"

        # Check if item is an outfit (has a spec)
        spec = get_outfit_spec(item.item_type)
        if spec is None:
            return False, f"{item.name} is not an outfit"

        # Check that item has an outfit capability (should always exist for outfits)
        if item.outfit_capability is None:
            return False, f"{item.name} has no outfit capability"

        # Unequip current outfit first if any
        if self._equipped_outfit is not None:
            old_item, _ = self._equipped_outfit
            # Check if we can store the old outfit (item taking its place)
            # Since we're removing one item and adding another of potentially
            # different size, we need to check capacity
            if not self.can_add_voluntary_item(old_item):
                return False, f"No room to unequip {old_item.name}"
            self._stored_items.append(old_item)
            self._equipped_outfit = None

        # Remove from stored items and equip using item's existing capability
        self._stored_items.remove(item)
        self._equipped_outfit = (item, item.outfit_capability)

        self._update_encumbrance_status()
        self._increment_revision()

        return True, f"Equipped {item.name}."

    def unequip_outfit(self) -> tuple[bool, str]:
        """Unequip the current outfit and store it in inventory.

        Returns:
            Tuple of (success, message).
        """
        if self._equipped_outfit is None:
            return False, "No outfit equipped"

        item, _ = self._equipped_outfit

        # No capacity check needed - moving from equipped to stored doesn't
        # change total used space (same size calculation applies to both).
        self._stored_items.append(item)
        self._equipped_outfit = None

        self._update_encumbrance_status()
        self._increment_revision()

        return True, f"Unequipped {item.name}."

    def set_starting_outfit(self, item: Item) -> None:
        """Set the starting outfit for a character (doesn't come from inventory).

        Used for character creation / spawning where the outfit doesn't need
        to be removed from inventory first. Uses the item's existing capability.

        Args:
            item: The outfit item with outfit_capability set.

        Raises:
            ValueError: If the item has no outfit_capability.
        """
        if item.outfit_capability is None:
            raise ValueError(f"{item.name} has no outfit capability")
        self._equipped_outfit = (item, item.outfit_capability)
        self._increment_revision()


class VisualEffectsComponent:
    """Handles visual feedback effects like flashing, continuous effects, and idle.

    Idle animations make characters feel alive between turns through:
    - Sub-tile drift: Gentle positional movement within the tile
    - Breathing pulse: Subtle brightness variation synchronized with breathing rate
    - Ambient particles: Optional breath vapor and dust effects

    All idle animation parameters are configured via an IdleAnimationProfile.
    """

    def __init__(self, idle_profile: IdleAnimationProfile | None = None) -> None:
        """Initialize the visual effects component.

        Args:
            idle_profile: Configuration for idle animations. If None, a default
                profile for medium-sized creatures is created on first access.
        """
        # Flash effect state
        self._flash_color: colors.Color | None = None
        self._flash_duration_frames: int = 0

        # Active continuous effects (from the effect library)
        self.continuous_effects: list[ContinuousEffect] = []

        # Idle animation state - uses real clock time for consistent animation speed
        self._idle_profile: IdleAnimationProfile | None = idle_profile
        self._creation_time: float = time.perf_counter()

        # Use id(self) as seed for visual randomness - doesn't affect game random state
        # This ensures actors are desynchronized without consuming global random numbers
        visual_seed = id(self) % 100000
        self._idle_timer_offset: float = (visual_seed % 1000) / 100.0  # 0-10 range

        # Weight-shifting direction state
        # The oscillation happens along a randomly chosen axis that changes periodically
        self._shift_direction_angle: float = (visual_seed % 628) / 100.0  # 0 to ~2pi
        self._last_direction_change: float = self._creation_time
        # Direction changes every 1-2 minutes (60-120 seconds)
        self._shift_direction_duration: float = 60.0 + (visual_seed % 6000) / 100.0

        # Per-actor speed variation (±20%) so not everyone sways at the same tempo
        # Range: 0.8 to 1.2
        self._speed_multiplier: float = 0.8 + ((visual_seed % 400) / 1000.0)

    @property
    def idle_profile(self) -> IdleAnimationProfile:
        """Get the idle animation profile, creating a default if needed."""
        if self._idle_profile is None:
            # Lazy import to avoid circular dependency
            from .idle_animation import IdleAnimationProfile

            self._idle_profile = IdleAnimationProfile()
        return self._idle_profile

    def set_idle_profile(self, profile: IdleAnimationProfile) -> None:
        """Change the idle animation profile.

        Args:
            profile: The new profile to use for idle animations.
        """
        self._idle_profile = profile

    def get_idle_drift_offset(self) -> tuple[float, float]:
        """Get the current positional drift offset for rendering.

        Returns the weight-shifting offset that makes characters subtly sway.
        The result should be added to the actor's rendered position.

        Uses real clock time for consistent animation speed regardless of
        frame rate or how often this method is called.

        Returns:
            Tuple of (x_offset, y_offset) in tile fractions.
        """
        profile = self.idle_profile
        x_offset = 0.0
        y_offset = 0.0

        # Use real time since creation for consistent speed
        current_time = time.perf_counter()
        elapsed = current_time - self._creation_time + self._idle_timer_offset

        # Weight-shifting drift - oscillates along a direction that changes periodically
        if profile.drift_enabled:
            # Check if it's time to change direction (every 1-2 minutes)
            time_since_direction_change = current_time - self._last_direction_change
            if time_since_direction_change >= self._shift_direction_duration:
                self._last_direction_change = current_time
                # Use time-based pseudo-random to pick new direction
                pseudo_seed = int(current_time * 1000) % 100000
                self._shift_direction_angle = (pseudo_seed % 628) / 100.0
                self._shift_direction_duration = 60.0 + (pseudo_seed % 6000) / 100.0

            # Oscillate back and forth (sine wave)
            # Apply per-actor speed variation so not everyone sways at the same tempo
            t = elapsed * profile.drift_speed * self._speed_multiplier
            oscillation = math.sin(t * 2 * math.pi) * profile.drift_amplitude
            # Apply oscillation along the current shift direction
            x_offset += oscillation * math.cos(self._shift_direction_angle)
            y_offset += oscillation * math.sin(self._shift_direction_angle)

        return (x_offset, y_offset)

    def flash(self, color: colors.Color, duration_frames: int = 5) -> None:
        """Start a flash effect with the given color and duration.

        Args:
            color: RGB color tuple for the flash
            duration_frames: How many frames the flash should last
        """
        self._flash_color = color
        self._flash_duration_frames = duration_frames

    def add_continuous_effect(self, effect: ContinuousEffect) -> None:
        """Add a continuous effect (like fire, aura, etc) to this component."""
        self.continuous_effects.append(effect)

    def update(self, delta_time: float = 0.016) -> None:
        """Update visual effects. Call this each frame.

        Args:
            delta_time: Time elapsed since last update in seconds.

        Note:
            Idle animations (drift, breathing) use real clock time via
            time.perf_counter() for consistent speed regardless of frame rate,
            so they don't need updating here. Direction changes for weight-
            shifting are handled lazily in get_idle_drift_offset().
        """
        # Update flash effect
        if self._flash_duration_frames > 0:
            self._flash_duration_frames -= 1
            if self._flash_duration_frames == 0:
                self._flash_color = None

        # Update all continuous effects (particles, etc.)
        for effect in self.continuous_effects:
            if hasattr(effect, "update"):
                effect.update(delta_time)

    def get_flash_color(self) -> colors.Color | None:
        """Get the current flash color for rendering, or None if no flash."""
        return self._flash_color if self._flash_duration_frames > 0 else None

    def has_continuous_effects(self) -> bool:
        """Check if this component has any continuous effects active."""
        return len(self.continuous_effects) > 0

    def remove_continuous_effect(self, effect: ContinuousEffect) -> None:
        """Remove a continuous effect from this component."""
        if effect in self.continuous_effects:
            if hasattr(effect, "stop"):
                effect.stop()
            self.continuous_effects.remove(effect)


class ModifiersComponent:
    """A facade for querying all of an actor's status effects and conditions.

    This component provides a single, unified interface for accessing modifiers
    from both temporary StatusEffects (stored on the Actor) and long-term
    Conditions (stored in the InventoryComponent). It centralizes the logic
    for combining these effects, simplifying systems like combat resolution,
    movement, and UI display.

    It does not have its own storage; it queries the underlying data sources
    on the actor directly.
    """

    __slots__ = ("actor", "revision")

    def __init__(self, actor: Actor) -> None:
        self.actor = actor
        # Revision counter increments whenever modifiers change
        self.revision = 0

    def _increment_revision(self) -> None:
        self.revision += 1

    def get_all_status_effects(self) -> list[StatusEffect]:
        """Returns a list of all active StatusEffect instances."""
        return self.actor.status_effects.get_all_status_effects()

    def get_all_conditions(self) -> list[Condition]:
        """Returns a list of all active Condition instances."""
        if not self.actor.conditions:
            return []
        return self.actor.conditions.get_all_conditions()

    def get_all_active_effects(self) -> list[StatusEffect | Condition]:
        """Returns a combined list of all active status effects and conditions.

        This provides a unified view of all temporary and long-term effects
        affecting the actor for UI display purposes.

        Returns:
            List containing all StatusEffect and Condition instances
        """
        all_effects: list[StatusEffect | Condition] = []
        all_effects.extend(self.get_all_status_effects())
        all_effects.extend(self.get_all_conditions())
        return all_effects

    def get_resolution_modifiers(self, stat_name: str) -> dict[str, bool]:
        """Aggregates resolution modifiers from all active effects and conditions.

        This is the primary method for determining advantage, disadvantage, or
        other modifications for an action roll.

        Args:
            stat_name: The primary statistic being used for the roll (e.g.,
                       "strength", "agility"), which some effects use to
                       determine their relevance.

        Returns:
            A dictionary of combined modifiers (e.g., {'has_disadvantage': True}).
        """
        # Start with a base context for the resolution.
        resolution_args: dict[str, Any] = {"stat_name": stat_name}

        # First, apply modifiers from temporary status effects.
        for effect in self.get_all_status_effects():
            resolution_args = effect.apply_to_resolution(resolution_args)

        # Second, apply modifiers from long-term conditions.
        for condition in self.get_all_conditions():
            resolution_args = condition.apply_to_resolution(resolution_args)

        # The calling system receives the final, combined set of modifiers.
        return resolution_args

    def get_movement_speed_multiplier(self) -> float:
        """Calculates the cumulative speed multiplier from all conditions.

        Includes penalties from:
        - Conditions (e.g., leg injury: 0.75x)
        - Exhaustion stacks (0.9x per stack)
        - Encumbrance (0.85^slots_over when carrying more than capacity)

        Returns:
            A float representing the final speed multiplier (e.g., 0.75 for a
            -25% penalty). A value of 1.0 means no modification.
        """
        multiplier = 1.0

        # Apply condition-based modifiers (e.g., leg injury)
        for condition in self.get_all_conditions():
            multiplier *= condition.get_movement_cost_modifier()

        # Apply exhaustion penalty: 0.9x per stack
        exhaustion_count = self.get_exhaustion_count()
        if exhaustion_count:
            multiplier *= (
                MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK**exhaustion_count
            )

        # Apply graduated encumbrance penalty: 0.85^slots_over
        # Only CharacterInventory has get_slots_over_capacity()
        inventory = self.actor.inventory
        if inventory is not None and hasattr(inventory, "get_slots_over_capacity"):
            slots_over = inventory.get_slots_over_capacity()
            if slots_over > 0:
                multiplier *= MovementConstants.ENCUMBRANCE_SPEED_BASE**slots_over

        return multiplier

    def get_exhaustion_count(self) -> int:
        """Returns the number of Exhaustion conditions the actor has."""
        return sum(1 for c in self.get_all_conditions() if isinstance(c, Exhaustion))

    def has_disadvantage_from_exhaustion(self) -> bool:
        """Checks if the actor is exhausted enough to suffer disadvantage on actions."""
        # The threshold is currently 2 stacks of exhaustion.
        return self.get_exhaustion_count() >= 2

    def get_exhaustion_energy_multiplier(self) -> float:
        """Calculate the cumulative energy accumulation reduction from exhaustion."""
        exhaustion_count = self.get_exhaustion_count()
        if exhaustion_count == 0:
            return 1.0
        return MovementConstants.EXHAUSTION_ENERGY_REDUCTION_PER_STACK**exhaustion_count


class StatusEffectsComponent:
    """Manage an actor's temporary status effects.

    The owning :class:`Actor` is passed in at construction so lifecycle hooks on
    :class:`StatusEffect` instances can reference the actor without every method
    needing it as a parameter.
    """

    def __init__(self, actor: Actor) -> None:
        self.actor = actor
        self._status_effects: list[StatusEffect] = []

    def __iter__(self):
        """Allow iteration over status effects."""
        return iter(self._status_effects)

    def __len__(self) -> int:
        """Return number of active status effects."""
        return len(self._status_effects)

    def apply_status_effect(self, effect: StatusEffect) -> None:
        """Add a status effect to the list and trigger its start hook."""
        if not effect.can_stack and self.has_status_effect(type(effect)):
            return
        self._status_effects.append(effect)
        effect.apply_on_start(self.actor)
        self.actor.modifiers._increment_revision()

    def remove_status_effect(
        self, effect_type: type[StatusEffect]
    ) -> list[StatusEffect]:
        """Remove effects of the given type and call their cleanup."""
        removed_effects: list[StatusEffect] = []
        for effect in self._status_effects[:]:
            if isinstance(effect, effect_type):
                self._status_effects.remove(effect)
                effect.remove_effect(self.actor)
                removed_effects.append(effect)
        if removed_effects:
            self.actor.modifiers._increment_revision()
        return removed_effects

    def has_status_effect(self, effect_type: type[StatusEffect]) -> bool:
        """Check if any status effect of the given type exists."""
        return any(isinstance(effect, effect_type) for effect in self._status_effects)

    def is_action_prevented(self) -> bool:
        """Check if any active status effect prevents this actor from acting.

        Used by the TurnManager to skip an actor's turn when they have effects
        like Tripped or Staggered active.
        """
        return any(effect.prevents_action for effect in self._status_effects)

    def get_status_effects_by_type(
        self, effect_type: type[StatusEffect]
    ) -> list[StatusEffect]:
        """Return all status effects of the given type."""
        return [e for e in self._status_effects if isinstance(e, effect_type)]

    def get_all_status_effects(self) -> list[StatusEffect]:
        """Return a copy of all active status effects."""
        return self._status_effects.copy()

    def update_turn(self) -> None:
        """Apply per-turn logic and handle expiration for all effects."""
        for effect in self._status_effects[:]:
            effect.apply_turn_effect(self.actor)
            if effect.duration > 0:
                effect.duration -= 1
            if effect.should_remove(self.actor):
                effect.remove_effect(self.actor)
                self._status_effects.remove(effect)
        self.actor.modifiers._increment_revision()


class ConditionsComponent:
    """Manages an actor's long-term conditions.

    Storage is delegated to an :class:`InventoryComponent` instance so that
    conditions continue to consume inventory space.
    """

    def __init__(self, inventory: InventoryComponent) -> None:
        self.inventory = inventory

    def __iter__(self):
        """Allow iteration over conditions."""
        return iter(self.get_all_conditions())

    def __len__(self) -> int:
        """Return the number of active conditions."""
        return len(self.get_all_conditions())

    # --- Query Helpers -------------------------------------------------
    def get_all_conditions(self) -> list[Condition]:
        """Return a list of all conditions stored in the inventory."""
        if not self.inventory:
            return []
        return [item for item in self.inventory if isinstance(item, Condition)]

    def has_condition(self, condition_type: type[Condition]) -> bool:
        """Check if any condition of the given type exists."""
        return any(isinstance(c, condition_type) for c in self.get_all_conditions())

    def get_conditions_by_type(
        self, condition_type: type[Condition]
    ) -> list[Condition]:
        """Return all conditions of the given type."""
        return [c for c in self.get_all_conditions() if isinstance(c, condition_type)]

    # --- Mutation Helpers ----------------------------------------------
    def add_condition(self, condition: Condition) -> tuple[bool, str, list[Item]]:
        """Add a condition, delegating to the inventory component."""
        return self.inventory.add_to_inventory(condition)

    def remove_condition(self, condition: Condition) -> bool:
        """Remove a specific condition from the inventory."""
        return self.inventory.remove_from_inventory(condition)

    def remove_conditions_by_type(
        self, condition_types: set[type[Condition]]
    ) -> list[Condition]:
        """Remove all conditions of the specified types."""
        removed: list[Condition] = []
        for condition_type in condition_types:
            conditions_of_type = self.get_conditions_by_type(condition_type)
            removed.extend([c for c in conditions_of_type if self.remove_condition(c)])
        return removed

    def apply_turn_effects(self, actor: Actor) -> None:
        """Apply per-turn effects for all conditions."""
        for condition in self.get_all_conditions():
            condition.apply_turn_effect(actor)


@dataclass(slots=True)
class EnergyComponent:
    """
    Manages an actor's energy for the Reactive Actor Framework (RAF).

    In RAF, energy accumulates proportionally to the actor's speed when the
    player acts. Faster actors get more energy per player action, ensuring
    they naturally get more turns over time while maintaining the turn-based
    nature of the game.

    Key RAF Energy Principles:
    - Energy accumulates when the player acts, not based on real time
    - Faster actors get proportionally more energy per player action
    - Energy is capped at maximum to prevent infinite buildup
    - Energy is spent when actions are taken
    - Game remains purely turn-based - nothing happens when player is idle

    Note:
        The actor reference is set after construction by Actor.__init__,
        following the late-binding pattern used by other components like
        ContainerStorage.
    """

    speed: int = DEFAULT_ACTOR_SPEED
    accumulated_energy: float = 0.0
    max_energy: int = 200  # Energy cap to prevent infinite accumulation
    actor: Actor | None = None  # Set by Actor.__init__ after construction

    @property
    def energy(self) -> float:
        """Alias for accumulated_energy for compatibility."""
        return self.accumulated_energy

    @energy.setter
    def energy(self, value: float) -> None:
        """Alias setter for accumulated_energy for compatibility."""
        self.accumulated_energy = value

    def accumulate_energy(self, amount: float) -> None:
        """Add energy up to the maximum cap.

        This is the core method for RAF energy accumulation. Energy is added
        smoothly up to the maximum, preventing infinite buildup while ensuring
        faster actors can accumulate energy more quickly per player action.

        Args:
            amount: Amount of energy to add (can be fractional)
        """
        if amount <= 0:
            return

        # Add energy but cap at maximum to prevent infinite accumulation
        self.accumulated_energy = min(self.max_energy, self.accumulated_energy + amount)

    def get_speed_based_energy_amount(self) -> float:
        """Calculate energy amount based on this actor's speed.

        This determines how much energy this actor should get per player action.
        Faster actors get proportionally more energy, ensuring they act more
        frequently over time.

        Returns:
            Energy amount proportional to actor speed

        Raises:
            AssertionError: If actor reference hasn't been set yet.
        """
        assert self.actor is not None, "EnergyComponent.actor must be set before use"

        # Base energy amount that gets scaled by speed
        # This is calibrated so that speed=100 actors get roughly 100 energy per action
        base_energy = 100.0

        # Apply movement speed multiplier from conditions (exhaustion, etc.)
        speed_multiplier = self.actor.modifiers.get_movement_speed_multiplier()

        # Apply exhaustion energy reduction
        exhaustion_multiplier = self.actor.modifiers.get_exhaustion_energy_multiplier()

        # Speed scaling factor - adjust this to tune the speed differences
        # Higher speeds will get proportionally more energy
        normalized_speed = self.speed / 100.0  # Normalize speed to reasonable range

        return base_energy * normalized_speed * speed_multiplier * exhaustion_multiplier

    def regenerate(self) -> None:
        """Legacy regeneration method (deprecated in RAF).

        This method is kept for compatibility during transition but should
        not be used in RAF. Use accumulate_energy() with speed-based amounts.

        For now, falls back to speed-based accumulation to maintain functionality.
        """
        # Fallback to speed-based accumulation for compatibility
        self.accumulate_energy(self.get_speed_based_energy_amount())

    def can_afford(self, cost: int) -> bool:
        """Return True if there is enough stored energy."""
        return self.accumulated_energy >= cost

    def spend(self, cost: int) -> None:
        """Spend accumulated energy."""
        self.accumulated_energy = max(0, self.accumulated_energy - cost)
