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

from catley import colors
from catley.config import DEFAULT_MAX_ARMOR

from .conditions import Condition
from .enums import ItemSize
from .items.item_core import Item


class StatsComponent:
    """Manages an Actor's core ability scores and derived stats."""

    def __init__(
        self,
        strength: int = 0,
        toughness: int = 0,
        agility: int = 0,
        observation: int = 0,
        intelligence: int = 0,
        demeanor: int = 0,
        weirdness: int = 0,
    ) -> None:
        # Core ability scores
        self.strength = strength
        self.toughness = toughness
        self.agility = agility
        self.observation = observation
        self.intelligence = intelligence
        self.demeanor = demeanor
        self.weirdness = weirdness

    @property
    def max_hp(self) -> int:
        """Derive max HP from toughness."""
        return self.toughness + 5

    @property
    def inventory_slots(self) -> int:
        """Derive inventory capacity from strength."""
        return self.strength + 5


class HealthComponent:
    """Handles physical integrity - HP, armor, damage from any source, healing."""

    def __init__(
        self, stats_component: StatsComponent, max_ap: int = DEFAULT_MAX_ARMOR
    ) -> None:
        self.stats = stats_component
        self.hp = self.stats.max_hp
        self.max_ap = max_ap
        self.ap = max_ap

    @property
    def max_hp(self) -> int:
        """Get max HP from stats component."""
        return self.stats.max_hp

    def take_damage(self, amount: int) -> None:
        """Handle damage to the actor, reducing AP first, then HP.

        Args:
            amount: Amount of damage to take
        """
        # First reduce AP if any
        if self.ap > 0:
            ap_damage = min(amount, self.ap)
            self.ap -= ap_damage
            amount -= ap_damage

        # Apply remaining damage to HP
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
    """Manages an actor's stored items and equipped gear.

    Stored items are tracked in inventory slots and can be items
    but can also be things like injuries or conditions.

    Equipped gear is stuff that the actor is holding or wearing. It may
    confer some mechanical advantage (like armor points).
    """

    def __init__(
        self, stats_component: StatsComponent, num_attack_slots: int = 2
    ) -> None:
        self.stats = stats_component
        self._stored_items: list[Item | Condition] = []

        self.attack_slots: list[Item | None] = [None] * num_attack_slots

    def __iter__(self):
        """Allow iteration over stored items."""
        return iter(self._stored_items)

    def __len__(self) -> int:
        """Return number of stored items."""
        return len(self._stored_items)

    def __contains__(self, item: Item | Condition) -> bool:
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

        # Count equipped items
        for equipped_item in self.attack_slots:
            if equipped_item:
                if equipped_item.size == ItemSize.TINY:
                    has_tiny_items = True
                elif equipped_item.size == ItemSize.NORMAL:
                    used_space += 1
                elif equipped_item.size == ItemSize.BIG:
                    used_space += 2
                elif equipped_item.size == ItemSize.HUGE:
                    used_space += 4

        if has_tiny_items:
            used_space += 1

        return used_space

    def can_add_to_inventory(self, item_to_add: Item | Condition) -> bool:
        """Check if an item can be added to inventory."""
        current_used_space = self.get_used_inventory_slots()
        additional_space_needed = 0

        if isinstance(item_to_add, Item):
            item = item_to_add
            if item.size == ItemSize.TINY:
                # If no tiny items exist yet, adding this one will create the tiny slot
                if not any(
                    isinstance(i, Item) and i.size == ItemSize.TINY
                    for i in self._stored_items
                ):
                    additional_space_needed = 1
            elif item.size == ItemSize.NORMAL:
                additional_space_needed = 1
            elif item.size == ItemSize.BIG:
                additional_space_needed = 2
            elif item.size == ItemSize.HUGE:
                additional_space_needed = 4
        elif isinstance(item_to_add, Condition):
            additional_space_needed = 1

        return (
            current_used_space + additional_space_needed
        ) <= self.total_inventory_slots

    def add_to_inventory(self, item: Item | Condition) -> bool:
        """Add item to inventory if space available."""
        if self.can_add_to_inventory(item):
            self._stored_items.append(item)
            return True
        return False

    def remove_from_inventory(self, item: Item | Condition) -> bool:
        """Remove item from inventory."""
        if item in self._stored_items:
            self._stored_items.remove(item)
            return True
        return False

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

        # Process equipped items
        for equipped_item in self.attack_slots:
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

        return slot_colors

    # === Equipment Management ===

    @property
    def num_attack_slots(self) -> int:
        """Get number of attack slots this character has"""
        return len(self.attack_slots)

    def equip_to_slot(self, item: Item, slot_index: int = 0) -> Item | None:
        """Equip item to specified slot, return what was there before"""
        if not (0 <= slot_index < len(self.attack_slots)):
            raise ValueError(f"Invalid slot index: {slot_index}")

        old_item = self.attack_slots[slot_index]
        self.attack_slots[slot_index] = item
        return old_item

    def unequip_slot(self, slot_index: int) -> Item | None:
        """Unequip specified slot, return what was equipped"""
        if not (0 <= slot_index < len(self.attack_slots)):
            raise ValueError(f"Invalid slot index: {slot_index}")

        item = self.attack_slots[slot_index]
        self.attack_slots[slot_index] = None
        return item

    def get_equipped_items(self) -> list[tuple[Item, int]]:
        """Get all equipped items with their slot index"""
        items = []
        for i, item in enumerate(self.attack_slots):
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
        for i, item in enumerate(self.attack_slots):
            display_name = self.get_slot_display_name(i)
            attacks.append((item, i, display_name))
        return attacks

    # FIXME: For backwards compatibility
    @property
    def equipped_weapon(self) -> Item | None:
        """For backwards compatibility - returns primary attack slot (slot 0)"""
        return self.attack_slots[0] if self.attack_slots else None

    # FIXME: For backwards compatibility
    @equipped_weapon.setter
    def equipped_weapon(self, value: Item) -> None:
        self.equip_to_slot(value, 0)

    # FIXME: For backwards compatibility
    def equip_weapon(self, weapon: Item) -> Item | None:
        """Equip a weapon, returning the previously equipped weapon."""
        return self.equip_to_slot(weapon, 0)

    # FIXME: For backwards compatibility
    def unequip_weapon(self) -> Item | None:
        """Unequip current weapon, returning it."""
        return self.unequip_slot(0)

    def equip_weapon_from_storage(self, weapon: Item) -> bool:
        """Move weapon from storage to equipped."""
        if weapon in self._stored_items:
            self.remove_from_inventory(weapon)
            old_weapon = self.equip_weapon(weapon)
            if old_weapon:
                # Put old weapon back in storage if there's space
                is_there_space = self.add_to_inventory(old_weapon)
                if not is_there_space:
                    # If no space, drop old weapon back to equipped and restore storage
                    self.equip_weapon(old_weapon)
                    self.add_to_inventory(weapon)
                    return False
            return True
        return False

    def unequip_weapon_to_storage(self) -> bool:
        """Move equipped weapon to storage if space available."""
        if self.equipped_weapon and self.can_add_to_inventory(self.equipped_weapon):
            weapon = self.unequip_weapon()
            self.add_to_inventory(weapon)
            return True
        return False


class VisualEffectsComponent:
    """Handles visual feedback effects like flashing, pulsing, etc."""

    def __init__(self) -> None:
        # Flash effect state
        self._flash_color: colors.Color | None = None
        self._flash_duration_frames: int = 0

    def flash(self, color: colors.Color, duration_frames: int = 5) -> None:
        """Start a flash effect with the given color and duration.

        Args:
            color: RGB color tuple for the flash
            duration_frames: How many frames the flash should last
        """
        self._flash_color = color
        self._flash_duration_frames = duration_frames

    def update(self) -> None:
        """Update visual effects. Call this each frame."""
        # Update flash effect
        if self._flash_duration_frames > 0:
            self._flash_duration_frames -= 1
            if self._flash_duration_frames == 0:
                self._flash_color = None

    def get_flash_color(self) -> colors.Color | None:
        """Get the current flash color for rendering, or None if no flash."""
        return self._flash_color if self._flash_duration_frames > 0 else None
