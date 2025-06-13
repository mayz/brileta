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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import Actor, StatusEffect

from catley import colors
from catley.config import DEFAULT_MAX_ARMOR
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
    """Handles physical integrity - HP, armor, damage from any source, healing."""

    stats: StatsComponent
    max_ap: int = DEFAULT_MAX_ARMOR
    hp: int = field(init=False)
    ap: int = field(init=False)

    def __post_init__(self) -> None:
        self.hp = self.stats.max_hp
        self.ap = self.max_ap

    @property
    def max_hp(self) -> int:
        """Get max HP from stats component."""
        return self.stats.max_hp

    def take_damage(self, amount: int, damage_type: str = "normal") -> None:
        """Handle damage to the actor, reducing AP first unless radiation.

        Args:
            amount: Amount of damage to take
            damage_type: "normal" or "radiation"
        """
        if damage_type == "radiation":
            # Radiation bypasses armor entirely
            self.hp = max(0, self.hp - amount)
            return

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
        self,
        stats_component: StatsComponent,
        num_attack_slots: int = 2,
        actor: Actor | None = None,
    ) -> None:
        self.stats = stats_component
        self.actor = actor  # Back-reference for status effects
        self._stored_items: list[Item | Condition] = []

        self.attack_slots: list[Item | None] = [None] * num_attack_slots
        self.active_weapon_slot: int = 0

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

    def is_encumbered(self) -> bool:
        """Return True if carrying more than base capacity."""
        return self.get_used_inventory_slots() > self.stats.inventory_slots

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
                    for eq_item in self.attack_slots
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

    def _find_droppable_item(self) -> Item | None:
        """Find the most recently added voluntary item to drop (LIFO)."""
        for item in reversed(self._stored_items):
            if isinstance(item, Item):
                return item
        return None

    def add_voluntary_item(self, item: Item) -> tuple[bool, str]:
        """Add a voluntary item if space allows."""
        if self.can_add_voluntary_item(item):
            self._stored_items.append(item)
            self._update_encumbrance_status()
            return True, f"Added {item.name}."
        return False, f"Cannot carry {item.name} - inventory full!"

    def add_condition(self, condition: Condition) -> tuple[list[Item], str]:
        dropped_items: list[Item] = []
        base_capacity = self.stats.inventory_slots

        # Keep dropping until we fit
        while True:
            current_used = self.get_used_inventory_slots()
            space_needed = self._calculate_space_needed(condition)

            if current_used + space_needed <= base_capacity:
                break  # We fit!

            droppable_item = self._find_droppable_item()
            if not droppable_item:
                break  # Nothing left to drop

            self._stored_items.remove(droppable_item)
            dropped_items.append(droppable_item)

        self._stored_items.append(condition)
        self._update_encumbrance_status()

        message_parts = [f"Added {condition.name}."]

        if dropped_items:
            dropped_names = [item.name for item in dropped_items]
            message_parts.append(f"Dropped: {', '.join(dropped_names)}.")

        if self.is_encumbered():
            message_parts.append("You are now encumbered!")

        return dropped_items, " ".join(message_parts)

    def _update_encumbrance_status(self) -> None:
        """Add or remove encumbrance status effect based on current load."""
        if not self.actor:
            return

        from catley import colors
        from catley.events import MessageEvent, publish_event

        currently_encumbered = self.actor.has_status_effect(EncumberedEffect)
        should_be_encumbered = self.is_encumbered()

        if should_be_encumbered and not currently_encumbered:
            self.actor.apply_status_effect(EncumberedEffect())
            publish_event(
                MessageEvent(f"{self.actor.name} is encumbered!", colors.YELLOW)
            )
        elif not should_be_encumbered and currently_encumbered:
            self.actor.remove_status_effect(EncumberedEffect)
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

    def switch_to_weapon_slot(self, slot_index: int) -> bool:
        """Switch to a specific weapon slot. Returns True if successful."""
        if 0 <= slot_index < len(self.attack_slots):
            self.active_weapon_slot = slot_index
            return True
        return False

    def get_active_weapon(self) -> Item | None:
        """Get the currently active weapon."""
        return self.attack_slots[self.active_weapon_slot]

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


@dataclass(slots=True)
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

    actor: Actor

    def get_all_status_effects(self) -> list[StatusEffect]:
        """Returns a list of all active StatusEffect instances."""
        return self.actor.status_effects

    def get_all_conditions(self) -> list[Condition]:
        """Scans the actor's inventory and returns a list of all Condition instances."""
        if not self.actor.inventory:
            return []
        # This is part of the facade's core logic: knowing that conditions
        # live inside the inventory.
        return [item for item in self.actor.inventory if isinstance(item, Condition)]

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

        Returns:
            A float representing the final speed multiplier (e.g., 0.75 for a
            -25% penalty). A value of 1.0 means no modification.
        """
        multiplier = 1.0
        # Iterate through all conditions that can affect movement.
        for condition in self.get_all_conditions():
            multiplier *= condition.get_movement_cost_modifier()
        return multiplier

    def get_exhaustion_count(self) -> int:
        """Returns the number of Exhaustion conditions the actor has."""
        return sum(1 for c in self.get_all_conditions() if isinstance(c, Exhaustion))

    def has_disadvantage_from_exhaustion(self) -> bool:
        """Checks if the actor is exhausted enough to suffer disadvantage on actions."""
        # The threshold is currently 2 stacks of exhaustion.
        return self.get_exhaustion_count() >= 2
