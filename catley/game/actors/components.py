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

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from catley.view.render.effects.particles import SubTileParticleSystem

    from .core import Actor
    from .status_effects import StatusEffect

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED, DEFAULT_MAX_ARMOR
from catley.constants.movement import MovementConstants
from catley.game.enums import BlendMode, ItemSize
from catley.game.items.item_core import Item
from catley.view.render.effects.particles import ParticleLayer

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

        # Revision counter for UI caching
        self.revision = 0

        self.attack_slots: list[Item | None] = [None] * num_attack_slots
        self.active_weapon_slot: int = 0

    def _increment_revision(self) -> None:
        """Bump the revision to signal a state change."""
        self.revision += 1

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
            self._increment_revision()
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
            self._increment_revision()

        self._stored_items.append(condition)
        self._update_encumbrance_status()
        self._increment_revision()

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
            self._increment_revision()
            return True
        return False

    def get_active_weapon(self) -> Item | None:
        """Get the currently active weapon."""
        if 0 <= self.active_weapon_slot < len(self.attack_slots):
            return self.attack_slots[self.active_weapon_slot]
        return None

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
        self._increment_revision()
        return old_item

    def unequip_slot(self, slot_index: int) -> Item | None:
        """Unequip specified slot, return what was equipped"""
        if not (0 <= slot_index < len(self.attack_slots)):
            raise ValueError(f"Invalid slot index: {slot_index}")

        item = self.attack_slots[slot_index]
        self.attack_slots[slot_index] = None
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
        if not (0 <= slot_index < len(self.attack_slots)):
            return False, f"Invalid slot index: {slot_index}"

        # Remove the item from stored inventory first
        self._stored_items.remove(item)

        old_item = self.attack_slots[slot_index]
        if old_item is not None:
            if not self.can_add_voluntary_item(old_item):
                # Revert and fail
                self._stored_items.append(item)
                return False, f"No room to unequip {old_item.name}"
            self._stored_items.append(old_item)

        self.attack_slots[slot_index] = item
        self._update_encumbrance_status()
        self._increment_revision()

        if old_item is not None:
            return True, f"Equipped {item.name}; unequipped {old_item.name}."
        return True, f"Equipped {item.name}."

    def unequip_to_inventory(self, slot_index: int) -> tuple[bool, str]:
        """Unequip the item in ``slot_index`` and store it if space allows."""

        if not (0 <= slot_index < len(self.attack_slots)):
            return False, f"Invalid slot index: {slot_index}"

        item = self.attack_slots[slot_index]
        if item is None:
            return False, "No item to unequip"

        if not self.can_add_voluntary_item(item):
            return False, f"No room to store {item.name}"

        self.attack_slots[slot_index] = None
        self._stored_items.append(item)
        self._update_encumbrance_status()
        self._increment_revision()
        return True, f"Unequipped {item.name}."

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


@dataclass
class ParticleEmitterPattern:
    """Configuration for a continuous particle emission pattern."""

    emission_type: str  # "directional_cone", "radial_burst", "background_tint"
    count: int
    colors_and_chars: list[tuple[colors.Color, str]]
    # Directional cone parameters
    direction_x: float = 0.0
    direction_y: float = -1.0  # Default upward
    cone_spread: float = 0.3
    # Speed and lifetime
    speed_range: tuple[float, float] = (1.0, 2.0)
    lifetime_range: tuple[float, float] = (0.1, 0.3)
    # Background tint parameters
    tint_color: colors.Color = (100, 100, 100)
    blend_mode: BlendMode = BlendMode.TINT
    upward_drift: float = -0.3
    # Other parameters
    gravity: float = 0.0
    layer: ParticleLayer = ParticleLayer.OVER_ACTORS
    # Emission control
    emission_probability: float = 1.0  # 0.0-1.0, chance to emit this pattern


class VisualEffectsComponent:
    """Handles visual feedback effects like flashing, particle emission, etc."""

    def __init__(self) -> None:
        # Flash effect state
        self._flash_color: colors.Color | None = None
        self._flash_duration_frames: int = 0

        # Particle emission state
        self.particle_emitters: list[ParticleEmitterPattern] = []
        self.emission_timer: float = 0.0
        self.emission_interval: float = 0.1  # Emit particles every 0.1 seconds

    def flash(self, color: colors.Color, duration_frames: int = 5) -> None:
        """Start a flash effect with the given color and duration.

        Args:
            color: RGB color tuple for the flash
            duration_frames: How many frames the flash should last
        """
        self._flash_color = color
        self._flash_duration_frames = duration_frames

    def add_fire_emitter(self) -> None:
        """Add fire particle emission patterns (flames, smoke, sparks)."""
        # Main flame particles - directional cone pointing up
        flame_pattern = ParticleEmitterPattern(
            emission_type="directional_cone",
            count=3,
            colors_and_chars=[
                ((255, 200, 0), "*"),  # Bright yellow
                ((255, 150, 0), "+"),  # Orange
                ((255, 100, 0), "^"),  # Dark orange
                ((200, 50, 0), "."),  # Red
            ],
            direction_x=0.0,
            direction_y=-1.0,  # Upward
            cone_spread=0.3,
            speed_range=(1.0, 2.0),
            lifetime_range=(0.1, 0.3),
            layer=ParticleLayer.OVER_ACTORS,
            emission_probability=1.0,
        )
        self.particle_emitters.append(flame_pattern)

        # Smoke particles - background tint with slow drift
        smoke_pattern = ParticleEmitterPattern(
            emission_type="background_tint",
            count=1,
            colors_and_chars=[((80, 80, 80), ".")],  # Not used for background tint
            speed_range=(0.2, 0.5),
            lifetime_range=(1.0, 2.0),
            tint_color=(80, 80, 80),  # Will be varied in emission
            blend_mode=BlendMode.TINT,
            upward_drift=-0.3,  # Negative is upward
            layer=ParticleLayer.OVER_ACTORS,
            emission_probability=0.3,  # 30% chance per emission
        )
        self.particle_emitters.append(smoke_pattern)

        # Occasional sparks - radial burst
        spark_pattern = ParticleEmitterPattern(
            emission_type="radial_burst",
            count=2,
            colors_and_chars=[((255, 255, 200), "*")],  # Bright white-yellow
            speed_range=(2.0, 3.0),
            lifetime_range=(0.05, 0.15),
            gravity=2.0,  # Sparks fall
            layer=ParticleLayer.OVER_ACTORS,
            emission_probability=0.1,  # 10% chance
        )
        self.particle_emitters.append(spark_pattern)

    def update(self) -> None:
        """Update visual effects. Call this each frame."""
        # Update flash effect
        if self._flash_duration_frames > 0:
            self._flash_duration_frames -= 1
            if self._flash_duration_frames == 0:
                self._flash_color = None

        # Update particle emission timer
        self.emission_timer += 0.016  # Approximate frame time (~60 FPS)

    def should_emit_particles(self) -> bool:
        """Check if it's time to emit particles."""
        if self.emission_timer >= self.emission_interval:
            self.emission_timer = 0.0
            return True
        return False

    def emit_particles(
        self, particle_system: SubTileParticleSystem, actor_x: int, actor_y: int
    ) -> None:
        """Emit particles for this actor using the configured patterns."""
        for pattern in self.particle_emitters:
            # Check emission probability
            if random.random() > pattern.emission_probability:
                continue

            if pattern.emission_type == "directional_cone":
                particle_system.emit_directional_cone(
                    tile_x=float(actor_x),
                    tile_y=float(actor_y),
                    direction_x=pattern.direction_x,
                    direction_y=pattern.direction_y,
                    count=pattern.count,
                    cone_spread=pattern.cone_spread,
                    speed_range=pattern.speed_range,
                    lifetime_range=pattern.lifetime_range,
                    colors_and_chars=pattern.colors_and_chars,
                    gravity=pattern.gravity,
                    layer=pattern.layer,
                )
            elif pattern.emission_type == "background_tint":
                # Vary the smoke color slightly for natural effect
                base_gray = pattern.tint_color[0]
                smoke_gray = base_gray + random.randint(-20, 20)
                smoke_gray = max(0, min(255, smoke_gray))
                varied_tint = (smoke_gray, smoke_gray, smoke_gray)

                particle_system.emit_background_tint(
                    tile_x=float(actor_x),
                    tile_y=float(actor_y),
                    count=pattern.count,
                    drift_speed=pattern.speed_range,
                    lifetime_range=pattern.lifetime_range,
                    tint_color=varied_tint,
                    blend_mode=pattern.blend_mode,
                    chars=[".", ",", " "],
                    upward_drift=pattern.upward_drift,
                    layer=pattern.layer,
                )
            elif pattern.emission_type == "radial_burst":
                particle_system.emit_radial_burst(
                    tile_x=float(actor_x),
                    tile_y=float(actor_y),
                    count=pattern.count,
                    speed_range=pattern.speed_range,
                    lifetime_range=pattern.lifetime_range,
                    colors_and_chars=pattern.colors_and_chars,
                    gravity=pattern.gravity,
                    layer=pattern.layer,
                )

    def get_flash_color(self) -> colors.Color | None:
        """Get the current flash color for rendering, or None if no flash."""
        return self._flash_color if self._flash_duration_frames > 0 else None

    def has_particle_emitters(self) -> bool:
        """Check if this component has any particle emitters configured."""
        return len(self.particle_emitters) > 0


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

        Returns:
            A float representing the final speed multiplier (e.g., 0.75 for a
            -25% penalty). A value of 1.0 means no modification.
        """
        multiplier = 1.0
        # Iterate through all conditions that can affect movement.
        for condition in self.get_all_conditions():
            multiplier *= condition.get_movement_cost_modifier()

        exhaustion_count = self.get_exhaustion_count()
        if exhaustion_count:
            multiplier *= (
                MovementConstants.EXHAUSTION_SPEED_REDUCTION_PER_STACK**exhaustion_count
            )

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
    """

    actor: Actor
    speed: int = DEFAULT_ACTOR_SPEED
    accumulated_energy: float = 0.0
    max_energy: int = 200  # Energy cap to prevent infinite accumulation

    def __post_init__(self) -> None:
        self.accumulated_energy = self.speed

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
        """
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
