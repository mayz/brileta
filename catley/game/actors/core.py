"""
Actors in the game world.

Defines the Actor class - the fundamental building block for all objects that exist
in the game world and participate in the turn-based simulation.

Actor:
    Any object with a position that can be rendered, interacted with, and updated
    each turn. Actors use a component-based architecture where different capabilities
    (health, inventory, AI, stats, visual effects) can be mixed and matched based on
    what each specific actor needs.

Examples of actors:
    - Player character: has stats, health, inventory, visual effects, no AI
    - NPCs and monsters: has stats, health, inventory, visual effects, AI
    - Interactive objects: doors with health, chests with inventory
    - Simple objects: decorative items with just position and appearance
    - Complex mechanisms: traps with AI timing and visual effects

The component system allows actors to be as simple or complex as needed:
    - A decorative statue: just position, character, and color
    - A treasure chest: position, appearance, inventory component
    - A breakable door: position, appearance, health component
    - A monster: all components for full agency and capability

This unified approach eliminates the need for separate actor hierarchies while
maintaining flexibility through optional components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED
from catley.game.actors import conditions
from catley.game.enums import Disposition, InjuryLocation
from catley.game.items.item_core import Item
from catley.view.render.effects.lighting import LightSource

from .ai import AIComponent, DispositionBasedAI
from .components import (
    ConditionsComponent,
    HealthComponent,
    InventoryComponent,
    ModifiersComponent,
    StatsComponent,
    StatusEffectsComponent,
    VisualEffectsComponent,
)
from .conditions import Condition, Injury
from .status_effects import StatusEffect

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameAction
    from catley.game.game_world import GameWorld


class Actor:
    """Any object that exists in the game world and participates in the simulation.

    Actors represent all interactive and non-interactive objects in the game world.
    They use a component-based architecture where different capabilities can be
    added or omitted based on what each specific actor needs to do.

    Note: Actors are distinct from Items (weapons, consumables, etc.). Items are pure
    gameplay objects with no world position - they exist within inventory systems.
    Actors are world objects with coordinates that can contain, drop, or represent
    Items. When you see "a sword on the ground," that's an Actor containing a sword
    Item.

    All actors have basic properties like position, appearance, and the ability to
    participate in the turn-based update cycle. Beyond that, actors can optionally have:

    - Stats: Ability scores like strength, toughness, intelligence
    - Health: Hit points, armor, damage/healing mechanics
    - Inventory: Item storage and equipment management
    - AI: Autonomous decision-making and behavior
    - Visual Effects: Rendering feedback like damage flashes
    - Light Source: Dynamic lighting that affects the game world

    The component system ensures that actors only pay the cost (memory, computation)
    for the capabilities they actually use, while maintaining a unified interface
    for game systems to interact with all objects in the world.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str = "<Unnamed Actor>",
        stats: StatsComponent | None = None,
        health: HealthComponent | None = None,
        inventory: InventoryComponent | None = None,
        visual_effects: VisualEffectsComponent | None = None,
        ai: AIComponent | None = None,
        modifiers: ModifiersComponent | None = None,
        # World and appearance
        game_world: GameWorld | None = None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        speed: int = DEFAULT_ACTOR_SPEED,
    ) -> None:
        self.x = x
        self.y = y
        self.ch = ch
        self.color = color
        self.gw = game_world
        self.light_source = light_source
        self.blocks_movement = blocks_movement
        if self.light_source and self.gw:
            self.light_source.attach(self, self.gw.lighting)

        self.name = name

        self.stats = stats
        self.health = health
        self.inventory = inventory
        self.visual_effects = visual_effects
        self.ai = ai

        # Initialize modifiers component - create one if not provided
        if modifiers is not None:
            self.modifiers = modifiers
        else:
            self.modifiers = ModifiersComponent(actor=self)

        self.speed = speed
        self.accumulated_energy: int = self.speed
        self._effective_speed_cache: int | None = None

        self.tricks: list = []
        self.status_effects = StatusEffectsComponent()

        if self.inventory is not None:
            self.conditions = ConditionsComponent(self.inventory)
        else:
            self.conditions = None

    def __repr__(self) -> str:
        """Return a debug representation of this actor."""
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({fields})"

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy

        if self.gw:
            # Notify the spatial index of this actor's new position.
            self.gw.actor_spatial_index.update(self)

        # Update the light source position when actor moves
        if self.light_source:
            self.light_source.position = (self.x, self.y)

    def take_damage(self, amount: int, damage_type: str = "normal") -> None:
        """Handle damage to the actor.

        That includes:
        - Update health math.
        - Visual feedback.
        - Handle death consequences, if any.

        Args:
            amount: Amount of damage to take
            damage_type: "normal" or "radiation"
        """
        # Visual feedback.
        if self.visual_effects:
            self.visual_effects.flash(colors.RED)

        if self.health:
            if damage_type == "radiation":
                initial_hp = self.health.hp
                self.health.take_damage(amount, damage_type="radiation")

                actual_damage = initial_hp - self.health.hp
                if self.inventory is not None and actual_damage > 0:
                    for _ in range(actual_damage):
                        if (
                            self.inventory.get_used_inventory_slots()
                            < self.inventory.total_inventory_slots
                        ):
                            self.add_condition(conditions.Rads())
                        else:
                            break
            else:
                # Delegate health math to the health component.
                self.health.take_damage(amount)

            if not self.health.is_alive():
                # Handle death consequences.
                self.ch = "x"
                self.color = colors.DEAD
                self.blocks_movement = False
                # If this actor was selected, deselect it.
                if self.gw and self.gw.selected_actor == self:
                    self.gw.selected_actor = None

    def update_turn(self, controller: Controller) -> None:
        """Advance ongoing status effects for this actor.

        This method should be called once at the *start* of each round. It
        processes active status effects, decrementing their duration and removing
        them when they expire. NPC AI or other per-turn logic could also be
        triggered here in the future.
        """
        # Delegate status effect updates to the component
        self.status_effects.update_turn(self)

        # Delegate condition turn effects to the component
        if self.conditions:
            self.conditions.apply_turn_effects(self)

    def get_next_action(self, controller: Controller) -> GameAction | None:
        """
        Determines the next action for this actor.
        """
        return None

    def _invalidate_effective_speed_cache(self) -> None:
        """Clear cached speed when conditions change."""
        self._effective_speed_cache = None

    def calculate_effective_speed(self) -> int:
        """Calculate speed after accounting for leg injuries and exhaustion."""
        if self._effective_speed_cache is not None:
            return self._effective_speed_cache

        speed = float(self.speed)
        speed *= self.modifiers.get_movement_speed_multiplier()

        self._effective_speed_cache = int(speed)
        return self._effective_speed_cache

    def regenerate_energy(self) -> None:
        """Regenerates energy for the actor based on their speed and conditions."""
        base_energy = self.calculate_effective_speed()

        if isinstance(self, Character):
            exhaustion_multiplier = self.modifiers.get_exhaustion_energy_multiplier()
            base_energy = int(base_energy * exhaustion_multiplier)

        self.accumulated_energy += base_energy

    def can_afford_action(self, cost: int) -> bool:
        """Checks if the actor has enough energy to perform an action."""
        return self.accumulated_energy >= cost

    def spend_energy(self, cost: int) -> None:
        self.accumulated_energy -= cost

    # === Status Effect Management ===

    def apply_status_effect(self, effect: StatusEffect) -> None:
        """Apply a status effect to this actor."""
        self.status_effects.apply_status_effect(effect)
        effect.apply_on_start(self)

    def remove_status_effect(self, effect_type: type[StatusEffect]) -> None:
        """Remove all status effects of the given type."""
        removed = self.status_effects.remove_status_effect(effect_type)
        for effect in removed:
            effect.remove_effect(self)

    def has_status_effect(self, effect_type: type[StatusEffect]) -> bool:
        """Check if actor has any status effect of the given type."""
        return self.status_effects.has_status_effect(effect_type)

    def get_status_effects_by_type(
        self, effect_type: type[StatusEffect]
    ) -> list[StatusEffect]:
        """Get all status effects of the given type."""
        return self.status_effects.get_status_effects_by_type(effect_type)

    # === Condition Management ===

    def has_condition(self, condition_type: type[Condition]) -> bool:
        if self.conditions is None:
            return False
        return self.conditions.has_condition(condition_type)

    def get_conditions(self) -> list[Condition]:
        if self.conditions is None:
            return []
        return self.conditions.get_all_conditions()

    def get_conditions_by_type(
        self, condition_type: type[Condition]
    ) -> list[Condition]:
        if self.conditions is None:
            return []
        return self.conditions.get_conditions_by_type(condition_type)

    def add_condition(self, condition: Condition) -> bool:
        if self.conditions is None:
            return False
        added, _msg, _dropped = self.conditions.add_condition(condition)
        if added:
            self._invalidate_effective_speed_cache()
        return added

    def remove_condition(self, condition: Condition) -> bool:
        if self.conditions is None:
            return False
        removed = self.conditions.remove_condition(condition)
        if removed:
            self._invalidate_effective_speed_cache()
        return removed


class Character(Actor):
    """A character (player, NPC, monster) with full capabilities.

    Characters have stats, health, inventory, and visual effects. They can think,
    fight, carry items, and participate fully in the simulation.

    Type-safe wrapper - guarantees certain components exist. All the actual
    functionality still comes from the components.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str,
        game_world: GameWorld | None = None,
        strength: int = 0,
        toughness: int = 0,
        agility: int = 0,
        observation: int = 0,
        intelligence: int = 0,
        demeanor: int = 0,
        weirdness: int = 0,
        ai: AIComponent | None = None,
        light_source: LightSource | None = None,
        starting_weapon: Item | None = None,
        num_attack_slots: int = 2,
        speed: int = DEFAULT_ACTOR_SPEED,
        **kwargs,
    ) -> None:
        """
        Instantiate Character.

        Args:
            x, y: Starting position
            ch: Character to display
            color: Display color
            name: Character name
            game_world: World to exist in
            strength, toughness, etc.: Ability scores
            ai: AI component for autonomous behavior (None for player)
            light_source: Optional light source
            starting_weapon: Initial equipped weapon
            num_attack_slots: The number of attack slots this character should have
            speed: Action speed (higher = more frequent actions)
            **kwargs: Additional Actor parameters
        """
        stats = StatsComponent(
            strength=strength,
            toughness=toughness,
            agility=agility,
            observation=observation,
            intelligence=intelligence,
            demeanor=demeanor,
            weirdness=weirdness,
        )

        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            game_world=game_world,
            stats=stats,
            health=HealthComponent(stats),
            inventory=InventoryComponent(stats, num_attack_slots, actor=self),
            visual_effects=VisualEffectsComponent(),
            ai=ai,
            light_source=light_source,
            speed=speed,
            **kwargs,
        )

        # Type narrowing - these are guaranteed to exist.
        self.stats: StatsComponent
        self.health: HealthComponent
        self.inventory: InventoryComponent
        self.visual_effects: VisualEffectsComponent
        self.modifiers: ModifiersComponent
        self.conditions: ConditionsComponent

        if starting_weapon:
            self.inventory.equip_to_slot(starting_weapon, 0)

    def can_use_two_handed_weapons(self) -> bool:
        """Return ``False`` if both arms are injured."""
        arm_injuries = [
            c
            for c in self.get_conditions_by_type(Injury)
            if isinstance(c, Injury)
            and c.injury_location in {InjuryLocation.LEFT_ARM, InjuryLocation.RIGHT_ARM}
        ]
        return len({c.injury_location for c in arm_injuries}) < 2

    # === Exhaustion Helpers ===

    def get_exhaustion_count(self) -> int:
        """Get the total number of exhaustion conditions affecting this character."""
        return self.modifiers.get_exhaustion_count()

    def has_exhaustion_disadvantage(self) -> bool:
        """Check if character has enough exhaustion for action disadvantage."""
        return self.modifiers.has_disadvantage_from_exhaustion()


class PC(Character):
    """A player character.

    Type-safe wrapper - guarantees certain components exist. All the actual
    functionality still comes from the components.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str,
        game_world: GameWorld | None = None,
        strength: int = 0,
        toughness: int = 0,
        agility: int = 0,
        observation: int = 0,
        intelligence: int = 0,
        demeanor: int = 0,
        weirdness: int = 0,
        light_source: LightSource | None = None,
        starting_weapon: Item | None = None,
        num_attack_slots: int = 2,
        speed: int = DEFAULT_ACTOR_SPEED,
    ) -> None:
        """Instantiate PC.

        Args:
            x, y: Starting position
            ch: Character to display
            color: Display color
            name: Character name
            game_world: World to exist in
            strength, toughness, etc.: Ability scores
            light_source: Optional light source
            starting_weapon: Initial equipped weapon
            num_attack_slots: The number of attack slots this character should have
            speed: Action speed (higher = more frequent actions)
        """
        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            game_world=game_world,
            strength=strength,
            toughness=toughness,
            agility=agility,
            observation=observation,
            intelligence=intelligence,
            demeanor=demeanor,
            weirdness=weirdness,
            light_source=light_source,
            starting_weapon=starting_weapon,
            num_attack_slots=num_attack_slots,
            speed=speed,
        )

    def get_next_action(self, controller: Controller) -> GameAction | None:
        """
        Determines the next action for this actor.
        """
        return controller.turn_manager.dequeue_player_action()


class NPC(Character):
    """An NPC or monster with full capabilities.

    NPCs have stats, health, inventory, and visual effects. They have AI, can
    fight, carry items, and participate fully in the simulation.

    Type-safe wrapper - guarantees certain components exist. All the actual
    functionality still comes from the components.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str,
        game_world: GameWorld | None = None,
        strength: int = 0,
        toughness: int = 0,
        agility: int = 0,
        observation: int = 0,
        intelligence: int = 0,
        demeanor: int = 0,
        weirdness: int = 0,
        light_source: LightSource | None = None,
        starting_weapon: Item | None = None,
        num_attack_slots: int = 2,
        disposition: Disposition = Disposition.WARY,
        speed: int = DEFAULT_ACTOR_SPEED,
        **kwargs,
    ) -> None:
        """Instantiate NPC.

        Args:
            x, y: Starting position
            ch: Character to display
            color: Display color
            name: Character name
            game_world: World to exist in
            strength, toughness, etc.: Ability scores
            light_source: Optional light source
            starting_weapon: Initial equipped weapon
            num_attack_slots: The number of attack slots this character should have
            disposition: Starting disposition toward player
            speed: Action speed (higher = more frequent actions)
            **kwargs: Additional Actor parameters
        """
        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            game_world=game_world,
            strength=strength,
            toughness=toughness,
            agility=agility,
            observation=observation,
            intelligence=intelligence,
            demeanor=demeanor,
            weirdness=weirdness,
            ai=DispositionBasedAI(disposition=disposition),
            light_source=light_source,
            starting_weapon=starting_weapon,
            num_attack_slots=num_attack_slots,
            speed=speed,
            **kwargs,
        )

        # Type narrowing - these are guaranteed to exist.
        self.ai: AIComponent

    def get_next_action(self, controller: Controller) -> GameAction | None:
        """
        Determines the next action for this actor.
        """
        if self.health.is_alive():
            # Allow AI to update its internal state before deciding on an action.
            self.ai.update(
                controller
            )  # Pass only controller as per AIComponent.update signature
            return self.ai.get_action(controller, self)

        return None
