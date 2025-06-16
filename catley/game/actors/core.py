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
from catley.util.coordinates import TileCoord, WorldTileCoord
from catley.view.render.effects.lighting import LightSource

from .ai import AIComponent, DispositionBasedAI
from .components import (
    ConditionsComponent,
    EnergyComponent,
    HealthComponent,
    InventoryComponent,
    ModifiersComponent,
    StatsComponent,
    StatusEffectsComponent,
    VisualEffectsComponent,
)
from .conditions import Injury

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.game_world import GameWorld


class Actor:
    """Any object that exists in the game world and participates in the simulation.

    Actors represent all interactive and non-interactive objects in the game world.
    They use a component-based architecture where different capabilities can be
    added or omitted based on what each specific actor needs to do.

    Note: Actors are distinct from Items (weapons, consumables, etc.). Items are pure
    gameplay objects with no world position; they exist within inventory systems.
    Actors are world objects with coordinates that can contain, drop, or represent
    Items. When you see "a sword on the ground," that's an Actor containing a sword
    Item.

    All actors have basic properties like position, appearance, and the ability to
    participate in the turn-based update cycle. Beyond that, actors can optionally
    have a suite of components to define their capabilities:

    Core Data Components:
    - Stats: Ability scores like strength, toughness, intelligence.
    - Health: Hit points, armor, damage/healing mechanics.
    - Inventory: Item storage and equipment management. Also stores Conditions.

    Action & Turn-Taking Components:
    - Energy: Manages the actor's action economy, including speed, energy
      accumulation, and the ability to take turns.

    Behavioral Components:
    - AI: Autonomous decision-making and behavior for NPCs.
    - VisualEffects: Manages rendering feedback like damage flashes.
    - LightSource: A dynamic light that affects the game world.

    Effect & Modifier Components:
    - Modifiers: The primary public interface for querying an actor's current
      state. This facade provides a unified view of all active StatusEffects
      and Conditions, answering questions like "does this actor have advantage?"
      or "what is their final movement speed?". **Most external systems should
      interact with `actor.modifiers` rather than the individual effect
      components.**
    - StatusEffects: Manages temporary, non-inventory effects (e.g., "Focused").
    - Conditions: Manages long-term conditions that take up inventory space
      (e.g., "Injured", "Poisoned"). This is a convenience wrapper around
      the inventory.

    This component system ensures that actors only pay the cost (memory,
    computation) for the capabilities they actually use, while maintaining a
    unified and clear interface for game systems.
    """

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        ch: str,
        color: colors.Color,
        name: str = "<Unnamed Actor>",
        stats: StatsComponent | None = None,
        health: HealthComponent | None = None,
        inventory: InventoryComponent | None = None,
        visual_effects: VisualEffectsComponent | None = None,
        ai: AIComponent | None = None,
        # World and appearance
        game_world: GameWorld | None = None,
        light_source: LightSource | None = None,
        blocks_movement: bool = True,
        speed: int = DEFAULT_ACTOR_SPEED,
    ) -> None:
        # === Core Identity & World Presence ===
        self.x: WorldTileCoord = x
        self.y: WorldTileCoord = y
        # Visual position (deliberately typed as floats but in world tile space)
        self.render_x: float = float(x)
        self.render_y: float = float(y)
        self.ch = ch
        self.color = color
        self.name = name
        self.gw = game_world
        self.blocks_movement = blocks_movement
        self.light_source = light_source

        # === Core Data Components ===
        self.stats = stats
        self.health = health
        self.inventory = inventory

        # === Dependent & Facade Components ===
        self.status_effects = StatusEffectsComponent(self)
        self.conditions = (
            ConditionsComponent(self.inventory) if self.inventory is not None else None
        )
        self.modifiers = ModifiersComponent(actor=self)
        self.energy = EnergyComponent(self, speed)

        # === Behavioral/Optional Components ===
        self.ai = ai
        self.visual_effects = visual_effects

        # === Final Setup & Registration ===
        # This should come last, ensuring the actor is fully constructed
        # before being registered with external systems.
        if self.light_source and self.gw:
            self.light_source.attach(self, self.gw.lighting)

    def __repr__(self) -> str:
        """Return a debug representation of this actor."""
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({fields})"

    def update_render_position(self, delta_time: float) -> None:
        """Smoothly move the visual position towards the logical position."""
        # A higher LERP_FACTOR makes the movement faster and snappier.
        # A lower value makes it smoother but introduces more "lag".
        LERP_FACTOR = 30.0 * delta_time
        self.render_x += (self.x - self.render_x) * LERP_FACTOR
        self.render_y += (self.y - self.render_y) * LERP_FACTOR

    def move(self, dx: TileCoord, dy: TileCoord) -> None:
        # The move method now only updates the logical position.
        self.x += dx
        self.y += dy

        if self.gw:
            # Notify the spatial index of this actor's new position.
            self.gw.actor_spatial_index.update(self)

        # Update the light source position when actor moves
        if self.light_source:
            self.light_source.position = (self.x, self.y)

    def teleport(self, x: WorldTileCoord, y: WorldTileCoord) -> None:
        """Instantly move the actor's logical and visual position."""
        self.x = x
        self.y = y
        self.render_x = float(x)
        self.render_y = float(y)
        if self.gw:
            self.gw.actor_spatial_index.update(self)
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
                            if self.conditions is not None:
                                self.conditions.add_condition(conditions.Rads())
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
        self.status_effects.update_turn()

        # Delegate condition turn effects to the component
        if self.conditions is not None:
            self.conditions.apply_turn_effects(self)

    def get_next_action(self, controller: Controller) -> GameIntent | None:
        """
        Determines the next action for this actor.
        """
        return None


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
            for c in self.conditions.get_conditions_by_type(Injury)
            if isinstance(c, Injury)
            and c.injury_location
            in {
                InjuryLocation.LEFT_ARM,
                InjuryLocation.RIGHT_ARM,
            }
        ]
        return len({c.injury_location for c in arm_injuries}) < 2


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

    def get_next_action(self, controller: Controller) -> GameIntent | None:
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

    def get_next_action(self, controller: Controller) -> GameIntent | None:
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
