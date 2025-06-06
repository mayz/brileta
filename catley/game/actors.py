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

Use one of the factory functions below to create actors:
    - make_pc()
    - make_npc()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED
from catley.view.effects.lighting import LightSource

from .ai import AIComponent, DispositionBasedAI
from .components import (
    HealthComponent,
    InventoryComponent,
    StatsComponent,
    VisualEffectsComponent,
)
from .enums import Disposition
from .items.item_core import Item

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.world.game_state import GameWorld

    from .actions import GameAction


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

    Use one of the factory functions below to create actors:
        - make_pc()
        - make_npc()
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

        self.speed = speed
        self.accumulated_energy: int = self.speed

        self.tricks: list = []

    def __repr__(self) -> str:
        """Return a debug representation of this actor."""
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({fields})"

    def move(self, dx: int, dy: int) -> None:
        self.x += dx
        self.y += dy
        # Update the light source position when actor moves
        if self.light_source:
            self.light_source.position = (self.x, self.y)

    def take_damage(self, amount: int) -> None:
        """Handle damage to the actor.

        That includes:
        - Update health math.
        - Visual feedback.
        - Handle death consequences, if any.

        Args:
            amount: Amount of damage to take
        """
        # Visual feedback.
        if self.visual_effects:
            self.visual_effects.flash(colors.RED)

        if self.health:
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
        """
        Handles turn-based logic for actors. For NPCs, this includes AI.
        For the player, this could handle passive effects like poison/regeneration.
        """
        # TODO: Implement passive effects here if any (e.g., poison, regeneration)

        # For example:
        # if self.has_condition("Poisoned"):
        #     self.take_damage(1)
        #     controller.message_log.add_message(
        #         f"{self.name} takes 1 poison damage.", colors.GREEN)
        #
        # Active actions (including AI for NPCs and player input) are now
        # handled via get_next_action() in the main game loop.
        pass

    def get_next_action(self, controller: Controller) -> GameAction | None:
        """
        Determines the next action for this actor.
        """
        return None

    def regenerate_energy(self) -> None:
        """Regenerates energy for the actor based on their speed."""
        self.accumulated_energy += self.speed

    def can_afford_action(self, cost: int) -> bool:
        """Checks if the actor has enough energy to perform an action."""
        return self.accumulated_energy >= cost

    def spend_energy(self, cost: int) -> None:
        self.accumulated_energy -= cost


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
            inventory=InventoryComponent(stats, num_attack_slots),
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

        if starting_weapon:
            self.inventory.equip_to_slot(starting_weapon, 0)


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
