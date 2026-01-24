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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from catley import colors
from catley.config import DEFAULT_ACTOR_SPEED
from catley.events import (
    FloatingTextEvent,
    FloatingTextSize,
    FloatingTextValence,
    publish_event,
)
from catley.game.actors import conditions
from catley.game.enums import CreatureSize, Disposition, InjuryLocation
from catley.game.items.item_core import Item
from catley.game.pathfinding_goal import PathfindingGoal
from catley.sound.emitter import SoundEmitter
from catley.types import TileCoord, WorldTileCoord
from catley.util.pathfinding import find_local_path
from catley.view.animation import MoveAnimation

from .ai import AIComponent, DispositionBasedAI
from .components import (
    CharacterInventory,
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
from .idle_animation import (
    IdleAnimationProfile,
    create_profile_for_size,
    scale_for_size,
)

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.game_world import GameWorld


@dataclass
class CharacterLayer:
    """A single character in a multi-character visual composition.

    Used to create rich visual representations of actors by layering multiple
    ASCII characters at sub-tile offsets. Similar to how decals and blood
    splatters use sub-tile positioning for visual richness.

    Attributes:
        char: The ASCII character to display.
        color: RGB color tuple for the character.
        offset_x: Sub-tile horizontal offset (-0.5 to 0.5, 0 = center).
        offset_y: Sub-tile vertical offset (-0.5 to 0.5, 0 = center).
        scale_x: Horizontal scale factor, multiplied with actor's visual_scale.
        scale_y: Vertical scale factor, multiplied with actor's visual_scale.
    """

    char: str
    color: colors.Color
    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0


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
        energy: EnergyComponent | None = None,
        # World and appearance
        game_world: GameWorld | None = None,
        blocks_movement: bool = True,
        visual_scale: float = 1.0,
        character_layers: list[CharacterLayer] | None = None,
    ) -> None:
        # === Core Identity & World Presence ===
        self.x: WorldTileCoord = x
        self.y: WorldTileCoord = y

        # INTERPOLATION TRACKING: Previous position for smooth movement
        # These are updated each fixed timestep to enable linear interpolation
        # between logic steps, creating smooth movement independent of visual framerate
        self.prev_x: WorldTileCoord = x
        self.prev_y: WorldTileCoord = y

        # Visual position (deliberately typed as floats but in world tile space)
        self.render_x: float = float(x)
        self.render_y: float = float(y)
        self.ch = ch
        self.color = color
        self.name = name
        self.visual_scale = visual_scale
        self.character_layers = character_layers  # Multi-char visual composition
        self.has_complex_visuals = False  # Flag for actors with particle effects, etc.
        self.gw = game_world
        self.blocks_movement = blocks_movement
        # Light source removed - handled by new lighting system

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

        # Energy is optional - static objects like containers don't need it.
        # Set the back-reference now that self exists.
        self.energy = energy
        if self.energy is not None:
            self.energy.actor = self

        # === Behavioral/Optional Components ===
        self.ai = ai
        self.visual_effects = visual_effects
        self.sound_emitters: list[SoundEmitter] | None = None

        # === Animation Control ===
        # Flag to indicate if this actor is under animation control
        self._animation_controlled: bool = False

        # === Final Setup & Registration ===
        # This should come last, ensuring the actor is fully constructed
        # before being registered with external systems.
        # Light source attachment removed in Phase 3 - new lighting system handles this

    def __repr__(self) -> str:
        """Return a debug representation of this actor."""
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({fields})"

    def move(
        self, dx: TileCoord, dy: TileCoord, controller: Controller | None = None
    ) -> None:
        """Move the actor and automatically create movement animation.

        Args:
            dx: Change in x coordinate
            dy: Change in y coordinate
            controller: Controller to queue animation with (if available)
        """
        # Store old position for animation
        old_x, old_y = self.x, self.y

        # Update logical position
        self.x += dx
        self.y += dy

        if self.gw:
            # Notify the spatial index of this actor's new position.
            self.gw.actor_spatial_index.update(self)

        # Update any lights owned by this actor
        if self.gw:
            self.gw.on_actor_moved(self)

        # Automatically create animation if controller available
        if controller and hasattr(controller, "animation_manager"):
            start_pos = (float(old_x), float(old_y))
            end_pos = (float(self.x), float(self.y))
            animation = MoveAnimation(self, start_pos, end_pos)
            controller.animation_manager.add(animation)  # pyright: ignore[reportAttributeAccessIssue]

    def teleport(self, x: WorldTileCoord, y: WorldTileCoord) -> None:
        """Instantly move the actor's logical and visual position."""
        self.x = x
        self.y = y
        self.render_x = float(x)
        self.render_y = float(y)
        if self.gw:
            self.gw.actor_spatial_index.update(self)
        # Update any lights owned by this actor
        if self.gw:
            self.gw.on_actor_moved(self)

    def take_damage(self, amount: int, damage_type: str = "normal") -> None:
        """Handle damage to the actor.

        That includes:
        - Update health math.
        - Visual feedback (flash and floating text).
        - Handle death consequences, if any.

        Args:
            amount: Amount of damage to take
            damage_type: "normal" or "radiation"
        """
        # Visual feedback.
        if self.visual_effects:
            self.visual_effects.flash(colors.RED)

        if self.health:
            actual_damage = amount

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

            # Emit floating text: skull for lethal damage, number otherwise
            if actual_damage > 0:
                died = not self.health.is_alive()
                publish_event(
                    FloatingTextEvent(
                        text="💀" if died else f"-{actual_damage}",
                        target_actor_id=id(self),
                        valence=FloatingTextValence.NEGATIVE,
                        size=FloatingTextSize.LARGE
                        if died
                        else FloatingTextSize.NORMAL,
                        duration=1.2 if died else None,
                        color=(200, 200, 200) if died else None,  # Light gray skull
                        world_x=self.x,
                        world_y=self.y,
                    )
                )

            if not self.health.is_alive():
                # Handle death consequences.
                self.ch = "x"
                self.color = colors.DEAD
                self.blocks_movement = False
                # If this actor was selected, deselect it.
                if self.gw and self.gw.selected_actor == self:
                    self.gw.selected_actor = None

    def heal(self, amount: int | None = None) -> int:
        """Heal the actor.

        Handles:
        - Health math (delegated to HealthComponent)
        - Visual feedback (green flash and floating text)

        Args:
            amount: Amount to heal. If None, restore to full HP.

        Returns:
            Actual amount healed (may be less than requested if at/near max HP).
        """
        if not self.health:
            return 0

        before = self.health.hp

        if amount is None:
            # Full restore
            self.health.hp = self.health.max_hp
        else:
            self.health.heal(amount)

        healed = self.health.hp - before

        if healed > 0:
            # Visual feedback
            if self.visual_effects:
                self.visual_effects.flash(colors.GREEN)

            # Emit floating text showing amount healed
            publish_event(
                FloatingTextEvent(
                    text=f"+{healed}",
                    target_actor_id=id(self),
                    valence=FloatingTextValence.POSITIVE,
                    world_x=self.x,
                    world_y=self.y,
                )
            )

        return healed

    def add_sound_emitter(self, emitter: SoundEmitter) -> None:
        """Add a sound emitter to this actor.

        Args:
            emitter: The SoundEmitter to add
        """
        if self.sound_emitters is None:
            self.sound_emitters = []
        self.sound_emitters.append(emitter)

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
        starting_weapon: Item | None = None,
        num_ready_slots: int = 2,
        speed: int = DEFAULT_ACTOR_SPEED,
        creature_size: CreatureSize = CreatureSize.MEDIUM,
        idle_profile: IdleAnimationProfile | None = None,
        visual_scale: float | None = None,
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
            starting_weapon: Initial equipped weapon
            num_ready_slots: The number of ready slots this character should have
            speed: Action speed (higher = more frequent actions)
            creature_size: Size category for idle animation and visual_scale defaults
            idle_profile: Custom idle animation profile (overrides creature_size)
            visual_scale: Rendering scale factor (overrides size-based default).
                If None, derived from creature_size via scale_for_size().
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

        # Create idle animation profile from size if not explicitly provided
        profile = idle_profile or create_profile_for_size(creature_size)
        visual_effects_component = VisualEffectsComponent(idle_profile=profile)

        # Derive visual_scale from creature_size if not explicitly provided
        effective_visual_scale = (
            visual_scale if visual_scale is not None else scale_for_size(creature_size)
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
            inventory=CharacterInventory(stats, num_ready_slots, actor=self),
            visual_effects=visual_effects_component,
            ai=ai,
            energy=EnergyComponent(speed=speed),
            visual_scale=effective_visual_scale,
            **kwargs,
        )

        self.pathfinding_goal: PathfindingGoal | None = None

        # Type narrowing - these are guaranteed to exist for Characters.
        self.stats: StatsComponent
        self.health: HealthComponent
        self.inventory: CharacterInventory
        self.visual_effects: VisualEffectsComponent
        self.modifiers: ModifiersComponent
        self.conditions: ConditionsComponent
        self.energy: EnergyComponent

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

    def get_next_action(self, controller: Controller) -> GameIntent | None:
        """Return a MoveIntent following this character's PathfindingGoal.

        For hierarchical paths (crossing regions), this method manages both the
        local path within the current region and the high-level region sequence.
        When the local path is exhausted, it computes the next segment.

        The autopilot revalidates the next step each turn. If the cached path is
        blocked, a new path is calculated. When no path can be found the goal is
        cleared and ``None`` is returned.
        """
        goal = self.pathfinding_goal
        if goal is None:
            return None

        path_is_valid = False
        if goal._cached_path:
            next_pos = goal._cached_path[0]
            validation = find_local_path(
                controller.gw.game_map,
                controller.gw.actor_spatial_index,
                self,
                (self.x, self.y),
                next_pos,
            )
            if validation:
                path_is_valid = True

        if not path_is_valid:
            recalculated = controller.start_actor_pathfinding(
                self, goal.target_pos, goal.final_intent
            )
            if not recalculated:
                return None

        # Defensive check: path may be empty if recalculation found no valid route
        if not self.pathfinding_goal or not self.pathfinding_goal._cached_path:
            self.pathfinding_goal = None
            return None

        next_pos = self.pathfinding_goal._cached_path.pop(0)
        dx, dy = next_pos[0] - self.x, next_pos[1] - self.y

        # After popping, check if we need to compute the next path segment
        self._advance_hierarchical_path(controller)

        from catley.game.actions.movement import MoveIntent

        return MoveIntent(controller, self, dx, dy)

    def _advance_hierarchical_path(self, controller: Controller) -> None:
        """Compute next path segment when current local path is exhausted.

        For hierarchical (cross-region) paths, when _cached_path becomes empty
        but high_level_path still has regions to traverse, this computes the
        local path to the next connection point (or to target_pos if we're in
        the final region).
        """
        goal = self.pathfinding_goal
        if goal is None:
            return

        # Only act if local path is exhausted and we have more regions to go
        if goal._cached_path:
            return
        if not goal.high_level_path:
            return

        gm = controller.gw.game_map
        asi = controller.gw.actor_spatial_index

        # Pop the region we just entered
        current_region_id = goal.high_level_path.pop(0)

        if goal.high_level_path:
            # More regions to traverse - path to next connection point
            next_region_id = goal.high_level_path[0]
            current_region = gm.regions.get(current_region_id)
            if current_region is None:
                self.pathfinding_goal = None
                return

            connection_point = current_region.connections.get(next_region_id)
            if connection_point is None:
                self.pathfinding_goal = None
                return

            # Compute local path to connection point
            # Note: We'll be at next_pos after moving, but we compute from current
            # position since the move hasn't happened yet. The path will be
            # validated on next turn anyway.
            new_path = find_local_path(
                gm, asi, self, (self.x, self.y), connection_point
            )
            if new_path:
                goal._cached_path = new_path
            else:
                # Can't reach connection - clear goal
                self.pathfinding_goal = None
        else:
            # We're in the final region - path to target
            new_path = find_local_path(gm, asi, self, (self.x, self.y), goal.target_pos)
            if new_path:
                goal._cached_path = new_path
            # If no path, leave _cached_path empty - goal will be cleared on next turn


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
        starting_weapon: Item | None = None,
        num_ready_slots: int = 2,
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
            num_ready_slots: The number of ready slots this character should have
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
            starting_weapon=starting_weapon,
            num_ready_slots=num_ready_slots,
            speed=speed,
        )

        # Give player starting energy so they can act immediately on game start.
        # NPCs don't need this - they get energy from on_player_action() before acting.
        self.energy.accumulated_energy = float(self.energy.speed)

    def get_next_action(self, controller: Controller) -> GameIntent | None:
        """Return the player's next action, prioritizing direct input."""

        if controller.turn_manager.has_pending_actions():
            if hasattr(controller, "stop_actor_pathfinding"):
                controller.stop_actor_pathfinding(self)
            return controller.turn_manager.dequeue_player_action()

        autopilot_action = super().get_next_action(controller)
        if autopilot_action:
            return autopilot_action

        return None


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
        starting_weapon: Item | None = None,
        num_ready_slots: int = 2,
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
            num_ready_slots: The number of ready slots this character should have
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
            starting_weapon=starting_weapon,
            num_ready_slots=num_ready_slots,
            speed=speed,
            **kwargs,
        )

        # Type narrowing - these are guaranteed to exist.
        self.ai: AIComponent

    def get_next_action(self, controller: Controller) -> GameIntent | None:
        """Return the next action for this NPC, including autopilot goals."""
        if not self.health.is_alive():
            return None

        autopilot_action = super().get_next_action(controller)
        if autopilot_action:
            return autopilot_action

        self.ai.update(controller)
        return self.ai.get_action(controller, self)
