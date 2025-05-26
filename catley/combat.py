from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import colors
from model import Actor
from play_mode import PlayMode

if TYPE_CHECKING:
    from engine import Controller


class CombatSide:
    PLAYER = "player"
    ENEMY = "enemy"
    # Future: NEUTRAL, FACTION_A, FACTION_B, etc.


@dataclass
class Combatant:
    """Wrapper for an actor in combat with combat-specific state."""

    actor: Actor
    side: str
    initiative: int
    movement_points_remaining: int = 0
    has_acted: bool = False

    def reset_turn(self) -> None:
        """Reset for a new turn."""
        # In Wastoid rules, movement is typically agility-based
        self.movement_points_remaining = max(1, self.actor.agility + 2)  # Example
        self.has_acted = False


class CombatManager:
    """Manages turn-based combat flow and state."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.combatants: list[Combatant] = []
        self.current_turn_index: int = 0
        self.round_number: int = 1
        self.combat_started: bool = False  # False = planning phase, True = turns active

    def start_planning_phase(self) -> None:
        """Start combat mode - player can plan but no turns yet."""
        self.combat_started = False
        self.controller.message_log.add_message(
            "Combat mode active. Select a target to begin combat.", colors.YELLOW
        )

    def begin_turn_based_combat(self, player_target: Actor) -> None:
        """Start actual turn-based combat with player attacking the target.

        Args:
            player_target: The actor the player chose to attack
        """
        if self.combat_started:
            return  # Already in turn-based combat

        self.combat_started = True

        # Find all actors in the player's current field of view
        participants = self._get_actors_in_fov()

        # Create combatants with player always first
        for actor in participants:
            side = self._determine_side(actor)
            combatant = Combatant(
                actor=actor, side=side, initiative=0
            )  # Initiative unused
            self.combatants.append(combatant)

        # Sort so player is always first
        self.combatants.sort(key=lambda c: 0 if c.side == CombatSide.PLAYER else 1)

        # Set up first turn (player's turn)
        self.current_turn_index = 0
        self._start_turn()

        self.controller.message_log.add_message(
            f"Combat begins! Round {self.round_number}", colors.RED
        )

        # Immediately perform player's attack action
        self._perform_player_attack_action(player_target)

    def _get_actors_in_fov(self) -> list[Actor]:
        """Get all living actors currently in the player's field of view."""
        actors_in_fov = []

        for entity in self.controller.model.entities:
            if (
                isinstance(entity, Actor)
                and entity.is_alive()
                and self.controller.fov.contains(entity.x, entity.y)
            ):
                actors_in_fov.append(entity)

        return actors_in_fov

    def _determine_side(self, actor: Actor) -> str:
        """Determine which side an actor is on."""
        if actor == self.controller.model.player:
            return CombatSide.PLAYER

        # For now, all NPCs are enemies
        # Later: check faction relationships
        return CombatSide.ENEMY

    def _perform_player_attack_action(self, target: Actor) -> None:
        """Perform the player's attack action, including movement if needed."""
        player = self.controller.model.player
        current_combatant = self.get_current_combatant()

        if not current_combatant or current_combatant.actor != player:
            return

        # Check if player needs to move to attack target
        weapon = player.equipped_weapon
        if weapon and weapon.melee:
            # Melee weapon - need to be adjacent
            distance = abs(player.x - target.x) + abs(
                player.y - target.y
            )  # Manhattan distance

            if distance > 1:
                # Need to move closer - use available movement points
                self._move_player_toward_target(target)

                # Check if we're now in range
                new_distance = abs(player.x - target.x) + abs(player.y - target.y)
                if new_distance > 1:
                    self.controller.message_log.add_message(
                        f"Moved closer to {target.name} but "
                        "couldn't reach them this turn.",
                        colors.YELLOW,
                    )
                    # Player used their action trying to close distance
                    current_combatant.has_acted = True
                    return

        # Player is in range - perform the attack
        from actions import AttackAction

        attack_action = AttackAction(
            controller=self.controller, attacker=player, defender=target
        )
        attack_action.execute()

        # Mark that player has acted
        current_combatant.has_acted = True

    def _move_player_toward_target(self, target: Actor) -> None:
        """Move player as close as possible to target using available movement."""
        player = self.controller.model.player
        current_combatant = self.get_current_combatant()

        if not current_combatant:
            return

        # Simple pathfinding - move one step at a time toward target
        while current_combatant.movement_points_remaining > 0:
            # Calculate direction to target
            dx = 0
            dy = 0

            if target.x > player.x:
                dx = 1
            elif target.x < player.x:
                dx = -1

            if target.y > player.y:
                dy = 1
            elif target.y < player.y:
                dy = -1

            # Try to move in that direction
            if dx != 0 or dy != 0:
                if self.move_current_combatant(dx, dy):
                    # Successfully moved
                    self.controller.fov.fov_needs_recomputing = True

                    # Check if we're adjacent now
                    distance = abs(player.x - target.x) + abs(player.y - target.y)
                    if distance <= 1:
                        break  # We're in range!
                else:
                    # Couldn't move (blocked) - stop trying
                    break
            else:
                # Already at same position? This shouldn't happen
                break

    def is_planning_phase(self) -> bool:
        """Check if we're in planning phase (combat mode but no turns yet)."""
        return not self.combat_started

    def is_turn_based_active(self) -> bool:
        """Check if turn-based combat is running."""
        return self.combat_started

    def get_current_combatant(self) -> Combatant | None:
        """Get the combatant whose turn it is."""
        if not self.is_turn_based_active() or not self.combatants:
            return None
        return self.combatants[self.current_turn_index]

    def is_player_turn(self) -> bool:
        """Check if it's the player's turn in combat."""
        current = self.get_current_combatant()
        return current is not None and current.actor == self.controller.model.player

    def end_current_turn(self) -> None:
        """End the current combatant's turn and advance to next."""
        if not self.is_turn_based_active():
            return

        # Clean up dead combatants before advancing turn
        self.cleanup_dead_combatants()

        if not self.combatants:  # All combatants dead
            return

        self.current_turn_index = (self.current_turn_index + 1) % len(self.combatants)

        # If we've cycled back to the first combatant, start new round
        if self.current_turn_index == 0:
            self.round_number += 1
            self.controller.message_log.add_message(
                f"Round {self.round_number}", colors.YELLOW
            )

        self._start_turn()

        # Check if combat should end
        if self.is_combat_over():
            self._end_combat()

    def start_player_turn(self) -> None:
        """Start the player's turn - called when it becomes player's turn again.

        In the future, this could show an action menu or wait for player input.
        For now, the player acts immediately when they click a target.
        """
        # TODO: Show action menu here in the future
        # For now, we wait for player to click a target or take other action
        pass

    def _start_turn(self) -> None:
        """Set up the current combatant's turn."""
        current = self.get_current_combatant()
        if current:
            current.reset_turn()
            if current.actor == self.controller.model.player:
                self.controller.message_log.add_message("Your turn!", colors.GREEN)
                self.start_player_turn()
            else:
                self.controller.message_log.add_message(
                    f"{current.actor.name}'s turn.", colors.WHITE
                )
                # TODO: Implement NPC AI turn logic here
                # For now, just end their turn immediately
                self.end_current_turn()

    def can_move(self, distance: int = 1) -> bool:
        """Check if current combatant can move the specified distance."""
        current = self.get_current_combatant()
        return current is not None and current.movement_points_remaining >= distance

    def move_current_combatant(self, dx: int, dy: int) -> bool:
        """Move current combatant and deduct movement points."""
        if not self.can_move():
            return False

        current = self.get_current_combatant()
        if not current:
            return False

        actor = current.actor
        newx = actor.x + dx
        newy = actor.y + dy

        # Check map boundaries and blocked tiles
        game_map = self.controller.model.game_map
        if (newx < 0 or newx >= game_map.width or
            newy < 0 or newy >= game_map.height or
            game_map.tiles[newx][newy].blocked):
            return False

        # Check for blocking entities (same logic as MoveAction)
        for entity in self.controller.model.entities:
            if (entity.blocks_movement
                and entity.x == newx
                and entity.y == newy):
                return False  # Cannot move into blocking entity

        # Move is valid - perform it
        actor.move(dx, dy)
        current.movement_points_remaining -= 1
        return True

    def can_act(self) -> bool:
        """Check if current combatant can take an action."""
        current = self.get_current_combatant()
        return current is not None and not current.has_acted

    def perform_action(self) -> None:
        """Mark that current combatant has performed their action."""
        current = self.get_current_combatant()
        if current:
            current.has_acted = True

    def is_combat_over(self) -> bool:
        """Check if combat should end."""
        if not self.is_turn_based_active():
            return False

        player_side_alive = any(
            c.actor.is_alive() for c in self.combatants if c.side == CombatSide.PLAYER
        )
        enemy_side_alive = any(
            c.actor.is_alive() for c in self.combatants if c.side == CombatSide.ENEMY
        )

        return not (player_side_alive and enemy_side_alive)

    def _end_combat(self) -> None:
        """End turn-based combat and return to exploration."""
        self.controller.message_log.add_message("Combat ended.", colors.GREEN)
        # Exit combat mode entirely by changing PlayMode
        self.controller.model.play_mode = PlayMode.ROAMING
        self.controller.model.combat_manager = None

    def cleanup_dead_combatants(self) -> None:
        """Remove dead combatants from combat."""
        if not self.is_turn_based_active():
            return

        initial_count = len(self.combatants)
        self.combatants = [c for c in self.combatants if c.actor.is_alive()]

        # Adjust current turn index if needed
        if self.current_turn_index >= len(self.combatants) and self.combatants:
            self.current_turn_index = 0

        # If we removed combatants, check if combat should end
        if len(self.combatants) < initial_count and self.is_combat_over():
            self._end_combat()

    def get_valid_targets(self) -> list[Actor]:
        """Get all valid targets for player actions."""
        valid_targets = []
        for entity in self.controller.model.entities:
            if (
                isinstance(entity, Actor)
                and entity != self.controller.model.player  # Can't target self
                and entity.is_alive()
                and self.controller.fov.contains(entity.x, entity.y)
            ):
                valid_targets.append(entity)

        return valid_targets
