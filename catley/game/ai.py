"""
AI system for autonomous actor behavior.

Implements decision-making components that determine what actions
NPCs should take each turn based on their personality, situation,
and goals.

AIComponent:
    Base class for all AI behavior. Determines what action an actor
    should take on their turn and handles any AI state updates.

DispositionBasedAI:
    AI manager that delegates to focused behavior components based on
    the actor's current disposition. This allows NPCs to change behavior
    dynamically without swapping the main AI component.

Behavior Components:
    HostileAI, WaryAI, etc. - Focused implementations for specific dispositions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from catley import colors

from .enums import Disposition

if TYPE_CHECKING:
    from catley.controller import Controller

    from .actions import GameAction
    from .actors import Actor
    from .components import HealthComponent


class AIComponent(ABC):
    """Base class for AI decision-making components.

    Each AI component is responsible for determining what action
    an actor should take on their turn, based on the current game
    state and the actor's goals/personality.
    """

    def __init__(self) -> None:
        self.actor: Actor | None = None

    @abstractmethod
    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        """Determine what action this AI wants to perform this turn.

        Args:
            controller: Game controller with access to world state
            actor: Actor this AI is controlling

        Returns:
            GameAction to perform, or None to do nothing this turn
        """
        pass

    def update(self, controller: Controller) -> None:
        """Called each turn to update AI internal state.

        Base implementation does nothing. Override for AI that needs
        to track state between turns.

        Args:
            controller: Game controller with access to world state
        """
        pass


class DispositionBasedAI(AIComponent):
    """AI manager that delegates to focused behavior components.

    This AI maintains a set of behavior components for different dispositions
    and delegates to the appropriate one based on the actor's current disposition.
    This allows for focused, single-responsibility behavior classes while still
    supporting dynamic disposition changes.
    """

    def __init__(
        self, disposition: Disposition = Disposition.WARY, aggro_radius: int = 8
    ) -> None:
        super().__init__()

        self.disposition = disposition

        # Create behavior delegates for each possible disposition
        self._behaviors: dict[Disposition, AIComponent] = {
            Disposition.HOSTILE: HostileAI(aggro_radius),
            Disposition.WARY: WaryAI(),
            Disposition.UNFRIENDLY: UnfriendlyAI(),
            Disposition.APPROACHABLE: ApproachableAI(),
            Disposition.FRIENDLY: FriendlyAI(),
            Disposition.ALLY: AllyAI(),
        }

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        """Delegate to the appropriate behavior based on the current disposition."""
        # Get the behavior for current disposition
        behavior = self._behaviors.get(self.disposition)
        if behavior:
            return behavior.get_action(controller, actor)

        return None

    def update(self, controller: Controller) -> None:
        """Update any behavior-specific state."""
        # Update all behaviors (in case they need to track state)
        for behavior in self._behaviors.values():
            behavior.update(controller)


# Focused behavior implementations for specific dispositions


class HostileAI(AIComponent):
    """Aggressive behavior: attack and chase the player."""

    def __init__(self, aggro_radius: int = 8) -> None:
        super().__init__()
        self.aggro_radius = aggro_radius

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        player = controller.gw.player

        # Don't act if actor or player is dead
        if not self._is_alive(actor) or not self._is_alive(player):
            return None

        # Calculate distance to player
        dx = player.x - actor.x
        dy = player.y - actor.y
        distance = abs(dx) + abs(dy)  # Manhattan distance

        # Import here to avoid circular imports
        from .actions import AttackAction

        # Adjacent to player - attack!
        if distance == 1:
            controller.message_log.add_message(
                f"{actor.name} lunges at {player.name}!", colors.RED
            )
            return AttackAction(controller, actor, player)

        # Within aggro range - chase the player
        if distance <= self.aggro_radius:
            return self._get_move_toward_player(controller, actor, player, dx, dy)

        # Too far away - just prowl menacingly
        controller.message_log.add_message(
            f"{actor.name} prowls menacingly.", colors.ORANGE
        )
        return None

    def _is_alive(self, actor: Actor) -> bool:
        """Check if an actor is alive (has health and HP > 0)."""
        if actor.health is None:
            return False
        health = cast("HealthComponent", actor.health)
        return health.is_alive()

    def _get_move_toward_player(
        self, controller: Controller, actor: Actor, player: Actor, dx: int, dy: int
    ) -> GameAction | None:
        """Calculate movement toward the player with basic pathfinding."""
        from .actions import MoveAction

        # Determine preferred movement direction
        move_dx = 0
        move_dy = 0
        if dx != 0:
            move_dx = 1 if dx > 0 else -1
        if dy != 0:
            move_dy = 1 if dy > 0 else -1

        # Try different movement options in order of preference
        potential_moves = []
        if move_dx != 0 and move_dy != 0:
            potential_moves.append((move_dx, move_dy))  # Diagonal first
        if move_dx != 0:
            potential_moves.append((move_dx, 0))  # Horizontal
        if move_dy != 0:
            potential_moves.append((0, move_dy))  # Vertical

        # Try each potential move until we find one that works
        for test_dx, test_dy in potential_moves:
            if self._can_move_to(controller, test_dx, test_dy, actor, player):
                controller.message_log.add_message(
                    f"{actor.name} charges towards {player.name}.", colors.ORANGE
                )
                return MoveAction(controller, actor, test_dx, test_dy)

        # No valid move found
        controller.message_log.add_message(
            f"{actor.name} snarls, unable to reach {player.name}.",
            colors.ORANGE,
        )
        return None

    def _can_move_to(
        self, controller: Controller, dx: int, dy: int, actor: Actor, player: Actor
    ) -> bool:
        """Check if the actor can move in the given direction."""
        target_x = actor.x + dx
        target_y = actor.y + dy

        # Check map boundaries
        if not (
            0 <= target_x < controller.gw.game_map.width
            and 0 <= target_y < controller.gw.game_map.height
        ):
            return False

        # Check if tile is blocked
        if controller.gw.game_map.tile_blocked[target_x, target_y]:
            return False

        # Check for blocking actors (except player, which we want to attack)
        blocking_actor = controller.gw.get_actor_at_location(target_x, target_y)
        return not (
            blocking_actor
            and blocking_actor != player
            and blocking_actor.blocks_movement
        )


class WaryAI(AIComponent):
    """Wary behavior: wait and watch, but ready to become hostile."""

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        # For now, just wait (do nothing)
        # Future: might back away if player gets too close
        return None


class UnfriendlyAI(AIComponent):
    """Unfriendly behavior: suspicious but not immediately hostile."""

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        # For now, just wait (do nothing)
        # Future: might warn player to keep distance, prepare to defend
        return None


class ApproachableAI(AIComponent):
    """Approachable behavior: neutral, might initiate interaction."""

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        # For now, just wait (do nothing)
        # Future: might greet player when they approach, offer quests
        return None


class FriendlyAI(AIComponent):
    """Friendly behavior: helpful and welcoming."""

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        # For now, just wait (do nothing)
        # Future: might follow player, offer help, trade
        return None


class AllyAI(AIComponent):
    """Ally behavior: actively helpful in combat and exploration."""

    def get_action(self, controller: Controller, actor: Actor) -> GameAction | None:
        # For now, just wait (do nothing)
        # Future: actively help in combat, follow player, coordinate attacks
        return None
