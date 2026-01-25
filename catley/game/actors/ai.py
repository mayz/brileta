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

import abc
from typing import TYPE_CHECKING

from catley import colors
from catley.constants.combat import CombatConstants as Combat
from catley.events import MessageEvent, publish_event
from catley.game import ranges
from catley.game.enums import Disposition

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.actions.base import GameIntent
    from catley.game.actions.movement import MoveIntent

    from . import NPC, Actor, Character


class AIComponent(abc.ABC):
    """Base class for AI decision-making components.

    Each AI component is responsible for determining what action
    an actor should take on their turn, based on the current game
    state and the actor's goals/personality.
    """

    def __init__(self) -> None:
        self.actor: Actor | None = None

    @property
    @abc.abstractmethod
    def disposition(self) -> Disposition:
        """The AI's current disposition toward the player."""
        ...

    @abc.abstractmethod
    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        """Determine what action this AI wants to perform this turn.

        Args:
            controller: Game controller with access to world state
            actor: Actor this AI is controlling

        Returns:
            GameIntent to perform, or None to do nothing this turn
        """
        pass

    def update(self, controller: Controller) -> None:  # noqa: B027
        """Called each turn to update AI internal state.

        Base implementation does nothing. Override for AI that needs
        to track state between turns.

        Args:
            controller: Game controller with access to world state
        """
        pass

    def _try_escape_hazard(
        self, controller: Controller, actor: NPC
    ) -> MoveIntent | None:
        """Check if standing on hazard and return escape move if possible.

        Finds the nearest safe adjacent tile and returns a MoveIntent to it.
        Returns None if not on a hazard or if no safe escape exists.

        Args:
            controller: Game controller with access to world state.
            actor: Actor to check for hazard escape.

        Returns:
            MoveIntent to escape, or None if no escape needed/possible.
        """
        from catley.environment.tile_types import get_tile_hazard_info
        from catley.game.actions.movement import MoveIntent

        # Check if currently standing on a hazard
        tile_id = int(controller.gw.game_map.tiles[actor.x, actor.y])
        damage_dice, _ = get_tile_hazard_info(tile_id)

        if not damage_dice:
            return None  # Not on a hazard, no escape needed

        # Find safe adjacent tiles (orthogonal and diagonal).
        # Track (dx, dy, distance) - prefer orthogonal (1) over diagonal (2).
        safe_tiles: list[tuple[int, int, int]] = []
        game_map = controller.gw.game_map

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = actor.x + dx, actor.y + dy

                # Check map boundaries
                if not (0 <= nx < game_map.width and 0 <= ny < game_map.height):
                    continue

                # Check if walkable
                if not game_map.walkable[nx, ny]:
                    continue

                # Check for blocking actors
                blocking_actor = controller.gw.get_actor_at_location(nx, ny)
                if blocking_actor and blocking_actor.blocks_movement:
                    continue

                # Check if this tile is also hazardous
                adj_tile_id = int(game_map.tiles[nx, ny])
                adj_damage, _ = get_tile_hazard_info(adj_tile_id)
                if adj_damage:
                    continue  # Skip hazardous tiles

                # This is a valid safe tile - track with distance preference
                dist = 1 if (dx == 0 or dy == 0) else 2
                safe_tiles.append((dx, dy, dist))

        if not safe_tiles:
            return None  # No safe escape exists

        # Pick the closest safe tile (prefer orthogonal)
        safe_tiles.sort(key=lambda t: t[2])
        dx, dy, _ = safe_tiles[0]

        # Publish escape message
        publish_event(
            MessageEvent(
                f"{actor.name} scrambles to escape the hazard!",
                colors.ORANGE,
            )
        )

        return MoveIntent(controller, actor, dx, dy)


class DispositionBasedAI(AIComponent):
    """AI manager that delegates to focused behavior components.

    This AI maintains a set of behavior components for different dispositions
    and delegates to the appropriate one based on the actor's current disposition.
    This allows for focused, single-responsibility behavior classes while still
    supporting dynamic disposition changes.
    """

    def __init__(
        self,
        disposition: Disposition = Disposition.WARY,
        aggro_radius: int = Combat.DEFAULT_AGGRO_RADIUS,
    ) -> None:
        super().__init__()

        self._disposition = disposition

        # Create behavior delegates for each possible disposition
        self._behaviors: dict[Disposition, AIComponent] = {
            Disposition.HOSTILE: HostileAI(aggro_radius),
            Disposition.WARY: WaryAI(),
            Disposition.UNFRIENDLY: UnfriendlyAI(),
            Disposition.APPROACHABLE: ApproachableAI(),
            Disposition.FRIENDLY: FriendlyAI(),
            Disposition.ALLY: AllyAI(),
        }

    @property
    def disposition(self) -> Disposition:
        return self._disposition

    @disposition.setter
    def disposition(self, value: Disposition) -> None:
        self._disposition = value

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        """Delegate to the appropriate behavior based on the current disposition.

        Priority: If standing on a hazardous tile, attempt to escape before
        any other action. Self-preservation overrides disposition behavior.
        """
        # Priority: Escape hazards before any other action
        escape_intent = self._try_escape_hazard(controller, actor)
        if escape_intent is not None:
            return escape_intent

        # Normal disposition-based behavior
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

    def __init__(self, aggro_radius: int = Combat.DEFAULT_AGGRO_RADIUS) -> None:
        super().__init__()
        self.aggro_radius = aggro_radius

    @property
    def disposition(self) -> Disposition:
        return Disposition.HOSTILE

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        from catley import config

        if not config.HOSTILE_AI_ENABLED:
            return None

        player = controller.gw.player

        if not actor.health.is_alive() or not player.health.is_alive():
            return None

        distance = ranges.calculate_distance(actor.x, actor.y, player.x, player.y)

        from catley.game.actions.combat import AttackIntent

        if distance == 1:
            controller.stop_walk_to_plan(actor)
            return AttackIntent(controller, actor, player)

        # Check if already walking toward a valid adjacent-to-player tile
        plan = actor.active_plan
        if plan is not None and plan.context.target_position is not None:
            gx, gy = plan.context.target_position
            if (
                ranges.calculate_distance(player.x, player.y, gx, gy) == 1
                and controller.gw.game_map.walkable[gx, gy]
            ):
                return None

        # Find the best adjacent-to-player tile to path toward.
        # Prefer tiles that are closer to the actor AND non-hazardous.
        from catley.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost

        best_dest: tuple[int, int] | None = None
        best_score = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = player.x + dx
                ty = player.y + dy
                if not (
                    0 <= tx < controller.gw.game_map.width
                    and 0 <= ty < controller.gw.game_map.height
                ):
                    continue
                if not controller.gw.game_map.walkable[tx, ty]:
                    continue
                actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
                if (
                    actor_at_tile
                    and actor_at_tile.blocks_movement
                    and actor_at_tile is not actor
                ):
                    continue

                # Score combines distance and hazard cost so AI prefers
                # non-hazardous tiles but will use them if no better option
                dist = ranges.calculate_distance(actor.x, actor.y, tx, ty)
                tile_id = int(controller.gw.game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)

                # Also check for fire actors (campfires, etc.) at this position
                damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
                if actor_at_tile and damage_per_turn > 0:
                    fire_cost = HAZARD_BASE_COST + damage_per_turn
                    hazard_cost = max(hazard_cost, fire_cost)

                score = dist + hazard_cost

                if score < best_score:
                    best_score = score
                    best_dest = (tx, ty)

        if best_dest and controller.start_walk_to_plan(actor, best_dest):
            return None

        return None

    def _get_move_toward_player(
        self,
        controller: Controller,
        actor: NPC,
        player: Character,
        dx: int,
        dy: int,
    ) -> MoveIntent | None:
        """Calculate movement toward the player with basic pathfinding."""
        from catley.game.actions.movement import MoveIntent

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
                publish_event(
                    MessageEvent(
                        f"{actor.name} charges towards {player.name}.", colors.ORANGE
                    )
                )
                return MoveIntent(controller, actor, test_dx, test_dy)

        # No valid move found
        publish_event(
            MessageEvent(
                f"{actor.name} snarls, unable to reach {player.name}.", colors.ORANGE
            )
        )
        return None

    def _can_move_to(
        self,
        controller: Controller,
        dx: int,
        dy: int,
        actor: Actor,
        player: Actor,
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
        if not controller.gw.game_map.walkable[target_x, target_y]:
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

    @property
    def disposition(self) -> Disposition:
        return Disposition.WARY

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        # For now, just wait (do nothing)
        # Future: might back away if player gets too close
        return None


class UnfriendlyAI(AIComponent):
    """Unfriendly behavior: suspicious but not immediately hostile."""

    @property
    def disposition(self) -> Disposition:
        return Disposition.UNFRIENDLY

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        # For now, just wait (do nothing)
        # Future: might warn player to keep distance, prepare to defend
        return None


class ApproachableAI(AIComponent):
    """Approachable behavior: neutral, might initiate interaction."""

    @property
    def disposition(self) -> Disposition:
        return Disposition.APPROACHABLE

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        # For now, just wait (do nothing)
        # Future: might greet player when they approach, offer quests
        return None


class FriendlyAI(AIComponent):
    """Friendly behavior: helpful and welcoming."""

    @property
    def disposition(self) -> Disposition:
        return Disposition.FRIENDLY

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        # For now, just wait (do nothing)
        # Future: might follow player, offer help, trade
        return None


class AllyAI(AIComponent):
    """Ally behavior: actively helpful in combat and exploration."""

    @property
    def disposition(self) -> Disposition:
        return Disposition.ALLY

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        # For now, just wait (do nothing)
        # Future: actively help in combat, follow player, coordinate attacks
        return None
