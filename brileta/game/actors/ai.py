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

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.events import MessageEvent, publish_event
from brileta.game import ranges
from brileta.game.action_plan import WalkToPlan
from brileta.game.actors.utility import (
    Action as UtilityAction,
)
from brileta.game.actors.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    ScoredAction,
    UtilityBrain,
    UtilityContext,
)
from brileta.game.enums import Disposition
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actions.movement import MoveIntent

    from . import NPC, Actor, Character

_rng = rng.get("npc.ai")


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
        from brileta.environment.tile_types import get_tile_hazard_info
        from brileta.game.actions.movement import MoveIntent

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

    @property
    def active_behavior(self) -> AIComponent | None:
        """Return the behavior component for the current disposition."""
        return self._behaviors.get(self._disposition)

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        """Delegate to the appropriate behavior based on the current disposition.

        Priority: If standing on a hazardous tile, attempt to escape before
        any other action. Self-preservation overrides disposition behavior.

        When the ai.force_hostile live variable is true, delegates to the
        hostile behavior regardless of actual disposition.
        """
        from brileta.util.live_vars import live_variable_registry

        # Priority: Escape hazards before any other action
        escape_intent = self._try_escape_hazard(controller, actor)
        if escape_intent is not None:
            return escape_intent

        # If force_hostile is on, always use hostile behavior
        force_var = live_variable_registry.get_variable("ai.force_hostile")
        if force_var is not None and force_var.get_value():
            hostile = self._behaviors.get(Disposition.HOSTILE)
            if hostile:
                return hostile.get_action(controller, actor)

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

        # Debug: cached results from the last utility evaluation.
        # Read by AI debug live variables for the debug stats overlay.
        self.last_chosen_action: str | None = None
        self.last_scores: list[ScoredAction] = []

        self.brain = UtilityBrain(
            [
                AttackAction(
                    base_score=1.0,
                    considerations=[
                        Consideration(
                            "health_percent",
                            ResponseCurve(ResponseCurveType.LINEAR),
                        ),
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.LINEAR),
                        ),
                    ],
                    preconditions=[_is_threat_present],
                ),
                FleeAction(
                    base_score=1.0,
                    considerations=[
                        Consideration(
                            "health_percent",
                            ResponseCurve(ResponseCurveType.INVERSE),
                            weight=2.0,
                        ),
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.LINEAR),
                        ),
                        Consideration(
                            "has_escape_route",
                            ResponseCurve(ResponseCurveType.STEP, threshold=0.5),
                        ),
                    ],
                    preconditions=[_is_threat_present],
                ),
                IdleAction(
                    base_score=0.2,
                    considerations=[
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                    ],
                ),
                WanderAction(
                    base_score=0.1,
                    considerations=[
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                    ],
                ),
                PatrolAction(
                    base_score=0.25,
                    considerations=[
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                    ],
                ),
            ]
        )

    @property
    def disposition(self) -> Disposition:
        return Disposition.HOSTILE

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        from brileta import config
        from brileta.game.actors.goals import ContinueGoalAction

        if not config.HOSTILE_AI_ENABLED:
            return None

        # Evaluate goal completion before any early exits. This ensures goals
        # are cleaned up even when the threat dies (player dead = no action
        # needed, but the goal should still be marked complete).
        if actor.current_goal is not None:
            actor.current_goal.evaluate_completion(actor, controller)
            if actor.current_goal.is_complete:
                actor.current_goal = None

        context = self._build_context(controller, actor)
        if not context.actor.health.is_alive() or not context.player.health.is_alive():
            return None

        # Score all actions including "continue current goal"
        action, scored = self.brain.select_action(context, actor.current_goal)

        # Cache debug info for the debug stats overlay
        self.last_scores = scored
        if action is None:
            self.last_chosen_action = None
            return None
        if isinstance(action, ContinueGoalAction):
            self.last_chosen_action = f"ContinueGoal({action.goal.goal_id.title()})"
        else:
            self.last_chosen_action = action.action_id.title()

        # ContinueGoalAction won - delegate to the goal for the next intent
        if isinstance(action, ContinueGoalAction):
            goal = action.goal
            goal.tick()
            return goal.get_next_action(actor, controller)

        # A different action won - abandon the current goal if active
        if actor.current_goal is not None and not actor.current_goal.is_complete:
            actor.current_goal.abandon()
            actor.current_goal = None

        # FleeAction and PatrolAction create goals instead of returning a
        # single move step
        if isinstance(action, FleeAction):
            return action.get_intent_with_goal(context, actor)
        if isinstance(action, PatrolAction):
            return action.get_intent_with_goal(context, actor)

        return action.get_intent(context)

    def _get_move_toward_player(
        self,
        controller: Controller,
        actor: NPC,
        player: Character,
        dx: int,
        dy: int,
    ) -> MoveIntent | None:
        """Calculate movement toward the player with basic pathfinding."""
        from brileta.game.actions.movement import MoveIntent

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

    def _build_context(self, controller: Controller, actor: NPC) -> UtilityContext:
        player = controller.gw.player
        distance = ranges.calculate_distance(actor.x, actor.y, player.x, player.y)
        health_percent = (
            actor.health.hp / actor.health.max_hp if actor.health.max_hp > 0 else 0.0
        )

        threat_level = 0.0
        if player.health.is_alive() and distance <= self.aggro_radius:
            threat_level = 1.0 - (distance / self.aggro_radius)
            threat_level = max(0.0, min(1.0, threat_level))

        best_attack_destination = self._select_attack_destination(
            controller, actor, player
        )
        best_flee_step = self._select_flee_step(controller, actor, player)

        return UtilityContext(
            controller=controller,
            actor=actor,
            player=player,
            distance_to_player=distance,
            health_percent=health_percent,
            threat_level=threat_level,
            can_attack=distance == 1,
            has_escape_route=best_flee_step is not None,
            best_attack_destination=best_attack_destination,
            best_flee_step=best_flee_step,
        )

    def _select_attack_destination(
        self, controller: Controller, actor: NPC, player: Character
    ) -> tuple[int, int] | None:
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost

        best_dest: tuple[int, int] | None = None
        best_score = float("inf")
        game_map = controller.gw.game_map

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = player.x + dx
                ty = player.y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                if not game_map.walkable[tx, ty]:
                    continue
                actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
                if (
                    actor_at_tile
                    and actor_at_tile.blocks_movement
                    and actor_at_tile is not actor
                ):
                    continue

                dist = ranges.calculate_distance(actor.x, actor.y, tx, ty)
                tile_id = int(game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)

                damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
                if actor_at_tile and damage_per_turn > 0:
                    fire_cost = HAZARD_BASE_COST + damage_per_turn
                    hazard_cost = max(hazard_cost, fire_cost)

                score = dist + hazard_cost

                if score < best_score:
                    best_score = score
                    best_dest = (tx, ty)

        return best_dest

    def _select_flee_step(
        self, controller: Controller, actor: NPC, player: Character
    ) -> tuple[int, int] | None:
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost

        game_map = controller.gw.game_map
        current_distance = ranges.calculate_distance(
            actor.x, actor.y, player.x, player.y
        )

        candidates: list[tuple[int, int, int, float, int]] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = actor.x + dx
                ty = actor.y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                if not game_map.walkable[tx, ty]:
                    continue
                actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
                if (
                    actor_at_tile
                    and actor_at_tile.blocks_movement
                    and actor_at_tile is not actor
                ):
                    continue

                distance_after = ranges.calculate_distance(tx, ty, player.x, player.y)
                if distance_after <= current_distance:
                    continue

                tile_id = int(game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)
                damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
                if actor_at_tile and damage_per_turn > 0:
                    fire_cost = HAZARD_BASE_COST + damage_per_turn
                    hazard_cost = max(hazard_cost, fire_cost)

                step_cost = 1 if (dx == 0 or dy == 0) else 2
                candidates.append(
                    (dx, dy, distance_after, hazard_cost + step_cost, step_cost)
                )

        if not candidates:
            return None

        candidates.sort(key=lambda c: (-c[2], c[3], c[4]))
        dx, dy, _, _, _ = candidates[0]
        return dx, dy


def _is_threat_present(context: UtilityContext) -> bool:
    return context.threat_level > 0.0


class AttackAction(UtilityAction):
    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="attack",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        from brileta.game.actions.combat import AttackIntent

        controller = context.controller
        actor = context.actor
        player = context.player

        if context.can_attack:
            controller.stop_plan(actor)
            return AttackIntent(controller, actor, player)

        plan = actor.active_plan
        if plan is not None and plan.context.target_position is not None:
            gx, gy = plan.context.target_position
            if (
                ranges.calculate_distance(player.x, player.y, gx, gy) == 1
                and controller.gw.game_map.walkable[gx, gy]
            ):
                return None

        if context.best_attack_destination and controller.start_plan(
            actor, WalkToPlan, target_position=context.best_attack_destination
        ):
            return None

        return None


class FleeAction(UtilityAction):
    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="flee",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Single-step flee fallback (used when goal system isn't active)."""
        from brileta.game.actions.movement import MoveIntent

        if context.best_flee_step is None:
            return None

        dx, dy = context.best_flee_step
        context.controller.stop_plan(context.actor)
        return MoveIntent(context.controller, context.actor, dx, dy)

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a FleeGoal and return the first flee action.

        Called by HostileAI when FleeAction wins scoring. Creates a persistent
        FleeGoal so the NPC continues fleeing across multiple turns until safe,
        rather than re-evaluating from scratch each tick.
        """
        from brileta.game.actors.goals import FleeGoal

        # Create and assign a flee goal targeting the player
        goal = FleeGoal(threat_actor_id=id(context.player))
        actor.current_goal = goal

        # Evaluate completion once (sets initial distance tracking)
        goal.evaluate_completion(actor, context.controller)
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        # If the goal immediately failed (e.g., cornered), clear it so the
        # NPC doesn't waste a tick holding a dead goal.
        if goal.is_complete:
            actor.current_goal = None

        return intent


class IdleAction(UtilityAction):
    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
    ) -> None:
        super().__init__(
            action_id="idle",
            base_score=base_score,
            considerations=considerations,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        return None


class WanderAction(UtilityAction):
    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
    ) -> None:
        super().__init__(
            action_id="wander",
            base_score=base_score,
            considerations=considerations,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        from brileta.environment.tile_types import get_hazard_cost
        from brileta.game.actions.movement import MoveIntent

        controller = context.controller
        actor = context.actor
        game_map = controller.gw.game_map

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        _rng.shuffle(directions)
        for dx, dy in directions:
            tx = actor.x + dx
            ty = actor.y + dy
            if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                continue
            if not game_map.walkable[tx, ty]:
                continue
            if get_hazard_cost(int(game_map.tiles[tx, ty])) > 0:
                continue
            blocking_actor = controller.gw.get_actor_at_location(tx, ty)
            if blocking_actor and blocking_actor.blocks_movement:
                continue
            return MoveIntent(controller, actor, dx, dy)
        return None


# How far from the NPC's current position to pick patrol waypoints.
_PATROL_RADIUS = 6

# How many candidate waypoints to pick for a patrol route.
_PATROL_WAYPOINT_COUNT = 3


class PatrolAction(UtilityAction):
    """Patrol nearby when no threat is present.

    When this action wins scoring and the NPC has no active PatrolGoal,
    creates one with random walkable waypoints near the NPC. If a
    PatrolGoal already exists, ContinueGoalAction handles continuation
    so this action won't create duplicates.
    """

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
    ) -> None:
        super().__init__(
            action_id="patrol",
            base_score=base_score,
            considerations=considerations,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in HostileAI.
        # This fallback should not be reached in normal flow.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a PatrolGoal with random nearby waypoints and start patrolling.

        Picks 2-3 random walkable tiles within _PATROL_RADIUS of the NPC's
        current position as waypoints. Uses the game_map.walkable array to
        validate tiles.
        """
        from brileta.game.actors.goals import PatrolGoal

        waypoints = _pick_patrol_waypoints(context.controller, actor)
        if len(waypoints) < 2:
            # Not enough walkable space - fall back to doing nothing
            return None

        goal = PatrolGoal(waypoints)
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        # If the goal immediately failed, clear it so the NPC doesn't
        # waste a tick holding a dead goal.
        if goal.is_complete:
            actor.current_goal = None

        return intent


def _pick_patrol_waypoints(controller: Controller, actor: NPC) -> list[tuple[int, int]]:
    """Pick random walkable tiles near the actor for patrol waypoints.

    Collects all walkable tiles within _PATROL_RADIUS of the actor,
    then samples _PATROL_WAYPOINT_COUNT of them.
    """
    game_map = controller.gw.game_map
    candidates: list[tuple[int, int]] = []

    for dx in range(-_PATROL_RADIUS, _PATROL_RADIUS + 1):
        for dy in range(-_PATROL_RADIUS, _PATROL_RADIUS + 1):
            if dx == 0 and dy == 0:
                continue
            tx = actor.x + dx
            ty = actor.y + dy
            if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                continue
            if not game_map.walkable[tx, ty]:
                continue
            candidates.append((tx, ty))

    if len(candidates) <= _PATROL_WAYPOINT_COUNT:
        return candidates

    return _rng.sample(candidates, _PATROL_WAYPOINT_COUNT)


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
