"""
AI system for autonomous actor behavior.

Implements decision-making components that determine what actions
NPCs should take each turn based on their personality, situation,
and goals.

AIComponent:
    Base class for all AI behavior. Determines what action an actor
    should take on their turn and handles any AI state updates.

UnifiedAI:
    Single AI component for all NPC dispositions. Owns a UtilityBrain
    that scores all possible behaviors (attack, flee, avoid, watch,
    idle, wander, patrol) every tick. Disposition is a numeric value
    (-100 to +100) that feeds into considerations as an input, not a
    selector for which class runs.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.events import CombatInitiatedEvent, MessageEvent, publish_event
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
from brileta.types import ActorId
from brileta.util import rng

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actions.movement import MoveIntent

    from . import NPC, Actor, Character

_rng = rng.get("npc.ai")

# ---------------------------------------------------------------------------
# Disposition: a continuous numeric value (-100 to +100).
#
# Behavior emerges from utility scoring, not from discrete bands. These
# band boundaries exist only for display labels (barks, target descriptions)
# and for the "turns hostile!" event threshold. They should never gate
# behavioral decisions - that's what the scoring curves are for.
# ---------------------------------------------------------------------------

# Each entry is (upper_bound, label). Ordered from most hostile to most allied.
DISPOSITION_BANDS: list[tuple[int, str]] = [
    (-51, "Hostile"),
    (-21, "Unfriendly"),
    (-1, "Wary"),
    (20, "Approachable"),
    (60, "Friendly"),
    (100, "Ally"),
]

# Threshold below which an NPC is considered hostile. Derived from
# DISPOSITION_BANDS so there is exactly one source of truth.
HOSTILE_UPPER = DISPOSITION_BANDS[0][0]


def disposition_label(value: int) -> str:
    """Return a human-readable label for a numeric disposition value."""
    for upper_bound, label in DISPOSITION_BANDS:
        if value <= upper_bound:
            return label
    return DISPOSITION_BANDS[-1][1]


def disposition_to_normalized(value: int) -> float:
    """Convert numeric disposition (-100..+100) to 0.0..1.0 for utility input."""
    return (value + 100) / 200.0


def escalate_hostility(
    attacker: Character, defender: Character, controller: Controller
) -> None:
    """Notify defender AI of aggression and escalate to hostile if needed.

    Called by combat and stunt executors after any aggressive act (attack,
    push, trip, kick, punch). Records the attacker for combat awareness,
    and if the defender isn't already hostile, sets them hostile and triggers
    combat mode for player-involved conflicts.
    """
    defender_ai = defender.ai
    if defender_ai is None:
        return  # No AI to update (e.g., player character)

    # Always record attacker for combat awareness.
    defender_ai.notify_attacked(attacker)

    # Only escalate if not already hostile.
    if defender_ai.disposition_toward(attacker) <= HOSTILE_UPPER:
        return

    defender_ai.set_hostile(attacker)
    publish_event(MessageEvent(f"{defender.name} turns hostile!", colors.ORANGE))

    player = controller.gw.player
    if player is None:
        return

    # Trigger auto-entry into combat mode only for player-involved conflict.
    if attacker is player or defender is player:
        publish_event(CombatInitiatedEvent(attacker=attacker, defender=defender))


class AIComponent(abc.ABC):
    """Base class for AI decision-making components.

    Each AI component is responsible for determining what action
    an actor should take on their turn, based on the current game
    state and the actor's goals/personality.
    """

    def __init__(self) -> None:
        self.actor: Actor | None = None

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

    @abc.abstractmethod
    def set_hostile(self, target: Actor) -> None:
        """Mark ``target`` as hostile for relationship-aware AI systems."""
        pass

    @abc.abstractmethod
    def disposition_toward(self, target: Actor) -> int:
        """Return numeric disposition toward ``target``."""
        pass

    @abc.abstractmethod
    def modify_disposition(self, target: Actor, delta: int) -> None:
        """Adjust disposition toward ``target`` by ``delta``."""
        pass

    @abc.abstractmethod
    def notify_attacked(self, attacker: Actor) -> None:
        """Record that ``attacker`` initiated aggression."""
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


class UnifiedAI(AIComponent):
    """Single AI component for all NPC dispositions.

    Owns a UtilityBrain that scores all possible behaviors every tick.
    Disposition is a numeric value (-100 to +100) fed into considerations
    as an input, not a selector for which class runs.

    Dispositions are relationship-scoped and keyed by target actor identity.
    Unknown relationships default to neutral (0).
    """

    def __init__(
        self,
        aggro_radius: int = Combat.DEFAULT_AGGRO_RADIUS,
    ) -> None:
        super().__init__()
        # Per-relationship disposition values keyed by ``target.actor_id``.
        # Unknown actors default to 0 (neutral).
        self._dispositions: dict[ActorId, int] = {}
        self.aggro_radius = aggro_radius

        # Combat awareness: identity of the most recent attacker. Allows the
        # NPC to treat that actor as a threat even outside normal aggro range.
        # Cleared when the attacker dies or is no longer reachable.
        # TODO: An NPC attacked by an *unseen* attacker (e.g., hidden sniper)
        # should panic/flee directionally rather than targeting the exact actor.
        self._last_attacker_id: ActorId | None = None

        # Debug: cached results from the last utility evaluation.
        # Read by AI debug live variables for the debug stats overlay.
        self.last_chosen_action: str | None = None
        self.last_scores: list[ScoredAction] = []
        self.last_threat_level: float | None = None
        self.last_target_actor_id: ActorId | None = None

        # Preconditions only gate physical impossibility (threat present,
        # escape route exists). Disposition influences behavior entirely
        # through scoring curves, enabling emergent behavior like a friendly
        # NPC fleeing when attacked rather than being locked out by a
        # disposition band check.
        self.brain = UtilityBrain(
            [
                # Attack: only when hostile (game rule) and threat present.
                # Disposition INVERSE curve ensures more hostile = higher score.
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
                        # Within the hostile band, more hostile values score higher.
                        Consideration(
                            "disposition",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                    ],
                    preconditions=[_is_threat_present, _is_hostile],
                ),
                # Flee: scores higher when hostile and low health.
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
                        # Flee scores higher when hostile (in danger of combat).
                        Consideration(
                            "disposition",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                    ],
                    preconditions=[_is_threat_present],
                ),
                # Avoid: move away from the target when unfriendly.
                AvoidAction(
                    base_score=0.7,
                    considerations=[
                        Consideration(
                            "disposition",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                        # Avoid peaks when the target is nearby, not at max aggro range.
                        Consideration(
                            "threat_level",
                            ResponseCurve(
                                ResponseCurveType.BELL,
                                peak=0.7,
                                width=0.5,
                            ),
                        ),
                    ],
                    preconditions=[
                        _is_threat_present,
                        _has_escape_route,
                        _is_not_hostile,
                    ],
                ),
                # Watch: stay put facing threat (INVERSE disposition curve).
                WatchAction(
                    base_score=0.35,
                    considerations=[
                        Consideration(
                            "disposition",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.LINEAR),
                        ),
                    ],
                    preconditions=[_is_threat_present, _is_not_hostile],
                ),
                # Idle: always a fallback, no disposition dependency.
                IdleAction(
                    base_score=0.1,
                    considerations=[
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                        # Idle favors cautious/neutral dispositions and fades out
                        # for friendlier NPCs that should keep moving.
                        Consideration(
                            "disposition",
                            ResponseCurve(
                                ResponseCurveType.BELL,
                                peak=0.45,
                                width=0.35,
                            ),
                        ),
                    ],
                ),
                # Wander: more when neutral/positive disposition (LINEAR curve).
                WanderAction(
                    base_score=0.18,
                    considerations=[
                        Consideration(
                            "threat_level",
                            ResponseCurve(ResponseCurveType.INVERSE),
                        ),
                        Consideration(
                            "disposition",
                            ResponseCurve(ResponseCurveType.LINEAR),
                        ),
                    ],
                    preconditions=[_is_no_threat, _is_not_hostile],
                ),
                # PatrolAction is defined but not scored here. It will be
                # wired up when guards / soldiers with assigned patrol routes
                # are implemented. See PatrolAction class below.
            ]
        )

    def set_hostile(self, target: Actor) -> None:
        """Set hostility toward ``target`` to hostile (-75)."""
        self._dispositions[target.actor_id] = -75

    def modify_disposition(self, target: Actor, delta: int) -> None:
        """Adjust disposition toward ``target`` by delta, clamped to [-100, +100]."""
        current = self.disposition_toward(target)
        self._dispositions[target.actor_id] = max(-100, min(100, current + delta))

    def disposition_toward(self, target: Actor) -> int:
        """Return numeric disposition toward ``target``."""
        return self._dispositions.get(target.actor_id, 0)

    def notify_attacked(self, attacker: Actor) -> None:
        """Record attacker for combat awareness.

        Allows the NPC to treat the attacker as a threat even beyond normal
        aggro range. The awareness radius uses ``_COMBAT_AWARENESS_RADIUS``
        so awareness tuning is independent from flee-goal safety distance.
        """
        self._last_attacker_id = attacker.actor_id

    def get_action(self, controller: Controller, actor: NPC) -> GameIntent | None:
        """Score all actions and execute the winner.

        Priority: If standing on a hazardous tile, attempt to escape before
        any other action. Self-preservation overrides disposition behavior.

        When the ai.force_hostile live variable is true, overrides disposition
        to -100 in the context so combat actions dominate scoring.
        """
        from brileta import config
        from brileta.game.actors.goals import ContinueGoalAction
        from brileta.util.live_vars import live_variable_registry

        if not config.HOSTILE_AI_ENABLED:
            return None

        # Priority: Escape hazards before any other action
        escape_intent = self._try_escape_hazard(controller, actor)
        if escape_intent is not None:
            return escape_intent

        # Evaluate goal completion before any early exits. This ensures goals
        # are cleaned up even when the threat dies (target dead = no action
        # needed, but the goal should still be marked complete).
        if actor.current_goal is not None:
            actor.current_goal.evaluate_completion(actor, controller)
            if actor.current_goal.is_complete:
                actor.current_goal = None

        # Check force_hostile override
        force_var = live_variable_registry.get_variable("ai.force_hostile")
        force_hostile = force_var is not None and force_var.get_value()

        context = self._build_context(controller, actor, force_hostile=force_hostile)
        self.last_threat_level = context.threat_level
        self.last_target_actor_id = context.target.actor_id
        if not context.actor.health.is_alive() or not context.target.health.is_alive():
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
            intent = goal.get_next_action(actor, controller)
            if goal.is_complete:
                actor.current_goal = None
            return intent

        # A different action won - abandon the current goal if active
        if actor.current_goal is not None and not actor.current_goal.is_complete:
            actor.current_goal.abandon()
            actor.current_goal = None

        # Flee/Wander actions create goals instead of returning a single
        # move step.
        if isinstance(action, FleeAction):
            return action.get_intent_with_goal(context, actor)
        if isinstance(action, WanderAction):
            return action.get_intent_with_goal(context, actor)
        return action.get_intent(context)

    def _build_context(
        self,
        controller: Controller,
        actor: NPC,
        *,
        force_hostile: bool = False,
    ) -> UtilityContext:
        """Build the utility evaluation context for this tick.

        Args:
            controller: Game controller with access to world state.
            actor: NPC being evaluated.
            force_hostile: When True, override disposition to -100.
        """
        target_actor = self._select_target_actor(controller, actor)
        distance = ranges.calculate_distance(
            actor.x, actor.y, target_actor.x, target_actor.y
        )
        health_percent = (
            actor.health.hp / actor.health.max_hp if actor.health.max_hp > 0 else 0.0
        )

        # Disposition for context: force_hostile overrides to -100
        effective_disposition = (
            -100 if force_hostile else self.disposition_toward(target_actor)
        )

        threat_level = 0.0
        if target_actor.health.is_alive():
            threat_level = self._compute_relationship_threat(
                distance=distance, disposition=effective_disposition
            )

        # Combat awareness: if the target is a known attacker outside normal
        # aggro range, compute an extended threat using
        # _COMBAT_AWARENESS_RADIUS as the effective radius.
        if (
            threat_level == 0.0
            and self._last_attacker_id == target_actor.actor_id
            and distance < _COMBAT_AWARENESS_RADIUS
        ):
            proximity = 1.0 - (distance / _COMBAT_AWARENESS_RADIUS)
            hostility = max(0.0, min(1.0, -effective_disposition / 50.0))
            threat_level = proximity * hostility

        best_attack_destination = self._select_attack_destination(
            controller, actor, target_actor
        )
        best_flee_step = self._select_flee_step(controller, actor, target_actor)

        return UtilityContext(
            controller=controller,
            actor=actor,
            target=target_actor,
            distance_to_target=distance,
            health_percent=health_percent,
            threat_level=threat_level,
            can_attack=distance == 1,
            has_escape_route=best_flee_step is not None,
            best_attack_destination=best_attack_destination,
            best_flee_step=best_flee_step,
            disposition=disposition_to_normalized(effective_disposition),
        )

    def _compute_relationship_threat(self, distance: int, disposition: int) -> float:
        """Compute threat from proximity and relationship hostility."""
        if distance > self.aggro_radius:
            return 0.0
        # Proximity captures spatial risk (nearer is higher).
        proximity_signal = 1.0 - (distance / self.aggro_radius)
        proximity_signal = max(0.0, min(1.0, proximity_signal))
        # Intent captures relationship hostility: <= 0 disposition contributes
        # threat, while neutral/friendly dispositions contribute none.
        # Scale over [-50, 0] so unfriendly actors still react strongly.
        hostility_signal = max(0.0, min(1.0, -disposition / 50.0))
        return proximity_signal * hostility_signal

    def _select_target_actor(self, controller: Controller, actor: NPC) -> Character:
        """Select the most threatening relationship target for this tick.

        Prefers living characters with non-zero relationship threat
        (proximity * hostility), which allows NPC-vs-NPC interactions when
        dispositions warrant it. Falls back to the player, then nearest living
        character, so context construction always has a concrete target.
        """
        from brileta.game.actors.core import Character

        best_target: Character | None = None
        best_threat = -1.0
        best_distance = float("inf")
        best_disposition = 101

        nearby_actors = controller.gw.actor_spatial_index.get_in_radius(
            actor.x, actor.y, self.aggro_radius
        )
        for other in nearby_actors:
            if other is actor or not isinstance(other, Character):
                continue
            if not other.health.is_alive():
                continue
            distance = ranges.calculate_distance(actor.x, actor.y, other.x, other.y)
            disposition = self.disposition_toward(other)
            threat = self._compute_relationship_threat(distance, disposition)
            if threat <= 0.0:
                continue
            if threat > best_threat or (
                threat == best_threat
                and (
                    distance < best_distance
                    or (distance == best_distance and disposition < best_disposition)
                )
            ):
                best_target = other
                best_threat = threat
                best_distance = float(distance)
                best_disposition = disposition

        if best_target is not None:
            return best_target

        # Combat awareness fallback: if no threat within aggro range but the
        # NPC was recently attacked, target the attacker regardless of distance.
        if self._last_attacker_id is not None:
            attacker = controller.gw.get_actor_by_id(self._last_attacker_id)
            if (
                attacker is not None
                and isinstance(attacker, Character)
                and attacker.health.is_alive()
            ):
                return attacker
            # Attacker is dead or gone - clear awareness.
            self._last_attacker_id = None

        player = controller.gw.player
        if player is not actor and player.health.is_alive():
            return player

        nearest_other: Character | None = None
        nearest_distance = float("inf")
        # TODO: Add a broader spatial-index nearest query so this fallback
        # doesn't scan all actors when no bounded threat exists.
        for other in controller.gw.actors:
            if other is actor or not isinstance(other, Character):
                continue
            if not other.health.is_alive():
                continue
            distance = ranges.calculate_distance(actor.x, actor.y, other.x, other.y)
            if distance < nearest_distance:
                nearest_other = other
                nearest_distance = float(distance)

        return nearest_other if nearest_other is not None else player

    def _select_attack_destination(
        self, controller: Controller, actor: NPC, target: Character
    ) -> tuple[int, int] | None:
        """Find the best tile adjacent to the target for attacking from."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost

        best_dest: tuple[int, int] | None = None
        best_score = float("inf")
        game_map = controller.gw.game_map

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = target.x + dx
                ty = target.y + dy
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
        self, controller: Controller, actor: NPC, target: Character
    ) -> tuple[int, int] | None:
        """Find the best adjacent tile that moves away from the target."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost

        game_map = controller.gw.game_map
        current_distance = ranges.calculate_distance(
            actor.x, actor.y, target.x, target.y
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

                distance_after = ranges.calculate_distance(tx, ty, target.x, target.y)
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


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------


def _is_threat_present(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is non-zero."""
    return context.threat_level > 0.0


def _is_no_threat(context: UtilityContext) -> bool:
    """Precondition: returns True when relationship-aware threat is zero."""
    return context.threat_level <= 0.0


def _has_escape_route(context: UtilityContext) -> bool:
    """Precondition: returns True when a valid flee step exists."""
    return context.has_escape_route


def _is_hostile(context: UtilityContext) -> bool:
    """Precondition: only initiate combat when hostile toward target.

    This is a game rule, not a scoring gate. Non-hostile NPCs do not start
    fights regardless of other considerations. Disposition scoring curves
    handle the intensity of hostile behavior (e.g., attack vs flee).
    """
    return context.disposition <= disposition_to_normalized(HOSTILE_UPPER)


def _is_not_hostile(context: UtilityContext) -> bool:
    """Precondition: only allow non-hostile social behaviors."""
    return context.disposition > disposition_to_normalized(HOSTILE_UPPER)


# ---------------------------------------------------------------------------
# Utility Actions
# ---------------------------------------------------------------------------


class AttackAction(UtilityAction):
    """Attack the target when adjacent, or pathfind toward them."""

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
        target = context.target

        if context.can_attack:
            controller.stop_plan(actor)
            return AttackIntent(controller, actor, target)

        plan = actor.active_plan
        if plan is not None and plan.context.target_position is not None:
            gx, gy = plan.context.target_position
            if (
                ranges.calculate_distance(target.x, target.y, gx, gy) == 1
                and controller.gw.game_map.walkable[gx, gy]
            ):
                return None

        if context.best_attack_destination and controller.start_plan(
            actor, WalkToPlan, target_position=context.best_attack_destination
        ):
            return None

        return None


class FleeAction(UtilityAction):
    """Flee from the target when health is low."""

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

        Called by UnifiedAI when FleeAction wins scoring. Creates a persistent
        FleeGoal so the NPC continues fleeing across multiple turns until safe,
        rather than re-evaluating from scratch each tick.
        """
        from brileta.game.actors.goals import FleeGoal

        # Create and assign a flee goal targeting the current threat.
        goal = FleeGoal(threat_actor_id=context.target.actor_id)
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


class AvoidAction(UtilityAction):
    """Move one step away from the target. Used by unfriendly NPCs."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="avoid",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        """Move away from the target using the best flee step."""
        from brileta.game.actions.movement import MoveIntent

        if context.best_flee_step is None:
            return None

        dx, dy = context.best_flee_step
        context.controller.stop_plan(context.actor)
        return MoveIntent(context.controller, context.actor, dx, dy)


class WatchAction(UtilityAction):
    """Stay put, facing the target. Used by wary/unfriendly NPCs."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="watch",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Do nothing - stay in place, watching.
        return None


class IdleAction(UtilityAction):
    """Do nothing this turn. Always a baseline fallback."""

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
    """Create/continue a wandering route goal when no threat is present."""

    def __init__(
        self,
        base_score: float,
        considerations: list[Consideration],
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="wander",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in UnifiedAI.
        # This fallback should not be reached in normal flow.
        return None

    def get_intent_with_goal(
        self, context: UtilityContext, actor: NPC
    ) -> GameIntent | None:
        """Create a heading-driven WanderGoal and return the first move intent."""
        from brileta.game.actors.goals import WanderGoal

        goal = WanderGoal(
            minimum_segment_steps=_WANDER_MIN_SEGMENT_STEPS,
            maximum_segment_steps=_WANDER_MAX_SEGMENT_STEPS,
            max_stuck_turns=_WANDER_MAX_STUCK_TURNS,
            heading_jitter_chance=_WANDER_HEADING_JITTER_CHANCE,
        )
        actor.current_goal = goal
        goal.tick()
        intent = goal.get_next_action(actor, context.controller)

        if goal.is_complete:
            actor.current_goal = None

        return intent


# How far from the NPC's current position to pick patrol waypoints.
_PATROL_RADIUS = 6

# How far combat awareness extends after an NPC is attacked.
_COMBAT_AWARENESS_RADIUS = 50

# How many candidate waypoints to pick for a patrol route.
_PATROL_WAYPOINT_COUNT = 3

# Wander segment tuning: heading is kept for a random budget of steps, then
# repicked. Blocked movement can force earlier heading resets.
_WANDER_MIN_SEGMENT_STEPS = 4
_WANDER_MAX_SEGMENT_STEPS = 12
_WANDER_MAX_STUCK_TURNS = 2
_WANDER_HEADING_JITTER_CHANCE = 0.15


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
        preconditions: list,
    ) -> None:
        super().__init__(
            action_id="patrol",
            base_score=base_score,
            considerations=considerations,
            preconditions=preconditions,
        )

    def get_intent(self, context: UtilityContext) -> GameIntent | None:
        # Intent generation is handled via get_intent_with_goal in UnifiedAI.
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
    return _pick_roam_waypoints(
        controller,
        actor,
        radius=_PATROL_RADIUS,
        waypoint_count=_PATROL_WAYPOINT_COUNT,
        require_unblocked=True,
    )


def _sample_roam_waypoints(
    candidates: list[tuple[int, int]], waypoint_count: int
) -> list[tuple[int, int]]:
    """Return up to ``waypoint_count`` sampled candidate waypoints."""
    if len(candidates) <= waypoint_count:
        return list(candidates)
    return _rng.sample(candidates, waypoint_count)


def _pick_roam_waypoints(
    controller: Controller,
    actor: NPC,
    *,
    radius: int,
    waypoint_count: int,
    require_unblocked: bool,
) -> list[tuple[int, int]]:
    """Pick walkable roam waypoints, preferring safe tiles when possible."""
    safe_candidates, hazardous_candidates = _collect_roam_candidates(
        controller, actor, radius=radius, require_unblocked=require_unblocked
    )
    candidates = safe_candidates + hazardous_candidates
    return _sample_roam_waypoints(candidates, waypoint_count)


def _collect_roam_candidates(
    controller: Controller,
    actor: NPC,
    radius: int,
    *,
    require_unblocked: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Collect nearby walkable roam candidates, partitioned by hazard.

    ``get_hazard_cost()`` returns 1 for safe tiles and >1 for hazardous tiles.
    """
    from brileta.environment.tile_types import get_hazard_cost

    game_map = controller.gw.game_map
    safe_candidates: list[tuple[int, int]] = []
    hazardous_candidates: list[tuple[int, int]] = []

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            tx = actor.x + dx
            ty = actor.y + dy
            if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                continue
            if not game_map.walkable[tx, ty]:
                continue
            if require_unblocked:
                blocker = controller.gw.get_actor_at_location(tx, ty)
                if blocker and blocker.blocks_movement and blocker is not actor:
                    continue
            if get_hazard_cost(int(game_map.tiles[tx, ty])) > 1:
                hazardous_candidates.append((tx, ty))
            else:
                safe_candidates.append((tx, ty))

    return safe_candidates, hazardous_candidates
