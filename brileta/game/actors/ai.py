"""
AI system for autonomous actor behavior.

Implements decision-making components that determine what actions
NPCs should take each turn based on their personality, situation,
and goals.

AIComponent:
    Single AI component for all NPC dispositions. Owns a UtilityBrain
    that scores all possible behaviors (attack, flee, avoid, watch,
    idle, wander, patrol) every tick. Disposition is a numeric value
    (-100 to +100) that feeds into considerations as an input, not a
    selector for which class runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.events import CombatInitiatedEvent, MessageEvent, publish_event
from brileta.game import ranges
from brileta.game.actors.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    ScoredAction,
    UtilityBrain,
    UtilityContext,
)
from brileta.types import ActorId, Direction, WorldTilePos
from brileta.util import rng

from .ai_actions import (
    AttackAction,
    AvoidAction,
    FleeAction,
    IdleAction,
    WanderAction,
    WatchAction,
    _has_escape_route,
    _is_hostile,
    _is_no_threat,
    _is_not_hostile,
    _is_threat_present,
)

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

# How far combat awareness extends after an NPC is attacked.
_COMBAT_AWARENESS_RADIUS = 50


@dataclass(slots=True)
class FleeCandidate:
    """One potential flee step scored for tactical desirability."""

    direction: Direction
    distance_after: float
    hazard_score: int
    step_cost: int


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
    if defender_ai.is_hostile_toward(attacker):
        return

    defender_ai.set_hostile(attacker)
    publish_event(MessageEvent(f"{defender.name} turns hostile!", colors.ORANGE))

    player = controller.gw.player
    if player is None:
        return

    # Trigger auto-entry into combat mode only for player-involved conflict.
    if attacker is player or defender is player:
        publish_event(CombatInitiatedEvent(attacker=attacker, defender=defender))


class AIComponent:
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
        self.actor: Actor | None = None
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
                # are implemented. See PatrolAction in ai_actions.py.
            ]
        )

    def update(self, controller: Controller) -> None:
        """Update per-turn AI state.

        AI state is currently event-driven, so no periodic updates are needed.
        """
        return

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

    def is_hostile_toward(self, target: Actor) -> bool:
        """Return True if disposition toward target is at or below hostile threshold."""
        return self.disposition_toward(target) <= HOSTILE_UPPER

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
        from brileta.util.pathfinding import probe_step

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

                # Passability check (bounds + walkable + actor blocking).
                # Door-capable NPCs can escape through closed doors.
                if (
                    probe_step(
                        game_map,
                        controller.gw,
                        nx,
                        ny,
                        can_open_doors=actor.can_open_doors,
                    )
                    is not None
                ):
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
    ) -> WorldTilePos | None:
        """Find the best tile adjacent to the target for attacking from."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
        from brileta.util.pathfinding import probe_step

        best_dest: WorldTilePos | None = None
        best_score = float("inf")
        game_map = controller.gw.game_map

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = target.x + dx
                ty = target.y + dy
                if (
                    probe_step(
                        game_map,
                        controller.gw,
                        tx,
                        ty,
                        exclude_actor=actor,
                        can_open_doors=actor.can_open_doors,
                    )
                    is not None
                ):
                    continue

                dist = ranges.calculate_distance(actor.x, actor.y, tx, ty)
                tile_id = int(game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)

                actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
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
    ) -> Direction | None:
        """Find the best adjacent tile that moves away from the target."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
        from brileta.util.pathfinding import probe_step

        game_map = controller.gw.game_map
        current_distance = ranges.calculate_distance(
            actor.x, actor.y, target.x, target.y
        )

        candidates: list[FleeCandidate] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                tx = actor.x + dx
                ty = actor.y + dy
                if (
                    probe_step(
                        game_map,
                        controller.gw,
                        tx,
                        ty,
                        exclude_actor=actor,
                        can_open_doors=actor.can_open_doors,
                    )
                    is not None
                ):
                    continue

                distance_after = ranges.calculate_distance(tx, ty, target.x, target.y)
                if distance_after <= current_distance:
                    continue

                tile_id = int(game_map.tiles[tx, ty])
                hazard_cost = get_hazard_cost(tile_id)
                actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
                damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
                if actor_at_tile and damage_per_turn > 0:
                    fire_cost = HAZARD_BASE_COST + damage_per_turn
                    hazard_cost = max(hazard_cost, fire_cost)

                step_cost = 1 if (dx == 0 or dy == 0) else 2
                candidates.append(
                    FleeCandidate(
                        direction=(dx, dy),
                        distance_after=distance_after,
                        hazard_score=hazard_cost + step_cost,
                        step_cost=step_cost,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda candidate: (
                -candidate.distance_after,
                candidate.hazard_score,
                candidate.step_cost,
            )
        )
        return candidates[0].direction
