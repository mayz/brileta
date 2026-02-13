"""
AIComponent: the core NPC decision-making driver.

Implements the AIComponent that determines what actions NPCs should take
each turn based on their personality, situation, and goals.

AIComponent:
    Single AI component for all NPC dispositions. Owns a UtilityBrain
    that scores all possible behaviors (attack, flee, avoid, watch,
    idle, wander, patrol) every tick. Disposition is a numeric value
    (-100 to +100) that feeds into considerations as an input, not a
    selector for which class runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from brileta import colors
from brileta.constants.combat import CombatConstants as Combat
from brileta.events import (
    CombatInitiatedEvent,
    FloatingTextEvent,
    FloatingTextValence,
    MessageEvent,
    publish_event,
)
from brileta.game import ranges
from brileta.types import DIRECTIONS, ActorId, Direction, WorldTilePos
from brileta.util import rng

from .actions import AttackAction, AvoidAction, IdleAction, WatchAction
from .behaviors.flee import FleeAction
from .behaviors.wander import WanderAction
from .perception import PerceivedActor, PerceptionComponent
from .utility import (
    ScoredAction,
    UtilityAction,
    UtilityBrain,
    UtilityContext,
    is_target_nearby,
    resolve_flee_from,
)

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.actions.base import GameIntent
    from brileta.game.actions.movement import MoveIntent
    from brileta.game.actors import NPC, Actor, Character

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
    Unknown relationships default to ``default_disposition``.
    """

    def __init__(
        self,
        aggro_radius: int = Combat.DEFAULT_AGGRO_RADIUS,
        default_disposition: int = 0,
        actions: list[UtilityAction] | None = None,
        perception: PerceptionComponent | None = None,
    ) -> None:
        self.actor: Actor | None = None
        # Per-relationship disposition values keyed by ``target.actor_id``.
        # Unknown actors default to self.default_disposition.
        self._dispositions: dict[ActorId, int] = {}
        self.aggro_radius = aggro_radius
        self.default_disposition = default_disposition
        # Perception gates awareness behind range + LOS checks. When None,
        # a default PerceptionComponent is created with radius 12.
        self.perception = perception or PerceptionComponent()

        # Combat awareness: identity of the most recent attacker. Allows the
        # NPC to treat that actor as a threat even outside normal aggro range.
        # Cleared when the attacker dies or is no longer reachable.
        # TODO: An NPC attacked by an *unseen* attacker (e.g., hidden sniper)
        # should panic/flee directionally rather than targeting the exact actor.
        self._last_attacker_id: ActorId | None = None

        # Tracks which action won last tick so we can detect transitions
        # and emit visual indicators (e.g., "!" for switching to attack).
        self._last_action_id: str | None = None

        # Debug: cached results from the last utility evaluation.
        # Read by AI debug live variables for the debug stats overlay.
        self.last_chosen_action: str | None = None
        self.last_scores: list[ScoredAction] = []
        self.last_threat_level: float | None = None
        self.last_target_actor_id: ActorId | None = None

        # Each action owns its own preconditions and considerations
        # internally. base_score is the per-archetype tuning knob: bump it
        # up to make this NPC type favor the action, lower it to suppress.
        # See each action class for its full scoring config.
        brain_actions = actions
        if brain_actions is None:
            brain_actions = [
                AttackAction(base_score=1.0),
                FleeAction(base_score=1.0),
                AvoidAction(base_score=0.7),
                WatchAction(base_score=0.35),
                IdleAction(base_score=0.1),
                WanderAction(base_score=0.18),
                # PatrolAction is defined but not scored here. It will be
                # wired up when guards / soldiers with assigned patrol routes
                # are implemented. See PatrolAction in behaviors/patrol.py.
            ]

        self.brain = UtilityBrain(brain_actions)
        # Some archetypes (e.g., skittish/predator) react to nearby actors
        # regardless of hostility. Those profiles need a non-hostile
        # proximity fallback target so target_proximity can be evaluated.
        self._uses_proximity_targeting = any(
            any(pre is is_target_nearby for pre in action.preconditions)
            or any(
                consideration.input_key == "target_proximity"
                for consideration in action.considerations
            )
            for action in brain_actions
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
        return self._dispositions.get(target.actor_id, self.default_disposition)

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
        from brileta.game.actors.ai.goals import ContinueGoalAction
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
        self.last_target_actor_id = (
            context.target.actor_id if context.target is not None else None
        )
        if not context.actor.health.is_alive():
            return None
        if context.target is not None and not context.target.health.is_alive():
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

        # Detect action transitions and emit visual indicators for
        # combat-relevant switches (attack, flee). Continuing the same
        # goal counts as the same action, so no indicator fires mid-flee.
        current_action_id = (
            action.goal.goal_id
            if isinstance(action, ContinueGoalAction)
            else action.action_id
        )
        if current_action_id != self._last_action_id:
            self._emit_action_transition_indicator(controller, actor, current_action_id)
        self._last_action_id = current_action_id

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

    # Action IDs that get a floating-text indicator on transition.
    # Maps action_id -> (text, color).
    _ACTION_TRANSITION_INDICATORS: ClassVar[dict[str, tuple[str, colors.Color]]] = {
        "attack": ("!", colors.RED),
        "flee": ("!", (255, 255, 150)),
    }

    def _emit_action_transition_indicator(
        self, controller: Controller, actor: NPC, action_id: str
    ) -> None:
        """Emit a floating text indicator when switching to a notable action.

        Only emits for combat-relevant transitions (attack, flee) and only
        when the NPC is within the player's field of view.
        """
        indicator = self._ACTION_TRANSITION_INDICATORS.get(action_id)
        if indicator is None:
            return

        # Only show indicators for NPCs the player can actually see.
        game_map = controller.gw.game_map
        if not game_map.visible[actor.x, actor.y]:
            return

        text, color = indicator
        publish_event(
            FloatingTextEvent(
                text=text,
                target_actor_id=actor.actor_id,
                valence=FloatingTextValence.NEGATIVE,
                duration=1.4,
                color=color,
                world_x=actor.x,
                world_y=actor.y,
            )
        )

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

        for dx, dy in DIRECTIONS:
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

        Three independent queries feed into context construction:
        1. Target selection - who is this NPC focused on? (may be None)
        2. Outgoing threat - how dangerous is the target to me?
        3. Incoming threat - is anything hostile approaching me?

        Args:
            controller: Game controller with access to world state.
            actor: NPC being evaluated.
            force_hostile: When True, override disposition to -100.
        """
        # Shared perception snapshot for this tick. Target selection and
        # incoming-threat scans both consume the same actor list so we only
        # pay range/LOS cost once per NPC decision.
        perceived = self._get_perceived_actors(controller, actor)

        # 1. Target selection and outgoing threat signals.
        target_actor = self._select_target_actor(
            controller,
            actor,
            perceived=perceived,
            allow_proximity_fallback=self._uses_proximity_targeting,
        )
        distance, threat_level, target_proximity, effective_disposition = (
            self._compute_outgoing_threat(
                controller, actor, target_actor, force_hostile
            )
        )

        # 2. Incoming threat: scans all perceived actors for hostility toward
        # this NPC. Independent of target - a neutral NPC can detect an
        # approaching predator even with no outgoing-threat target.
        incoming_threat, threat_source = self._compute_incoming_threat(
            controller, actor, perceived=perceived
        )

        # 3. Tactical decisions that depend on the above.
        health_percent = (
            actor.health.hp / actor.health.max_hp if actor.health.max_hp > 0 else 0.0
        )
        flee_from = resolve_flee_from(target_actor, threat_level, threat_source)
        best_attack_destination = (
            self._select_attack_destination(controller, actor, target_actor)
            if target_actor is not None
            else None
        )
        best_flee_step = (
            self._select_flee_step(controller, actor, flee_from)
            if flee_from is not None
            else None
        )

        return UtilityContext(
            controller=controller,
            actor=actor,
            target=target_actor,
            distance_to_target=distance,
            health_percent=health_percent,
            threat_level=threat_level,
            can_attack=target_actor is not None and distance == 1,
            has_escape_route=best_flee_step is not None,
            best_attack_destination=best_attack_destination,
            best_flee_step=best_flee_step,
            threat_source=threat_source,
            flee_from=flee_from,
            disposition=disposition_to_normalized(effective_disposition),
            target_proximity=target_proximity,
            incoming_threat=incoming_threat,
        )

    def _compute_outgoing_threat(
        self,
        controller: Controller,
        actor: NPC,
        target: Character | None,
        force_hostile: bool,
    ) -> tuple[int, float, float, int]:
        """Compute outgoing threat signals toward a specific target.

        Returns (distance, threat_level, target_proximity, effective_disposition).
        When target is None, returns safe defaults indicating no threat.
        """
        if target is None:
            return 0, 0.0, 0.0, 0

        distance = ranges.calculate_distance(actor.x, actor.y, target.x, target.y)
        effective_disposition = (
            -100 if force_hostile else self.disposition_toward(target)
        )

        target_proximity = 0.0
        if self.aggro_radius > 0:
            target_proximity = max(0.0, 1.0 - distance / self.aggro_radius)

        threat_level = 0.0
        if target.health.is_alive():
            threat_level = self._compute_relationship_threat(
                distance=distance, disposition=effective_disposition
            )

        # Combat awareness: if the target is a known attacker outside normal
        # aggro range, compute an extended threat.
        if (
            threat_level == 0.0
            and self._last_attacker_id == target.actor_id
            and distance < _COMBAT_AWARENESS_RADIUS
        ):
            proximity = 1.0 - (distance / _COMBAT_AWARENESS_RADIUS)
            hostility = max(0.0, min(1.0, -effective_disposition / 50.0))
            threat_level = proximity * hostility

        return distance, threat_level, target_proximity, effective_disposition

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

    def _compute_incoming_threat(
        self,
        controller: Controller,
        actor: NPC,
        *,
        perceived: list[PerceivedActor] | None = None,
    ) -> tuple[float, Character | None]:
        """Compute the strongest incoming danger this NPC perceives.

        Scans ALL perceived actors to find any that are hostile toward this
        NPC (via their AI disposition). Returns the highest threat signal
        found (hostility * perception_strength) and the actor producing it.

        This is the "I can see something hostile approaching me" signal. It
        is separate from threat_level, which measures our own hostility toward
        a specific target. A neutral resident can have zero threat_level
        toward a scorpion while still perceiving high incoming_threat from it.
        """
        if perceived is None:
            perceived = self._get_perceived_actors(controller, actor)
        max_threat = 0.0
        source: Character | None = None

        for p in perceived:
            other_ai = getattr(p.actor, "ai", None)
            if other_ai is None:
                continue

            # Check the other actor's disposition toward this NPC.
            their_disposition = other_ai.disposition_toward(actor)
            if their_disposition >= 0:
                continue  # Neutral or friendly toward us.

            if p.perception_strength <= 0.0:
                continue

            # Hostility signal: maps disposition [-100, 0] to [1.0, 0.0].
            hostility = min(1.0, -their_disposition / 100.0)
            threat = hostility * p.perception_strength
            if threat > max_threat:
                max_threat = threat
                source = p.actor

        return max_threat, source

    def _get_perceived_actors(
        self, controller: Controller, actor: NPC
    ) -> list[PerceivedActor]:
        """Return actors this NPC can currently perceive.

        Uses the spatial index for candidate collection, then filters
        through the PerceptionComponent's range + LOS checks.
        """
        # Query a radius large enough to cover perception awareness.
        radius = self.perception.awareness_radius
        nearby = controller.gw.actor_spatial_index.get_in_radius(
            actor.x, actor.y, radius
        )
        return self.perception.get_perceived_actors(
            actor, controller.gw.game_map, nearby
        )

    def _select_target_actor(
        self,
        controller: Controller,
        actor: NPC,
        *,
        perceived: list[PerceivedActor] | None = None,
        allow_proximity_fallback: bool = False,
    ) -> Character | None:
        """Select the most threatening perceived target for this tick.

        Only actors that pass the perception filter (range + LOS) are
        considered as threat candidates. This means NPCs cannot target
        actors they cannot actually detect.

        Returns None when no threat is perceived and no combat awareness
        exists, unless allow_proximity_fallback is True and at least one
        actor is perceived. Proximity fallback is used by behavior profiles
        that score on raw nearness rather than hostility.
        """
        best_target: Character | None = None
        best_threat = -1.0
        best_distance = float("inf")
        best_disposition = 101

        # Only consider actors that pass perception checks (range + LOS).
        if perceived is None:
            perceived = self._get_perceived_actors(controller, actor)
        for p in perceived:
            other = p.actor
            disposition = self.disposition_toward(other)
            threat = self._compute_relationship_threat(p.distance, disposition)
            if threat <= 0.0:
                continue
            if threat > best_threat or (
                threat == best_threat
                and (
                    p.distance < best_distance
                    or (p.distance == best_distance and disposition < best_disposition)
                )
            ):
                best_target = other
                best_threat = threat
                best_distance = float(p.distance)
                best_disposition = disposition

        if best_target is not None:
            return best_target

        # Combat awareness fallback: if no threat within perception range
        # but the NPC was recently attacked, target the attacker regardless
        # of distance (they know who hit them).
        if self._last_attacker_id is not None:
            from brileta.game.actors.core import Character

            attacker = controller.gw.get_actor_by_id(self._last_attacker_id)
            if (
                attacker is not None
                and isinstance(attacker, Character)
                and attacker.health.is_alive()
            ):
                return attacker
            # Attacker is dead or gone - clear awareness.
            self._last_attacker_id = None

        # Proximity fallback: when an archetype uses proximity-only scoring
        # (e.g., skittish/predator), provide the nearest perceived actor so
        # target_proximity and is_target_nearby can function without requiring
        # a hostile relationship.
        if allow_proximity_fallback and perceived:
            return perceived[0].actor

        return None

    def _select_attack_destination(
        self, controller: Controller, actor: NPC, target: Character
    ) -> WorldTilePos | None:
        """Find the best tile adjacent to the target for attacking from."""
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
        from brileta.util.pathfinding import probe_step

        best_dest: WorldTilePos | None = None
        best_score = float("inf")
        game_map = controller.gw.game_map

        for dx, dy in DIRECTIONS:
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
        candidates = self._collect_flee_candidates(
            controller,
            actor,
            target,
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

    def _collect_flee_candidates(
        self,
        controller: Controller,
        actor: NPC,
        target: Character,
    ) -> list[FleeCandidate]:
        """Enumerate passable adjacent tiles that don't move toward the threat.

        Prefers tiles that increase distance. If none exist (e.g., against a
        wall), includes lateral tiles (same distance) so the NPC can slide
        along obstacles to find an opening.
        """
        from brileta.environment.tile_types import HAZARD_BASE_COST, get_hazard_cost
        from brileta.util.pathfinding import probe_step

        game_map = controller.gw.game_map
        current_distance = ranges.calculate_distance(
            actor.x, actor.y, target.x, target.y
        )
        increasing: list[FleeCandidate] = []
        lateral: list[FleeCandidate] = []

        for dx, dy in DIRECTIONS:
            tx = actor.x + dx
            ty = actor.y + dy
            block_reason = probe_step(
                game_map,
                controller.gw,
                tx,
                ty,
                exclude_actor=actor,
                can_open_doors=actor.can_open_doors,
            )
            if block_reason is not None:
                continue

            distance_after = ranges.calculate_distance(tx, ty, target.x, target.y)
            if distance_after < current_distance:
                continue  # Moving toward threat is never acceptable.

            tile_id = int(game_map.tiles[tx, ty])
            hazard_cost = get_hazard_cost(tile_id)
            actor_at_tile = controller.gw.get_actor_at_location(tx, ty)
            damage_per_turn = getattr(actor_at_tile, "damage_per_turn", 0)
            if actor_at_tile and damage_per_turn > 0:
                fire_cost = HAZARD_BASE_COST + damage_per_turn
                hazard_cost = max(hazard_cost, fire_cost)

            step_cost = 1 if (dx == 0 or dy == 0) else 2
            candidate = FleeCandidate(
                direction=(dx, dy),
                distance_after=distance_after,
                hazard_score=hazard_cost + step_cost,
                step_cost=step_cost,
            )
            if distance_after > current_distance:
                increasing.append(candidate)
            else:
                lateral.append(candidate)

        return increasing or lateral
