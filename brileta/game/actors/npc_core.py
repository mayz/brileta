"""NPC type templates and tag-driven AI composition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import colors
from brileta.config import DEFAULT_ACTOR_SPEED
from brileta.game.actors import NPC
from brileta.game.actors.ai import AIComponent
from brileta.game.actors.ai.actions import (
    AttackAction,
    AvoidAction,
    IdleAction,
    WatchAction,
)
from brileta.game.actors.ai.behaviors.flee import FleeAction
from brileta.game.actors.ai.behaviors.request_help import (
    RequestHelpAction,
    TradeAction,
)
from brileta.game.actors.ai.behaviors.routine import RoutineAction
from brileta.game.actors.ai.behaviors.surrender import SurrenderAction
from brileta.game.actors.ai.behaviors.wander import WanderAction
from brileta.game.actors.ai.perception import PerceptionComponent
from brileta.game.actors.ai.personality import (
    TRAIT_AVERAGE,
    TRAIT_MAX,
    TRAIT_MIN,
    PersonalityComponent,
)
from brileta.game.actors.ai.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    has_escape_route,
    is_any_threat_perceived,
    is_target_nearby,
)
from brileta.game.actors.identity import Gender, NPCIdentity, identity_for_gender
from brileta.game.enums import CreatureSize
from brileta.game.items.item_core import ItemType
from brileta.sprites.characters import (
    FEM_PRESENTATION,
    MASC_PRESENTATION,
    CharacterPresentationProfile,
)
from brileta.types import WorldTileCoord
from brileta.util import rng
from brileta.util.rng import RNG

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld
    from brileta.sprites.quadrupeds import QuadrupedPreset

_npc_type_rng = rng.get("npc.types")
# Personality sampling draws from its own stream so adding it does not shift the
# stat stream's draw sequence, keeping existing worlds' stats reproducible.
_npc_personality_rng = rng.get("npc.personality")
_npc_identity_rng = rng.get("npc.identity")


@dataclass(frozen=True, slots=True)
class StatDistribution:
    """Normal distribution parameters for generating a stat value.

    Defaults to the ability score range [-5, 5]. Override min_val/max_val
    for other stat scales (e.g. personality traits use 0-10).
    """

    mean: int = 0
    std_dev: float = 1.0
    min_val: int = -5
    max_val: int = 5

    def sample(self, random_stream: RNG) -> int:
        """Sample a stat value from a Gaussian distribution, clamped to range."""
        sampled = round(random_stream.gauss(self.mean, self.std_dev))
        return max(self.min_val, min(self.max_val, int(sampled)))


def personality_trait(
    mean: int = TRAIT_AVERAGE, spread: float = 1.5
) -> StatDistribution:
    """Build a 0-10 personality-trait distribution centered on ``mean``.

    Thin wrapper over StatDistribution with the trait scale baked in, so NPC
    type definitions read as ``personality_trait(mean=8, spread=1.5)`` rather
    than repeating the 0-10 bounds each time.
    """
    return StatDistribution(
        mean=mean, std_dev=spread, min_val=TRAIT_MIN, max_val=TRAIT_MAX
    )


# Shared default personality distribution (human-average, moderate spread) used
# for NPCType trait fields a type does not override. Safe to share because
# StatDistribution is frozen/immutable.
_DEFAULT_TRAIT_DIST = personality_trait()


# Semantic alias for NPC behavior tags.
type NPCTag = str
type GenderWeights = tuple[tuple[Gender, float], ...]

# Each tag maps to the utility actions it contributes. When an NPCType lists
# multiple tags, actions are composed in order - later tags override earlier
# ones when they share an action_id (e.g. predator's Attack overrides
# combatant's Attack).
TAG_ACTIONS: dict[NPCTag, list[UtilityAction]] = {
    # Default baseline behaviors for all NPCs.
    "base": [IdleAction(0.1), WanderAction(0.18)],
    # Combat-capable behavior package.
    "combatant": [AttackAction(1.0), FleeAction(1.0)],
    # Social awareness for intelligent NPCs. Overrides the combatant flee
    # with a version that also responds to incoming threats (hostile actors
    # approaching), not just outgoing combat danger. Also adds Surrender
    # (NUBS 7): a cornered, hurt, frightened sapient yields instead of dying.
    "sapient": [
        WatchAction(0.35),
        AvoidAction(0.7),
        SurrenderAction(1.4),
        # Replaces combatant flee (same action_id="flee"). Triggers on
        # either outgoing threat (I'm hostile and hurt) or incoming threat
        # (something hostile is approaching me). Uses sapient_flee_urgency
        # to preserve panic fleeing for non-hostile NPCs while keeping
        # hostile combatants from fleeing at full health.
        FleeAction(
            base_score=1.0,
            preconditions=[is_any_threat_perceived, has_escape_route],
            considerations=[
                Consideration(
                    "sapient_flee_urgency",
                    ResponseCurve(ResponseCurveType.LINEAR),
                ),
                # Neuroticism sets the flight threshold: anxious sapients panic
                # and run sooner, steady ones hold their ground longer. Centered
                # on 1.0 so an average-neuroticism sapient flees unchanged.
                Consideration(
                    "neuroticism",
                    ResponseCurve(ResponseCurveType.CENTERED, gain=0.8),
                ),
            ],
        ),
    ],
    # Predator package: attack by proximity instead of hostility.
    "predator": [
        AttackAction(
            base_score=0.8,
            preconditions=[is_target_nearby],
            considerations=[
                Consideration(
                    "target_proximity",
                    ResponseCurve(ResponseCurveType.LINEAR),
                ),
                Consideration(
                    "health_percent",
                    ResponseCurve(ResponseCurveType.LINEAR),
                ),
            ],
        ),
    ],
    # Social package: help-seeking and (Phase 7 stub) trade for settlement
    # NPCs. RequestHelp lets an NPC with an urgent unmet need approach a nearby
    # helper and ask; Trade is registered but scored 0 until the conversation UI.
    "social": [RequestHelpAction(0.9), TradeAction(0.0)],
    # Daily-routine package: home/workplace schedule for settlement dwellers.
    # Outscores wander so residents pursue their routine by default; shares
    # wander's no-threat preconditions so combat/flee always win over it.
    "routine": [RoutineAction(0.4)],
    # Skittish package: flee from nearby entities, with the flight threshold
    # set by personality. The tag still supplies the flee-from-proximity
    # mechanism, but Neuroticism now drives *whether* an individual actually
    # bolts (Phase 5 reinterpretation of the old stand-in tag): a nervous animal
    # runs from anyone close, a bold one lets its wander/idle win and stays near.
    # A future Extraversion-driven approach action (Phase 6 social) would let
    # gregarious animals close the distance instead; for now high Extraversion
    # simply reads as a calmer, stay-put animal via reduced flee competition.
    "skittish": [
        FleeAction(
            base_score=1.2,
            preconditions=[is_target_nearby, has_escape_route],
            considerations=[
                Consideration(
                    "target_proximity",
                    ResponseCurve(ResponseCurveType.LINEAR),
                    weight=2.0,
                ),
                # Centered on 1.0 with a strong slope so a skittish (high
                # neuroticism) animal bolts and a bold one stays, while an
                # average dog flees exactly as the plain proximity rule did.
                Consideration(
                    "neuroticism",
                    ResponseCurve(ResponseCurveType.CENTERED, gain=1.5),
                ),
            ],
        ),
    ],
}


def compose_actions(tags: Sequence[NPCTag]) -> list[UtilityAction]:
    """Compose actions for ordered tags, allowing later tags to override."""
    action_by_id: dict[str, UtilityAction] = {}

    for tag in tags:
        tag_actions = TAG_ACTIONS.get(tag)
        if tag_actions is None:
            msg = f"Unknown NPC tag: {tag}"
            raise ValueError(msg)

        for action in tag_actions:
            # Later tags intentionally override earlier action definitions.
            action_by_id[action.action_id] = action

    return list(action_by_id.values())


@dataclass(eq=False)
class NPCType:
    """Defines an NPC archetype and creates configured NPC instances."""

    id: str
    tags: tuple[NPCTag, ...]
    glyph: str
    color: colors.Color
    creature_size: CreatureSize
    default_disposition: int = 0
    can_open_doors: bool = True
    starting_weapon: ItemType | None = None
    speed: int = DEFAULT_ACTOR_SPEED
    # Omnidirectional detection range in tiles (Chebyshev distance).
    # Controls how far the NPC can perceive other actors via range + LOS.
    awareness_radius: int = 12
    role: str | None = None
    display_name: str = ""
    # Opt-in to procedural quadruped sprites: set to a species preset (e.g.
    # DOG_PRESET) and the controller generates and assigns a critter
    # sprite pose set. Left None, the NPC renders as its text glyph. This is one
    # of the only two places dog-specific knowledge lives (the other is the
    # preset itself); later species opt in by setting this, no controller change.
    critter_preset: QuadrupedPreset | None = None
    # Optional gender distribution for NPCs that need dialogue pronouns and
    # human visual presentation. Empty means this archetype does not carry
    # human identity data.
    identity_weights: GenderWeights = ()
    strength_dist: StatDistribution = StatDistribution()
    toughness_dist: StatDistribution = StatDistribution()
    agility_dist: StatDistribution = StatDistribution()
    observation_dist: StatDistribution = StatDistribution()
    intelligence_dist: StatDistribution = StatDistribution()
    demeanor_dist: StatDistribution = StatDistribution()
    weirdness_dist: StatDistribution = StatDistribution()
    # OCEAN personality distributions, sampled at spawn (Phase 5). Each defaults
    # to the human average (mean 5) with a moderate spread, so a type that does
    # not care about a trait produces middling, varied individuals. The shared
    # frozen default instance is safe because StatDistribution is immutable.
    openness_dist: StatDistribution = _DEFAULT_TRAIT_DIST
    conscientiousness_dist: StatDistribution = _DEFAULT_TRAIT_DIST
    extraversion_dist: StatDistribution = _DEFAULT_TRAIT_DIST
    agreeableness_dist: StatDistribution = _DEFAULT_TRAIT_DIST
    neuroticism_dist: StatDistribution = _DEFAULT_TRAIT_DIST

    def __post_init__(self) -> None:
        # Derive display_name from id if not explicitly provided.
        if not self.display_name:
            self.display_name = self.id.replace("_", " ").title()

    def __hash__(self) -> int:  # pragma: no cover - identity hash
        return id(self)

    def _sample_identity(self) -> NPCIdentity | None:
        """Sample optional identity using the dedicated identity RNG stream."""
        if not self.identity_weights:
            return None

        genders = tuple(gender for gender, _weight in self.identity_weights)
        weights = tuple(weight for _gender, weight in self.identity_weights)
        gender = _npc_identity_rng.choices(genders, weights=weights, k=1)[0]
        return identity_for_gender(gender)

    def _presentation_for_identity(
        self, identity: NPCIdentity | None
    ) -> CharacterPresentationProfile | None:
        """Return the default visual presentation profile for identity."""
        if identity is None:
            return None
        if identity.gender is Gender.FEMALE:
            return FEM_PRESENTATION
        return MASC_PRESENTATION

    def create(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
        name: str,
        game_world: GameWorld | None = None,
    ) -> NPC:
        """Create a concrete NPC instance from this archetype."""
        npc_ai = AIComponent(
            actions=compose_actions(self.tags),
            default_disposition=self.default_disposition,
            perception=PerceptionComponent(awareness_radius=self.awareness_radius),
        )
        starting_weapon = (
            self.starting_weapon.create() if self.starting_weapon is not None else None
        )
        # Sample personality from the type's per-trait distributions using the
        # dedicated seeded stream, so worlds stay reproducible.
        personality = PersonalityComponent(
            openness=self.openness_dist.sample(_npc_personality_rng),
            conscientiousness=self.conscientiousness_dist.sample(_npc_personality_rng),
            extraversion=self.extraversion_dist.sample(_npc_personality_rng),
            agreeableness=self.agreeableness_dist.sample(_npc_personality_rng),
            neuroticism=self.neuroticism_dist.sample(_npc_personality_rng),
        )
        identity = self._sample_identity()

        return NPC(
            x=x,
            y=y,
            ch=self.glyph,
            color=self.color,
            name=name,
            game_world=game_world,
            ai=npc_ai,
            strength=self.strength_dist.sample(_npc_type_rng),
            toughness=self.toughness_dist.sample(_npc_type_rng),
            agility=self.agility_dist.sample(_npc_type_rng),
            observation=self.observation_dist.sample(_npc_type_rng),
            intelligence=self.intelligence_dist.sample(_npc_type_rng),
            demeanor=self.demeanor_dist.sample(_npc_type_rng),
            weirdness=self.weirdness_dist.sample(_npc_type_rng),
            starting_weapon=starting_weapon,
            speed=self.speed,
            creature_size=self.creature_size,
            can_open_doors=self.can_open_doors,
            personality=personality,
            identity=identity,
            character_presentation=self._presentation_for_identity(identity),
            critter_preset=self.critter_preset,
        )
