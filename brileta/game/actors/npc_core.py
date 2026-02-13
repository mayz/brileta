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
from brileta.game.actors.ai.behaviors.wander import WanderAction
from brileta.game.actors.ai.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    UtilityAction,
    has_escape_route,
    is_target_nearby,
)
from brileta.game.enums import CreatureSize
from brileta.game.items.item_core import ItemType
from brileta.types import WorldTileCoord
from brileta.util import rng
from brileta.util.rng import RNG

if TYPE_CHECKING:
    from brileta.game.game_world import GameWorld

_npc_type_rng = rng.get("npc.types")


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


# Semantic alias for NPC behavior tags.
type NPCTag = str

# Each tag maps to the utility actions it contributes. When an NPCType lists
# multiple tags, actions are composed in order - later tags override earlier
# ones when they share an action_id (e.g. predator's Attack overrides
# combatant's Attack).
TAG_ACTIONS: dict[NPCTag, list[UtilityAction]] = {
    # Default baseline behaviors for all NPCs.
    "base": [IdleAction(0.1), WanderAction(0.18)],
    # Combat-capable behavior package.
    "combatant": [AttackAction(1.0), FleeAction(1.0)],
    # Basic social awareness for intelligent NPCs.
    "sapient": [WatchAction(0.35), AvoidAction(0.7)],
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
    # Skittish package: flee from nearby entities regardless of disposition.
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
                Consideration(
                    "has_escape_route",
                    ResponseCurve(ResponseCurveType.STEP),
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
    role: str | None = None
    display_name: str = ""
    strength_dist: StatDistribution = StatDistribution()
    toughness_dist: StatDistribution = StatDistribution()
    agility_dist: StatDistribution = StatDistribution()
    observation_dist: StatDistribution = StatDistribution()
    intelligence_dist: StatDistribution = StatDistribution()
    demeanor_dist: StatDistribution = StatDistribution()
    weirdness_dist: StatDistribution = StatDistribution()

    def __post_init__(self) -> None:
        # Derive display_name from id if not explicitly provided.
        if not self.display_name:
            self.display_name = self.id.replace("_", " ").title()

    def __hash__(self) -> int:  # pragma: no cover - identity hash
        return id(self)

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
        )
        starting_weapon = (
            self.starting_weapon.create() if self.starting_weapon is not None else None
        )

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
        )
