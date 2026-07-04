"""Tests for OCEAN personality: sampling, consideration curves, and behavior."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from brileta.game.actors.ai.actions import AttackAction, IdleAction
from brileta.game.actors.ai.behaviors.flee import FleeAction
from brileta.game.actors.ai.behaviors.routine import (
    errand_chance_multiplier,
    schedule_offset,
)
from brileta.game.actors.ai.personality import (
    TRAIT_MAX,
    TRAIT_MIN,
    PersonalityComponent,
    normalize_trait,
)
from brileta.game.actors.ai.utility import (
    Consideration,
    ResponseCurve,
    ResponseCurveType,
    UtilityContext,
)
from brileta.game.actors.npc_core import personality_trait
from brileta.game.actors.npc_types import (
    BRIGAND_TYPE,
    DOG_TYPE,
    GIANT_SCORPION_TYPE,
    RESIDENT_TYPE,
    TROG_TYPE,
)
from brileta.game.game_world import GameWorld
from brileta.util import rng
from tests.helpers import DummyGameWorld, make_ai_world

# Normalized trait input keys the utility context exposes.
_TRAIT_KEYS = {
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
}

# The skittish flee's considerations, mirrored here so the dog test scores the
# same curve DOG_TYPE uses without reaching into the tag table.
_DOG_SKITTISH_CONSIDERATIONS = [
    Consideration(
        "target_proximity", ResponseCurve(ResponseCurveType.LINEAR), weight=2.0
    ),
    Consideration("neuroticism", ResponseCurve(ResponseCurveType.CENTERED, gain=1.5)),
]


def _flee_context(*, neuroticism: float, health_percent: float = 0.9) -> UtilityContext:
    """Build a combat context that differs only in the neuroticism input."""
    controller, player, npc = make_ai_world(npc_x=2, npc_y=0, disposition=-75)
    return UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=2,
        health_percent=health_percent,
        threat_level=0.8,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
        disposition=0.125,  # -75 normalized
        neuroticism=neuroticism,
    )


# ---------------------------------------------------------------------------
# Trait sampling
# ---------------------------------------------------------------------------


def test_personality_defaults_to_average() -> None:
    """An unspecified personality is average (5) in every trait."""
    p = PersonalityComponent()
    assert (
        p.openness
        == p.conscientiousness
        == p.extraversion
        == p.agreeableness
        == p.neuroticism
        == 5
    )


def test_personality_trait_distribution_clamps_to_scale() -> None:
    """personality_trait samples stay within the 0-10 trait scale."""
    stream = rng.get("test.personality")
    dist = personality_trait(mean=9, spread=3.0)
    samples = [dist.sample(stream) for _ in range(200)]
    assert all(TRAIT_MIN <= s <= TRAIT_MAX for s in samples)
    # A mean-9 distribution should skew high.
    assert sum(samples) / len(samples) > 6.5


def test_create_samples_personality_within_type_range() -> None:
    """Spawning from a type produces a personality in range, sampled at spawn."""
    gw = DummyGameWorld(width=20, height=20)
    npc = BRIGAND_TYPE.create(5, 5, "Brigand", game_world=cast(GameWorld, gw))
    for trait in (
        npc.personality.openness,
        npc.personality.conscientiousness,
        npc.personality.extraversion,
        npc.personality.agreeableness,
        npc.personality.neuroticism,
    ):
        assert TRAIT_MIN <= trait <= TRAIT_MAX


def test_personality_sampling_is_reproducible() -> None:
    """Same seed -> same sampled personality (seeded RNG stream)."""
    rng.init(1234)
    gw = DummyGameWorld(width=20, height=20)
    first = DOG_TYPE.create(1, 1, "Dog", game_world=cast(GameWorld, gw)).personality

    rng.init(1234)
    gw2 = DummyGameWorld(width=20, height=20)
    second = DOG_TYPE.create(1, 1, "Dog", game_world=cast(GameWorld, gw2)).personality

    assert first == second


# ---------------------------------------------------------------------------
# Neutrality at the average (no accidental rebalance)
# ---------------------------------------------------------------------------


def test_centered_curve_is_neutral_at_midpoint() -> None:
    """CENTERED evaluates to exactly 1.0 at the midpoint, and moves either way."""
    for gain in (0.8, -0.8, 1.5):
        curve = ResponseCurve(ResponseCurveType.CENTERED, gain=gain)
        assert curve.evaluate(0.5) == pytest.approx(1.0)

    rising = ResponseCurve(ResponseCurveType.CENTERED, gain=1.0)
    assert rising.evaluate(0.8) > 1.0
    assert rising.evaluate(0.2) < 1.0
    # A steep gain never drives the multiplier negative.
    assert ResponseCurve(ResponseCurveType.CENTERED, gain=3.0).evaluate(0.0) >= 0.0


def test_average_personality_leaves_every_action_score_unchanged() -> None:
    """An all-average NPC scores identically to pre-Phase-5 on every action.

    Every personality consideration must evaluate to exactly 1.0 for an
    average (trait 5 -> 0.5) NPC, so it contributes a x1.0 factor and the
    action's score is what it was before personality was wired in. This guards
    against reintroducing the uneven dampening a non-centered curve caused.
    """
    average_ctx = _flee_context(neuroticism=0.5)  # all traits default to 0.5
    assert all(getattr(average_ctx, key) == 0.5 for key in _TRAIT_KEYS), (
        "context should be all-average"
    )

    seen: set[str] = set()
    for npc_type in (
        BRIGAND_TYPE,
        DOG_TYPE,
        RESIDENT_TYPE,
        GIANT_SCORPION_TYPE,
        TROG_TYPE,
    ):
        npc = npc_type.create(0, 0, "Test")
        for action in npc.ai.brain.actions:
            for consideration in action.considerations:
                if consideration.input_key in _TRAIT_KEYS:
                    seen.add(consideration.input_key)
                    assert consideration.evaluate(average_ctx) == pytest.approx(1.0)

    # Confirm the sweep actually exercised the traits wired as considerations.
    assert {"neuroticism", "agreeableness", "extraversion", "openness"} <= seen


# ---------------------------------------------------------------------------
# Consideration curves
# ---------------------------------------------------------------------------


def test_neuroticism_raises_flee_score() -> None:
    """A high-neuroticism NPC scores flee higher than a calm one, all else equal."""
    calm = FleeAction(base_score=1.0).score(_flee_context(neuroticism=0.2))
    anxious = FleeAction(base_score=1.0).score(_flee_context(neuroticism=0.8))
    assert anxious > calm


def test_low_neuroticism_holds_ground_longer_before_fleeing() -> None:
    """At moderate health a calm NPC still prefers attack while an anxious one flees."""
    attack = AttackAction(base_score=1.0)
    flee = FleeAction(base_score=1.0)

    # Health sits in the band where the flight threshold separates the two:
    # the anxious NPC flees here, the calm one still fights.
    calm_ctx = _flee_context(neuroticism=0.1, health_percent=0.35)
    anxious_ctx = _flee_context(neuroticism=0.9, health_percent=0.35)

    # The anxious NPC's flee outscores its attack; the calm NPC's does not.
    assert flee.score(anxious_ctx) > attack.score(anxious_ctx)
    assert flee.score(calm_ctx) < attack.score(calm_ctx)


def test_agreeableness_lowers_attack_score() -> None:
    """A cooperative NPC is slower to attack than a confrontational one."""
    hostile_ctx = _flee_context(neuroticism=0.5)
    agreeable_ctx = replace(hostile_ctx, agreeableness=0.9)
    confrontational_ctx = replace(hostile_ctx, agreeableness=0.1)

    attack = AttackAction(base_score=1.0)
    assert attack.score(confrontational_ctx) > attack.score(agreeable_ctx)


def test_extraversion_lowers_idle_score() -> None:
    """An extravert idles less than an introvert (drives the ambient move mix)."""
    base = _flee_context(neuroticism=0.5)
    # Idle needs no threat; neutralize it so the extraversion term is isolated.
    calm = replace(base, threat_level=0.0, disposition=0.5)
    introvert = replace(calm, extraversion=0.1)
    extravert = replace(calm, extraversion=0.9)

    idle = IdleAction(base_score=0.1)
    assert idle.score(introvert) > idle.score(extravert)


def test_conscientiousness_scales_persistence() -> None:
    """Persistence scale rises with conscientiousness; average maps to 1.0."""
    base = _flee_context(neuroticism=0.5)
    erratic = replace(base, conscientiousness=0.0)
    average = replace(base, conscientiousness=0.5)
    diligent = replace(base, conscientiousness=1.0)

    assert erratic.conscientiousness_persistence_scale() == 0.5
    assert average.conscientiousness_persistence_scale() == 1.0
    assert diligent.conscientiousness_persistence_scale() == 1.5


# ---------------------------------------------------------------------------
# Routine timing
# ---------------------------------------------------------------------------


def test_conscientious_npc_leaves_earlier() -> None:
    """Higher conscientiousness biases the workday offset earlier (more negative)."""
    stream = rng.get("test.routine")
    diligent = PersonalityComponent(conscientiousness=10)
    lax = PersonalityComponent(conscientiousness=0)
    # Average many draws so the shared residual jitter cancels out.
    diligent_mean = sum(schedule_offset(diligent, stream) for _ in range(200)) / 200
    lax_mean = sum(schedule_offset(lax, stream) for _ in range(200)) / 200
    assert diligent_mean < lax_mean


def test_extravert_runs_more_errands_than_introvert() -> None:
    """Errand chance multiplier rises with extraversion, falls with diligence."""
    extravert = errand_chance_multiplier(
        PersonalityComponent(extraversion=10, conscientiousness=2)
    )
    homebody = errand_chance_multiplier(
        PersonalityComponent(extraversion=2, conscientiousness=10)
    )
    assert extravert > homebody


# ---------------------------------------------------------------------------
# End-to-end through AIComponent._build_context
# ---------------------------------------------------------------------------


def test_two_npcs_same_threat_different_neuroticism_score_flee_differently() -> None:
    """Two identical NPCs differing only in neuroticism produce different flee scores.

    Exercises the full path: personality on the NPC -> _build_context ->
    normalized trait input -> FleeAction scoring.
    """
    # Damage the NPC so the flee health curve is non-zero and neuroticism can
    # move the score (an undamaged NPC's flee is zero regardless of trait).
    controller, _player, npc = make_ai_world(
        npc_x=2, npc_y=0, disposition=-75, npc_hp_damage=4
    )
    flee = FleeAction(base_score=1.0)

    npc.personality = PersonalityComponent(neuroticism=1)
    calm_score = flee.score(npc.ai._build_context(controller, npc))

    npc.personality = PersonalityComponent(neuroticism=9)
    anxious_score = flee.score(npc.ai._build_context(controller, npc))

    assert anxious_score > calm_score


def test_bold_dog_stays_while_skittish_dog_flees() -> None:
    """A low-neuroticism dog suppresses flee; a high-neuroticism one keeps it high."""
    controller, player, npc = make_ai_world(npc_x=2, npc_y=0, disposition=20)
    # Use the dog's skittish flee (proximity + neuroticism driven).
    skittish_flee = FleeAction(
        base_score=1.2,
        preconditions=[],
        considerations=_DOG_SKITTISH_CONSIDERATIONS,
    )
    npc.personality = PersonalityComponent(neuroticism=1)
    bold = skittish_flee.score(_dog_proximity_context(controller, player, npc))
    npc.personality = PersonalityComponent(neuroticism=9)
    skittish = skittish_flee.score(_dog_proximity_context(controller, player, npc))
    assert skittish > bold


def _dog_proximity_context(controller, player, npc) -> UtilityContext:
    """Proximity-only context (no hostility) like a dog sensing a passerby."""
    return UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=2,
        health_percent=1.0,
        threat_level=0.0,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
        disposition=0.6,
        target_proximity=0.7,
        neuroticism=normalize_trait(npc.personality.neuroticism),
    )
