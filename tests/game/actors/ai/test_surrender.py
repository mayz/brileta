"""Tests for NUBS 7 surrender: SurrenderAction scoring, goal, and composition.

Validates:
- Surrender scores only when cornered (no escape route) with a threat present.
- Low health and fear raise the score; a healthy NPC barely surrenders.
- Agreeableness modulates it: a fanatic (low agreeableness) surrenders less
  than an agreeable NPC in the same corner.
- The action adopts a SurrenderGoal that cowers (no intent) and stays sticky.
- accept() pacifies the NPC (large disposition swing, completes); refuse()
  drops it back into a fight (fails, stays hostile).
- Tag composition: sapient NPCs get surrender, non-sapient creatures do not.
"""

from __future__ import annotations

from brileta.game.actors.ai.actions import AttackAction
from brileta.game.actors.ai.behaviors.surrender import (
    SurrenderAction,
    SurrenderGoal,
)
from brileta.game.actors.ai.goals import GoalState
from brileta.game.actors.ai.utility import UtilityContext
from brileta.game.actors.npc_core import compose_actions
from brileta.game.actors.npc_types import (
    BRIGAND_TYPE,
    DOG_TYPE,
    GIANT_SCORPION_TYPE,
    RESIDENT_TYPE,
    TROG_TYPE,
)
from tests.helpers import make_ai_world


def _surrender_context(
    *,
    health_percent: float = 0.15,
    has_escape_route: bool = False,
    threat_level: float = 0.8,
    agreeableness: float = 0.5,
) -> UtilityContext:
    """Build a UtilityContext wired for surrender scoring."""
    controller, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=-75)
    return UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=1,
        health_percent=health_percent,
        threat_level=threat_level,
        can_attack=True,
        has_escape_route=has_escape_route,
        best_attack_destination=None,
        best_flee_step=None,
        disposition=(-75 + 100) / 200.0,
        agreeableness=agreeableness,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_surrender_scores_when_cornered_hurt_and_afraid() -> None:
    assert SurrenderAction().score(_surrender_context()) > 0.0


def test_surrender_scores_zero_with_an_escape_route() -> None:
    # If flee is possible, surrender is off the table (flee handles it).
    assert SurrenderAction().score(_surrender_context(has_escape_route=True)) == 0.0


def test_surrender_scores_zero_with_no_threat() -> None:
    context = _surrender_context(threat_level=0.0)
    assert SurrenderAction().score(context) == 0.0


def test_healthy_npc_barely_surrenders() -> None:
    hurt = SurrenderAction().score(_surrender_context(health_percent=0.1))
    healthy = SurrenderAction().score(_surrender_context(health_percent=0.9))
    assert hurt > healthy


def test_fanatic_surrenders_less_than_agreeable() -> None:
    fanatic = SurrenderAction().score(_surrender_context(agreeableness=0.0))
    agreeable = SurrenderAction().score(_surrender_context(agreeableness=1.0))
    assert agreeable > fanatic


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------


def test_action_adopts_surrender_goal_and_cowers() -> None:
    context = _surrender_context()
    npc = context.actor
    intent = SurrenderAction().get_intent_with_goal(context, npc)
    assert isinstance(npc.current_goal, SurrenderGoal)
    assert intent is None  # Cowering: no action this tick.


def test_accept_pacifies_and_completes() -> None:
    _, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=-75)
    goal = SurrenderGoal()
    npc.current_goal = goal

    goal.accept(npc, player)

    assert goal.state is GoalState.COMPLETED
    assert not npc.ai.is_hostile_toward(player)  # Disposition swung positive.
    assert npc.indicator is None


def test_refuse_fails_and_keeps_hostility() -> None:
    _, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=-75)
    goal = SurrenderGoal()
    npc.current_goal = goal

    goal.refuse(npc, player)

    assert goal.state is GoalState.FAILED
    assert npc.ai.is_hostile_toward(player)  # Still hostile; fights on.
    assert npc.ai.was_surrender_refused_by(player)


def test_refused_surrender_is_suppressed_toward_that_target() -> None:
    """After a refusal, Surrender scores 0 toward the refuser so Attack wins.

    Regression for the load-bearing bug: no scoring input changes on a refusal
    (the NPC is still cornered, hurt, and afraid), so without this suppression
    SurrenderAction would out-score Attack and re-adopt a fresh SurrenderGoal on
    the very next tick - the white flag comes right back and "fights on" never
    happens. Ticks the decision layer past the refusal to prove it does not.
    """
    context = _surrender_context()
    npc = context.actor
    player = context.target
    assert player is not None
    # Give the NPC a two-action brain so selection is a clean Surrender-vs-Attack
    # decision (the default make_ai_world brain has no Surrender action).
    npc.ai.brain.actions = [AttackAction(1.0), SurrenderAction(1.4)]

    # Before refusal: Surrender wins the NPC's action selection (it out-scores
    # attacking while cornered, hurt, and afraid).
    action, _ = npc.ai.brain.select_action(context)
    assert action is not None and action.action_id == "surrender"

    # The player refuses the surrender.
    npc.ai.record_surrender_refused(player)

    # Next tick, same cornered/hurt/afraid context: Surrender is suppressed
    # (scores 0) and Attack takes over - the NPC actually fights on.
    assert SurrenderAction().score(context) == 0.0
    action, _ = npc.ai.brain.select_action(context)
    assert action is not None and action.action_id == "attack"


def test_surrender_refusal_is_per_target() -> None:
    """A refusal from one actor doesn't suppress surrender toward another."""
    context = _surrender_context()
    npc = context.actor
    other = make_ai_world(npc_x=9, npc_y=9)[2]  # a distinct, unrelated NPC

    npc.ai.record_surrender_refused(other)

    # The refuser was `other`, but the threat in this context is the player, so
    # surrender is still on the table.
    assert SurrenderAction().score(context) > 0.0


def test_surrender_refusal_persists_while_threat_present() -> None:
    """A refusal holds through the rest of the fight, not just one tick.

    While the refuser is still a perceived threat, the flag must stay set so
    SurrenderAction keeps being suppressed and the NPC fights on.
    """
    controller, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=-75)
    npc.ai.record_surrender_refused(player)

    # Hostile and adjacent: the player is a live threat, so the refusal holds.
    npc.ai.get_action(controller, npc)

    assert npc.ai.was_surrender_refused_by(player)


def test_surrender_refusal_clears_when_threat_gone() -> None:
    """Once no threat is perceived, the engagement is over and refusals reset.

    This scopes the flag to the fight (not "never again forever"): a brigand
    cornered again much later can beg again. Safe - with no threat, surrender's
    own precondition already fails, so clearing can't re-trigger the loop.
    """
    controller, player, npc = make_ai_world(npc_x=1, npc_y=0, disposition=0)
    npc.ai.record_surrender_refused(player)

    # Neutral disposition, no incoming hostility: the NPC perceives no threat.
    npc.ai.get_action(controller, npc)

    assert not npc.ai.was_surrender_refused_by(player)


# ---------------------------------------------------------------------------
# Tag composition
# ---------------------------------------------------------------------------


def test_sapient_npcs_get_surrender_creatures_do_not() -> None:
    for sapient in (BRIGAND_TYPE, RESIDENT_TYPE):
        ids = {a.action_id for a in compose_actions(sapient.tags)}
        assert "surrender" in ids

    for creature in (TROG_TYPE, GIANT_SCORPION_TYPE, DOG_TYPE):
        ids = {a.action_id for a in compose_actions(creature.tags)}
        assert "surrender" not in ids
