"""Unit tests for utility input and precondition helpers."""

from __future__ import annotations

import pytest

from brileta.game.actors.ai.utility import UtilityContext, is_any_threat_perceived
from tests.helpers import make_ai_world


def _make_context(
    *,
    threat_level: float,
    incoming_threat: float,
    health_percent: float = 1.0,
    disposition: float = 0.5,
) -> UtilityContext:
    """Build a minimal UtilityContext for helper-function tests."""
    controller, player, npc = make_ai_world(npc_x=3, npc_y=0, disposition=0)
    return UtilityContext(
        controller=controller,
        actor=npc,
        target=player,
        distance_to_target=3,
        health_percent=health_percent,
        threat_level=threat_level,
        can_attack=False,
        has_escape_route=True,
        best_attack_destination=None,
        best_flee_step=(1, 0),
        disposition=disposition,
        incoming_threat=incoming_threat,
    )


@pytest.mark.parametrize(
    ("threat_level", "incoming_threat", "expected"),
    [
        (0.2, 0.7, 0.7),
        (0.9, 0.4, 0.9),
        (0.0, 0.0, 0.0),
    ],
)
def test_max_threat_input_returns_highest_signal(
    threat_level: float,
    incoming_threat: float,
    expected: float,
) -> None:
    """max_threat should expose the stronger of outgoing/incoming danger."""
    context = _make_context(
        threat_level=threat_level,
        incoming_threat=incoming_threat,
    )
    assert context.get_input("max_threat") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("threat_level", "incoming_threat", "expected"),
    [
        (0.0, 0.0, False),
        (0.4, 0.0, True),
        (0.0, 0.4, True),
        (0.3, 0.8, True),
    ],
)
def test_is_any_threat_perceived_checks_both_signals(
    threat_level: float,
    incoming_threat: float,
    expected: bool,
) -> None:
    """is_any_threat_perceived should pass when either threat channel is active."""
    context = _make_context(
        threat_level=threat_level,
        incoming_threat=incoming_threat,
    )
    assert is_any_threat_perceived(context) is expected


def test_sapient_flee_urgency_non_hostile_ignores_health_gate() -> None:
    """Non-hostile panic should react to threat regardless of health."""
    healthy = _make_context(
        threat_level=0.0,
        incoming_threat=0.7,
        health_percent=1.0,
        disposition=0.5,
    )
    injured = _make_context(
        threat_level=0.0,
        incoming_threat=0.7,
        health_percent=0.2,
        disposition=0.5,
    )

    assert healthy.get_input("sapient_flee_urgency") == pytest.approx(0.7)
    assert injured.get_input("sapient_flee_urgency") == pytest.approx(0.7)


def test_sapient_flee_urgency_hostile_is_health_gated() -> None:
    """Hostile retreat urgency should be zero at full health and rise with injury."""
    full_health = _make_context(
        threat_level=0.7,
        incoming_threat=0.0,
        health_percent=1.0,
        disposition=0.125,  # hostile band
    )
    injured = _make_context(
        threat_level=0.7,
        incoming_threat=0.0,
        health_percent=0.4,
        disposition=0.125,  # hostile band
    )

    assert full_health.get_input("sapient_flee_urgency") == pytest.approx(0.0)
    assert injured.get_input("sapient_flee_urgency") == pytest.approx(0.252)
