"""Tests for PlayerStatusView."""

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character, conditions, status_effects
from catley.game.actors.status_effects import EncumberedEffect
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.view.render.graphics import GraphicsContext
from catley.view.views.player_status_view import PlayerStatusView
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any = None


def make_world() -> tuple[DummyController, Character, PlayerStatusView]:
    """Create a test world with a player and PlayerStatusView."""
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.player = actor
    gw.add_actor(actor)
    renderer = MagicMock(spec=GraphicsContext)
    renderer.root_console = MagicMock()
    renderer.tile_dimensions = (8, 16)
    controller = DummyController(gw=gw, graphics=renderer)
    view = PlayerStatusView(cast(Controller, controller), renderer)
    return controller, actor, view


def test_no_conditions_or_effects() -> None:
    """Methods return empty lists when no effects are active."""
    _controller, _actor, view = make_world()
    assert view._get_conditions() == []
    assert view._get_status_effects() == []


def test_condition_grouping_and_colors() -> None:
    """Conditions group by name with 'x2' count when duplicated."""
    _controller, actor, view = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    actor.conditions.add_condition(injury)

    lines = sorted(view._get_conditions())
    assert ("Exhaustion x2", colors.LIGHT_BLUE) in lines
    assert (injury.name, colors.RED) in lines


def test_status_effects_with_colors() -> None:
    """Status effects return name with appropriate color (no duration)."""
    _controller, actor, view = make_world()
    effect = status_effects.OffBalanceEffect()
    effect.duration = 1
    actor.status_effects.apply_status_effect(effect)

    lines = view._get_status_effects()
    # PlayerStatusView returns just the name (no duration), with LIGHT_GREY color
    assert lines == [("Off Balance", colors.LIGHT_GREY)]


def test_encumbered_severity_colors() -> None:
    """EncumberedEffect uses RED/ORANGE/YELLOW based on slots over capacity."""
    _controller, actor, view = make_world()

    # Apply encumbered effect
    effect = EncumberedEffect()
    actor.status_effects.apply_status_effect(effect)

    # Test severe encumbrance (3+ slots over) -> RED
    actor.inventory.get_slots_over_capacity = MagicMock(return_value=3)
    lines = view._get_status_effects()
    assert lines == [("Encumbered", colors.RED)]

    # Test moderate encumbrance (1-2 slots over) -> ORANGE
    actor.inventory.get_slots_over_capacity = MagicMock(return_value=1)
    lines = view._get_status_effects()
    assert lines == [("Encumbered", colors.ORANGE)]

    # Test mild encumbrance (0 slots over but still encumbered) -> YELLOW
    actor.inventory.get_slots_over_capacity = MagicMock(return_value=0)
    lines = view._get_status_effects()
    assert lines == [("Encumbered", colors.YELLOW)]
