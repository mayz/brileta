from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character, conditions, status_effects
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.types import InterpolationAlpha
from catley.view.render.graphics import GraphicsContext
from catley.view.views.status_view import StatusView
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld
    graphics: Any = None


def make_world() -> tuple[DummyController, Character, StatusView]:
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.player = actor
    gw.add_actor(actor)
    renderer = MagicMock(spec=GraphicsContext)
    renderer.root_console = MagicMock()
    renderer.tile_dimensions = (8, 16)
    controller = DummyController(gw=gw, graphics=renderer)
    view = StatusView(cast(Controller, controller))
    return controller, actor, view


def test_no_conditions_or_effects() -> None:
    _controller, actor, view = make_world()
    assert view._get_condition_lines(actor) == []
    assert view._get_status_effect_lines(actor) == []


def test_condition_grouping_and_colors() -> None:
    _controller, actor, view = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    actor.conditions.add_condition(injury)

    lines = sorted(view._get_condition_lines(actor))
    assert ("Exhaustion x2", colors.LIGHT_BLUE) in lines
    assert (injury.name, colors.RED) in lines


def test_status_effect_duration_format() -> None:
    _controller, actor, view = make_world()
    effect = status_effects.OffBalanceEffect()
    effect.duration = 1
    actor.status_effects.apply_status_effect(effect)

    lines = view._get_status_effect_lines(actor)
    # Returns (text, color) tuples; OffBalanceEffect uses LIGHT_GREY
    assert lines == [("Off Balance (1 turn)", colors.LIGHT_GREY)]


def test_displayed_counts_initialized() -> None:
    """StatusView should initialize display count tracking variables."""
    _controller, _actor, view = make_world()
    assert view._num_status_effects_displayed == 0
    assert view._num_conditions_displayed == 0


def test_status_effects_displayed_before_conditions() -> None:
    """Status effects should appear before conditions in the display order."""
    _controller, actor, view = make_world()

    # Add a status effect
    effect = status_effects.OffBalanceEffect()
    effect.duration = 2
    actor.status_effects.apply_status_effect(effect)

    # Add a condition
    injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    actor.conditions.add_condition(injury)

    # Get the display lists (both return (text, color) tuples)
    status_effect_lines = view._get_status_effect_lines(actor)
    condition_lines = view._get_condition_lines(actor)

    # Status effects come first in the combined display
    assert len(status_effect_lines) == 1
    assert "Off Balance" in status_effect_lines[0][0]  # text is first element of tuple
    assert len(condition_lines) == 1
    assert condition_lines[0][0] == "Sprained Ankle"


def test_vertical_layout_tracks_displayed_counts() -> None:
    """After drawing, the view should track how many items of each type are shown."""
    _controller, actor, view = make_world()
    view.height = 10  # Enough room for all
    view.width = 30
    view.tile_dimensions = (8, 16)

    # Add status effect and conditions
    effect = status_effects.OffBalanceEffect()
    effect.duration = 2
    actor.status_effects.apply_status_effect(effect)

    actor.conditions.add_condition(
        conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    )
    actor.conditions.add_condition(
        conditions.Injury(InjuryLocation.TORSO, "Combat Wound")
    )

    # Trigger a draw (with mocked graphics)
    view.canvas = MagicMock()
    view.draw_content(MagicMock(), InterpolationAlpha(0.0))

    # Should have tracked 1 status effect and 2 conditions
    assert view._num_status_effects_displayed == 1
    assert view._num_conditions_displayed == 2
