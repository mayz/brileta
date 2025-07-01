from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character, conditions, status_effects
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
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
    controller, actor, view = make_world()
    assert view._get_condition_lines(actor) == []
    assert view._get_status_effect_lines(actor) == []


def test_condition_grouping_and_colors() -> None:
    controller, actor, view = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    actor.conditions.add_condition(injury)

    lines = sorted(view._get_condition_lines(actor))
    assert ("Exhaustion x2", colors.LIGHT_BLUE) in lines
    assert (injury.name, colors.RED) in lines


def test_status_effect_duration_format() -> None:
    controller, actor, view = make_world()
    effect = status_effects.OffBalanceEffect()
    effect.duration = 1
    actor.status_effects.apply_status_effect(effect)

    lines = view._get_status_effect_lines(actor)
    assert lines == ["Off Balance (1 turn)"]
