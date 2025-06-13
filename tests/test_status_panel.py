from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

from catley import colors
from catley.controller import Controller
from catley.game.actors import Character, conditions, status_effects
from catley.game.enums import InjuryLocation
from catley.game.game_world import GameWorld
from catley.view.panels.status_panel import StatusPanel
from catley.view.render.renderer import Renderer
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    gw: DummyGameWorld


def make_world() -> tuple[DummyController, Character, StatusPanel]:
    gw = DummyGameWorld()
    actor = Character(0, 0, "A", colors.WHITE, "Act", game_world=cast(GameWorld, gw))
    gw.player = actor
    gw.add_actor(actor)
    controller = DummyController(gw=gw)
    renderer = MagicMock(spec=Renderer)
    renderer.root_console = MagicMock()
    renderer.tile_dimensions = (8, 16)
    panel = StatusPanel(cast(Controller, controller), renderer=renderer)
    return controller, actor, panel


def test_no_conditions_or_effects() -> None:
    controller, actor, panel = make_world()
    assert panel._get_condition_lines(actor) == []
    assert panel._get_status_effect_lines(actor) == []


def test_condition_grouping_and_colors() -> None:
    controller, actor, panel = make_world()
    actor.conditions.add_condition(conditions.Exhaustion())
    actor.conditions.add_condition(conditions.Exhaustion())
    injury = conditions.Injury(InjuryLocation.LEFT_LEG, "Sprained Ankle")
    actor.conditions.add_condition(injury)

    lines = sorted(panel._get_condition_lines(actor))
    assert ("Exhaustion x2", colors.LIGHT_BLUE) in lines
    assert (injury.name, colors.RED) in lines


def test_status_effect_duration_format() -> None:
    controller, actor, panel = make_world()
    effect = status_effects.OffBalanceEffect()
    effect.duration = 1
    actor.status_effects.apply_status_effect(effect)

    lines = panel._get_status_effect_lines(actor)
    assert lines == ["Off Balance (1 turn)"]
