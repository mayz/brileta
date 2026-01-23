"""Tests for actor outline rendering in WorldView."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from catley import colors
from catley.config import CONTEXTUAL_OUTLINE_ALPHA
from catley.controller import Controller
from catley.game.actors import Actor
from catley.view.render.effects.screen_shake import ScreenShake
from catley.view.views.world_view import WorldView


class DummyActor:
    """Test actor stub."""

    def __init__(self, x: int, y: int, *, has_complex_visuals: bool = False) -> None:
        self.x = x
        self.y = y
        self.ch = "@"
        self.color = (255, 255, 255)
        self.character_layers: list[object] = []
        self.has_complex_visuals = has_complex_visuals


class DummyGameMap:
    """Test game map stub."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visible = np.zeros((width, height), dtype=bool)


class DummyGW:
    """Test game world stub."""

    def __init__(self) -> None:
        self.player = DummyActor(0, 0)
        self.actors = [self.player]
        self.game_map = DummyGameMap(20, 20)


@dataclass
class DummyController:
    """Test controller stub."""

    gw: DummyGW
    graphics: MagicMock
    clock: object
    active_mode: object | None
    is_combat_mode: Callable[[], bool]
    selected_target: DummyActor | None = None
    hovered_actor: DummyActor | None = None


def make_controller(*, is_combat: bool) -> DummyController:
    """Create a test controller with the specified combat mode state."""
    gw = DummyGW()
    gw.game_map.visible[:] = True
    graphics = MagicMock()
    graphics.console_to_screen_coords = lambda x, y: (x, y)
    graphics.tile_dimensions = (16, 16)  # Standard tile size for tests
    clock = SimpleNamespace(last_delta_time=0.016, last_time=0.0)
    return DummyController(
        gw=gw,
        graphics=graphics,
        clock=clock,
        active_mode=None,
        is_combat_mode=lambda: is_combat,
    )


def _setup_view(controller: DummyController) -> WorldView:
    """Create and configure a WorldView for testing."""
    view = WorldView(cast(Controller, controller), ScreenShake())
    view.set_bounds(0, 0, 10, 10)
    player = controller.gw.player
    player.x = 5
    player.y = 5
    view.viewport_system.update_camera(
        cast(Actor, player),
        controller.gw.game_map.width,
        controller.gw.game_map.height,
    )
    view.viewport_system.camera.set_position(player.x, player.y)
    return view


class TestSelectionOutline:
    """Tests for selected target outline rendering."""

    def test_selected_target_renders_golden_outline(self) -> None:
        """Selected target should render with golden outline."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view._render_selection_and_hover_outlines()

        vp_x, vp_y = view.viewport_system.world_to_screen(target.x, target.y)
        expected_root_x = view.x + vp_x
        expected_root_y = view.y + vp_y
        controller.graphics.draw_actor_outline.assert_called_once_with(
            target.ch,
            expected_root_x,
            expected_root_y,
            colors.SELECTION_OUTLINE,
            float(CONTEXTUAL_OUTLINE_ALPHA),
            scale_x=1.0,
            scale_y=1.0,
        )

    def test_selected_target_skips_in_combat_mode(self) -> None:
        """Selection outline should not render in combat mode."""
        controller = make_controller(is_combat=True)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view._render_selection_and_hover_outlines()

        controller.graphics.draw_actor_outline.assert_not_called()

    def test_selected_target_skips_when_not_visible(self) -> None:
        """Selection outline should not render for non-visible targets."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target
        controller.gw.game_map.visible[target.x, target.y] = False

        view._render_selection_and_hover_outlines()

        controller.graphics.draw_actor_outline.assert_not_called()


class TestHoverOutline:
    """Tests for hover actor outline rendering."""

    def test_hover_renders_white_outline(self) -> None:
        """Hovered actor should render with white outline at 50% opacity."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.hovered_actor = target

        view._render_selection_and_hover_outlines()

        vp_x, vp_y = view.viewport_system.world_to_screen(target.x, target.y)
        expected_root_x = view.x + vp_x
        expected_root_y = view.y + vp_y
        controller.graphics.draw_actor_outline.assert_called_once_with(
            target.ch,
            expected_root_x,
            expected_root_y,
            colors.HOVER_OUTLINE,
            0.50,
            scale_x=1.0,
            scale_y=1.0,
        )

    def test_selected_takes_priority_over_hover(self) -> None:
        """Selected target should take priority over hovered actor."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target
        controller.hovered_actor = target  # Same actor is both selected and hovered

        view._render_selection_and_hover_outlines()

        # Should only render once (selected takes priority)
        controller.graphics.draw_actor_outline.assert_called_once()
        # Should be golden (selection color), not grey (hover color)
        call_args = controller.graphics.draw_actor_outline.call_args
        # call_args[0] is positional args: (ch, x, y, color, alpha, ...)
        assert call_args[0][3] == colors.SELECTION_OUTLINE


@pytest.mark.parametrize(
    ("is_combat", "is_visible"),
    [
        (True, True),
        (False, False),
    ],
)
def test_hover_outline_skips_when_combat_or_not_visible(
    is_combat: bool, is_visible: bool
) -> None:
    """Hover outline should not render in combat or when target not visible."""
    controller = make_controller(is_combat=is_combat)
    view = _setup_view(controller)

    target = DummyActor(6, 5)
    controller.gw.actors.append(target)
    controller.hovered_actor = target
    controller.gw.game_map.visible[target.x, target.y] = is_visible

    view._render_selection_and_hover_outlines()

    controller.graphics.draw_actor_outline.assert_not_called()


class TestComplexVisualsOutline:
    """Tests for has_complex_visuals flag affecting outline rendering."""

    def test_complex_visuals_gets_full_tile_outline(self) -> None:
        """Actors with has_complex_visuals=True should get full-tile outline."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        # Actor with complex visuals (e.g., fire with particle effects)
        target = DummyActor(6, 5, has_complex_visuals=True)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view._render_selection_and_hover_outlines()

        # Should call draw_rect_outline for full-tile outline, not draw_actor_outline
        controller.graphics.draw_rect_outline.assert_called_once()
        controller.graphics.draw_actor_outline.assert_not_called()

    def test_regular_actor_gets_glyph_outline(self) -> None:
        """Regular actors (has_complex_visuals=False) should get glyph outline."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        # Regular actor without complex visuals
        target = DummyActor(6, 5, has_complex_visuals=False)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view._render_selection_and_hover_outlines()

        # Should call draw_actor_outline for glyph-based outline
        controller.graphics.draw_actor_outline.assert_called_once()
        controller.graphics.draw_rect_outline.assert_not_called()


class TestContainedFireHasComplexVisuals:
    """Tests verifying ContainedFire has has_complex_visuals=True."""

    def test_campfire_has_complex_visuals(self) -> None:
        """ContainedFire (campfire) should have has_complex_visuals=True."""
        from catley.game.actors.environmental import ContainedFire

        campfire = ContainedFire.create_campfire(5, 5)
        assert campfire.has_complex_visuals is True

    def test_barrel_fire_has_complex_visuals(self) -> None:
        """ContainedFire (barrel fire) should have has_complex_visuals=True."""
        from catley.game.actors.environmental import ContainedFire

        barrel_fire = ContainedFire.create_barrel_fire(5, 5)
        assert barrel_fire.has_complex_visuals is True

    def test_torch_has_complex_visuals(self) -> None:
        """ContainedFire (torch) should have has_complex_visuals=True."""
        from catley.game.actors.environmental import ContainedFire

        torch = ContainedFire.create_torch(5, 5)
        assert torch.has_complex_visuals is True
