"""Tests for actor outline rendering via ActorRenderer (delegated from WorldView)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from brileta import colors
from brileta.controller import Controller
from brileta.game.actors import Actor
from brileta.view.render.effects.screen_shake import ScreenShake
from brileta.view.views.world_view import WorldView


class DummyActor:
    """Test actor stub."""

    def __init__(self, x: int, y: int, *, has_complex_visuals: bool = False) -> None:
        self.x = x
        self.y = y
        self.ch = "@"
        self.color = (255, 255, 255)
        self.character_layers: list[object] = []
        self.has_complex_visuals = has_complex_visuals
        self.visual_scale = 1.0
        self.sprite_uv = None
        self._animation_controlled = False
        self.render_x = float(x)
        self.render_y = float(y)
        # Interpolation sources and optional components read by
        # compute_actor_screen_position.
        self.prev_x = float(x)
        self.prev_y = float(y)
        self.visual_effects = None
        self.health = None


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


class DummyController:
    """Test controller stub."""

    def __init__(
        self,
        gw: DummyGW,
        graphics: MagicMock,
        clock: object,
        active_mode: object | None,
        is_combat_mode_fn: Callable[[], bool],
    ) -> None:
        self.gw = gw
        self.graphics = graphics
        self.clock = clock
        self.active_mode = active_mode
        self._is_combat_mode_fn = is_combat_mode_fn
        self.selected_target: DummyActor | None = None
        self.hovered_actor: DummyActor | None = None

    def is_combat_mode(self) -> bool:
        return self._is_combat_mode_fn()


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
        is_combat_mode_fn=lambda: is_combat,
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


def _outline_kwargs(view: WorldView) -> dict:
    """Return common keyword arguments for outline rendering calls."""
    return {
        "game_world": view.controller.gw,
        "controller": view.controller,
        "interpolation_alpha": 1.0,
        "camera_frac_offset": view.camera_frac_offset,
        "view_origin": (float(view.x), float(view.y)),
    }


class TestSelectionOutline:
    """Tests for selected target outline rendering."""

    def test_selected_target_renders_golden_ring(self) -> None:
        """Selected entity should render a golden ground ring."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_called_once()
        # Ring color is the golden selection color (positional index 4).
        call_args = controller.graphics.draw_ellipse_outline.call_args
        assert call_args[0][4] == colors.SELECTION_OUTLINE

    def test_selected_target_skips_in_combat_mode(self) -> None:
        """Selection outline should not render in combat mode."""
        controller = make_controller(is_combat=True)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_not_called()

    def test_selected_target_skips_when_not_visible(self) -> None:
        """Selection outline should not render for non-visible targets."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target
        controller.gw.game_map.visible[target.x, target.y] = False

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_not_called()


class TestHoverOutline:
    """Tests for hover actor outline rendering."""

    def test_hover_renders_ring(self) -> None:
        """Hovered entity should render a subtle ground ring."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.hovered_actor = target

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_called_once()
        call_args = controller.graphics.draw_ellipse_outline.call_args
        assert call_args[0][4] == colors.HOVER_OUTLINE

    def test_selected_takes_priority_over_hover(self) -> None:
        """Selected target should take priority over hovered actor."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        target = DummyActor(6, 5)
        controller.gw.actors.append(target)
        controller.selected_target = target
        controller.hovered_actor = target  # Same actor is both selected and hovered

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        # Should only render once (selected takes priority)
        controller.graphics.draw_ellipse_outline.assert_called_once()
        # Should be golden (selection color), not the hover color
        call_args = controller.graphics.draw_ellipse_outline.call_args
        assert call_args[0][4] == colors.SELECTION_OUTLINE


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

    view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

    controller.graphics.draw_ellipse_outline.assert_not_called()


class TestComplexVisualsOutline:
    """Entities get a ground ring regardless of visual composition."""

    def test_complex_visuals_gets_ring(self) -> None:
        """Actors with has_complex_visuals get the same ring as any entity."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        # Actor with complex visuals (e.g., fire with particle effects)
        target = DummyActor(6, 5, has_complex_visuals=True)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_called_once()
        controller.graphics.draw_rect_outline.assert_not_called()

    def test_regular_actor_gets_ring(self) -> None:
        """Regular actors also get the ground ring."""
        controller = make_controller(is_combat=False)
        view = _setup_view(controller)

        # Regular actor without complex visuals
        target = DummyActor(6, 5, has_complex_visuals=False)
        controller.gw.actors.append(target)
        controller.selected_target = target

        view.actor_renderer.render_selection_and_hover_outlines(**_outline_kwargs(view))

        controller.graphics.draw_ellipse_outline.assert_called_once()
        controller.graphics.draw_rect_outline.assert_not_called()


class TestContainedFireHasComplexVisuals:
    """Tests verifying ContainedFire has has_complex_visuals=True."""

    def test_campfire_has_complex_visuals(self) -> None:
        """ContainedFire (campfire) should have has_complex_visuals=True."""
        from brileta.game.actors.environmental import ContainedFire

        campfire = ContainedFire.create_campfire(5, 5)
        assert campfire.has_complex_visuals is True

    def test_barrel_fire_has_complex_visuals(self) -> None:
        """ContainedFire (barrel fire) should have has_complex_visuals=True."""
        from brileta.game.actors.environmental import ContainedFire

        barrel_fire = ContainedFire.create_barrel_fire(5, 5)
        assert barrel_fire.has_complex_visuals is True

    def test_torch_has_complex_visuals(self) -> None:
        """ContainedFire (torch) should have has_complex_visuals=True."""
        from brileta.game.actors.environmental import ContainedFire

        torch = ContainedFire.create_torch(5, 5)
        assert torch.has_complex_visuals is True
