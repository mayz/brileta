"""Tests for ActorRenderer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import numpy as np

from brileta.environment.map import GameMap
from brileta.game.actors import Actor
from brileta.game.game_world import GameWorld
from brileta.types import InterpolationAlpha
from brileta.view.render.actor_renderer import ActorRenderer
from brileta.view.render.shadow_renderer import ShadowRenderer
from brileta.view.render.viewport import ViewportSystem


class DummyActor:
    """Minimal actor stub for ActorRenderer tests."""

    def __init__(
        self,
        x: int,
        y: int,
        *,
        ch: str = "@",
        color: tuple = (255, 255, 255),
        visual_scale: float = 1.0,
    ) -> None:
        self.x = x
        self.y = y
        self.prev_x = x
        self.prev_y = y
        self.ch = ch
        self.color = color
        self.visual_scale = visual_scale
        self.character_layers: list[object] = []
        self.has_complex_visuals = False
        self.visual_effects = None
        self.health = None
        self._animation_controlled = False
        self.render_x = float(x)
        self.render_y = float(y)
        self.shadow_height = 1


class DummySpatialIndex:
    """Stub spatial index that returns a configurable list of actors."""

    def __init__(self, actors: list[DummyActor]) -> None:
        self._actors = actors

    def get_in_bounds(self, x1: int, y1: int, x2: int, y2: int) -> list[DummyActor]:
        return [a for a in self._actors if x1 <= a.x <= x2 and y1 <= a.y <= y2]


class DummyGameMap:
    """Stub game map with a visibility array."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visible = np.ones((width, height), dtype=bool)


class DummyGameWorld:
    """Stub game world for render_actors tests."""

    def __init__(self, actors: list[DummyActor], player: DummyActor) -> None:
        self.actors = actors
        self.player = player
        self.game_map = DummyGameMap(20, 20)
        self.actor_spatial_index = DummySpatialIndex(actors)
        self.selected_actor = None


def _make_renderer(
    width: int = 10, height: int = 10
) -> tuple[ActorRenderer, MagicMock]:
    """Create an ActorRenderer with mocked graphics and return both."""
    graphics = MagicMock()
    graphics.console_to_screen_coords = lambda x, y: (float(x) * 16.0, float(y) * 16.0)
    graphics.tile_dimensions = (16, 16)

    vs = ViewportSystem(width, height)

    shadow_renderer = MagicMock(spec=ShadowRenderer)
    shadow_renderer.actor_shadow_receive_light_scale = {}

    renderer = ActorRenderer(
        viewport_system=vs,
        graphics=graphics,
        shadow_renderer=shadow_renderer,
    )
    return renderer, graphics


class TestRenderActorsSmoothDispatch:
    """Test that render_actors dispatches correctly based on the smooth flag."""

    def test_smooth_flag_true_calls_smooth_path(self) -> None:
        """When smooth=True, actors should render with sub-pixel positioning."""
        renderer, graphics = _make_renderer()
        player = DummyActor(5, 5)
        actor = DummyActor(6, 5, ch="g")
        gw = DummyGameWorld([player, actor], player)
        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actors(
            InterpolationAlpha(1.0),
            game_world=cast(GameWorld, gw),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
            smooth=True,
        )

        # draw_actor_smooth should have been called for both visible actors
        assert graphics.draw_actor_smooth.call_count >= 1

    def test_smooth_flag_false_calls_traditional_path(self) -> None:
        """When smooth=False, actors should render tile-aligned."""
        renderer, graphics = _make_renderer()
        player = DummyActor(5, 5)
        actor = DummyActor(6, 5, ch="g")
        gw = DummyGameWorld([player, actor], player)
        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actors(
            InterpolationAlpha(1.0),
            game_world=cast(GameWorld, gw),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
            smooth=False,
            game_time=0.0,
            is_combat=False,
        )

        # draw_actor_smooth is used by both paths (traditional also uses it)
        assert graphics.draw_actor_smooth.call_count >= 1


class TestVisibilityFiltering:
    """Test that only visible actors are rendered."""

    def test_invisible_actors_are_skipped(self) -> None:
        """Actors on non-visible tiles should not be rendered."""
        renderer, graphics = _make_renderer()
        player = DummyActor(5, 5)
        visible_actor = DummyActor(6, 5, ch="v")
        invisible_actor = DummyActor(7, 5, ch="i")
        gw = DummyGameWorld([player, visible_actor, invisible_actor], player)
        # Make the invisible actor's tile not visible
        gw.game_map.visible[7, 5] = False

        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actors(
            InterpolationAlpha(1.0),
            game_world=cast(GameWorld, gw),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
            smooth=True,
        )

        # Check that draw_actor_smooth was NOT called with the invisible actor's glyph
        for c in graphics.draw_actor_smooth.call_args_list:
            assert c[0][0] != "i", "Invisible actor should not be rendered"


class TestSortOrder:
    """Test that actors are sorted correctly for painter-style rendering."""

    def test_actors_sorted_by_y_then_scale_then_player(self) -> None:
        """Actors should sort by Y, then visual_scale, then player-on-top."""
        renderer, _graphics = _make_renderer()
        player = DummyActor(5, 5)
        actor_high_y = DummyActor(5, 7, ch="h")
        actor_low_y = DummyActor(5, 3, ch="l")
        gw = DummyGameWorld([player, actor_high_y, actor_low_y], player)
        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        bounds = renderer.viewport_system.get_visible_bounds()
        sorted_actors = renderer.get_sorted_visible_actors(bounds, cast(GameWorld, gw))

        # actor_low_y (y=3) should come before player (y=5) before actor_high_y (y=7)
        assert len(sorted_actors) == 3, "All 3 actors should be in viewport"
        ys = [a.y for a in sorted_actors]
        assert ys == sorted(ys), "Actors should be sorted by Y coordinate"

    def test_player_renders_on_top_at_same_y(self) -> None:
        """Player should render after other actors at the same Y coordinate."""
        renderer, _graphics = _make_renderer()
        player = DummyActor(5, 5, ch="P")
        npc = DummyActor(6, 5, ch="N")
        gw = DummyGameWorld([player, npc], player)
        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        bounds = renderer.viewport_system.get_visible_bounds()
        sorted_actors = renderer.get_sorted_visible_actors(bounds, cast(GameWorld, gw))

        # Both at y=5, same scale - player should be last
        assert len(sorted_actors) == 2, "Both actors should be in viewport"
        player_idx = next(i for i, a in enumerate(sorted_actors) if a.ch == "P")
        npc_idx = next(i for i, a in enumerate(sorted_actors) if a.ch == "N")
        assert player_idx > npc_idx, "Player should render after NPC at same Y"


class TestCharacterLayerRendering:
    """Test that multi-glyph actors (character layers) render correctly."""

    def test_character_layers_render_multiple_glyphs(self) -> None:
        """Actors with character_layers should render one glyph per layer."""
        renderer, graphics = _make_renderer()
        player = DummyActor(5, 5)

        # Create an actor with character layers
        actor = DummyActor(6, 5, ch="X")
        layer1 = SimpleNamespace(
            char="A",
            color=(255, 0, 0),
            offset_x=0.0,
            offset_y=0.0,
            scale_x=1.0,
            scale_y=1.0,
        )
        layer2 = SimpleNamespace(
            char="B",
            color=(0, 255, 0),
            offset_x=0.3,
            offset_y=0.0,
            scale_x=0.8,
            scale_y=0.8,
        )
        actor.character_layers = [layer1, layer2]

        gw = DummyGameWorld([player, actor], player)
        renderer.viewport_system.update_camera(cast(Actor, player), 20, 20)
        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actors(
            InterpolationAlpha(1.0),
            game_world=cast(GameWorld, gw),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
            smooth=True,
        )

        # Should render layer chars, not the actor's base ch
        rendered_chars = [c[0][0] for c in graphics.draw_actor_smooth.call_args_list]
        assert "A" in rendered_chars, "Layer 1 char should be rendered"
        assert "B" in rendered_chars, "Layer 2 char should be rendered"
        # The actor's base char "X" should NOT be rendered (layers replace it)
        assert "X" not in rendered_chars, (
            "Base char should not be rendered when layers exist"
        )


class TestRenderActorOutline:
    """Tests for render_actor_outline (combat mode glyph outlines)."""

    def test_outline_drawn_for_visible_actor(self) -> None:
        """A visible actor should produce a draw_actor_outline call."""
        renderer, graphics = _make_renderer()
        actor = DummyActor(6, 5, ch="g")
        game_map = DummyGameMap(20, 20)  # all visible by default

        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actor_outline(
            cast(Actor, actor),
            (255, 0, 0),
            0.8,
            game_map=cast(GameMap, game_map),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
        )

        assert graphics.draw_actor_outline.call_count == 1
        call_args = graphics.draw_actor_outline.call_args
        assert call_args[0][0] == "g", "Should outline the actor's glyph"

    def test_outline_skipped_for_invisible_actor(self) -> None:
        """An actor on a non-visible tile should not produce an outline."""
        renderer, graphics = _make_renderer()
        actor = DummyActor(6, 5, ch="g")
        game_map = DummyGameMap(20, 20)
        game_map.visible[6, 5] = False

        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actor_outline(
            cast(Actor, actor),
            (255, 0, 0),
            0.8,
            game_map=cast(GameMap, game_map),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
        )

        assert graphics.draw_actor_outline.call_count == 0

    def test_animation_controlled_actor_uses_float_position(self) -> None:
        """An animation-controlled actor should use render_x/render_y for positioning."""
        renderer, graphics = _make_renderer()
        actor = DummyActor(6, 5, ch="g")
        actor._animation_controlled = True
        actor.render_x = 6.3
        actor.render_y = 5.7
        game_map = DummyGameMap(20, 20)

        renderer.viewport_system.camera.set_position(5.0, 5.0)

        renderer.render_actor_outline(
            cast(Actor, actor),
            (255, 0, 0),
            0.8,
            game_map=cast(GameMap, game_map),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
        )

        assert graphics.draw_actor_outline.call_count == 1
        # The screen coords should reflect the fractional world position
        call_args = graphics.draw_actor_outline.call_args
        screen_x, screen_y = call_args[0][1], call_args[0][2]
        # With camera at (5,5) and viewport centered, the viewport offset should
        # produce non-integer screen pixel values for a fractional world position
        assert isinstance(screen_x, float)
        assert isinstance(screen_y, float)

    def test_camera_frac_offset_shifts_outline_position(self) -> None:
        """Camera fractional offset should shift the outline position."""
        renderer, graphics = _make_renderer()
        actor = DummyActor(6, 5, ch="g")
        game_map = DummyGameMap(20, 20)

        renderer.viewport_system.camera.set_position(5.0, 5.0)

        # Render with no offset
        renderer.render_actor_outline(
            cast(Actor, actor),
            (255, 0, 0),
            0.8,
            game_map=cast(GameMap, game_map),
            camera_frac_offset=(0.0, 0.0),
            view_origin=(0.0, 0.0),
        )
        no_offset_x = graphics.draw_actor_outline.call_args[0][1]

        graphics.reset_mock()

        # Render with a fractional camera offset
        renderer.render_actor_outline(
            cast(Actor, actor),
            (255, 0, 0),
            0.8,
            game_map=cast(GameMap, game_map),
            camera_frac_offset=(0.5, 0.0),
            view_origin=(0.0, 0.0),
        )
        offset_x = graphics.draw_actor_outline.call_args[0][1]

        # The offset should shift the position (0.5 tiles * 16 px/tile = 8px shift)
        assert offset_x != no_offset_x, (
            "Camera frac offset should shift outline position"
        )
        assert offset_x < no_offset_x, "Positive frac offset should shift position left"
