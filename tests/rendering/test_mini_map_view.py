"""Tests for MiniMapView."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np

from brileta import colors
from brileta.controller import Controller, GameWorld
from brileta.game.actors import NPC, Character
from brileta.game.actors.boulder import Boulder
from brileta.game.actors.trees import Tree
from brileta.types import InterpolationAlpha
from brileta.view.render.graphics import GraphicsContext
from brileta.view.render.viewport import ViewportSystem
from brileta.view.views.mini_map_view import MiniMapView
from tests.helpers import DummyGameWorld


@dataclass
class DummyController:
    """Minimal controller stub for MiniMapView tests."""

    gw: DummyGameWorld
    graphics: Any


def make_world() -> tuple[DummyController, Character, MiniMapView]:
    """Create a mini-map view backed by a dummy world and renderer."""
    gw = DummyGameWorld(width=30, height=20)
    player = Character(
        5,
        5,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    gw.player = player
    gw.add_actor(player)

    graphics = MagicMock(spec=GraphicsContext)
    graphics.tile_dimensions = (8, 16)
    graphics.release_texture = MagicMock()

    controller = DummyController(gw=gw, graphics=graphics)
    viewport = ViewportSystem(10, 8)
    viewport.viewport.map_width = gw.game_map.width
    viewport.viewport.map_height = gw.game_map.height
    viewport.camera.set_position(float(player.x), float(player.y))

    view = MiniMapView(cast(Controller, controller), viewport)
    view.tile_dimensions = graphics.tile_dimensions
    view.set_bounds(0, 0, 30, 10)
    return controller, player, view


def test_cache_key_changes_on_exploration_revision_bump() -> None:
    """Exploration progress must invalidate the mini-map texture cache."""
    controller, _player, view = make_world()

    key_before = view.get_cache_key()
    controller.gw.game_map.exploration_revision += 1
    key_after = view.get_cache_key()

    assert key_before != key_after


def test_cache_key_changes_on_player_position_change() -> None:
    """Player movement must invalidate the player marker position."""
    _controller, player, view = make_world()

    key_before = view.get_cache_key()
    player.x += 1
    key_after = view.get_cache_key()

    assert key_before != key_after


def test_cache_key_changes_on_structural_revision_change() -> None:
    """Map structure changes must invalidate the terrain layer."""
    controller, _player, view = make_world()

    key_before = view.get_cache_key()
    controller.gw.game_map.structural_revision += 1
    key_after = view.get_cache_key()

    assert key_before != key_after


def test_reset_for_new_world_forces_full_rebuild() -> None:
    """After reset_for_new_world, the terrain buffer rebuilds even if
    structural_revision is the same as before (both maps at 0)."""
    controller, _player, view = make_world()

    # Simulate having already rendered - mark revisions as current.
    view._map_rgb_revision = controller.gw.game_map.structural_revision
    view._map_rgb_exploration_revision = controller.gw.game_map.exploration_revision
    view._feature_layer_rev = controller.gw.game_map.structural_revision

    # Swap in a "new" game map with the same structural_revision (0).
    new_gw = DummyGameWorld(width=30, height=20)
    assert new_gw.game_map.structural_revision == 0
    controller.gw = new_gw

    # Without reset, _ensure_map_rgb_current would see 0 == 0 and skip rebuild.
    view.reset_for_new_world()

    # After reset, revisions are -1 so the next ensure call forces a full rebuild.
    assert view._map_rgb_revision == -1
    assert view._map_rgb_exploration_revision == -1
    assert view._feature_layer_rev == -1


def test_explored_colors_are_strictly_dimmer_than_visible_colors() -> None:
    """Explored terrain colors should be a dimmed version of visible terrain."""
    _controller, _player, view = make_world()

    for visible, explored in zip(
        view._visible_colors, view._explored_colors, strict=True
    ):
        assert all(exp <= vis for exp, vis in zip(explored, visible, strict=True))
        assert sum(explored) < sum(visible)


def test_visibility_toggle() -> None:
    """MiniMapView should honor inherited hide/show visibility controls."""
    _controller, _player, view = make_world()

    assert view.visible is True
    view.hide()
    assert view.visible is False
    view.show()
    assert view.visible is True


def test_dead_visible_npc_is_excluded_from_markers() -> None:
    """Dead NPCs should not leave stale dots on the mini-map."""
    controller, player, view = make_world()

    npc = NPC(
        8,
        6,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)
    npc.ai.set_hostile(player)
    assert npc.health is not None
    npc.health._hp = 0
    controller.gw.game_map.visible[npc.x, npc.y] = True

    assert (npc.x, npc.y, True) not in view._iter_visible_character_markers()


def test_live_visible_hostile_npc_appears_in_markers() -> None:
    """Visible hostile NPCs should appear as hostile mini-map markers."""
    controller, player, view = make_world()

    npc = NPC(
        8,
        6,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)
    npc.ai.set_hostile(player)
    controller.gw.game_map.visible[npc.x, npc.y] = True

    assert (npc.x, npc.y, True) in view._iter_visible_character_markers()


def test_actor_marker_changes_do_not_invalidate_terrain_layer_key() -> None:
    """NPC movement should invalidate overlay only, not the terrain layer."""
    controller, player, view = make_world()

    npc = NPC(
        8,
        6,
        "r",
        colors.RED,
        "Raider",
        game_world=cast(GameWorld, controller.gw),
    )
    controller.gw.add_actor(npc)
    npc.ai.set_hostile(player)
    controller.gw.game_map.visible[npc.x, npc.y] = True

    terrain_key_before = view._get_terrain_cache_key()
    overlay_key_before = view._get_overlay_cache_key()

    controller.gw.game_map.visible[npc.x, npc.y] = False
    npc.x = 9
    npc.y = 6
    controller.gw.game_map.visible[npc.x, npc.y] = True

    terrain_key_after = view._get_terrain_cache_key()
    overlay_key_after = view._get_overlay_cache_key()

    assert terrain_key_before == terrain_key_after
    assert overlay_key_before != overlay_key_after


def test_terrain_cache_key_ignores_player_position() -> None:
    """Terrain cache invalidation should depend on exploration revision, not position."""
    _controller, player, view = make_world()

    terrain_key_before = view._get_terrain_cache_key()
    player.x += 1
    terrain_key_after = view._get_terrain_cache_key()

    assert terrain_key_before == terrain_key_after


def test_scale_selection_expands_to_available_space() -> None:
    """Mini-map scale should grow to fill bounds and respect height limits."""
    _controller, _player, view = make_world()

    view.set_bounds(0, 0, 30, 10)  # 240x160 px -> min((238//30), (158//20)) = 7
    assert view._get_pixels_per_tile() == 7

    view.set_bounds(0, 0, 20, 10)  # 160x160 px -> width-limited to 5
    assert view._get_pixels_per_tile() == 5

    view.set_bounds(0, 0, 40, 4)  # 320x64 px -> height-limited to 3
    assert view._get_pixels_per_tile() == 3


# ---------------------------------------------------------------------------
# _render_terrain_pixels tests
# ---------------------------------------------------------------------------


def test_terrain_pixels_shape_and_contiguity() -> None:
    """Terrain buffer must be (H, W, 4) RGBA, C-contiguous for GPU upload."""
    _controller, _player, view = make_world()

    buf = view._render_terrain_pixels()

    assert buf.ndim == 3
    assert buf.shape[2] == 4  # RGBA
    assert buf.dtype == np.uint8
    assert buf.flags["C_CONTIGUOUS"]


def test_terrain_pixels_visible_tiles_use_visible_colors() -> None:
    """Visible floor tiles should render with the visible color from the LUT."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    # Ensure everything is visible.
    game_map.visible[:] = True
    game_map.explored[:] = True

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    # Sample a tile well inside the map.
    from brileta.environment.tile_types import TileTypeID

    expected_rgb = view._visible_colors[TileTypeID.FLOOR]
    px = buf[origin_y + 1, origin_x + 1]
    assert tuple(px[:3]) == expected_rgb
    assert px[3] == 255


def test_terrain_pixels_explored_only_tiles_use_explored_colors() -> None:
    """Explored-but-not-visible tiles should use the dimmer explored palette."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = False
    game_map.explored[:] = True

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    from brileta.environment.tile_types import TileTypeID

    expected_rgb = view._explored_colors[TileTypeID.FLOOR]
    px = buf[origin_y + 1, origin_x + 1]
    assert tuple(px[:3]) == expected_rgb


def test_terrain_pixels_visible_tree_uses_tree_color() -> None:
    """A visible tree should paint its pixel with the tree minimap color."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = True
    game_map.explored[:] = True

    # Place a tree at a known position.
    tree = Tree(10, 8, "#", colors.GREEN, game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(tree)

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    px = buf[origin_y + tree.y, origin_x + tree.x]
    assert tuple(px[:3]) == MiniMapView._FEATURE_COLORS[Tree]


def test_terrain_pixels_visible_boulder_uses_boulder_color() -> None:
    """A visible boulder should paint its pixel with the boulder minimap color."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = True
    game_map.explored[:] = True

    boulder = Boulder(12, 6, game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(boulder)

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    px = buf[origin_y + boulder.y, origin_x + boulder.x]
    assert tuple(px[:3]) == MiniMapView._FEATURE_COLORS[Boulder]


def test_terrain_pixels_explored_tree_uses_dimmed_color() -> None:
    """An explored-but-not-visible tree should use the dimmed tree color."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = False
    game_map.explored[:] = True

    tree = Tree(10, 8, "#", colors.GREEN, game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(tree)

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    expected = view._dim_color(
        MiniMapView._FEATURE_COLORS[Tree], view._EXPLORED_DIM_FACTOR
    )
    px = buf[origin_y + tree.y, origin_x + tree.x]
    assert tuple(px[:3]) == expected


def test_terrain_pixels_unexplored_tree_stays_black() -> None:
    """An unexplored tree should not appear on the minimap."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = False
    game_map.explored[:] = False

    tree = Tree(10, 8, "#", colors.GREEN, game_world=cast(GameWorld, controller.gw))
    controller.gw.add_actor(tree)

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    px = buf[origin_y + tree.y, origin_x + tree.x]
    assert tuple(px[:3]) == (0, 0, 0)


def test_terrain_pixels_unexplored_tiles_are_black() -> None:
    """Unexplored tiles should stay black (RGB 0,0,0)."""
    controller, _player, view = make_world()
    game_map = controller.gw.game_map
    game_map.visible[:] = False
    game_map.explored[:] = False

    buf = view._render_terrain_pixels()
    _, _ma_x, _ma_y, tile_origin_x, tile_origin_y = view._map_layout()
    px_per_tile = view._get_pixels_per_tile()
    origin_x = round(tile_origin_x / px_per_tile)
    origin_y = round(tile_origin_y / px_per_tile)

    px = buf[origin_y + 1, origin_x + 1]
    assert tuple(px[:3]) == (0, 0, 0)
    assert px[3] == 255  # Opaque black, not transparent.


# ---------------------------------------------------------------------------
# player is None path
# ---------------------------------------------------------------------------


def test_draw_content_without_player_skips_markers() -> None:
    """draw_content should not crash when the game has no player."""
    controller, _player, view = make_world()
    controller.gw.player = None

    graphics = MagicMock(spec=GraphicsContext)
    graphics.tile_dimensions = (8, 16)

    view.canvas.begin_frame()
    # Should not raise.
    view.draw_content(graphics, InterpolationAlpha(0.0))
    view.canvas.end_frame()
