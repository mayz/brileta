"""Tests for hovered actor tracking in Controller.

The hovered_actor field tracks which actor is under the mouse for visual
feedback (the hover ring) and the ActionPanel's hover target. Hover is resolved
by hit-testing the cursor's fractional world position against actors'
interpolated (rendered) footprints, so a walking actor is hoverable wherever
its sprite currently appears, not only on its logical destination tile.
"""

from __future__ import annotations

from brileta import colors
from brileta.game.actors import Character
from brileta.game.actors.core import CharacterLayer
from tests.helpers import get_controller_with_dummy_world


def _hover_at(controller, wx: float, wy: float) -> None:
    """Point the cursor at a fractional world position and refresh hover."""
    controller.gw.mouse_world_pos_f = (wx, wy)
    controller.update_hovered_actor()


class TestHoveredActorFromMouseMovement:
    """Tests for hovered_actor updates when the cursor moves."""

    def test_cursor_over_actor_sets_hovered_actor(self) -> None:
        """Pointing at an actor's tile sets hovered_actor."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is npc

    def test_cursor_over_empty_tile_clears_hovered_actor(self) -> None:
        """Pointing at an empty tile clears hovered_actor."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is npc

        _hover_at(controller, 7.5, 5.5)
        assert controller.hovered_actor is None

    def test_cursor_none_clears_hovered_actor(self) -> None:
        """A None cursor position (outside the map) clears hovered_actor."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is npc

        gw.mouse_world_pos_f = None
        controller.update_hovered_actor()
        assert controller.hovered_actor is None

    def test_cursor_over_non_visible_tile_returns_none(self) -> None:
        """An actor on a non-visible tile is not hovered."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)
        gw.game_map.visible[6, 5] = False

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is None

    def test_player_is_hoverable(self) -> None:
        """Hovering the player sets hovered_actor (consistent with selection)."""
        controller = get_controller_with_dummy_world()
        player = controller.gw.player

        _hover_at(controller, player.x + 0.5, player.y + 0.5)
        assert controller.hovered_actor is player

    def test_overlapping_npc_preferred_over_player(self) -> None:
        """An NPC sharing the player's tile is hovered, not the player."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        player = gw.player
        npc = Character(player.x, player.y, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, player.x + 0.5, player.y + 0.5)
        assert controller.hovered_actor is npc


class TestHoveredActorFollowsInterpolatedPosition:
    """The hit test tracks the sprite's interpolated position mid-walk."""

    def test_hover_hits_interpolated_cell_not_logical_tile(self) -> None:
        """A mid-walk actor is hoverable where its sprite is drawn."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw

        # NPC's logical tile is (6, 5) but it is halfway through a step from
        # (6, 6), so its sprite renders around y = 5.5. Force mid-step interp.
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        npc.prev_x, npc.prev_y = 6, 6
        gw.add_actor(npc)
        controller.frame_manager.world_view.interpolation_alpha = 0.5

        # The interpolated cell is roughly [6, 7) x [5.5, 6.5): a point in the
        # visually-occupied lower tile hits, even though it is not the logical
        # tile (6, 5).
        _hover_at(controller, 6.5, 6.0)
        assert controller.hovered_actor is npc

        # The now-vacated logical tile (its upper half) does not hit.
        _hover_at(controller, 6.5, 5.1)
        assert controller.hovered_actor is None


class TestHoveredActorCoversDrawnExtent:
    """Selection covers a sprite's full drawn area, not just its base tile."""

    def test_tall_sprite_hoverable_above_its_base_tile(self) -> None:
        """A scaled-up sprite (like a tree) is hoverable across its canopy."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw

        # A tree-like sprite at (10, 10) drawn 3 tiles tall, feet at the tile.
        # Its drawn box spans roughly x[9, 12), y[8, 11).
        tree = Character(10, 10, "T", colors.RED, "Tree", game_world=gw)
        tree.visual_scale = 3.0
        tree.sprite_ground_anchor_y = 1.0
        gw.add_actor(tree)

        # Two tiles above the base (up in the canopy) still hits.
        _hover_at(controller, 10.5, 8.5)
        assert controller.hovered_actor is tree

        # The trunk tile hits.
        _hover_at(controller, 10.5, 10.5)
        assert controller.hovered_actor is tree

        # Above the canopy top does not.
        _hover_at(controller, 10.5, 7.0)
        assert controller.hovered_actor is None

    def test_flat_sprite_not_selected_above_its_silhouette(self) -> None:
        """A flat sprite (boulder) is not selectable in its transparent padding."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw

        # A wide, flat boulder: large visual_scale (mostly padding) but its
        # opaque content fills only the bottom sliver of the quad.
        boulder = Character(10, 10, "O", colors.RED, "Boulder", game_world=gw)
        boulder.visual_scale = 3.0
        boulder.sprite_ground_anchor_y = 1.0
        boulder.sprite_content_bbox = (0.1, 0.7, 0.9, 1.0)
        gw.add_actor(boulder)

        # Above the silhouette, in the quad's transparent padding: no selection.
        _hover_at(controller, 10.5, 8.5)
        assert controller.hovered_actor is None

        # On the boulder's visible body: selected.
        _hover_at(controller, 10.5, 10.6)
        assert controller.hovered_actor is boulder

    def test_multi_layer_actor_extent_covers_its_layers(self) -> None:
        """A multi-layer glyph actor (bookcase) is clickable across its layers."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw

        # A bookcase drawn as a tall composition: a double-height layer whose
        # box reaches half a tile above the base tile top (y=10).
        bookcase = Character(10, 10, "H", colors.RED, "Bookcase", game_world=gw)
        bookcase.character_layers = [
            CharacterLayer(char="=", color=colors.RED, scale_y=2.0),
        ]
        gw.add_actor(bookcase)

        # Above the base tile, inside the tall layer, is clickable.
        _hover_at(controller, 10.5, 9.7)
        assert controller.hovered_actor is bookcase

        # The base tile is clickable.
        _hover_at(controller, 10.5, 10.5)
        assert controller.hovered_actor is bookcase


class TestHoveredActorAfterActorMovement:
    """Tests for hovered_actor updates when actors move under a still cursor."""

    def test_actor_moves_into_cursor_position(self) -> None:
        """When an actor moves under the cursor, hovered_actor updates."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(7, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is None

        gw.actor_spatial_index.remove(npc)
        npc.x, npc.y = 6, 5
        npc.prev_x, npc.prev_y = 6, 5
        gw.actor_spatial_index.add(npc)

        controller.update_hovered_actor()
        assert controller.hovered_actor is npc

    def test_actor_moves_away_from_cursor_position(self) -> None:
        """When an actor moves off the cursor, hovered_actor clears."""
        controller = get_controller_with_dummy_world()
        gw = controller.gw
        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=gw)
        gw.add_actor(npc)

        _hover_at(controller, 6.5, 5.5)
        assert controller.hovered_actor is npc

        gw.actor_spatial_index.remove(npc)
        npc.x, npc.y = 7, 5
        npc.prev_x, npc.prev_y = 7, 5
        gw.actor_spatial_index.add(npc)

        controller.update_hovered_actor()
        assert controller.hovered_actor is None
