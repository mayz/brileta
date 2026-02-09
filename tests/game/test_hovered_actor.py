"""Tests for hovered actor tracking in Controller.

The hovered_actor field tracks which actor is under the mouse for visual
feedback (subtle hover outline). It must be updated in two scenarios:
1. When the mouse moves (via update_hovered_actor called from InputHandler)
2. When actors move (via update_hovered_actor called after turn processing)
"""

from __future__ import annotations

from typing import cast

from brileta import colors
from brileta.game.actors import Character
from brileta.game.game_world import GameWorld
from tests.helpers import DummyGameWorld


class DummyController:
    """Minimal controller stub for testing hovered actor logic."""

    def __init__(self, gw: DummyGameWorld) -> None:
        self.gw = gw
        self.hovered_actor: Character | None = None

    def update_hovered_actor(self, mouse_pos: tuple[int, int] | None) -> None:
        """Update the hovered actor based on mouse position."""
        if mouse_pos is None:
            self.hovered_actor = None
            return

        mouse_x, mouse_y = mouse_pos
        self.hovered_actor = self._get_visible_actor_at_tile(mouse_x, mouse_y)

    def _get_visible_actor_at_tile(self, x: int, y: int) -> Character | None:
        """Return the first visible actor at a tile, if any."""
        gm = self.gw.game_map
        if not (0 <= x < gm.width and 0 <= y < gm.height):
            return None
        if not gm.visible[x, y]:
            return None

        actors_at_tile = self.gw.actor_spatial_index.get_at_point(x, y)
        for actor in actors_at_tile:
            if actor is not self.gw.player:
                return cast(Character, actor)
        return None


class TestHoveredActorFromMouseMovement:
    """Tests for hovered_actor updates when mouse moves."""

    def test_mouse_over_actor_sets_hovered_actor(self) -> None:
        """Moving mouse over an actor should set hovered_actor."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        controller = DummyController(gw)

        controller.update_hovered_actor((6, 5))
        assert controller.hovered_actor is npc

    def test_mouse_over_empty_tile_clears_hovered_actor(self) -> None:
        """Moving mouse over empty tile should clear hovered_actor."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        controller = DummyController(gw)

        # First hover over NPC
        controller.update_hovered_actor((6, 5))
        assert controller.hovered_actor is npc

        # Then move to empty tile
        controller.update_hovered_actor((7, 5))
        assert controller.hovered_actor is None

    def test_mouse_none_clears_hovered_actor(self) -> None:
        """Mouse position None (outside game area) should clear hovered_actor."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        controller = DummyController(gw)

        controller.update_hovered_actor((6, 5))
        assert controller.hovered_actor is npc

        controller.update_hovered_actor(None)
        assert controller.hovered_actor is None

    def test_mouse_over_non_visible_tile_returns_none(self) -> None:
        """Mouse over non-visible tile should not set hovered_actor."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        # Make the NPC's tile not visible
        gw.game_map.visible[6, 5] = False

        controller = DummyController(gw)

        controller.update_hovered_actor((6, 5))
        assert controller.hovered_actor is None

    def test_player_is_not_hovered(self) -> None:
        """Hovering over the player should not set hovered_actor."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        controller = DummyController(gw)

        controller.update_hovered_actor((5, 5))
        assert controller.hovered_actor is None


class TestHoveredActorAfterActorMovement:
    """Tests for hovered_actor updates when actors move (turn processing)."""

    def test_actor_moves_into_mouse_position(self) -> None:
        """When an actor moves to the mouse position, hovered_actor should update."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(7, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        controller = DummyController(gw)

        # Mouse is at (6, 5) - empty
        gw.mouse_tile_location_on_map = (6, 5)
        controller.update_hovered_actor(gw.mouse_tile_location_on_map)
        assert controller.hovered_actor is None

        # NPC moves to (6, 5)
        gw.actor_spatial_index.remove(npc)
        npc.x, npc.y = 6, 5
        gw.actor_spatial_index.add(npc)

        # Simulate turn completion refresh
        controller.update_hovered_actor(gw.mouse_tile_location_on_map)
        assert controller.hovered_actor is npc

    def test_actor_moves_away_from_mouse_position(self) -> None:
        """When an actor moves away from mouse position, hovered_actor should clear."""
        gw = DummyGameWorld()
        player = Character(
            5, 5, "@", colors.WHITE, "Player", game_world=cast(GameWorld, gw)
        )
        gw.player = player
        gw.add_actor(player)

        npc = Character(6, 5, "N", colors.RED, "NPC", game_world=cast(GameWorld, gw))
        gw.add_actor(npc)

        controller = DummyController(gw)

        # Mouse is at (6, 5) - where NPC is
        gw.mouse_tile_location_on_map = (6, 5)
        controller.update_hovered_actor(gw.mouse_tile_location_on_map)
        assert controller.hovered_actor is npc

        # NPC moves away to (7, 5)
        gw.actor_spatial_index.remove(npc)
        npc.x, npc.y = 7, 5
        gw.actor_spatial_index.add(npc)

        # Simulate turn completion refresh
        controller.update_hovered_actor(gw.mouse_tile_location_on_map)
        assert controller.hovered_actor is None
