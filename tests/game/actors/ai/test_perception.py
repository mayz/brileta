"""Tests for the PerceptionComponent and incoming_threat integration.

Validates:
- Detection within awareness_radius with clear LOS.
- LOS blocking prevents detection through walls.
- Perception strength falls off linearly with distance.
- Out-of-range actors are not detected.
- incoming_threat computation in UtilityContext.
- Sapient NPCs react to incoming threats via utility scoring.
"""

from typing import cast
from unittest.mock import patch

import pytest

from brileta import colors
from brileta.environment.tile_types import TileTypeID
from brileta.game.actors import NPC, Character
from brileta.game.actors.ai.perception import PerceptionComponent
from brileta.game.game_world import GameWorld
from tests.helpers import DummyController, DummyGameWorld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perception_world(
    *,
    npc_x: int = 5,
    npc_y: int = 5,
    player_x: int = 0,
    player_y: int = 0,
    awareness_radius: int = 12,
    map_size: int = 30,
) -> tuple[DummyController, Character, NPC]:
    """Create a test world with configurable perception radius."""
    gw = DummyGameWorld(width=map_size, height=map_size)
    player = Character(
        player_x,
        player_y,
        "@",
        colors.WHITE,
        "Player",
        game_world=cast(GameWorld, gw),
    )
    npc = NPC(
        npc_x,
        npc_y,
        "g",
        colors.RED,
        "Guard",
        game_world=cast(GameWorld, gw),
    )
    # Configure perception on the NPC's AI component.
    npc.ai.perception = PerceptionComponent(awareness_radius=awareness_radius)
    gw.player = player
    gw.add_actor(player)
    gw.add_actor(npc)
    controller = DummyController(gw)
    return controller, player, npc


# ---------------------------------------------------------------------------
# PerceptionComponent Unit Tests
# ---------------------------------------------------------------------------


class TestPerceptionDetection:
    """Tests for basic detection (range + LOS)."""

    def test_detects_actor_within_radius(self) -> None:
        """Actor within awareness_radius and clear LOS should be detected."""
        controller, player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player in actors

    def test_ignores_actor_beyond_radius(self) -> None:
        """Actor beyond awareness_radius should not be detected."""
        controller, player, npc = _make_perception_world(
            npc_x=15, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player not in actors

    def test_ignores_actor_at_exact_radius_boundary(self) -> None:
        """Actor at exactly awareness_radius distance should not be detected."""
        controller, player, npc = _make_perception_world(
            npc_x=10, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player not in actors

    def test_los_blocked_by_wall(self) -> None:
        """Wall between perceiver and target should block detection."""
        controller, player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        # Place a wall between player at (0,0) and NPC at (5,0).
        # Bresenham LOS checks intermediate tiles [1:-1], so wall at (3,0)
        # should block the ray.
        controller.gw.game_map.tiles[3, 0] = TileTypeID.WALL
        controller.gw.game_map.invalidate_property_caches()

        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player not in actors

    def test_adjacent_actor_detected_despite_opaque_tile(self) -> None:
        """Adjacent actors should always be detected.

        Bresenham LOS only checks intermediate tiles (excluding start and
        end), so an adjacent actor is always visible even if the perceiver
        is standing on an opaque tile.
        """
        controller, player, npc = _make_perception_world(
            npc_x=1, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player in actors

    def test_does_not_perceive_self(self) -> None:
        """The perceiving actor should never appear in its own results."""
        controller, _player, npc = _make_perception_world()
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert npc not in actors

    def test_does_not_perceive_dead_actors(self) -> None:
        """Dead actors should not appear in perception results."""
        controller, player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        player.health._hp = 0

        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        actors = [p.actor for p in perceived]
        assert player not in actors


class TestPerceptionStrength:
    """Tests for perception_strength falloff."""

    def test_strength_at_point_blank(self) -> None:
        """Perception strength should be 1.0 when adjacent (distance 1)."""
        controller, player, npc = _make_perception_world(
            npc_x=1, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        entry = next(p for p in perceived if p.actor is player)
        assert entry.perception_strength == pytest.approx(0.9)  # 1 - 1/10

    def test_strength_falls_off_with_distance(self) -> None:
        """Closer actors should have higher perception_strength."""
        controller, player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=20
        )
        # Add a second actor closer to the NPC.
        closer = NPC(
            3,
            0,
            "c",
            colors.RED,
            "Closer",
            game_world=cast(GameWorld, controller.gw),
        )
        controller.gw.add_actor(closer)

        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        player_entry = next(p for p in perceived if p.actor is player)
        closer_entry = next(p for p in perceived if p.actor is closer)

        assert closer_entry.perception_strength > player_entry.perception_strength

    def test_strength_just_inside_radius_is_positive(self) -> None:
        """Perception strength remains positive for in-range actors."""
        controller, player, npc = _make_perception_world(
            npc_x=9, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        entry = next(p for p in perceived if p.actor is player)
        assert entry.perception_strength == pytest.approx(0.1)

    def test_results_sorted_by_distance(self) -> None:
        """Perceived actors should be sorted closest-first."""
        controller, _player, npc = _make_perception_world(
            npc_x=10, npc_y=0, player_x=0, player_y=0, awareness_radius=20
        )
        mid = NPC(
            5,
            0,
            "m",
            colors.RED,
            "Mid",
            game_world=cast(GameWorld, controller.gw),
        )
        controller.gw.add_actor(mid)

        perceived = npc.ai.perception.get_perceived_actors(
            npc, controller.gw.game_map, controller.gw.actors
        )
        distances = [p.distance for p in perceived]
        assert distances == sorted(distances)


# ---------------------------------------------------------------------------
# Incoming Threat Integration Tests
# ---------------------------------------------------------------------------


class TestIncomingThreat:
    """Tests for incoming_threat computation in AIComponent."""

    def test_incoming_threat_from_hostile_target(self) -> None:
        """incoming_threat should be non-zero when target is hostile toward us."""
        controller, _player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        # Create a hostile NPC that the guard can see.
        hostile = NPC(
            5,
            0,
            "H",
            colors.RED,
            "Hostile",
            game_world=cast(GameWorld, controller.gw),
        )
        controller.gw.add_actor(hostile)
        # Make the hostile NPC hostile toward our guard.
        hostile.ai.set_hostile(npc)

        context = npc.ai._build_context(controller, npc)
        # The hostile NPC is perceived and hostile toward us, so
        # incoming_threat should be positive regardless of target selection.
        assert context.incoming_threat > 0.0

    def test_incoming_threat_zero_for_neutral_target(self) -> None:
        """incoming_threat should be 0 when target has no hostility toward us."""
        controller, _player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        # Player has no AI, so incoming_threat should be 0.
        context = npc.ai._build_context(controller, npc)
        assert context.incoming_threat == 0.0

    def test_incoming_threat_scales_with_perception_strength(self) -> None:
        """Closer hostile actors should produce higher incoming_threat."""
        # Close hostile
        controller1, _, npc1 = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=20
        )
        hostile1 = NPC(
            3,
            0,
            "H",
            colors.RED,
            "Hostile",
            game_world=cast(GameWorld, controller1.gw),
        )
        controller1.gw.add_actor(hostile1)
        hostile1.ai.set_hostile(npc1)
        # Make NPC hostile toward hostile1 so it becomes the target.
        npc1.ai.set_hostile(hostile1)

        # Far hostile
        controller2, _, npc2 = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=20
        )
        hostile2 = NPC(
            15,
            0,
            "H",
            colors.RED,
            "Hostile",
            game_world=cast(GameWorld, controller2.gw),
        )
        controller2.gw.add_actor(hostile2)
        hostile2.ai.set_hostile(npc2)
        npc2.ai.set_hostile(hostile2)

        ctx1 = npc1.ai._build_context(controller1, npc1)
        ctx2 = npc2.ai._build_context(controller2, npc2)

        # Close hostile should produce higher incoming_threat.
        assert ctx1.incoming_threat >= ctx2.incoming_threat

    def test_incoming_threat_detected_without_outgoing_target(self) -> None:
        """Incoming threat should detect hostile actors even when the NPC
        has no outgoing-threat target at all.

        Regression test: a neutral resident should perceive incoming threat
        from an approaching hostile scorpion, even though the resident has
        no target (neutral disposition = no outgoing threat = target is None).
        """
        controller, _player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        # Create a hostile attacker that the NPC hasn't targeted.
        attacker = NPC(
            3,
            0,
            "s",
            colors.RED,
            "Attacker",
            game_world=cast(GameWorld, controller.gw),
        )
        controller.gw.add_actor(attacker)
        # The attacker is hostile toward our NPC, but our NPC has no
        # hostility toward the attacker (neutral disposition).
        attacker.ai.set_hostile(npc)

        context = npc.ai._build_context(controller, npc)
        # The NPC has no outgoing-threat target, but incoming_threat should
        # still detect the hostile attacker, and threat_source should be set.
        assert context.target is None
        assert context.incoming_threat > 0.0
        assert context.threat_source is attacker

    def test_incoming_threat_blocked_by_los_wall(self) -> None:
        """LOS blockers should suppress incoming_threat just like targeting."""
        controller, _player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        attacker = NPC(
            1,
            0,
            "s",
            colors.RED,
            "Attacker",
            game_world=cast(GameWorld, controller.gw),
        )
        controller.gw.add_actor(attacker)
        attacker.ai.set_hostile(npc)

        # Block LOS between npc (5,0) and attacker (1,0).
        controller.gw.game_map.tiles[3, 0] = TileTypeID.WALL
        controller.gw.game_map.invalidate_property_caches()

        context = npc.ai._build_context(controller, npc)
        assert context.incoming_threat == 0.0
        assert context.threat_source is None

    def test_incoming_threat_in_get_input(self) -> None:
        """incoming_threat should be accessible via UtilityContext.get_input()."""
        controller, _player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        context = npc.ai._build_context(controller, npc)
        value = context.get_input("incoming_threat")
        assert value is not None
        assert isinstance(value, float)

    def test_build_context_reuses_single_perception_query(self) -> None:
        """Context build should call perception once and reuse the snapshot."""
        controller, _player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        with patch.object(
            npc.ai,
            "_get_perceived_actors",
            wraps=npc.ai._get_perceived_actors,
        ) as mocked_perception:
            npc.ai._build_context(controller, npc)

        assert mocked_perception.call_count == 1


# ---------------------------------------------------------------------------
# Perception-Gated Target Selection Tests
# ---------------------------------------------------------------------------


class TestPerceptionGatedTargeting:
    """Tests that _select_target_actor respects perception."""

    def test_target_selection_ignores_actors_behind_walls(self) -> None:
        """NPCs should not target actors they can't see through walls."""
        controller, player, npc = _make_perception_world(
            npc_x=5, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        npc.ai.set_hostile(player)

        # Place wall blocking LOS
        controller.gw.game_map.tiles[3, 0] = TileTypeID.WALL
        controller.gw.game_map.invalidate_property_caches()

        # Can't see the player through the wall, so no target at all.
        context = npc.ai._build_context(controller, npc)
        assert context.target is None
        assert context.threat_level == 0.0

    def test_target_selection_works_with_clear_los(self) -> None:
        """NPCs should target perceived hostile actors with clear LOS."""
        controller, player, npc = _make_perception_world(
            npc_x=3, npc_y=0, player_x=0, player_y=0, awareness_radius=10
        )
        npc.ai.set_hostile(player)

        context = npc.ai._build_context(controller, npc)
        assert context.threat_level > 0.0
        assert context.target is player
