"""Movement/pathfinding scenario driven through the headless SimHarness.

Proves the harness can drive the player (start a walk plan), pump the game loop
deterministically, and observe the resulting position - the core loop that every
other scenario builds on.
"""

from __future__ import annotations

from brileta.game.ranges import calculate_distance
from brileta.testing import SimHarness
from tests.helpers import find_tile_near, make_reachable_predicate


def test_player_walks_to_reachable_tile() -> None:
    """The player walks to a distant reachable tile and arrives there."""
    sim = SimHarness(seed="acid-helm-pivot")
    gw = sim.controller.gw
    start = sim.player_pos
    min_steps = 6

    reachable = make_reachable_predicate(gw, sim.player, start, min_steps=min_steps)
    target = find_tile_near(gw.game_map, start, reachable, min_radius=min_steps)
    assert target != start

    sim.walk_to(*target)
    # Pump enough steps for the plan to path across the distance. WalkToPlan
    # stops on the target tile (stop_distance=0), so on arrival the plan clears.
    sim.wait(turns=300)

    # Arrived at the target (or immediately adjacent, if the final tile became
    # transiently occupied). Either satisfies "walk drove the player there".
    assert calculate_distance(*sim.player_pos, *target) <= 1
    assert sim.player.active_plan is None
