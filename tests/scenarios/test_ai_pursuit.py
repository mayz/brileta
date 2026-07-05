"""Hostile-pursuit scenario driven through the headless SimHarness.

The combat scenario (``test_combat.py``) drives the *player* attacking an NPC.
This covers the missing half of the combat loop: a hostile NPC perceiving the
idle player, pathing across open ground to reach it, and dealing damage - all
emergent from the utility AI run tick by tick through the real loop.

Unit tests (``tests/game/actors/ai/test_hostile_ai.py``) score a single AI
decision with the target pre-placed adjacent; none drive the pursuit-across-
distance + turn pacing + strike that this asserts end to end.
"""

from __future__ import annotations

from brileta.game import ranges
from brileta.game.actors.npc_types import BRIGAND_TYPE
from brileta.testing import SimHarness
from brileta.util.pathfinding import probe_step
from tests.helpers import find_tile_near


def test_hostile_brigand_pursues_and_damages_player() -> None:
    """An idle player is hunted down: the brigand closes distance and strikes.

    Spawns a hostile brigand a few tiles away with clear line of sight (it must
    perceive the player to pursue), then pumps the loop while the player stands
    still. The brigand's AI paths toward the player each step and eventually
    lands a hit, which auto-enters combat and drops the player's HP.
    """
    sim = SimHarness(seed="pursuit-test-one")
    gw = sim.controller.gw
    player = sim.player
    px, py = sim.player_pos

    # Spawn on an open tile with line of sight to the player, at least a few
    # tiles out, so the brigand has real ground to cover before it can strike.
    spot = find_tile_near(
        gw.game_map,
        (px, py),
        lambda x, y: (
            probe_step(gw.game_map, gw, x, y) is None
            and ranges.has_line_of_sight(gw.game_map, x, y, px, py)
        ),
        min_radius=3,
    )
    brigand = sim.spawn(BRIGAND_TYPE, *spot, name="Hunter")
    assert brigand.ai.is_hostile_toward(player)

    start_hp = player.health.hp
    start_distance = ranges.calculate_distance(brigand.x, brigand.y, px, py)

    # Pump until the brigand reaches the player and lands a hit. The player has
    # no plan, so it stands still and takes the pursuit; the dice stream is
    # seeded, so this converges deterministically.
    for _ in range(400):
        sim.tick(1)
        if player.health.hp < start_hp:
            break

    # It actually closed the gap (pursuit), and it actually connected (damage).
    final_distance = ranges.calculate_distance(brigand.x, brigand.y, *sim.player_pos)
    assert final_distance < start_distance, "brigand never closed on the player"
    assert player.health.hp < start_hp, "brigand never damaged the player"
    # Landing a hit on the player auto-enters combat mode.
    assert sim.controller.is_combat_mode()
