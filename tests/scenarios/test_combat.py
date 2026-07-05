"""Combat scenarios driven through the headless SimHarness.

Proves the harness can spawn a hostile NPC, drive the player to attack it through
the real intent pipeline (enter combat + MeleeAttackPlan), and observe combat
outcomes: damage dealt, combat mode engaged, and the canonical NUBS Phase 1
flee-when-hurt behavior.
"""

from __future__ import annotations

from brileta.game import ranges
from brileta.game.actors.indicators import IndicatorKind
from brileta.game.actors.npc_types import BRIGAND_TYPE
from brileta.testing import SimHarness
from brileta.util.pathfinding import probe_step
from tests.helpers import find_tile_near


def test_player_attacks_brigand_deals_damage() -> None:
    """Attacking a spawned brigand engages combat and drops its HP."""
    sim = SimHarness(seed="brawl-test-one")
    gw = sim.controller.gw

    spot = find_tile_near(
        gw.game_map,
        sim.player_pos,
        lambda x, y: probe_step(gw.game_map, gw, x, y) is None,
        min_radius=2,
    )
    brigand = sim.spawn(BRIGAND_TYPE, *spot, name="Brigand")

    # Brigands are hostile on sight (default disposition -75).
    assert sim.disposition(brigand) <= -51
    start_hp = brigand.health.hp

    # A single fist strike can miss, so re-issue the melee plan until a hit
    # lands. The dice stream is seeded, so this converges deterministically.
    for _ in range(15):
        if not brigand.health.is_alive() or brigand.health.hp < start_hp:
            break
        sim.attack(brigand)
        sim.wait(turns=60)

    assert sim.controller.is_combat_mode()
    assert brigand.health.hp < start_hp


def test_hurt_brigand_flees() -> None:
    """A badly wounded, anxious brigand adopts flee behavior (NUBS Phase 1).

    Drives the canonical showcase: damage the brigand to near death and it
    breaks and runs rather than fighting to the death - raising the FLEE
    indicator and opening distance from the player over several ticks.
    """
    sim = SimHarness(seed="flee-test-one")
    gw = sim.controller.gw
    player = sim.player
    px, py = sim.player_pos

    # Spawn on an open tile with clear line of sight to the player, so the
    # brigand is guaranteed to perceive the player as a threat regardless of
    # seed (flee requires a perceived threat - a walled-off spawn never would).
    spot = find_tile_near(
        gw.game_map,
        (px, py),
        lambda x, y: (
            probe_step(gw.game_map, gw, x, y) is None
            and ranges.has_line_of_sight(gw.game_map, x, y, px, py)
        ),
        min_radius=1,
    )
    brigand = sim.spawn(BRIGAND_TYPE, *spot, name="Coward")

    # Force the deterministic corner of the flee decision: max neuroticism (breaks
    # early) plus near-death HP, so flee reliably outscores attack in utility.
    brigand.personality.neuroticism = 10
    while brigand.health.hp > 1:
        brigand.take_damage(1)
    # Register the player as the aggressor so the threat is present even if the
    # brigand's line of sight is momentarily broken while it runs.
    brigand.ai.notify_attacked(player)

    start_distance = ranges.calculate_distance(brigand.x, brigand.y, player.x, player.y)

    saw_flee_indicator = False
    for _ in range(120):
        sim.tick(1)
        if brigand.indicator == IndicatorKind.FLEE:
            saw_flee_indicator = True

    # The FLEE indicator proves the flee action actually won scoring (not just
    # that the brigand wandered off). Distance opening corroborates it acted.
    assert saw_flee_indicator, "brigand never raised the FLEE indicator"
    assert brigand.health.is_alive(), "brigand died instead of fleeing"
    final_distance = ranges.calculate_distance(brigand.x, brigand.y, player.x, player.y)
    assert final_distance > start_distance
