"""Use-consumable-on-target scenario driven through the headless SimHarness.

Proves the targeted-consumable pipeline end to end: the player holds a healing
item, issues it against a wounded ally a few tiles away, and the plan drives the
approach (``ApproachStep``) before ``UseConsumableOnTargetExecutor`` applies the
heal - so the ally's HP rises only after the player has closed to adjacency.

Unit tests (``tests/game/actions/test_use_consumable_on_target.py`` and
``tests/game/items/test_consumable_heal.py``) apply the effect directly with the
target pre-placed; none drive the approach + heal wiring through the real loop.
"""

from __future__ import annotations

from brileta.game.actors.npc_types import RESIDENT_TYPE
from brileta.testing import SimHarness
from tests.helpers import find_tile_near, make_healing_item, make_reachable_predicate


def test_heal_ally_across_distance() -> None:
    """Player walks to a wounded ally and heals it with a consumable.

    The ally starts a few tiles off, so the plan must approach to adjacency
    before the heal applies; the rising HP proves the whole approach -> heal
    pipeline ran, not just the effect in isolation.
    """
    sim = SimHarness(seed="heal-ally-one")
    gw = sim.controller.gw
    start = sim.player_pos

    # A wounded, non-hostile resident a few reachable steps away, so the player
    # has to walk to it before the consumable can land.
    reachable = make_reachable_predicate(gw, sim.player, start, min_steps=3)
    spot = find_tile_near(gw.game_map, start, reachable, min_radius=2)
    ally = sim.spawn(RESIDENT_TYPE, *spot, name="Mara")
    ally.take_damage(2)
    assert ally.health.is_alive()
    assert ally.health.hp < ally.health.max_hp, "ally should start wounded"
    before = ally.health.hp

    salve = make_healing_item(heal=2)
    sim.player.inventory.add_to_inventory(salve)

    sim.use_item_on(ally, salve)
    # Pump enough steps for the player to path adjacent and apply the salve.
    sim.wait(turns=200)

    assert ally.health.hp > before, "ally was never healed"
    assert sim.player.active_plan is None, "plan should complete after healing"
