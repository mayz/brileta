from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from brileta.environment.generators import GeneratedMapData
from brileta.environment.map import GameMap

if TYPE_CHECKING:
    from brileta.environment.map import MapRegion
from brileta.game import ranges
from brileta.game.items.capabilities import RangedAttack
from brileta.game.items.item_core import Item
from brileta.game.items.properties import ItemProperty, WeaponProperty


@dataclass
class DummyRangedAttack:
    optimal_range: int
    max_range: int
    properties: set[ItemProperty] | None = None


@dataclass
class DummyItem(Item):
    ranged_attack: RangedAttack | None = None


class DummyMap(GameMap):
    def __init__(self, transparent: np.ndarray) -> None:
        width, height = transparent.shape
        tiles = np.full((width, height), 1, dtype=np.uint8)
        regions: dict[int, MapRegion] = {}
        map_data = GeneratedMapData(
            tiles=tiles,
            regions=regions,
            tile_to_region_id=np.full((width, height), -1, dtype=np.int16),
        )
        super().__init__(width, height, map_data)
        self._transparent_map_cache = transparent


def test_get_range_category():
    attack = DummyRangedAttack(optimal_range=3, max_range=5)
    item = DummyItem(ranged_attack=cast(RangedAttack, attack))
    assert ranges.get_range_category(1, item) == "adjacent"
    assert ranges.get_range_category(3, item) == "close"
    assert ranges.get_range_category(5, item) == "far"
    assert ranges.get_range_category(6, item) == "out_of_range"
    melee_only = DummyItem()
    assert ranges.get_range_category(2, melee_only) == "melee_only"


def test_get_range_modifier_scoped():
    """Scoped weapons are awkward up close but steady at distance."""
    attack = DummyRangedAttack(
        optimal_range=3, max_range=5, properties={WeaponProperty.SCOPED}
    )
    item = DummyItem(ranged_attack=cast(RangedAttack, attack))
    # Scoped weapons have disadvantage at close range
    assert ranges.get_range_modifier(item, "adjacent") == {"has_disadvantage": True}
    assert ranges.get_range_modifier(item, "close") == {"has_disadvantage": True}
    # Scoped weapons have no modifier at far range (scope compensates for distance)
    assert ranges.get_range_modifier(item, "far") == {}


def test_has_line_of_sight():
    transparent = np.ones((5, 5), dtype=bool)
    transparent[2, 2] = False
    game_map = DummyMap(transparent)
    assert not ranges.has_line_of_sight(game_map, 0, 0, 4, 4)
    transparent[2, 2] = True
    assert ranges.has_line_of_sight(game_map, 0, 0, 4, 4)


def test_calculate_distance():
    assert ranges.calculate_distance(0, 0, 3, 4) == 4
    assert ranges.calculate_distance(2, 2, 3, 3) == 1


def test_get_line():
    assert ranges.get_line(0, 0, 3, 3) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
