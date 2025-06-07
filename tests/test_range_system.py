from dataclasses import dataclass

import numpy as np

from catley.game import range_system
from catley.game.items.item_core import Item
from catley.game.items.properties import ItemProperty, WeaponProperty
from catley.world.map import GameMap


@dataclass
class DummyRangedAttack:
    optimal_range: int
    max_range: int
    properties: set[ItemProperty] | None = None


@dataclass
class DummyItem(Item):
    ranged_attack: DummyRangedAttack | None = None


class DummyMap(GameMap):
    def __init__(self, transparent: np.ndarray) -> None:
        super().__init__(transparent.shape[0], transparent.shape[1])
        self._transparent_map_cache = transparent


def test_get_range_category():
    attack = DummyRangedAttack(optimal_range=3, max_range=5)
    item = DummyItem(ranged_attack=attack)
    assert range_system.get_range_category(1, item) == "adjacent"
    assert range_system.get_range_category(3, item) == "close"
    assert range_system.get_range_category(5, item) == "far"
    assert range_system.get_range_category(6, item) == "out_of_range"
    melee_only = DummyItem()
    assert range_system.get_range_category(2, melee_only) == "melee_only"


def test_get_range_modifier_scoped():
    attack = DummyRangedAttack(
        optimal_range=3, max_range=5, properties={WeaponProperty.SCOPED}
    )
    item = DummyItem(ranged_attack=attack)
    mod = range_system.get_range_modifier(item, "far")
    assert mod == {"has_advantage": True}


def test_has_line_of_sight():
    transparent = np.ones((5, 5), dtype=bool)
    transparent[2, 2] = False
    game_map = DummyMap(transparent)
    assert not range_system.has_line_of_sight(game_map, 0, 0, 4, 4)
    transparent[2, 2] = True
    assert range_system.has_line_of_sight(game_map, 0, 0, 4, 4)


def test_calculate_distance():
    assert range_system.calculate_distance(0, 0, 3, 4) == 4
    assert range_system.calculate_distance(2, 2, 3, 3) == 1


def test_get_line():
    assert range_system.get_line(0, 0, 3, 3) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
