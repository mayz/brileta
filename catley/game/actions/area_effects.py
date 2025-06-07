"""
Area effect actions for weapons and abilities that affect multiple tiles.

Handles explosions, area-of-effect attacks, and other abilities that impact
multiple targets or tiles simultaneously.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game import range_system
from catley.game.actions.base import GameAction
from catley.game.actors import Character
from catley.game.items.capabilities import AreaEffect, RangedAttack
from catley.game.items.item_core import Item
from catley.game.items.properties import TacticalProperty, WeaponProperty

if TYPE_CHECKING:
    from catley.controller import Controller


# Convenience type aliases for area-effect calculations
Coord = tuple[int, int]
# Maps an (x, y) tile coordinate to the distance from the effect's origin.
DistanceByTile = dict[Coord, int]


class AreaEffectAction(GameAction):
    """Action for executing an item's area effect."""

    def __init__(
        self,
        controller: Controller,
        attacker: Character,
        target_x: int,
        target_y: int,
        weapon: Item,
    ) -> None:
        super().__init__(controller, attacker)
        self.attacker = attacker
        self.target_x = target_x
        self.target_y = target_y
        self.weapon = weapon

    def execute(self) -> None:
        effect = self.weapon.area_effect
        if effect is None:
            return

        # 1. Check ammo requirements if weapon uses ammo
        ranged = self.weapon.ranged_attack
        if ranged and ranged.current_ammo <= 0:
            self.controller.message_log.add_message(
                f"{self.weapon.name} is out of ammo!",
                colors.RED,
            )
            return

        # 2. Determine which tiles are affected by the effect
        tiles = self._calculate_tiles(effect)
        if not tiles:
            self.controller.message_log.add_message("Nothing is affected.", colors.GREY)
            return

        # 3. Apply damage or healing to actors in the affected tiles
        hits = self._apply_damage(tiles, effect)

        # 4. Consume ammo if necessary
        if ranged:
            self._consume_ammo(ranged)

        # 5. Log messages about the outcome of the effect
        self._log_effect_results(hits)

        # 6. Trigger the appropriate visual effect
        self._trigger_visual_effect(effect)

    def _calculate_tiles(self, effect: AreaEffect) -> DistanceByTile:
        match effect.area_type:
            case "circle":
                return self._circle_tiles(effect)
            case "line":
                return self._line_tiles(effect)
            case "cone":
                return self._cone_tiles(effect)
            case _:
                return {}

    def _circle_tiles(self, effect: AreaEffect) -> DistanceByTile:
        game_map = self.controller.gw.game_map
        tiles: dict[tuple[int, int], int] = {}
        for dx in range(-effect.size, effect.size + 1):
            for dy in range(-effect.size, effect.size + 1):
                tx = self.target_x + dx
                ty = self.target_y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                distance = max(abs(dx), abs(dy))
                if distance > effect.size:
                    continue
                if effect.requires_line_of_sight and not range_system.has_line_of_sight(
                    game_map,
                    self.attacker.x,
                    self.attacker.y,
                    tx,
                    ty,
                ):
                    continue
                if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                    continue
                tiles[(tx, ty)] = distance
        return tiles

    def _line_tiles(self, effect: AreaEffect) -> DistanceByTile:
        game_map = self.controller.gw.game_map
        tiles: dict[tuple[int, int], int] = {}
        line = range_system.get_line(
            self.attacker.x,
            self.attacker.y,
            self.target_x,
            self.target_y,
        )
        for i, (tx, ty) in enumerate(line[1 : effect.size + 1], start=1):
            if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                break
            if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                break
            if effect.requires_line_of_sight and not range_system.has_line_of_sight(
                game_map,
                self.attacker.x,
                self.attacker.y,
                tx,
                ty,
            ):
                continue
            tiles[(tx, ty)] = i
        return tiles

    def _cone_tiles(self, effect: AreaEffect) -> DistanceByTile:
        game_map = self.controller.gw.game_map
        tiles: dict[tuple[int, int], int] = {}

        dir_x = self.target_x - self.attacker.x
        dir_y = self.target_y - self.attacker.y
        length = (dir_x**2 + dir_y**2) ** 0.5 or 1.0
        dir_x /= length
        dir_y /= length
        cos_limit = 0.707  # ~45 degrees spread

        for dx in range(-effect.size, effect.size + 1):
            for dy in range(-effect.size, effect.size + 1):
                tx = self.attacker.x + dx
                ty = self.attacker.y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                distance = (dx**2 + dy**2) ** 0.5
                if distance == 0 or distance > effect.size:
                    continue
                dot = dx * dir_x + dy * dir_y
                if dot <= 0:
                    continue
                cos_angle = dot / distance
                if cos_angle < cos_limit:
                    continue
                if (
                    effect._spec.requires_line_of_sight
                    and not range_system.has_line_of_sight(
                        game_map,
                        self.attacker.x,
                        self.attacker.y,
                        tx,
                        ty,
                    )
                ):
                    continue
                if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                    continue
                tiles[(tx, ty)] = round(distance)
        return tiles

    def _apply_damage(
        self, tiles: DistanceByTile, effect: AreaEffect
    ) -> list[tuple[Character, int]]:
        """Apply damage or healing to actors within the affected tiles."""
        base_damage = effect.damage_dice.roll()
        hits: list[tuple[Character, int]] = []
        for actor in self.controller.gw.actors:
            if not isinstance(actor, Character) or not actor.health.is_alive():
                continue
            if (actor.x, actor.y) not in tiles:
                continue
            distance = tiles[(actor.x, actor.y)]
            damage = base_damage
            if effect.damage_falloff:
                falloff = max(0.0, 1.0 - (distance / max(1, effect.size)))
                damage = round(base_damage * falloff)
            if damage == 0:
                continue
            if damage > 0:
                actor.take_damage(damage)
            else:
                actor.health.heal(-damage)
            hits.append((actor, damage))
        return hits

    def _consume_ammo(self, ranged_attack: RangedAttack) -> None:
        """Consume ammo for weapons with a ranged attack."""
        ammo_used = 1
        if WeaponProperty.AUTOMATIC in ranged_attack.properties:
            ammo_used = min(3, ranged_attack.current_ammo)
        ranged_attack.current_ammo -= ammo_used

    def _log_effect_results(self, hits: list[tuple[Character, int]]) -> None:
        """Log messages summarizing the effect results."""
        for actor, dmg in hits:
            if dmg > 0:
                self.controller.message_log.add_message(
                    f"{actor.name} takes {dmg} damage.", colors.ORANGE
                )
            else:
                self.controller.message_log.add_message(
                    f"{actor.name} recovers {-dmg} HP.", colors.GREEN
                )
        if not hits:
            self.controller.message_log.add_message(
                "The effect hits nothing.", colors.GREY
            )

    def _trigger_visual_effect(self, effect: AreaEffect) -> None:
        """Emit particle effects based on the effect properties."""
        if TacticalProperty.EXPLOSIVE in effect.properties:
            self.frame_manager.create_effect(
                "explosion", x=self.target_x, y=self.target_y
            )
        elif TacticalProperty.SMOKE in effect.properties:
            self.frame_manager.create_effect(
                "smoke_cloud", x=self.target_x, y=self.target_y
            )
