from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.constants.combat import CombatConstants as Combat
from catley.events import EffectEvent, MessageEvent, publish_event
from catley.game import ranges
from catley.game.actions.base import GameActionResult
from catley.game.actions.executors.base import ActionExecutor
from catley.game.actors import Character
from catley.game.enums import AreaType
from catley.game.items.capabilities import AreaEffect, RangedAttack
from catley.game.items.properties import TacticalProperty, WeaponProperty
from catley.types import WorldTilePos

if TYPE_CHECKING:
    from catley.game.actions.area_effects import AreaEffectIntent

# Convenience type aliases for area-effect calculations
Coord = WorldTilePos
# Maps an (x, y) tile coordinate to the distance from the effect's origin.
DistanceByTile = dict[Coord, int]


class WeaponAreaEffectExecutor(ActionExecutor):
    """Executes weapon-based area effect intents like grenades and flamethrowers.

    This executor handles area-of-effect attacks that originate from weapons,
    including ammo consumption, range validation, and weapon-specific effects.
    For environmental damage (fire, radiation zones), use EnvironmentalDamageExecutor.
    """

    def __init__(self) -> None:
        """Create a WeaponAreaEffectExecutor without requiring a controller."""
        pass

    def execute(self, intent: AreaEffectIntent) -> GameActionResult | None:  # type: ignore[override]
        effect = intent.weapon.area_effect
        if effect is None:
            return GameActionResult(succeeded=False)

        # 1. Check ammo requirements if weapon uses ammo
        ranged = intent.weapon.ranged_attack
        if ranged and ranged.current_ammo <= 0:
            publish_event(
                MessageEvent(f"{intent.weapon.name} is out of ammo!", colors.RED)
            )
            return GameActionResult(succeeded=False)

        # 2. Determine which tiles are affected by the effect
        tiles = self._calculate_tiles(intent, effect)
        if not tiles:
            publish_event(MessageEvent("Nothing is affected.", colors.GREY))
            return GameActionResult(succeeded=False)

        # 3. Apply damage or healing to actors in the affected tiles
        hits = self._apply_damage(intent, tiles, effect)

        # 4. Consume ammo if necessary
        if ranged:
            self._consume_ammo(ranged)
            intent.attacker.inventory._increment_revision()

        # 5. Log messages about the outcome of the effect
        self._log_effect_results(hits)

        # 6. Trigger the appropriate visual effect
        self._trigger_visual_effect(intent, effect)
        return GameActionResult()

    def _calculate_tiles(
        self, intent: AreaEffectIntent, effect: AreaEffect
    ) -> DistanceByTile:
        match effect.area_type:
            case AreaType.CIRCLE:
                return self._circle_tiles(intent, effect)
            case AreaType.LINE:
                return self._line_tiles(intent, effect)
            case AreaType.CONE:
                return self._cone_tiles(intent, effect)
            case _:
                return {}

    def _circle_tiles(
        self, intent: AreaEffectIntent, effect: AreaEffect
    ) -> DistanceByTile:
        game_map = intent.controller.gw.game_map
        tiles: dict[WorldTilePos, int] = {}
        for dx in range(-effect.size, effect.size + 1):
            for dy in range(-effect.size, effect.size + 1):
                tx = intent.target_x + dx
                ty = intent.target_y + dy
                if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                    continue
                distance = max(abs(dx), abs(dy))
                if distance > effect.size:
                    continue
                if effect.requires_line_of_sight and not ranges.has_line_of_sight(
                    game_map,
                    intent.attacker.x,
                    intent.attacker.y,
                    tx,
                    ty,
                ):
                    continue
                if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                    continue
                tiles[(tx, ty)] = distance
        return tiles

    def _line_tiles(
        self, intent: AreaEffectIntent, effect: AreaEffect
    ) -> DistanceByTile:
        game_map = intent.controller.gw.game_map
        tiles: dict[WorldTilePos, int] = {}
        line = ranges.get_line(
            intent.attacker.x,
            intent.attacker.y,
            intent.target_x,
            intent.target_y,
        )
        for i, (tx, ty) in enumerate(line[1 : effect.size + 1], start=1):
            if not (0 <= tx < game_map.width and 0 <= ty < game_map.height):
                break
            if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                break
            if effect.requires_line_of_sight and not ranges.has_line_of_sight(
                game_map,
                intent.attacker.x,
                intent.attacker.y,
                tx,
                ty,
            ):
                continue
            tiles[(tx, ty)] = i
        return tiles

    def _cone_tiles(
        self, intent: AreaEffectIntent, effect: AreaEffect
    ) -> DistanceByTile:
        game_map = intent.controller.gw.game_map
        tiles: dict[WorldTilePos, int] = {}

        dir_x = intent.target_x - intent.attacker.x
        dir_y = intent.target_y - intent.attacker.y
        length = (dir_x**2 + dir_y**2) ** 0.5 or 1.0
        dir_x /= length
        dir_y /= length
        cos_limit = Combat.CONE_SPREAD_COSINE

        for dx in range(-effect.size, effect.size + 1):
            for dy in range(-effect.size, effect.size + 1):
                tx = intent.attacker.x + dx
                ty = intent.attacker.y + dy
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
                if effect._spec.requires_line_of_sight and not ranges.has_line_of_sight(
                    game_map,
                    intent.attacker.x,
                    intent.attacker.y,
                    tx,
                    ty,
                ):
                    continue
                if not effect.penetrates_walls and not game_map.transparent[tx, ty]:
                    continue
                tiles[(tx, ty)] = round(distance)
        return tiles

    def _apply_damage(
        self, intent: AreaEffectIntent, tiles: DistanceByTile, effect: AreaEffect
    ) -> list[tuple[Character, int]]:
        """Apply damage or healing to actors within the affected tiles."""
        base_damage = effect.damage_dice.roll()
        hits: list[tuple[Character, int]] = []

        # Determine damage type
        damage_type = "normal"
        if TacticalProperty.RADIATION in effect.properties:
            damage_type = "radiation"

        actors = intent.controller.gw.actors
        for actor in actors:
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
                actor.take_damage(damage, damage_type=damage_type)
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
                publish_event(
                    MessageEvent(f"{actor.name} takes {dmg} damage.", colors.ORANGE)
                )
            else:
                publish_event(
                    MessageEvent(f"{actor.name} recovers {-dmg} HP.", colors.GREEN)
                )
        if not hits:
            publish_event(MessageEvent("The effect hits nothing.", colors.GREY))

    def _trigger_visual_effect(
        self, intent: AreaEffectIntent, effect: AreaEffect
    ) -> None:
        """Emit particle effects based on the effect properties."""
        if TacticalProperty.EXPLOSIVE in effect.properties:
            publish_event(EffectEvent("explosion", intent.target_x, intent.target_y))
        elif TacticalProperty.SMOKE in effect.properties:
            publish_event(EffectEvent("smoke_cloud", intent.target_x, intent.target_y))
