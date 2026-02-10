from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from brileta import config
from brileta.game import ranges
from brileta.game.actors import Character

if TYPE_CHECKING:
    from brileta.controller import Controller


@dataclass
class ActionContext:
    """Context information for action discovery."""

    tile_x: int
    tile_y: int

    nearby_actors: list[Character]
    items_on_ground: list

    in_combat: bool = False
    selected_actor: Character | None = None
    interaction_mode: str = "normal"


class ActionContextBuilder:
    """Utility to construct :class:`ActionContext` from game state."""

    def build_context(self, controller: Controller, actor: Character) -> ActionContext:
        gm = controller.gw.game_map
        potential_actors = controller.gw.actor_spatial_index.get_in_radius(
            actor.x, actor.y, radius=config.ACTION_CONTEXT_RADIUS
        )

        nearby: list[Character] = []
        for other in potential_actors:
            if other is actor or not isinstance(other, Character):
                continue
            if not other.health.is_alive():
                continue
            if (
                0 <= other.x < gm.width
                and 0 <= other.y < gm.height
                and gm.visible[other.x, other.y]
                and ranges.has_line_of_sight(gm, actor.x, actor.y, other.x, other.y)
            ):
                nearby.append(other)

        items_on_ground = controller.gw.get_pickable_items_at_location(actor.x, actor.y)
        in_combat = any(
            o.ai is not None and o.ai.is_hostile_toward(actor) for o in nearby
        )
        selected_actor = controller.gw.selected_actor
        selected_actor = (
            selected_actor if isinstance(selected_actor, Character) else None
        )

        return ActionContext(
            tile_x=actor.x,
            tile_y=actor.y,
            nearby_actors=nearby,
            items_on_ground=items_on_ground,
            in_combat=in_combat,
            selected_actor=selected_actor,
        )

    def get_nearby_actors(
        self, controller: Controller, actor: Character
    ) -> list[Character]:
        return self.build_context(controller, actor).nearby_actors

    def calculate_combat_probability(
        self,
        controller: Controller,
        actor: Character,
        target: Character,
        stat_name: str,
        range_modifiers: dict | None = None,
    ) -> float:
        resolution_modifiers = actor.modifiers.get_resolution_modifiers(stat_name)
        has_advantage = (
            range_modifiers and range_modifiers.get("has_advantage", False)
        ) or resolution_modifiers.get("has_advantage", False)
        has_disadvantage = (
            range_modifiers and range_modifiers.get("has_disadvantage", False)
        ) or resolution_modifiers.get("has_disadvantage", False)

        resolver = controller.create_resolver(
            ability_score=getattr(actor.stats, stat_name),
            roll_to_exceed=target.stats.agility + 10,
            has_advantage=has_advantage,
            has_disadvantage=has_disadvantage,
        )
        return resolver.calculate_success_probability()

    def get_adjacent_cover_bonus(
        self, controller: Controller, defender: Character
    ) -> int:
        """Return the highest cover bonus adjacent to the defender."""
        from brileta.environment import tile_types

        game_map = controller.gw.game_map
        max_bonus = 0
        x, y = defender.x, defender.y

        # Cover checks are infrequent, so we prioritize memory usage over
        # lookup speed by querying tile types directly rather than caching a
        # full map of bonuses.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < game_map.width and 0 <= ny < game_map.height:
                    tile_id = game_map.tiles[nx, ny]
                    tile_data = tile_types.get_tile_type_data_by_id(int(tile_id))
                    bonus = int(tile_data["cover_bonus"])
                    max_bonus = max(max_bonus, bonus)

        return max_bonus
