"""Environmental actors like fires, torches, and other interactive world objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from catley import colors
from catley.game.actions.base import GameIntent
from catley.game.actors.components import VisualEffectsComponent
from catley.game.actors.core import Actor
from catley.game.lights import DynamicLight
from catley.view.render.effects.effects import FireEffect

if TYPE_CHECKING:
    from catley.controller import Controller
    from catley.game.game_world import GameWorld


class FireTickIntent(GameIntent):
    """A no-op intent that allows fires to participate in the turn system for damage."""

    def __init__(self, controller: Controller, actor: Actor) -> None:
        super().__init__(controller, actor)


class ContainedFire(Actor):
    """A fire object that burns continuously with particle effects and dynamic lighting.

    Examples: campfires, barrel fires, torches. These are specialized actors that
    emit particles for visual effect and provide illumination through dynamic lighting.
    """

    def __init__(
        self,
        x: int,
        y: int,
        ch: str,
        color: colors.Color,
        name: str = "Fire",
        game_world: GameWorld | None = None,
        light_radius: int = 5,
        light_color: colors.Color = (255, 150, 50),  # Orange
        damage_per_turn: int = 5,
    ) -> None:
        """Initialize a contained fire.

        Args:
            x, y: Position in the world
            ch: Character to display (e.g. 'Ω' for campfire)
            color: Display color of the fire actor
            name: Name of the fire object
            game_world: Game world reference
            light_radius: Radius of light emission
            light_color: Color of the emitted light
            damage_per_turn: Damage dealt to actors on the same tile
        """
        # Create visual effects component
        visual_effects = VisualEffectsComponent()

        super().__init__(
            x=x,
            y=y,
            ch=ch,
            color=color,
            name=name,
            game_world=game_world,
            blocks_movement=False,  # Can walk through fire (but take damage)
            visual_effects=visual_effects,
        )

        self.damage_per_turn = damage_per_turn

        # Add the fire effect
        self.fire_effect = FireEffect()
        if self.visual_effects is not None:
            self.visual_effects.add_continuous_effect(self.fire_effect)

        # Create dynamic light for this fire
        if game_world:
            self.light_source = DynamicLight(
                position=(x, y),
                radius=light_radius,
                color=light_color,
                flicker_enabled=True,
                flicker_speed=3.0,
                min_brightness=0.7,
                max_brightness=1.0,
                owner=self,
            )
            game_world.add_light(self.light_source)

    def get_next_action(self, controller: Controller) -> FireTickIntent | None:
        """Return a no-op action to allow fire to participate in turn system."""
        return FireTickIntent(controller, self)

    def update_turn(self, controller: Controller) -> None:
        """Update the fire each turn - damage actors on the same tile."""
        super().update_turn(controller)

        # Damage actors on the same tile
        if self.gw:
            actors_here = self.gw.actor_spatial_index.get_at_point(self.x, self.y)
            for actor in actors_here:
                if actor is not self and hasattr(actor, "take_damage"):
                    actor.take_damage(self.damage_per_turn, damage_type="fire")

    @staticmethod
    def create_campfire(
        x: int, y: int, game_world: GameWorld | None = None
    ) -> ContainedFire:
        """Create a standard campfire."""
        return ContainedFire(
            x=x,
            y=y,
            ch=".",  # Small dot to indicate something is here
            color=(100, 50, 0),  # Dark brown (subtle)
            name="Campfire",
            game_world=game_world,
            light_radius=6,
            light_color=(255, 180, 80),  # Warm orange light
        )

    @staticmethod
    def create_barrel_fire(
        x: int, y: int, game_world: GameWorld | None = None
    ) -> ContainedFire:
        """Create a barrel fire."""
        return ContainedFire(
            x=x,
            y=y,
            ch="o",  # Small circle for barrel base
            color=(80, 40, 20),  # Dark brown (subtle)
            name="Burning Barrel",
            game_world=game_world,
            light_radius=4,
            light_color=(255, 140, 40),  # Slightly dimmer orange
        )

    @staticmethod
    def create_torch(
        x: int, y: int, game_world: GameWorld | None = None
    ) -> ContainedFire:
        """Create a wall-mounted torch."""
        return ContainedFire(
            x=x,
            y=y,
            ch="|",  # Vertical line for torch post
            color=(60, 30, 15),  # Dark brown (subtle)
            name="Torch",
            game_world=game_world,
            light_radius=3,
            light_color=(255, 200, 100),  # Yellower light
            damage_per_turn=3,  # Less damage than full fires
        )
