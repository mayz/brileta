"""Environmental actors like fires, torches, and other interactive world objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta import colors
from brileta.config import DEFAULT_ACTOR_SPEED
from brileta.game.actions.environmental import EnvironmentalDamageIntent
from brileta.game.actors.components import EnergyComponent, VisualEffectsComponent
from brileta.game.actors.core import Actor
from brileta.game.lights import DynamicLight
from brileta.sound.emitter import SoundEmitter
from brileta.types import WorldTileCoord
from brileta.view.render.effects.effects import FireEffect

if TYPE_CHECKING:
    from brileta.controller import Controller
    from brileta.game.game_world import GameWorld


class ContainedFire(Actor):
    """A fire object that burns continuously with particle effects and dynamic lighting.

    Examples: campfires, barrel fires, torches. These are specialized actors that
    emit particles for visual effect and provide illumination through dynamic lighting.
    """

    def __init__(
        self,
        x: WorldTileCoord,
        y: WorldTileCoord,
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
            energy=EnergyComponent(speed=DEFAULT_ACTOR_SPEED),
            shadow_height=0,  # Light sources don't cast projected shadows.
        )

        self.damage_per_turn = damage_per_turn

        # Type narrowing - energy is always set for ContainedFire.
        self.energy: EnergyComponent

        # Add the fire effect
        self.fire_effect = FireEffect()
        if self.visual_effects is not None:
            self.visual_effects.add_continuous_effect(self.fire_effect)

        # Mark as having complex visuals so outline rendering uses full-tile outline
        self.has_complex_visuals = True

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

        # Add fire crackling sound
        self.add_sound_emitter(SoundEmitter("fire_ambient"))

    def get_next_action(
        self, controller: Controller
    ) -> EnvironmentalDamageIntent | None:
        """Return environmental damage intent to deal fire damage."""
        return EnvironmentalDamageIntent(
            controller=controller,
            source_actor=self,
            damage_amount=self.damage_per_turn,
            damage_type="fire",
            affected_coords=[(self.x, self.y)],
            source_description=self.name.lower(),
        )

    def update_turn(self, controller: Controller) -> None:
        """Update the fire each turn."""
        super().update_turn(controller)

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
