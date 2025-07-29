"""Sound definitions and data structures."""

from __future__ import annotations

from dataclasses import dataclass

from catley.types import SoundId


@dataclass
class SoundLayer:
    """A single layer of sound within a sound definition.

    Represents one audio file that can be played as part of a sound effect.
    Multiple layers can be combined to create complex sounds.
    """

    file: str  # Path to the sound file relative to assets/sounds/
    volume: float = 1.0  # Volume multiplier for this layer (0.0-1.0)
    loop: bool = True  # Whether this sound loops continuously
    interval: tuple[float, float] | None = (
        None  # Random interval for non-looping sounds
    )


@dataclass
class SoundDefinition:
    """Complete definition of a sound effect.

    Defines all properties of a sound including its layers, volume falloff,
    and distance parameters.
    """

    sound_id: SoundId  # Unique identifier for this sound
    layers: list[SoundLayer]  # All sound layers that make up this effect
    base_volume: float = 1.0  # Base volume at minimum distance
    falloff_start: float = 2.0  # Distance at which volume starts to fall off
    max_distance: float = 10.0  # Maximum distance at which sound is audible
    priority: int = 5  # Priority for sound culling (1-10, higher = more important)


# Built-in sound definitions
# In a real implementation, these would be loaded from a data file
SOUND_DEFINITIONS = {
    "fire_ambient": SoundDefinition(
        sound_id="fire_ambient",
        layers=[
            SoundLayer(file="fire_crackle_loop.ogg", volume=1.0, loop=True),
            SoundLayer(
                file="fire_pops.ogg",
                volume=0.3,
                loop=False,
                interval=(2.0, 8.0),  # Random pops every 2-8 seconds
            ),
        ],
        base_volume=0.7,
        falloff_start=2.0,
        max_distance=12.0,
        priority=6,
    ),
    "waterfall_ambient": SoundDefinition(
        sound_id="waterfall_ambient",
        layers=[SoundLayer(file="water_flow_loop.ogg", volume=1.0, loop=True)],
        base_volume=0.8,
        falloff_start=3.0,
        max_distance=15.0,
        priority=5,
    ),
    "engine_loop": SoundDefinition(
        sound_id="engine_loop",
        layers=[SoundLayer(file="engine_idle_loop.ogg", volume=1.0, loop=True)],
        base_volume=0.6,
        falloff_start=1.0,
        max_distance=8.0,
        priority=4,
    ),
    "steam_hiss": SoundDefinition(
        sound_id="steam_hiss",
        layers=[SoundLayer(file="steam_release_loop.ogg", volume=0.8, loop=True)],
        base_volume=0.5,
        falloff_start=1.5,
        max_distance=6.0,
        priority=3,
    ),
}


def get_sound_definition(sound_id: SoundId) -> SoundDefinition | None:
    """Get a sound definition by sound ID."""
    return SOUND_DEFINITIONS.get(sound_id)
