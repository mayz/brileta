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
    pitch_variation: tuple[float, float] | None = (
        None  # Random pitch variation range (e.g., (0.9, 1.1) for ±10%)
    )
    variants: list[str] | None = None  # List of alternative files for random variation
    delay: float = 0.0  # Delay in seconds before playing this layer


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
    rolloff_factor: float = (
        1.0  # How quickly sound attenuates with distance (0=none, 2=sharp)
    )


# Built-in sound definitions
# These should eventually be loaded from a data file
SOUND_DEFINITIONS = {
    "fire_ambient": SoundDefinition(
        sound_id="fire_ambient",
        layers=[
            SoundLayer(file="fire_crackle_loop.ogg", volume=1.0, loop=True),
            # Multiple pop layers for more organic randomness
            SoundLayer(
                file="fire_pops.ogg",
                volume=0.4,
                loop=False,
                interval=(1.5, 6.0),  # Primary pops
                pitch_variation=(0.8, 1.6),  # -20% to +60% pitch variation
            ),
            SoundLayer(
                file="fire_pops.ogg",
                volume=0.2,
                loop=False,
                interval=(3.0, 12.0),  # Secondary pops (quieter, less frequent)
                pitch_variation=(0.85, 1.3),  # -15% to +30% pitch variation
            ),
            SoundLayer(
                file="fire_pops.ogg",
                volume=0.15,
                loop=False,
                interval=(0.8, 4.0),  # Tertiary pops (very quiet, more frequent)
                pitch_variation=(0.9, 1.2),  # -10% to +20% for most variety
            ),
        ],
        base_volume=0.7,
        falloff_start=2.8,
        max_distance=17.0,
        priority=6,
        rolloff_factor=0.7,  # Gentle falloff for ambient sounds
    ),
    "waterfall_ambient": SoundDefinition(
        sound_id="waterfall_ambient",
        layers=[SoundLayer(file="water_flow_loop.ogg", volume=1.0, loop=True)],
        base_volume=0.8,
        falloff_start=3.0,
        max_distance=15.0,
        priority=5,
        rolloff_factor=0.6,  # Even gentler falloff for water sounds
    ),
    "engine_loop": SoundDefinition(
        sound_id="engine_loop",
        layers=[SoundLayer(file="engine_idle_loop.ogg", volume=1.0, loop=True)],
        base_volume=0.6,
        falloff_start=1.0,
        max_distance=8.0,
        priority=4,
        rolloff_factor=1.2,  # Sharper falloff for mechanical sounds
    ),
    "steam_hiss": SoundDefinition(
        sound_id="steam_hiss",
        layers=[SoundLayer(file="steam_release_loop.ogg", volume=0.8, loop=True)],
        base_volume=0.5,
        falloff_start=1.5,
        max_distance=6.0,
        priority=3,
        rolloff_factor=1.5,  # Sharp falloff for hissing sounds
    ),
    # Gunfire sounds - one per weapon type (Fallout 1/2 style)
    "gun_fire_pistol": SoundDefinition(
        sound_id="gun_fire_pistol",
        layers=[
            SoundLayer(
                file="gunfire_pistol_01.ogg",
                variants=["gunfire_pistol_02.ogg", "gunfire_pistol_03.ogg"],
                volume=1.0,
                loop=False,
                pitch_variation=(0.92, 1.08),
            ),
        ],
        base_volume=0.8,
        falloff_start=3.0,
        max_distance=18.0,
        priority=8,
        rolloff_factor=0.8,
    ),
    "gun_fire_rifle": SoundDefinition(
        sound_id="gun_fire_rifle",
        layers=[
            SoundLayer(
                file="gunfire_rifle_01.ogg",
                variants=["gunfire_rifle_02.ogg", "gunfire_rifle_03.ogg"],
                volume=1.0,
                loop=False,
                pitch_variation=(0.88, 1.05),
            ),
        ],
        base_volume=0.9,
        falloff_start=4.0,
        max_distance=25.0,
        priority=9,
        rolloff_factor=0.7,
    ),
    "gun_fire_shotgun": SoundDefinition(
        sound_id="gun_fire_shotgun",
        layers=[
            SoundLayer(
                file="gunfire_shotgun_01.ogg",
                variants=["gunfire_shotgun_02.ogg", "gunfire_shotgun_03.ogg"],
                volume=1.0,
                loop=False,
                pitch_variation=(0.85, 1.0),
                delay=0.0,  # Blast plays immediately
            ),
            SoundLayer(
                file="gunfire_shotgun_rack.ogg",
                volume=0.9,  # Slightly quieter than blast
                loop=False,
                pitch_variation=None,  # No pitch modulation on rack
                delay=0.5,  # Rack plays 500ms after blast starts
            ),
        ],
        base_volume=0.95,
        falloff_start=3.5,
        max_distance=22.0,
        priority=9,
        rolloff_factor=0.75,
    ),
    # Dry fire click - universal for all weapons
    "gun_dry_fire": SoundDefinition(
        sound_id="gun_dry_fire",
        layers=[
            SoundLayer(
                file="gunfire_dry_fire.ogg",
                volume=1.0,
                loop=False,
                pitch_variation=(0.95, 1.05),
            ),
        ],
        base_volume=0.6,
        falloff_start=1.5,
        max_distance=8.0,
        priority=7,
        rolloff_factor=1.0,
    ),
    # Shotgun reload - 5 shell inserts followed by pump rack
    # FIXME: This always plays 5 shell inserts regardless of how many shells are
    # actually being loaded (e.g., reloading 1 shell still plays 5 insert sounds).
    # To fix: generate sounds dynamically in ReloadExecutor based on
    # (max_ammo - current_ammo) shells needed.
    #
    # Important: when implementing dynamic reload sounds, the shell inserts can
    # have pitch variation, but the final pump rack sound must NOT have pitch
    # variation (it sounds unrealistic). Ensure the rack sound is emitted without
    # pitch_jitter - see system.py where delayed layers intentionally don't
    # inherit event pitch_jitter for this reason.
    "gun_reload_shotgun": SoundDefinition(
        sound_id="gun_reload_shotgun",
        layers=[
            # Shell inserts (5 shells, 400ms apart, subtle pitch variation)
            SoundLayer(
                file="shotgun_shell_insert.ogg",
                volume=0.9,
                loop=False,
                delay=0.0,
                pitch_variation=(0.97, 1.03),
            ),
            SoundLayer(
                file="shotgun_shell_insert.ogg",
                volume=0.9,
                loop=False,
                delay=0.4,
                pitch_variation=(0.97, 1.03),
            ),
            SoundLayer(
                file="shotgun_shell_insert.ogg",
                volume=0.9,
                loop=False,
                delay=0.8,
                pitch_variation=(0.97, 1.03),
            ),
            SoundLayer(
                file="shotgun_shell_insert.ogg",
                volume=0.9,
                loop=False,
                delay=1.2,
                pitch_variation=(0.97, 1.03),
            ),
            SoundLayer(
                file="shotgun_shell_insert.ogg",
                volume=0.9,
                loop=False,
                delay=1.6,
                pitch_variation=(0.97, 1.03),
            ),
            # Pump rack at the end (no pitch variation)
            SoundLayer(
                file="gunfire_shotgun_rack.ogg",
                volume=0.9,
                loop=False,
                delay=2.1,
            ),
        ],
        base_volume=0.8,
        falloff_start=2.0,
        max_distance=12.0,
        priority=7,
        rolloff_factor=1.0,
    ),
}


def get_sound_definition(sound_id: SoundId) -> SoundDefinition | None:
    """Get a sound definition by sound ID."""
    return SOUND_DEFINITIONS.get(sound_id)
