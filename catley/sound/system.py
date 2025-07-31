"""Main sound system that manages all sound playback and processing."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from catley.types import WorldTileCoord

from .audio_backend import AudioBackend, AudioChannel, LoadedSound
from .definitions import SoundDefinition, SoundLayer, get_sound_definition

if TYPE_CHECKING:
    from catley.game.actors.core import Actor
    from catley.sound.emitter import SoundEmitter

logger = logging.getLogger(__name__)


@dataclass
class PlayingSound:
    """Represents an actively playing sound instance."""

    emitter: SoundEmitter
    layer: SoundLayer
    channel: AudioChannel | None = None  # Audio backend channel
    next_trigger_time: float = 0.0  # For interval-based sounds
    loaded_sound: LoadedSound | None = None  # Cached sound data


class SoundSystem:
    """Manages all sound playback, distance calculations, and audio mixing.

    The SoundSystem is responsible for:
    - Collecting all active sound emitters from actors
    - Calculating distance-based volume for each sound
    - Managing sound playback through the audio backend
    - Handling sound instance pooling and priority
    """

    def __init__(self, max_concurrent_sounds: int = 16) -> None:
        """Initialize the sound system.

        Args:
            max_concurrent_sounds: Maximum number of sounds that can play at once
        """
        self.max_concurrent_sounds = max_concurrent_sounds
        self.playing_sounds: list[PlayingSound] = []
        self.audio_backend: AudioBackend | None = None
        self.current_time: float = 0.0
        self._sound_cache: dict[str, LoadedSound] = {}  # Cache loaded sounds
        self._assets_path: Path | None = None

        # Audio listener interpolation for smooth volume transitions
        self.audio_listener_x: WorldTileCoord = 0.0
        self.audio_listener_y: WorldTileCoord = 0.0
        self.transition_duration: float = 1.0  # seconds for transition
        self._listener_initialized: bool = False

    def update(
        self,
        listener_x: WorldTileCoord,
        listener_y: WorldTileCoord,
        actors: list[Actor],
        delta_time: float,
    ) -> None:
        """Update all sounds based on listener position and active emitters.

        Args:
            listener_x: X position of the listener (usually the player)
            listener_y: Y position of the listener
            actors: All actors in the game world
            delta_time: Time elapsed since last update in seconds
        """
        self.current_time += delta_time

        # Initialize or interpolate audio listener position
        if not self._listener_initialized:
            # First update - set audio listener to actual player position
            self.audio_listener_x = listener_x
            self.audio_listener_y = listener_y
            self._listener_initialized = True
        else:
            # Interpolate audio listener toward actual player position
            if self.transition_duration > 0:
                # Calculate distance to target
                dx = listener_x - self.audio_listener_x
                dy = listener_y - self.audio_listener_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance > 0.01:  # Small threshold to avoid division by zero
                    # Calculate speed needed to complete transition in fixed duration
                    required_speed = distance / self.transition_duration
                    max_movement = required_speed * delta_time

                    if distance > max_movement:
                        # Move toward target at calculated speed
                        ratio = max_movement / distance
                        self.audio_listener_x += dx * ratio
                        self.audio_listener_y += dy * ratio
                    else:
                        # Close enough - snap to target
                        self.audio_listener_x = listener_x
                        self.audio_listener_y = listener_y
                else:
                    # Already at target
                    self.audio_listener_x = listener_x
                    self.audio_listener_y = listener_y
            else:
                # No transition - instant movement
                self.audio_listener_x = listener_x
                self.audio_listener_y = listener_y

        # Collect all active emitters with their positions
        active_emitters: list[tuple[SoundEmitter, WorldTileCoord, WorldTileCoord]] = []
        for actor in actors:
            # Handle actors without sound_emitters attribute (e.g., test actors)
            if hasattr(actor, "sound_emitters") and actor.sound_emitters:
                active_emitters.extend(
                    (emitter, actor.x, actor.y)
                    for emitter in actor.sound_emitters
                    if emitter.active
                )

        if active_emitters and self.audio_backend:
            logger.debug(
                f"SoundSystem update: {len(active_emitters)} active emitters, "
                f"listener at ({listener_x}, {listener_y})"
            )

        # Keep track of which emitters are being processed
        processed_emitters = set()

        # Calculate volumes and update playing sounds
        for emitter, emitter_x, emitter_y in active_emitters:
            processed_emitters.add(emitter)
            sound_def = get_sound_definition(emitter.sound_id)
            if not sound_def:
                logger.warning(f"No sound definition for: {emitter.sound_id}")
                continue

            # Calculate distance-based volume
            # Use interpolated listener for continuous sounds, actual for one-off
            use_interpolated = self._sound_has_continuous_layers(sound_def)
            volume = self._calculate_volume(
                emitter_x,
                emitter_y,
                listener_x,
                listener_y,
                sound_def,
                emitter.volume_multiplier,
                use_interpolated_listener=use_interpolated,
            )

            if volume > 0.0:
                logger.debug(
                    f"Sound {emitter.sound_id} at ({emitter_x}, {emitter_y}), "
                    f"distance from player ({listener_x}, {listener_y}), "
                    f"volume: {volume:.3f}"
                )
                self._ensure_sound_playing(emitter, sound_def, volume)
            else:
                self._stop_sound_if_playing(emitter)

        # Stop sounds for any emitters that are no longer active or out of range
        all_emitters = []
        for actor in actors:
            if hasattr(actor, "sound_emitters") and actor.sound_emitters:
                all_emitters.extend(actor.sound_emitters)

        for emitter in all_emitters:
            if emitter not in processed_emitters and emitter.playing_instances:
                self._stop_sound_if_playing(emitter)

        # Update audio backend volume for all playing sounds
        if self.audio_backend:
            for playing in self.playing_sounds:
                if playing.channel and playing.channel.is_playing():
                    # Find current volume for this emitter
                    emitter_x = emitter_y = 0
                    for actor in actors:
                        if (
                            hasattr(actor, "sound_emitters")
                            and actor.sound_emitters
                            and playing.emitter in actor.sound_emitters
                        ):
                            emitter_x, emitter_y = actor.x, actor.y
                            break

                    sound_def = get_sound_definition(playing.emitter.sound_id)
                    if sound_def:
                        # Use interpolated listener for looping sounds, actual for one
                        use_interpolated = playing.layer.loop
                        volume = self._calculate_volume(
                            emitter_x,
                            emitter_y,
                            listener_x,
                            listener_y,
                            sound_def,
                            playing.emitter.volume_multiplier,
                            use_interpolated_listener=use_interpolated,
                        )
                        playing.channel.set_volume(volume * playing.layer.volume)

        # Clean up finished sounds
        self._cleanup_finished_sounds()

        # Update audio backend
        if self.audio_backend:
            self.audio_backend.update()

    def _calculate_volume(
        self,
        emitter_x: WorldTileCoord,
        emitter_y: WorldTileCoord,
        listener_x: WorldTileCoord,
        listener_y: WorldTileCoord,
        sound_def: SoundDefinition,
        volume_multiplier: float,
        use_interpolated_listener: bool = True,
    ) -> float:
        """Calculate volume based on distance using inverse square law.

        Args:
            emitter_x: X position of the sound source
            emitter_y: Y position of the sound source
            listener_x: X position of the listener (actual player position)
            listener_y: Y position of the listener (actual player position)
            sound_def: Sound definition with falloff parameters
            volume_multiplier: Emitter-specific volume adjustment
            use_interpolated_listener: If True, use interpolated audio listener position
                                     If False, use actual listener position

        Returns:
            Final volume (0.0 to 1.0)
        """
        # Use appropriate listener position based on flag
        if use_interpolated_listener:
            effective_listener_x = self.audio_listener_x
            effective_listener_y = self.audio_listener_y
        else:
            effective_listener_x = listener_x
            effective_listener_y = listener_y

        # Calculate Euclidean distance
        dx = emitter_x - effective_listener_x
        dy = emitter_y - effective_listener_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Beyond max distance, no sound
        if distance > sound_def.max_distance:
            return 0.0

        # Within falloff start distance, full volume
        if distance <= sound_def.falloff_start:
            return sound_def.base_volume * volume_multiplier

        # Apply inverse square law for falloff
        falloff_distance = distance - sound_def.falloff_start
        # Add small constant to prevent division by zero and extreme volume near source
        volume_factor = 1.0 / (1.0 + falloff_distance**2)

        return sound_def.base_volume * volume_multiplier * volume_factor

    def _sound_has_continuous_layers(self, sound_def: SoundDefinition) -> bool:
        """Check if a sound definition has any continuous (looping) layers.

        Args:
            sound_def: The sound definition to check

        Returns:
            True if the sound has looping layers, False otherwise
        """
        return any(layer.loop for layer in sound_def.layers)

    def _ensure_sound_playing(
        self, emitter: SoundEmitter, sound_def: SoundDefinition, volume: float
    ) -> None:
        """Ensure a sound is playing for the given emitter.

        Args:
            emitter: The sound emitter
            sound_def: The sound definition
            volume: The calculated volume
        """
        # For now, just track that we want this sound playing
        # Full implementation will handle the audio backend

        # Check each layer in the sound definition
        for layer in sound_def.layers:
            # Find or create playing instance for this layer
            playing = None
            for p in emitter.playing_instances:
                if p.layer == layer:
                    playing = p
                    break

            if layer.loop:
                # Looping sounds should always have an instance
                if not playing:
                    playing = PlayingSound(emitter=emitter, layer=layer)

                    # Start playing through audio backend
                    if self.audio_backend:
                        loaded_sound = self._load_sound(layer.file)
                        if loaded_sound:
                            channel = self.audio_backend.play_sound(
                                loaded_sound, volume=volume * layer.volume, loop=True
                            )
                            if channel:
                                playing.channel = channel
                                playing.loaded_sound = loaded_sound
                                logger.debug(
                                    f"Started loop: {sound_def.sound_id} - "
                                    f"{layer.file} at volume "
                                    f"{volume * layer.volume:.2f}"
                                )
                            else:
                                logger.warning(
                                    f"No channels available for: {layer.file}"
                                )

                    emitter.playing_instances.append(playing)
                    self.playing_sounds.append(playing)
            else:
                # Interval-based sounds
                if layer.interval:
                    if not playing:
                        # Create new instance and schedule first play
                        playing = PlayingSound(emitter=emitter, layer=layer)
                        playing.next_trigger_time = self.current_time + random.uniform(
                            *layer.interval
                        )
                        emitter.playing_instances.append(playing)
                        self.playing_sounds.append(playing)
                    elif self.current_time >= playing.next_trigger_time:
                        # Time to play this sound again
                        playing.next_trigger_time = self.current_time + random.uniform(
                            *layer.interval
                        )

                        # Play one-shot through audio backend
                        if self.audio_backend:
                            loaded_sound = playing.loaded_sound or self._load_sound(
                                layer.file
                            )
                            if loaded_sound:
                                # Add volume randomization for more natural variation
                                # Vary volume by ±20% for more organic sound
                                volume_variation = random.uniform(0.8, 1.2)
                                final_volume = volume * layer.volume * volume_variation

                                channel = self.audio_backend.play_sound(
                                    loaded_sound,
                                    volume=final_volume,
                                    loop=False,
                                )
                                if channel:
                                    # Store the channel for volume updates
                                    playing.channel = channel
                                    logger.debug(
                                        f"Triggered one-shot: "
                                        f"{sound_def.sound_id} - "
                                        f"{layer.file} at volume "
                                        f"{final_volume:.2f}"
                                    )
                                playing.loaded_sound = loaded_sound

    def _stop_sound_if_playing(self, emitter: SoundEmitter) -> None:
        """Stop all sounds for the given emitter.

        Args:
            emitter: The sound emitter to stop
        """
        for playing in emitter.playing_instances[:]:
            # Stop through audio backend
            if playing.channel:
                playing.channel.stop()
                logger.debug(f"Stopped: {playing.layer.file}")

            emitter.playing_instances.remove(playing)
            if playing in self.playing_sounds:
                self.playing_sounds.remove(playing)

    def _cleanup_finished_sounds(self) -> None:
        """Remove finished one-shot sounds from tracking."""
        if not self.audio_backend:
            return

        # Remove sounds that are no longer playing
        finished = [
            playing
            for playing in self.playing_sounds
            if (
                playing.channel
                and not playing.channel.is_playing()
                and not playing.layer.loop
            )
        ]

        for playing in finished:
            if playing in playing.emitter.playing_instances:
                playing.emitter.playing_instances.remove(playing)
            if playing in self.playing_sounds:
                self.playing_sounds.remove(playing)
            logger.debug(f"Cleaned up finished sound: {playing.layer.file}")

    def _load_sound(self, file_name: str) -> LoadedSound | None:
        """Load a sound file, using cache if available.

        Args:
            file_name: Name of the sound file

        Returns:
            LoadedSound object or None if loading fails
        """
        if not self.audio_backend:
            return None

        # Check cache first
        if file_name in self._sound_cache:
            return self._sound_cache[file_name]

        # Determine full path
        if self._assets_path:
            file_path = self._assets_path / "sounds" / file_name
        else:
            file_path = Path("assets/sounds") / file_name

        try:
            loaded_sound = self.audio_backend.load_sound(file_path)
            self._sound_cache[file_name] = loaded_sound
            logger.info(f"Loaded sound: {file_name}")
            return loaded_sound
        except Exception as e:
            logger.error(f"Failed to load sound {file_name}: {e}")
            return None

    def set_audio_backend(self, backend: AudioBackend | None) -> None:
        """Set the audio backend for sound playback.

        Args:
            backend: The audio backend to use, or None to disable audio
        """
        # Stop all sounds if changing backend
        if self.audio_backend:
            for playing in self.playing_sounds:
                if playing.channel:
                    playing.channel.stop()

        self.audio_backend = backend
        self.playing_sounds.clear()
        self._sound_cache.clear()

        # Clear playing instances from all emitters
        # This will be repopulated on next update
        for playing in self.playing_sounds:
            playing.emitter.playing_instances.clear()

        logger.info(
            f"Audio backend set: {type(backend).__name__ if backend else 'None'}"
        )

    def set_assets_path(self, assets_path: Path) -> None:
        """Set the base path for audio assets.

        Args:
            assets_path: Base path for assets
        """
        self._assets_path = assets_path
        logger.debug(f"Assets path set: {assets_path}")

    def set_transition_duration(self, duration: float) -> None:
        """Set the audio listener transition duration.

        Args:
            duration: Transition duration in seconds. Set to 0 for instant transitions.
        """
        self.transition_duration = max(0.0, duration)
        logger.debug(f"Audio listener transition duration set: {duration} seconds")

    def set_master_volume(self, volume: float) -> None:
        """Set the master volume for all audio.

        Args:
            volume: Master volume level (0.0 to 1.0)
        """
        if self.audio_backend:
            self.audio_backend.set_master_volume(volume)
