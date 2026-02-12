"""Main sound system that manages all sound playback and processing."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from brileta.events import SoundEvent, subscribe_to_event
from brileta.types import DeltaTime, FloatRange, WorldTileCoord
from brileta.util import rng
from brileta.util.spatial import SpatialIndex

from .audio_backend import AudioBackend, AudioChannel, LoadedSound
from .definitions import (
    SOUND_DEFINITIONS,
    SoundDefinition,
    SoundLayer,
    get_sound_definition,
)
from .loader import AudioLoader

if TYPE_CHECKING:
    from brileta.game.actors.core import Actor
    from brileta.sound.emitter import SoundEmitter

logger = logging.getLogger(__name__)

_rng = rng.get("audio.system")


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
        self._assets_path: Path | None = None
        self._max_audio_distance: float | None = None  # Cached maximum audio distance

        # Audio listener position (interpolated world coordinates for smooth volume
        # transitions - floats rather than WorldTileCoord since they're smoothed)
        self.audio_listener_x: float = 0.0
        self.audio_listener_y: float = 0.0
        self.transition_duration: float = 1.0  # seconds for transition
        self._listener_initialized: bool = False

        # Velocity-based smoothing parameters
        self.listener_smoothing_factor: float = 0.15  # 0-1, higher = less smoothing
        self.teleport_threshold: float = 10.0  # Distance to consider as teleportation
        self.previous_listener_x: float = 0.0
        self.previous_listener_y: float = 0.0

        # Track last played variant per sound_id to avoid repeats
        self._last_played_variant: dict[str, str] = {}

        # Track delayed sounds as (play_at_time, event) tuples
        self._delayed_sounds: list[tuple[float, SoundEvent]] = []

        # Subscribe to sound events
        subscribe_to_event(SoundEvent, self.handle_sound_event)

    def _calculate_max_audio_distance(self) -> float:
        """Calculate the maximum audible distance across all sound definitions.

        Returns:
            Maximum distance at which any sound can be heard, plus a buffer.
        """
        if self._max_audio_distance is not None:
            return self._max_audio_distance

        max_dist = 0.0
        for sound_def in SOUND_DEFINITIONS.values():
            max_dist = max(max_dist, sound_def.max_distance)

        # Add buffer for smooth transitions (currently max is 17.0 for campfire)
        self._max_audio_distance = max_dist + 5.0  # ~22.0 tiles
        return self._max_audio_distance

    def update(
        self,
        listener_x: WorldTileCoord,
        listener_y: WorldTileCoord,
        actor_spatial_index: SpatialIndex[Actor],
        delta_time: DeltaTime,
    ) -> None:
        """Update all sounds based on listener position and active emitters.

        Args:
            listener_x: X position of the listener (usually the player)
            listener_y: Y position of the listener
            actor_spatial_index: Spatial index of all actors in the game world
            delta_time: Time elapsed since last update in seconds
        """
        self.current_time += delta_time

        # Initialize or interpolate audio listener position
        if not self._listener_initialized:
            # First update - set audio listener to actual player position
            self.audio_listener_x = listener_x
            self.audio_listener_y = listener_y
            self.previous_listener_x = listener_x
            self.previous_listener_y = listener_y
            self._listener_initialized = True
        else:
            # Detect teleportation
            movement_distance = math.sqrt(
                (listener_x - self.previous_listener_x) ** 2
                + (listener_y - self.previous_listener_y) ** 2
            )

            if movement_distance > self.teleport_threshold:
                # Instant update for teleportation
                self.audio_listener_x = listener_x
                self.audio_listener_y = listener_y
            else:
                # Smooth interpolation using exponential smoothing
                alpha = 1.0 - math.exp(-delta_time * 5.0)  # 5.0 = smoothing speed
                self.audio_listener_x += (listener_x - self.audio_listener_x) * alpha
                self.audio_listener_y += (listener_y - self.audio_listener_y) * alpha

            # Update previous position for next frame
            self.previous_listener_x = listener_x
            self.previous_listener_y = listener_y

        # Query only actors within maximum audio distance
        audio_cull_distance = self._calculate_max_audio_distance()
        nearby_actors = actor_spatial_index.get_in_radius(
            int(self.audio_listener_x),  # Use interpolated position
            int(self.audio_listener_y),
            radius=int(audio_cull_distance),
        )

        # Collect active emitters from nearby actors only
        # Also build a position lookup dict for volume updates later
        active_emitters: list[tuple[SoundEmitter, WorldTileCoord, WorldTileCoord]] = []
        emitter_positions: dict[
            SoundEmitter, tuple[WorldTileCoord, WorldTileCoord]
        ] = {}
        for actor in nearby_actors:
            if hasattr(actor, "sound_emitters") and actor.sound_emitters:
                # Additional distance check with actual Euclidean distance
                dx = actor.x - self.audio_listener_x
                dy = actor.y - self.audio_listener_y
                if dx * dx + dy * dy <= audio_cull_distance * audio_cull_distance:
                    for emitter in actor.sound_emitters:
                        if emitter.active:
                            active_emitters.append((emitter, actor.x, actor.y))
                            emitter_positions[emitter] = (actor.x, actor.y)

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
            volume = self._calculate_volume(
                emitter_x,
                emitter_y,
                sound_def,
                emitter.volume_multiplier,
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
        # Check all playing sounds and stop those whose emitters were not processed
        # (either because they're out of range or inactive)
        for playing in self.playing_sounds[
            :
        ]:  # Use slice to avoid modification during iteration
            if playing.emitter not in processed_emitters:
                # This emitter is either out of range or inactive
                self._stop_sound_if_playing(playing.emitter)

        # Update audio backend volume for all playing sounds
        if self.audio_backend:
            for playing in self.playing_sounds:
                if playing.channel and playing.channel.is_playing():
                    # Look up emitter position from the dict we built earlier
                    # Skip volume update if emitter is out of range (not in dict)
                    position = emitter_positions.get(playing.emitter)
                    if position is None:
                        # Emitter's actor is out of range - skip update.
                        # Sound will be stopped by processed_emitters check.
                        continue

                    emitter_x, emitter_y = position
                    sound_def = get_sound_definition(playing.emitter.sound_id)
                    if sound_def:
                        volume = self._calculate_volume(
                            emitter_x,
                            emitter_y,
                            sound_def,
                            playing.emitter.volume_multiplier,
                        )

                        playing.channel.set_volume(volume * playing.layer.volume)

        # Clean up finished sounds
        self._cleanup_finished_sounds()

        # Process any delayed sounds that are now due
        self._process_delayed_sounds()

        # Update audio backend
        if self.audio_backend:
            self.audio_backend.update()

    def _calculate_volume(
        self,
        emitter_x: WorldTileCoord,
        emitter_y: WorldTileCoord,
        sound_def: SoundDefinition,
        volume_multiplier: float,
    ) -> float:
        """Calculate volume based on distance using inverse square law.

        Args:
            emitter_x: X position of the sound source
            emitter_y: Y position of the sound source
            sound_def: Sound definition with falloff parameters
            volume_multiplier: Emitter-specific volume adjustment

        Returns:
            Final volume (0.0 to 1.0)
        """
        # Always use interpolated listener position for smooth audio transitions
        effective_listener_x = self.audio_listener_x
        effective_listener_y = self.audio_listener_y

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

        # Apply industry-standard distance attenuation model
        falloff_distance = distance - sound_def.falloff_start
        reference_distance = sound_def.falloff_start
        rolloff_factor = sound_def.rolloff_factor

        # Industry-standard rolloff formula used by Unity, Unreal, FMOD, Wwise
        # This provides predictable, tunable falloff that designers understand
        if falloff_distance <= 0:
            volume_factor = 1.0
        else:
            # Defensive check: ensure we don't divide by zero
            # This should never happen with valid sound definitions, but better safe
            denominator = reference_distance + rolloff_factor * falloff_distance
            if denominator <= 0:
                volume_factor = 0.0  # Fallback to silence if invalid configuration
            else:
                volume_factor = reference_distance / denominator

        return sound_def.base_volume * volume_multiplier * volume_factor

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
                        playing.next_trigger_time = self.current_time + _rng.uniform(
                            *layer.interval
                        )
                        emitter.playing_instances.append(playing)
                        self.playing_sounds.append(playing)
                    elif self.current_time >= playing.next_trigger_time:
                        # Time to play this sound again
                        playing.next_trigger_time = self.current_time + _rng.uniform(
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
                                volume_variation = _rng.uniform(0.8, 1.2)
                                final_volume = volume * layer.volume * volume_variation

                                # Apply pitch variation if specified
                                sound_to_play = loaded_sound
                                if layer.pitch_variation:
                                    pitch_factor = _rng.uniform(*layer.pitch_variation)
                                    sound_to_play = AudioLoader.pitch_shift(
                                        loaded_sound, pitch_factor
                                    )

                                channel = self.audio_backend.play_sound(
                                    sound_to_play,
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

    def _process_delayed_sounds(self) -> None:
        """Process delayed sounds that are due to play.

        Checks all scheduled sounds and plays any whose delay has expired.
        """
        if not self._delayed_sounds:
            return

        # Separate sounds that are due from those still waiting
        due_sounds = [(t, e) for t, e in self._delayed_sounds if t <= self.current_time]
        self._delayed_sounds = [
            (t, e) for t, e in self._delayed_sounds if t > self.current_time
        ]

        # Play each due sound
        for _, event in due_sounds:
            self.play_one_shot(
                sound_id=event.sound_id,
                x=event.x,
                y=event.y,
                layer_index=event.layer,
                volume_jitter=event.volume_jitter,
                pitch_jitter=event.pitch_jitter,
                params=event.params,
            )

    def _load_sound(self, file_name: str) -> LoadedSound | None:
        """Load a sound file via the audio backend.

        The backend's AudioLoader handles caching internally, so repeated
        calls for the same file are O(1) after the first decode.

        Args:
            file_name: Name of the sound file

        Returns:
            LoadedSound object or None if loading fails
        """
        if not self.audio_backend:
            return None

        if self._assets_path:
            file_path = self._assets_path / "sounds" / file_name
        else:
            file_path = Path("assets/sounds") / file_name

        try:
            return self.audio_backend.load_sound(file_path)
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

        # Clear playing instances from all emitters
        # This will be repopulated on next update
        for playing in self.playing_sounds:
            playing.emitter.playing_instances.clear()

        self.audio_backend = backend
        self.playing_sounds.clear()

        logger.info(
            f"Audio backend set: {type(backend).__name__ if backend else 'None'}"
        )

    def reset(self) -> None:
        """Reset the sound system state for a new game or save load.

        This clears cached sounds and playing instances without changing
        the audio backend. Call this when starting a new game or loading
        a save to prevent stale audio data from accumulating.
        """
        # Stop all currently playing sounds
        for playing in self.playing_sounds:
            if playing.channel:
                playing.channel.stop()

        # Clear playing instances from all emitters
        for playing in self.playing_sounds:
            playing.emitter.playing_instances.clear()

        # Reset state
        self.playing_sounds.clear()
        self._last_played_variant.clear()
        self._delayed_sounds.clear()
        self.current_time = 0.0

        # Reset listener position so it re-initializes on next update
        self._listener_initialized = False
        self.audio_listener_x = 0.0
        self.audio_listener_y = 0.0
        self.previous_listener_x = 0.0
        self.previous_listener_y = 0.0

        logger.info("Sound system reset")

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

    def handle_sound_event(self, event: SoundEvent) -> None:
        """Handle a sound event from the event bus.

        Args:
            event: The sound event to handle
        """
        if event.delay > 0:
            # Schedule sound to play after delay
            play_at = self.current_time + event.delay
            self._delayed_sounds.append((play_at, event))
        else:
            self.play_one_shot(
                sound_id=event.sound_id,
                x=event.x,
                y=event.y,
                layer_index=event.layer,
                volume_jitter=event.volume_jitter,
                pitch_jitter=event.pitch_jitter,
                params=event.params,
            )

    def _get_repeat_count(
        self,
        layer: SoundLayer,
        params: dict[str, int | float] | None,
    ) -> int:
        """Get the effective repeat count for a layer.

        Args:
            layer: The sound layer to get repeat count for.
            params: Runtime parameters from the SoundEvent.

        Returns:
            Number of times to play this layer (minimum 1).
        """
        # Parameter takes precedence if set and available
        if layer.repeat_count_param and params:
            param_value = params.get(layer.repeat_count_param)
            if param_value is not None:
                return max(1, int(param_value))
            logger.warning(
                f"Sound param '{layer.repeat_count_param}' not found, "
                f"falling back to repeat_count={layer.repeat_count}"
            )

        # Fall back to static repeat_count, or 1 if not set
        if layer.repeat_count is not None:
            return max(1, layer.repeat_count)

        return 1

    def _play_layer_instance(
        self,
        layer: SoundLayer,
        sound_id: str,
        volume: float,
        volume_jitter: FloatRange | None,
        pitch_jitter: FloatRange | None,
    ) -> None:
        """Play a single instance of a sound layer.

        Args:
            layer: The sound layer to play.
            sound_id: ID of the parent sound (for variant tracking).
            volume: Base volume after distance calculation.
            volume_jitter: Optional volume variation (min, max).
            pitch_jitter: Optional pitch variation (min, max).
        """
        if not self.audio_backend:
            return

        # Determine which file to play (handle variants)
        all_files = [layer.file]
        if layer.variants:
            all_files.extend(layer.variants)

        # Avoid playing the same variant twice in a row
        last_played = self._last_played_variant.get(sound_id)
        if last_played in all_files and len(all_files) > 1:
            available_files = [f for f in all_files if f != last_played]
        else:
            available_files = all_files

        file_to_play = _rng.choice(available_files)
        self._last_played_variant[sound_id] = file_to_play

        loaded_sound = self._load_sound(file_to_play)
        if not loaded_sound:
            return

        # Calculate final volume with jitter
        layer_volume = volume * layer.volume
        if volume_jitter:
            layer_volume *= _rng.uniform(*volume_jitter)

        # Apply pitch variation
        sound_to_play = loaded_sound

        # Combine layer pitch variation with event pitch jitter
        pitch_factor = 1.0
        if layer.pitch_variation:
            pitch_factor *= _rng.uniform(*layer.pitch_variation)
        if pitch_jitter:
            pitch_factor *= _rng.uniform(*pitch_jitter)

        if pitch_factor != 1.0:
            sound_to_play = AudioLoader.pitch_shift(loaded_sound, pitch_factor)

        self.audio_backend.play_sound(
            sound_to_play,
            volume=layer_volume,
            loop=False,
        )

        logger.debug(
            f"Played one-shot: {sound_id} ({file_to_play}), "
            f"vol={layer_volume:.2f}, pitch={pitch_factor:.2f}"
        )

    def play_one_shot(
        self,
        sound_id: str,
        x: int,
        y: int,
        layer_index: int | None = None,
        volume_jitter: FloatRange | None = None,
        pitch_jitter: FloatRange | None = None,
        params: dict[str, int | float] | None = None,
    ) -> None:
        """Play a one-shot sound effect at a specific location.

        Args:
            sound_id: ID of the sound to play.
            x: X position of the sound.
            y: Y position of the sound.
            layer_index: Optional index of specific layer to play. If None, plays
                all layers with repeat and timing logic applied.
            volume_jitter: Optional volume variation (min, max).
            pitch_jitter: Optional pitch variation (min, max).
            params: Runtime parameters for parameterized sounds.
        """
        if not self.audio_backend:
            return

        sound_def = get_sound_definition(sound_id)
        if not sound_def:
            logger.warning(f"No sound definition for one-shot: {sound_id}")
            return

        # Calculate volume based on distance
        volume = self._calculate_volume(
            x,
            y,
            sound_def,
            volume_multiplier=1.0,
        )

        if volume <= 0.0:
            return

        # If playing a specific layer (from delayed scheduling), just play it directly
        if layer_index is not None:
            if 0 <= layer_index < len(sound_def.layers):
                layer = sound_def.layers[layer_index]
                self._play_layer_instance(
                    layer, sound_id, volume, volume_jitter, pitch_jitter
                )
            return

        # Playing all layers - handle repeats and after_layer dependencies
        layers = sound_def.layers

        # First pass: calculate completion times for each layer (for after_layer)
        # Completion time = time when the last repeat of this layer plays
        layer_completion_times: dict[int, float] = {}
        for idx, layer in enumerate(layers):
            repeat_count = self._get_repeat_count(layer, params)
            # Last repeat plays at (repeat_count - 1) * repeat_delay from layer start
            completion = (repeat_count - 1) * layer.repeat_delay
            layer_completion_times[idx] = completion

        # Second pass: schedule all layer instances
        for idx, layer in enumerate(layers):
            # Calculate base delay for this layer
            base_delay = layer.delay

            # Add completion time of referenced layer if using after_layer
            if layer.after_layer is not None:
                if layer.after_layer in layer_completion_times:
                    base_delay += layer_completion_times[layer.after_layer]
                else:
                    logger.warning(
                        f"Layer {idx} references invalid after_layer="
                        f"{layer.after_layer}"
                    )

            repeat_count = self._get_repeat_count(layer, params)

            for i in range(repeat_count):
                instance_delay = base_delay + (i * layer.repeat_delay)

                if instance_delay > 0:
                    # Schedule for later playback
                    play_at = self.current_time + instance_delay
                    delayed_event = SoundEvent(
                        sound_id=sound_id,
                        x=x,
                        y=y,
                        layer=idx,
                        volume_jitter=volume_jitter,
                        pitch_jitter=None,  # Delayed layers use own pitch_variation
                        delay=0.0,
                        params=params,
                    )
                    self._delayed_sounds.append((play_at, delayed_event))
                    logger.debug(
                        f"Scheduled layer {idx} repeat {i + 1}/{repeat_count} of "
                        f"{sound_id} at {play_at:.2f}s (delay: {instance_delay:.2f}s)"
                    )
                else:
                    # Play immediately
                    self._play_layer_instance(
                        layer, sound_id, volume, volume_jitter, pitch_jitter
                    )
