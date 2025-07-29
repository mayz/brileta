"""Main sound system that manages all sound playback and processing."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from catley.types import WorldTileCoord

from .definitions import SoundDefinition, SoundLayer, get_sound_definition

if TYPE_CHECKING:
    from catley.game.actors.core import Actor
    from catley.sound.emitter import SoundEmitter


@dataclass
class PlayingSound:
    """Represents an actively playing sound instance."""

    emitter: SoundEmitter
    layer: SoundLayer
    channel: object | None = None  # Audio backend channel/handle
    next_trigger_time: float = 0.0  # For interval-based sounds


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
        self.audio_backend: object | None = None  # TODO: Add pygame/other audio backend
        self.current_time: float = 0.0

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

        # Keep track of which emitters are being processed
        processed_emitters = set()

        # Calculate volumes and update playing sounds
        for emitter, emitter_x, emitter_y in active_emitters:
            processed_emitters.add(emitter)
            sound_def = get_sound_definition(emitter.sound_id)
            if not sound_def:
                continue

            # Calculate distance-based volume
            volume = self._calculate_volume(
                emitter_x,
                emitter_y,
                listener_x,
                listener_y,
                sound_def,
                emitter.volume_multiplier,
            )

            if volume > 0.0:
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

        # Clean up finished sounds
        self._cleanup_finished_sounds()

    def _calculate_volume(
        self,
        emitter_x: WorldTileCoord,
        emitter_y: WorldTileCoord,
        listener_x: WorldTileCoord,
        listener_y: WorldTileCoord,
        sound_def: SoundDefinition,
        volume_multiplier: float,
    ) -> float:
        """Calculate volume based on distance using inverse square law.

        Args:
            emitter_x: X position of the sound source
            emitter_y: Y position of the sound source
            listener_x: X position of the listener
            listener_y: Y position of the listener
            sound_def: Sound definition with falloff parameters
            volume_multiplier: Emitter-specific volume adjustment

        Returns:
            Final volume (0.0 to 1.0)
        """
        # Calculate Euclidean distance
        dx = emitter_x - listener_x
        dy = emitter_y - listener_y
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
                    emitter.playing_instances.append(playing)
                    self.playing_sounds.append(playing)
                    # TODO: Start playing through audio backend
                    # DEBUG: Log when sound would start
                    if __debug__:
                        print(
                            f"[SOUND] Starting loop: {sound_def.sound_id} - "
                            f"{layer.file} at volume {volume:.2f}"
                        )
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
                        # TODO: Play one-shot through audio backend
                        # DEBUG: Log when sound would trigger
                        if __debug__:
                            print(
                                f"[SOUND] Triggering one-shot: {sound_def.sound_id} - "
                                f"{layer.file} at volume {volume:.2f}"
                            )

    def _stop_sound_if_playing(self, emitter: SoundEmitter) -> None:
        """Stop all sounds for the given emitter.

        Args:
            emitter: The sound emitter to stop
        """
        for playing in emitter.playing_instances[:]:
            # TODO: Stop through audio backend
            # DEBUG: Log when sound would stop
            if __debug__:
                print(f"[SOUND] Stopping: {playing.layer.file}")
            emitter.playing_instances.remove(playing)
            if playing in self.playing_sounds:
                self.playing_sounds.remove(playing)

    def _cleanup_finished_sounds(self) -> None:
        """Remove finished one-shot sounds from tracking."""
        # TODO: Implement when audio backend is added
        pass
