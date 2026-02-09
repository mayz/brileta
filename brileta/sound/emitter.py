"""Sound emitter component for actors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brileta.types import SoundId

if TYPE_CHECKING:
    from .system import PlayingSound


class SoundEmitter:
    """Represents a logical sound source attached to an actor.

    A SoundEmitter is a component that can be attached to actors to make them
    emit sounds. Each emitter references a sound definition and manages its
    playback state.

    Attributes:
        sound_id: ID of the sound definition to use
        volume_multiplier: Actor-specific volume adjustment
        active: Whether this emitter is currently active
        playing_instances: Active sound playback handles managed by the sound system
    """

    def __init__(
        self, sound_id: SoundId, volume_multiplier: float = 1.0, active: bool = True
    ) -> None:
        """Initialize a sound emitter.

        Args:
            sound_id: ID of the sound definition from the sound library
            volume_multiplier: Volume adjustment for this specific emitter (0.0-1.0)
            active: Whether the emitter starts active
        """
        self.sound_id = sound_id
        self.volume_multiplier = volume_multiplier
        self.active = active
        self.playing_instances: list[PlayingSound] = []

    def set_active(self, active: bool) -> None:
        """Enable or disable this emitter."""
        self.active = active

    def set_volume_multiplier(self, multiplier: float) -> None:
        """Adjust the volume multiplier for this emitter."""
        self.volume_multiplier = max(0.0, min(1.0, multiplier))
