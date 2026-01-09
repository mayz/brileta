"""Action Presentation Layer for staggered combat feedback.

This module provides a presentation queue that staggers visual and audio feedback
from combat actions. When multiple actors act in a single turn, their effects play
out sequentially rather than all firing simultaneously.

This creates a tactical "call and response" rhythm that makes combat readable and
satisfying, similar to games like XCOM, Cogmind, and Into the Breach.

Key design principles:
- Player actions present immediately (no delay) - responsiveness is paramount
- NPC actions are staggered with configurable timing
- Queue management prevents backlog during rapid player input
- Game state is never blocked - presentation is purely cosmetic
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from catley.events import (
    EffectEvent,
    GameEvent,
    SoundEvent,
    publish_event,
    subscribe_to_event,
)
from catley.types import FixedTimestep

logger = logging.getLogger(__name__)


@dataclass
class PresentationEvent(GameEvent):
    """Bundles visual and audio feedback for a single game action.

    This event is published by executors (e.g., AttackExecutor) instead of
    directly publishing EffectEvent/SoundEvent. The PresentationManager
    receives these events and staggers their playback over time.

    Attributes:
        effect_events: Visual effects to trigger (particles, muzzle flash, etc.)
        sound_events: Sounds to play (gunfire, impacts, etc.)
        source_x: World X coordinate of the action (for distance-based priority)
        source_y: World Y coordinate of the action
        is_player_action: If True, presents immediately with no delay
    """

    effect_events: list[EffectEvent] = field(default_factory=list)
    sound_events: list[SoundEvent] = field(default_factory=list)
    source_x: int = 0
    source_y: int = 0
    is_player_action: bool = False


@dataclass
class _QueuedPresentation:
    """Internal representation of a queued presentation with timing info."""

    presentation: PresentationEvent
    scheduled_time: (
        float  # Time when this should be presented (relative to batch start)
    )


class PresentationManager:
    """Manages staggered playback of action presentations.

    This manager subscribes to PresentationEvent and queues them for staggered
    playback. Player actions always present immediately, while NPC actions are
    spread out over time to create readable combat rhythm.

    Queue overflow behavior (rapid player input):
    1. Player presentations ALWAYS play immediately (never queued, never dropped)
    2. When player acts again while NPC queue is playing: queue is flushed
    3. If queue exceeds max_queue_size: oldest NPC presentations are dropped
    4. If batch exceeds max_presentation_window: timing is compressed

    This ensures: mashing arrow keys 10 times results in instant game response,
    older NPC feedback gets dropped, and no infinite backlog builds up.
    """

    # =========================================================================
    # TUNING: Adjust these timing values based on playtesting feel.
    # - npc_base_delay: How long after player acts before first NPC effect
    # - npc_stagger: Gap between each subsequent NPC effect
    # - max_presentation_window: Max time to play out a batch (compresses if exceeded)
    # =========================================================================

    def __init__(
        self,
        npc_base_delay: float = 0.05,
        npc_stagger: float = 0.10,
        max_queue_size: int = 8,
        max_presentation_window: float = 0.6,
        flush_on_player_action: bool = True,
    ) -> None:
        """Initialize the presentation manager.

        Args:
            npc_base_delay: Delay before first NPC presentation (seconds).
            npc_stagger: Delay between consecutive NPC presentations (seconds).
            max_queue_size: Maximum NPC presentations to queue before dropping oldest.
            max_presentation_window: Maximum time to play out a batch (seconds).
                If exceeded, timing is compressed.
            flush_on_player_action: If True, clear NPC queue when player acts again.
        """
        self.npc_base_delay = npc_base_delay
        self.npc_stagger = npc_stagger
        self.max_queue_size = max_queue_size
        self.max_presentation_window = max_presentation_window
        self.flush_on_player_action = flush_on_player_action

        # Queue of pending NPC presentations (player presentations are immediate)
        self._queue: deque[_QueuedPresentation] = deque()

        # Time elapsed since the current batch started
        self._batch_elapsed: float = 0.0

        # Track the scheduled time for the next NPC presentation
        self._next_npc_time: float = 0.0

        # Subscribe to presentation events
        subscribe_to_event(PresentationEvent, self._handle_presentation_event)

    def _handle_presentation_event(self, event: PresentationEvent) -> None:
        """Handle incoming presentation events.

        Player actions are dispatched immediately. NPC actions are queued
        for staggered playback.
        """
        if event.is_player_action:
            # Player actions are always immediate - responsiveness is paramount
            self._dispatch_presentation(event)

            # Optionally flush the NPC queue when player acts
            if self.flush_on_player_action:
                self._flush_queue()
        else:
            # Queue NPC presentation for staggered playback
            self._enqueue_npc_presentation(event)

    def _enqueue_npc_presentation(self, event: PresentationEvent) -> None:
        """Add an NPC presentation to the queue with appropriate timing."""
        # Calculate scheduled time for this presentation
        if len(self._queue) == 0:
            # First NPC in batch - use base delay
            scheduled_time = self._batch_elapsed + self.npc_base_delay
        else:
            # Subsequent NPCs - stagger from previous
            scheduled_time = self._next_npc_time + self.npc_stagger

        # Update next NPC time tracker
        self._next_npc_time = scheduled_time

        # Check if we need timing compression (include new item's time)
        if scheduled_time > self.max_presentation_window:
            # Compress timing: scale all scheduled times to fit within window
            scale = self._compress_timing(scheduled_time)
            # The new item's time is also scaled to fit within the window
            scheduled_time = scheduled_time * scale
            self._next_npc_time = scheduled_time

        # Enforce max queue size - drop oldest if exceeded
        while len(self._queue) >= self.max_queue_size:
            dropped = self._queue.popleft()
            logger.debug(
                f"Dropped oldest NPC presentation at ({dropped.presentation.source_x}, "
                f"{dropped.presentation.source_y}) due to queue overflow"
            )

        # Add to queue
        self._queue.append(
            _QueuedPresentation(presentation=event, scheduled_time=scheduled_time)
        )

    def _compress_timing(self, new_item_time: float) -> float:
        """Compress all scheduled times to fit within max_presentation_window.

        Args:
            new_item_time: The scheduled time of the item about to be added.
                This is used to determine the scale factor since it may be
                larger than any time currently in the queue.

        Returns:
            The scale factor applied (1.0 if no compression needed).
        """
        # Find the maximum time including the new item
        if self._queue:
            max_time = max(max(q.scheduled_time for q in self._queue), new_item_time)
        else:
            max_time = new_item_time

        if max_time <= self.max_presentation_window:
            return 1.0

        # Scale factor to fit everything within the window
        scale = self.max_presentation_window / max_time

        # Apply compression to all queued presentations
        for queued in self._queue:
            queued.scheduled_time *= scale

        # Update the next NPC time tracker
        if self._queue:
            self._next_npc_time = self._queue[-1].scheduled_time

        logger.debug(
            f"Compressed presentation timing by factor {scale:.2f} to fit "
            f"within {self.max_presentation_window}s window"
        )

        return scale

    def _flush_queue(self) -> None:
        """Clear all pending NPC presentations and reset timing."""
        if self._queue:
            logger.debug(f"Flushing {len(self._queue)} pending NPC presentations")
        self._queue.clear()
        self._batch_elapsed = 0.0
        self._next_npc_time = 0.0

    def _dispatch_presentation(self, event: PresentationEvent) -> None:
        """Dispatch a presentation's effects and sounds to the event bus."""
        for effect in event.effect_events:
            publish_event(effect)

        for sound in event.sound_events:
            publish_event(sound)

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update the presentation manager and dispatch due presentations.

        Call this method each logic step (typically from Controller.update_logic_step).

        Args:
            fixed_timestep: Fixed time step duration in seconds.
        """
        if not self._queue:
            # No pending presentations - nothing to do
            return

        # Advance time
        self._batch_elapsed += fixed_timestep

        # Dispatch all presentations that are due
        while self._queue:
            next_presentation = self._queue[0]
            if next_presentation.scheduled_time <= self._batch_elapsed:
                self._queue.popleft()
                self._dispatch_presentation(next_presentation.presentation)
            else:
                # Next presentation isn't due yet
                break

        # If queue is empty, reset batch timing for next batch
        if not self._queue:
            self._batch_elapsed = 0.0
            self._next_npc_time = 0.0

    def clear(self) -> None:
        """Clear all pending presentations. Use for game state resets."""
        self._flush_queue()

    @property
    def has_pending_presentations(self) -> bool:
        """Check if there are any pending NPC presentations."""
        return len(self._queue) > 0

    @property
    def queue_size(self) -> int:
        """Get the current number of queued NPC presentations."""
        return len(self._queue)
