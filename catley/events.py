"""Global event system for decoupling UI notifications and cross-system communication.

This event bus is designed for UI feedback and cross-system notifications only.
It uses a global instance for simplicity and to reduce Controller dependencies.

USE FOR:
- Messages to message log
- Visual effects (particles, screen shake)
- Audio cues and sound effects
- Cross-system notifications (actor death, level changes)
- UI state changes that multiple systems care about

DO NOT USE FOR:
- Core game mechanics (combat resolution, movement, turn processing)
- Performance-critical operations (rendering loops, input processing)
- Internal component updates or data structure modifications
- Operations that need immediate return values or synchronous confirmation
- Error handling or exception propagation

The event bus is fire-and-forget: publish an event without expecting return values
or confirmations. All handlers execute immediately (synchronously). If you need a
return value or confirmation, use direct method calls.
"""

import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from catley import colors
from catley.types import DeltaTime

logger = logging.getLogger(__name__)


@dataclass
class GameEvent:
    """Base class for all game events."""

    pass


@dataclass
class MessageEvent(GameEvent):
    """Event for adding messages to the message log."""

    text: str
    color: colors.Color = colors.WHITE
    stack: bool = True


@dataclass
class EffectEvent(GameEvent):
    """Event for creating visual effects."""

    effect_name: str
    x: int
    y: int
    intensity: float = 1.0
    direction_x: float = 0.0
    direction_y: float = 0.0
    ray_count: int | None = (
        None  # For blood splatter: None = 1 (default), 2-4 for shotgun
    )


@dataclass
class ActorDeathEvent(GameEvent):
    """Event for when an actor dies."""

    actor: Any  # Avoid circular imports


@dataclass
class ScreenShakeEvent(GameEvent):
    """Event for triggering screen shake."""

    intensity: float
    duration: DeltaTime


@dataclass
class SoundEvent(GameEvent):
    """Event for triggering a one-shot sound effect.

    Attributes:
        sound_id: ID of the sound definition to play.
        x, y: World position of the sound source.
        layer: Optional specific layer index to play (None = all layers).
        volume_jitter: Optional (min, max) volume multiplier range.
        pitch_jitter: Optional (min, max) pitch multiplier range.
        delay: Delay in seconds before playing.
        params: Runtime parameters for parameterized sounds. Used by layers with
            repeat_count_param to determine dynamic repeat counts.
    """

    sound_id: str
    x: int
    y: int
    layer: int | None = None
    volume_jitter: tuple[float, float] | None = None
    pitch_jitter: tuple[float, float] | None = None
    delay: float = 0.0
    params: dict[str, int | float] | None = None


class EventBus:
    """Simple event bus for publish/subscribe pattern."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Callable]] = {}

    def subscribe(self, event_type: type, handler: Callable) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type, handler: Callable) -> None:
        """Unsubscribe a handler from an event type."""
        if event_type in self._handlers:
            with suppress(ValueError):
                self._handlers[event_type].remove(handler)

    def publish(self, event: GameEvent) -> None:
        """Publish an event to all subscribed handlers."""
        event_type = type(event)
        if event_type in self._handlers:
            for handler in self._handlers[event_type]:
                try:
                    handler(event)
                except Exception:
                    logger.exception(f"Error handling event {event_type.__name__}")


# Global event bus instance
_global_event_bus = EventBus()


# Public API functions


def subscribe_to_event(event_type: type, handler: Callable) -> None:
    """Subscribe to an event type globally."""
    _global_event_bus.subscribe(event_type, handler)


def unsubscribe_from_event(event_type: type, handler: Callable) -> None:
    """Unsubscribe from an event type globally."""
    _global_event_bus.unsubscribe(event_type, handler)


def publish_event(event: GameEvent) -> None:
    """Publish an event globally."""
    _global_event_bus.publish(event)


def reset_event_bus_for_testing() -> None:
    """Reset the global event bus. Use only in tests."""
    global _global_event_bus
    _global_event_bus = EventBus()
