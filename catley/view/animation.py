"""Animation framework for smooth visual transitions.

This module provides the foundational animation system for decoupling game logic
from visual presentation. It enables smooth, time-based transitions for game
actions like movement, making them feel more natural and easier to follow.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from catley.game.actors.core import Actor


class Animation(ABC):
    """Abstract base class for all animations.

    Animations represent visual transitions that occur over time, allowing
    the game to show smooth changes rather than instantaneous updates.
    """

    @abstractmethod
    def update(self, delta_time: float) -> bool:
        """Update the animation state based on elapsed time.

        Args:
            delta_time: Time elapsed since last update in seconds.

        Returns:
            True if the animation is finished, False if it should continue.
        """
        pass


class MoveAnimation(Animation):
    """Animation for smoothly moving an actor between two positions.

    Interpolates an actor's render_x and render_y coordinates from a start
    position to an end position over a fixed duration, creating smooth
    movement that's easy for players to track visually.
    """

    def __init__(
        self,
        actor: "Actor",
        start_pos: tuple[float, float],
        end_pos: tuple[float, float],
    ) -> None:
        """Initialize a movement animation.

        Args:
            actor: The actor to animate.
            start_pos: Starting position as (x, y) coordinates.
            end_pos: Target position as (x, y) coordinates.
        """
        self.actor = actor
        self.start_x, self.start_y = start_pos
        self.end_x, self.end_y = end_pos
        self.duration = 0.15  # Animation duration in seconds
        self.elapsed_time = 0.0

        # Take control of the actor's render position
        self.actor._animation_controlled = True

        # Set initial render position
        self.actor.render_x = self.start_x
        self.actor.render_y = self.start_y

    def update(self, delta_time: float) -> bool:
        """Update the movement animation.

        Args:
            delta_time: Time elapsed since last update in seconds.

        Returns:
            True if the animation is finished, False otherwise.
        """
        self.elapsed_time += delta_time

        if self.elapsed_time >= self.duration:
            # Animation complete - snap to final position and release control
            self.actor.render_x = self.end_x
            self.actor.render_y = self.end_y
            self.actor._animation_controlled = False
            return True

        # Linear interpolation between start and end positions
        progress = self.elapsed_time / self.duration
        self.actor.render_x = self.start_x + (self.end_x - self.start_x) * progress
        self.actor.render_y = self.start_y + (self.end_y - self.start_y) * progress

        return False


class AnimationManager:
    """Manage a queue of animations and process them simultaneously."""

    def __init__(self) -> None:
        """Initialize an empty animation manager."""
        self._queue: deque[Animation] = deque()

    def add(self, animation: Animation) -> None:
        """Add an animation to the end of the queue.

        Args:
            animation: The animation to add to the queue.
        """
        self._queue.append(animation)

    def is_queue_empty(self) -> bool:
        """Check if the animation queue is empty.

        Returns:
            True if no animations are queued, False otherwise.
        """
        return len(self._queue) == 0

    def update(self, delta_time: float) -> None:
        """Update all animations simultaneously and remove finished ones.

        Args:
            delta_time: Time elapsed since last update in seconds.
        """
        # Process all animations simultaneously for responsive gameplay
        if not self._queue:
            return

        # Update all animations and collect those still running
        remaining_animations = deque()
        for animation in self._queue:
            if not animation.update(delta_time):
                remaining_animations.append(animation)

        self._queue = remaining_animations
