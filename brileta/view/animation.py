"""Animation framework for smooth visual transitions.

This module provides the foundational animation system for decoupling game logic
from visual presentation. It enables smooth, time-based transitions for game
actions like movement, making them feel more natural and easier to follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from brileta import config
from brileta.types import FixedTimestep

if TYPE_CHECKING:
    from brileta.game.actors.core import Actor

# Glides at least this long advance the walk cycle at their midpoint too;
# below it, the per-step walk-frame flip alone is fast enough.
_MIDSTEP_FLIP_MIN_DURATION_S = 0.25


class Animation(ABC):
    """Abstract base class for all animations.

    Animations represent visual transitions that occur over time, allowing
    the game to show smooth changes rather than instantaneous updates.
    """

    @abstractmethod
    def update(self, fixed_timestep: FixedTimestep) -> bool:
        """Update the animation state based on elapsed time.

        Args:
            fixed_timestep: Fixed time step duration for deterministic animation timing.

        Returns:
            True if the animation is finished, False if it should continue.
        """
        pass


class MoveAnimation(Animation):
    """Animation for smoothly moving an actor between two positions.

    Interpolates an actor's render_x and render_y coordinates from a start
    position to an end position over a configurable duration, creating smooth
    movement that's easy for players to track visually.
    """

    def __init__(
        self,
        actor: Actor,
        start_pos: tuple[float, float],
        end_pos: tuple[float, float],
        duration: float = 0.15,
        ease_power: float = config.DEFAULT_MOVE_EASE_POWER,
    ) -> None:
        """Initialize a movement animation.

        Args:
            actor: The actor to animate.
            start_pos: Starting position as (x, y) coordinates.
            end_pos: Target position as (x, y) coordinates.
            duration: Animation duration in seconds. Defaults to 0.15s for
                backward compatibility.
            ease_power: Ease-out strength. 1.0 is linear (constant velocity,
                good for chained ambient strolls), 2.0 is quadratic ease-out
                (a punchy accent for single steps); values between blend the
                two feels.
        """
        self.actor = actor
        self.start_x, self.start_y = start_pos
        self.end_x, self.end_y = end_pos
        self.duration = duration
        self.ease_power = ease_power
        self.elapsed_time = 0.0
        # Whether the mid-glide walk-frame flip has fired. The flip applies to
        # any glide of at least _MIDSTEP_FLIP_MIN_DURATION_S, eased or not:
        # skating is about how long one leg pose stays on screen, which is a
        # function of duration, not of the easing curve.
        self._midstep_flipped = False

        # Take control of the actor's render position. Registering as the
        # actor's active glide supersedes any older in-flight MoveAnimation,
        # which will drop out silently on its next update instead of fighting
        # this one for the render position.
        self.actor._animation_controlled = True
        self.actor._active_move_animation = self

        # Set initial render position
        self.actor.render_x = self.start_x
        self.actor.render_y = self.start_y

    def update(self, fixed_timestep: FixedTimestep) -> bool:
        """Update the movement animation.

        Args:
            fixed_timestep: Fixed time step duration for deterministic animation timing.

        Returns:
            True if the animation is finished, False otherwise.
        """
        # If the actor died mid-animation, terminate immediately.
        # The render position was already snapped when the actor died.
        # We check health directly rather than _animation_controlled to avoid
        # interfering with queued animations for the same actor.
        if self.actor.health is not None and not self.actor.health.is_alive():
            return True

        # A newer glide has taken over this actor (rapid chained steps).
        # Finish without snapping or releasing control so the new glide's
        # render position and state are left untouched.
        if self.actor._active_move_animation is not self:
            return True

        self.elapsed_time += fixed_timestep

        if self.elapsed_time >= self.duration:
            # Animation complete - snap to final position and release control.
            # The sprite is deliberately NOT reset to the standing pose here: the
            # actor holds its last walk frame, so continuous walking never flickers
            # to stand between tiles. The idle-settle timer (ticked by the
            # animation manager) is what eventually settles a stopped actor to
            # stand, only after it has been still past the idle delay.
            self.actor.render_x = self.end_x
            self.actor.render_y = self.end_y
            self.actor._animation_controlled = False
            self.actor._active_move_animation = None
            return True

        # Ease-out interpolation starts promptly and eases into the stop,
        # avoiding a perceived "hesitation" at the start of short moves.
        # ease_power 1.0 keeps velocity constant so back-to-back tile glides
        # (ambient NPC strolling) read as one continuous walk.
        progress = self.elapsed_time / self.duration
        eased = 1.0 - (1.0 - progress) ** self.ease_power

        # Long glides advance the walk cycle mid-step as well as at step
        # time; one pose held across a ~450ms slide reads as ice skating.
        # Short chained hops (held-key repeats) are excluded - their per-step
        # flip already runs the walk cycle plenty fast.
        if (
            self.duration >= _MIDSTEP_FLIP_MIN_DURATION_S
            and not self._midstep_flipped
            and progress >= 0.5
        ):
            self._midstep_flipped = True
            self.actor.walk_frame = 1 - self.actor.walk_frame
            self.actor._update_active_sprite_uv(moving=True)
        self.actor.render_x = self.start_x + (self.end_x - self.start_x) * eased
        self.actor.render_y = self.start_y + (self.end_y - self.start_y) * eased

        return False


class AnimationManager:
    """Manage a queue of animations and process them simultaneously."""

    def __init__(self) -> None:
        """Initialize an empty animation manager."""
        self._queue: deque[Animation] = deque()
        # Actors that recently stepped and are counting down to idle. Ticked every
        # logic step (even with an empty animation queue) so a stopped actor
        # settles from its last walk frame to stand once the idle delay lapses.
        self._settling_actors: set[Actor] = set()

    def add(self, animation: Animation) -> None:
        """Add an animation to the end of the queue.

        Args:
            animation: The animation to add to the queue.
        """
        self._queue.append(animation)

    def track_idle(self, actor: Actor) -> None:
        """Register an actor whose idle-settle timer needs ticking down."""
        self._settling_actors.add(actor)

    def is_queue_empty(self) -> bool:
        """Check if the animation queue is empty.

        Returns:
            True if no animations are queued, False otherwise.
        """
        return len(self._queue) == 0

    def clear(self) -> None:
        """Clear all animations without completing them.

        Use this when the game world is being regenerated and the animated
        actors no longer exist. For normal gameplay interruption, use
        finish_all_and_clear() instead to properly snap actors to positions.
        """
        self._queue.clear()
        self._settling_actors.clear()

    def update(self, fixed_timestep: FixedTimestep) -> None:
        """Update all animations simultaneously and remove finished ones.

        Args:
            fixed_timestep: Fixed time step duration for deterministic animation timing.
        """
        # Tick idle-settle timers first, even when the animation queue is empty:
        # an actor that took its last step has no queued animation but must still
        # count down and settle from its walk frame to stand once it stops.
        if self._settling_actors:
            settled = {
                actor
                for actor in self._settling_actors
                if not actor.tick_idle_settle(fixed_timestep)
            }
            self._settling_actors -= settled

        # Process all animations simultaneously for responsive gameplay
        if not self._queue:
            return

        # Update all animations and collect those still running
        remaining_animations = deque()
        for animation in self._queue:
            if not animation.update(fixed_timestep):
                remaining_animations.append(animation)

        self._queue = remaining_animations

    def finish_all_and_clear(self) -> None:
        """Instantly complete all animations and clear the queue.

        This "snap" mechanism forces every queued animation to finish
        immediately. Actors involved in ``MoveAnimation`` instances are
        positioned at their final coordinates and released from animation
        control. The animation queue is emptied afterwards.
        """

        for animation in self._queue:
            if isinstance(animation, MoveAnimation):
                animation.actor.render_x = animation.end_x
                animation.actor.render_y = animation.end_y
                animation.actor._animation_controlled = False

        # A hard snap ends all motion, so settle every counting-down actor
        # straight to its standing pose.
        for actor in self._settling_actors:
            actor._idle_settle_time = 0.0
            actor._update_active_sprite_uv(moving=False)

        self._queue.clear()
        self._settling_actors.clear()

    def interrupt_player_animations(self, player_actor: Actor) -> None:
        """Remove all animations belonging to ``player_actor`` from the queue.

        Snaps the player to their animation's end position before removing,
        preventing visual pops when manual input interrupts movement.
        """
        remaining_animations: deque[Animation] = deque()

        for animation in self._queue:
            if isinstance(animation, MoveAnimation) and animation.actor is player_actor:
                # Snap to end position before releasing control. The sprite frame
                # is left as-is (the last walk frame) rather than reset to stand,
                # so an interrupted walk does not snap to a neutral pose.
                animation.actor.render_x = animation.end_x
                animation.actor.render_y = animation.end_y
                animation.actor._animation_controlled = False
            else:
                remaining_animations.append(animation)

        self._queue = remaining_animations
