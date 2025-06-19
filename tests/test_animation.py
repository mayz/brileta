"""Unit tests for the animation framework."""

from typing import Any, cast

import pytest

from catley.view.animation import Animation, AnimationManager, MoveAnimation


class DummyActor:
    """Test actor with render coordinates."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.render_x = float(x)
        self.render_y = float(y)


class TestAnimation:
    """Test Animation base class behavior."""

    def test_animation_is_abstract(self) -> None:
        """Animation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Animation()  # type: ignore[abstract]


class TestMoveAnimation:
    """Test MoveAnimation functionality."""

    def test_initialization(self) -> None:
        """MoveAnimation initializes with proper values."""
        actor = DummyActor(5, 5)
        start_pos = (3.0, 4.0)
        end_pos = (7.0, 8.0)

        animation = MoveAnimation(cast(Any, actor), start_pos, end_pos)

        assert animation.actor is actor
        assert animation.start_x == 3.0
        assert animation.start_y == 4.0
        assert animation.end_x == 7.0
        assert animation.end_y == 8.0
        assert animation.duration == 0.15
        assert animation.elapsed_time == 0.0

        # Actor should be set to start position
        assert actor.render_x == 3.0
        assert actor.render_y == 4.0

    def test_animation_progression(self) -> None:
        """Animation progresses correctly over time."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (10.0, 20.0))

        # At 50% completion (0.075 seconds)
        finished = animation.update(0.075)
        assert not finished
        assert actor.render_x == pytest.approx(5.0)
        assert actor.render_y == pytest.approx(10.0)

        # At 75% completion (additional 0.0375 seconds = 0.1125 total)
        finished = animation.update(0.0375)
        assert not finished
        assert actor.render_x == pytest.approx(7.5)
        assert actor.render_y == pytest.approx(15.0)

    def test_animation_completion(self) -> None:
        """Animation completes and snaps to final position."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (1.0, 2.0), (5.0, 6.0))

        # Complete the animation
        finished = animation.update(0.15)
        assert finished
        assert actor.render_x == 5.0
        assert actor.render_y == 6.0

    def test_animation_overtime(self) -> None:
        """Animation handles time exceeding duration."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (4.0, 8.0))

        # Update with time much longer than duration
        finished = animation.update(1.0)
        assert finished
        assert actor.render_x == 4.0
        assert actor.render_y == 8.0

    def test_zero_distance_animation(self) -> None:
        """Animation works with zero distance moves."""
        actor = DummyActor(3, 3)
        animation = MoveAnimation(cast(Any, actor), (5.0, 5.0), (5.0, 5.0))

        # Should complete immediately but still respect timing
        finished = animation.update(0.05)
        assert not finished
        assert actor.render_x == 5.0
        assert actor.render_y == 5.0

        finished = animation.update(0.15)
        assert finished

    def test_negative_coordinates(self) -> None:
        """Animation works with negative coordinates."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (-2.0, -3.0), (2.0, 3.0))

        # At 50% completion
        finished = animation.update(0.075)
        assert not finished
        assert actor.render_x == pytest.approx(0.0)
        assert actor.render_y == pytest.approx(0.0)


class TestAnimationManager:
    """Test AnimationManager functionality."""

    def test_initialization(self) -> None:
        """AnimationManager initializes empty."""
        manager = AnimationManager()
        assert manager.is_queue_empty()

    def test_add_animation(self) -> None:
        """Animations can be added to the queue."""
        manager = AnimationManager()
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (1.0, 1.0))

        manager.add(animation)
        assert not manager.is_queue_empty()

    def test_single_animation_processing(self) -> None:
        """Single animation is processed correctly."""
        manager = AnimationManager()
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (10.0, 10.0))

        manager.add(animation)

        # Animation in progress
        manager.update(0.075)  # 50% completion
        assert not manager.is_queue_empty()
        assert actor.render_x == pytest.approx(5.0)
        assert actor.render_y == pytest.approx(5.0)

        # Animation completes
        manager.update(0.075)  # 100% completion
        assert manager.is_queue_empty()
        assert actor.render_x == 10.0
        assert actor.render_y == 10.0

    def test_multiple_animation_queue(self) -> None:
        """Multiple animations are processed sequentially."""
        manager = AnimationManager()
        actor1 = DummyActor(0, 0)
        actor2 = DummyActor(5, 5)

        # Add two animations
        anim1 = MoveAnimation(cast(Any, actor1), (0.0, 0.0), (2.0, 2.0))
        anim2 = MoveAnimation(cast(Any, actor2), (5.0, 5.0), (7.0, 7.0))

        manager.add(anim1)
        manager.add(anim2)

        # First animation should run
        manager.update(0.075)  # 50% of first animation
        assert not manager.is_queue_empty()
        assert actor1.render_x == pytest.approx(1.0)
        assert actor1.render_y == pytest.approx(1.0)
        assert actor2.render_x == 5.0  # Second actor unchanged
        assert actor2.render_y == 5.0

        # Complete first animation, start second
        manager.update(0.075)  # Complete first animation
        assert not manager.is_queue_empty()
        assert actor1.render_x == 2.0  # First actor at final position
        assert actor1.render_y == 2.0
        assert actor2.render_x == 5.0  # Second actor at start position
        assert actor2.render_y == 5.0

        # Progress second animation
        manager.update(0.075)  # 50% of second animation
        assert not manager.is_queue_empty()
        assert actor2.render_x == pytest.approx(6.0)
        assert actor2.render_y == pytest.approx(6.0)

        # Complete second animation
        manager.update(0.075)  # Complete second animation
        assert manager.is_queue_empty()
        assert actor2.render_x == 7.0
        assert actor2.render_y == 7.0

    def test_empty_queue_update(self) -> None:
        """Updating empty queue does nothing."""
        manager = AnimationManager()

        # Should not raise any exceptions
        manager.update(0.1)
        assert manager.is_queue_empty()

    def test_queue_order_preservation(self) -> None:
        """Animations are processed in FIFO order."""
        manager = AnimationManager()
        actor = DummyActor(0, 0)

        # Add animations with different durations but same end result
        # This tests that the first added is processed first
        anim1 = MoveAnimation(cast(Any, actor), (0.0, 0.0), (1.0, 0.0))
        anim2 = MoveAnimation(cast(Any, actor), (1.0, 0.0), (1.0, 1.0))
        anim3 = MoveAnimation(cast(Any, actor), (1.0, 1.0), (0.0, 1.0))

        manager.add(anim1)
        manager.add(anim2)
        manager.add(anim3)

        # Complete all animations
        for _ in range(3):
            manager.update(0.15)

        assert manager.is_queue_empty()
        assert actor.render_x == 0.0
        assert actor.render_y == 1.0
