"""Unit tests for the animation framework."""

from typing import Any, cast

import pytest

from catley.types import FixedTimestep
from catley.view.animation import Animation, AnimationManager, MoveAnimation


class DummyActor:
    """Test actor with render coordinates."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.render_x = float(x)
        self.render_y = float(y)
        self._animation_controlled = False


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
        finished = animation.update(FixedTimestep(0.075))
        assert not finished
        assert actor.render_x == pytest.approx(5.0)
        assert actor.render_y == pytest.approx(10.0)

        # At 75% completion (additional 0.0375 seconds = 0.1125 total)
        finished = animation.update(FixedTimestep(0.0375))
        assert not finished
        assert actor.render_x == pytest.approx(7.5)
        assert actor.render_y == pytest.approx(15.0)

    def test_animation_completion(self) -> None:
        """Animation completes and snaps to final position."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (1.0, 2.0), (5.0, 6.0))

        # Complete the animation
        finished = animation.update(FixedTimestep(0.15))
        assert finished
        assert actor.render_x == 5.0
        assert actor.render_y == 6.0

    def test_animation_overtime(self) -> None:
        """Animation handles time exceeding duration."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (4.0, 8.0))

        # Update with time much longer than duration
        finished = animation.update(FixedTimestep(1.0))
        assert finished
        assert actor.render_x == 4.0
        assert actor.render_y == 8.0

    def test_zero_distance_animation(self) -> None:
        """Animation works with zero distance moves."""
        actor = DummyActor(3, 3)
        animation = MoveAnimation(cast(Any, actor), (5.0, 5.0), (5.0, 5.0))

        # Should complete immediately but still respect timing
        finished = animation.update(FixedTimestep(0.05))
        assert not finished
        assert actor.render_x == 5.0
        assert actor.render_y == 5.0

        finished = animation.update(FixedTimestep(0.15))
        assert finished

    def test_negative_coordinates(self) -> None:
        """Animation works with negative coordinates."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (-2.0, -3.0), (2.0, 3.0))

        # At 50% completion
        finished = animation.update(FixedTimestep(0.075))
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
        manager.update(FixedTimestep(0.075))  # 50% completion
        assert not manager.is_queue_empty()
        assert actor.render_x == pytest.approx(5.0)
        assert actor.render_y == pytest.approx(5.0)

        # Animation completes
        manager.update(FixedTimestep(0.075))  # 100% completion
        assert manager.is_queue_empty()
        assert actor.render_x == 10.0
        assert actor.render_y == 10.0

    def test_multiple_animation_queue(self) -> None:
        """Multiple animations are processed simultaneously."""
        manager = AnimationManager()
        actor1 = DummyActor(0, 0)
        actor2 = DummyActor(5, 5)

        # Add two animations
        anim1 = MoveAnimation(cast(Any, actor1), (0.0, 0.0), (2.0, 2.0))
        anim2 = MoveAnimation(cast(Any, actor2), (5.0, 5.0), (7.0, 7.0))

        manager.add(anim1)
        manager.add(anim2)

        # Both animations should run simultaneously
        manager.update(FixedTimestep(0.075))  # 50% of both animations
        assert not manager.is_queue_empty()
        # First actor moves from (0,0) to (2,2), so 50% progress = (1,1)
        assert actor1.render_x == pytest.approx(1.0)
        assert actor1.render_y == pytest.approx(1.0)
        # Second actor moves from (5,5) to (7,7), so 50% progress = (6,6)
        assert actor2.render_x == pytest.approx(6.0)
        assert actor2.render_y == pytest.approx(6.0)

        # Complete both animations
        manager.update(FixedTimestep(0.075))  # Complete both animations simultaneously
        assert manager.is_queue_empty()
        assert actor1.render_x == 2.0
        assert actor1.render_y == 2.0
        assert actor2.render_x == 7.0
        assert actor2.render_y == 7.0

    def test_empty_queue_update(self) -> None:
        """Updating empty queue does nothing."""
        manager = AnimationManager()

        # Should not raise any exceptions
        manager.update(FixedTimestep(0.1))
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
            manager.update(FixedTimestep(0.15))

        assert manager.is_queue_empty()
        assert actor.render_x == 0.0
        assert actor.render_y == 1.0


@pytest.fixture
def manager() -> AnimationManager:
    """Return a fresh AnimationManager for interruption tests."""

    return AnimationManager()


class TestAnimationManagerInterruption:
    """Tests for the new interruption and snap methods."""

    def test_finish_all_and_clear_snaps_actors(self, manager: AnimationManager) -> None:
        actor1 = DummyActor(0, 0)
        actor2 = DummyActor(5, 5)

        anim1 = MoveAnimation(cast(Any, actor1), (0.0, 0.0), (2.0, 3.0))
        anim2 = MoveAnimation(cast(Any, actor2), (5.0, 5.0), (8.0, 9.0))
        manager.add(anim1)
        manager.add(anim2)

        manager.finish_all_and_clear()

        assert actor1.render_x == 2.0
        assert actor1.render_y == 3.0
        assert actor2.render_x == 8.0
        assert actor2.render_y == 9.0
        assert not actor1._animation_controlled
        assert not actor2._animation_controlled

    def test_finish_all_and_clear_clears_queue(self, manager: AnimationManager) -> None:
        actor1 = DummyActor(0, 0)
        actor2 = DummyActor(1, 1)

        manager.add(MoveAnimation(cast(Any, actor1), (0.0, 0.0), (1.0, 1.0)))
        manager.add(MoveAnimation(cast(Any, actor2), (1.0, 1.0), (2.0, 2.0)))

        manager.finish_all_and_clear()

        assert manager.is_queue_empty()

    def test_finish_all_and_clear_on_empty_queue(
        self, manager: AnimationManager
    ) -> None:
        manager.finish_all_and_clear()
        assert manager.is_queue_empty()

    def test_interrupt_player_animations_removes_only_player(
        self, manager: AnimationManager
    ) -> None:
        player_actor = DummyActor(0, 0)
        npc_actor = DummyActor(5, 5)

        player_anim = MoveAnimation(cast(Any, player_actor), (0.0, 0.0), (1.0, 1.0))
        npc_anim = MoveAnimation(cast(Any, npc_actor), (5.0, 5.0), (6.0, 6.0))
        manager.add(player_anim)
        manager.add(npc_anim)

        manager.interrupt_player_animations(cast(Any, player_actor))

        assert len(manager._queue) == 1
        remaining = cast(MoveAnimation, manager._queue[0])
        assert remaining.actor is npc_actor
        assert not player_actor._animation_controlled
        assert npc_actor._animation_controlled

    def test_interrupt_player_animations_on_empty_queue(
        self, manager: AnimationManager
    ) -> None:
        player_actor = DummyActor(0, 0)
        manager.interrupt_player_animations(cast(Any, player_actor))
        assert manager.is_queue_empty()

    def test_interrupt_player_animations_with_no_player_anims(
        self, manager: AnimationManager
    ) -> None:
        player_actor = DummyActor(0, 0)
        npc_actor = DummyActor(1, 1)

        npc_anim = MoveAnimation(cast(Any, npc_actor), (1.0, 1.0), (2.0, 2.0))
        manager.add(npc_anim)

        manager.interrupt_player_animations(cast(Any, player_actor))

        assert len(manager._queue) == 1
        assert manager._queue[0] is npc_anim
        assert npc_actor._animation_controlled
