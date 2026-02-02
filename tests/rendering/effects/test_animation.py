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
        self.health: Any = None  # Optional health component for death tests


class TestAnimation:
    """Test Animation base class behavior."""

    def test_animation_is_abstract(self) -> None:
        """Animation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Animation()


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
        assert animation.duration == 0.15  # Default duration
        assert animation.elapsed_time == 0.0

        # Actor should be set to start position
        assert actor.render_x == 3.0
        assert actor.render_y == 4.0

    def test_custom_duration(self) -> None:
        """MoveAnimation accepts custom duration parameter."""
        actor = DummyActor(0, 0)

        # Test with 70ms duration (held-key movement)
        animation = MoveAnimation(
            cast(Any, actor), (0.0, 0.0), (10.0, 10.0), duration=0.07
        )
        assert animation.duration == 0.07

        # At 50% time (0.035 seconds), ease-out gives 75% of distance.
        finished = animation.update(FixedTimestep(0.035))
        assert not finished
        assert actor.render_x == pytest.approx(7.5)

        # Complete the animation
        finished = animation.update(FixedTimestep(0.035))
        assert finished
        assert actor.render_x == 10.0

    def test_animation_progression(self) -> None:
        """Animation progresses correctly over time with ease-out curve."""
        actor = DummyActor(0, 0)
        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (10.0, 20.0))

        # At 50% time (0.075 seconds), ease-out gives 75% of distance.
        finished = animation.update(FixedTimestep(0.075))
        assert not finished
        assert actor.render_x == pytest.approx(7.5)
        assert actor.render_y == pytest.approx(15.0)

        # At 75% time (0.1125 total), ease-out gives 93.75% of distance.
        finished = animation.update(FixedTimestep(0.0375))
        assert not finished
        assert actor.render_x == pytest.approx(9.375)
        assert actor.render_y == pytest.approx(18.75)

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

        # At 50% time, ease-out gives 75% of distance.
        # From (-2,-3) to (2,3) = distance (4,6), so 75% = (1.0, 1.5)
        finished = animation.update(FixedTimestep(0.075))
        assert not finished
        assert actor.render_x == pytest.approx(1.0)
        assert actor.render_y == pytest.approx(1.5)


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

        # At 50% time, ease-out gives 75% of distance.
        manager.update(FixedTimestep(0.075))
        assert not manager.is_queue_empty()
        assert actor.render_x == pytest.approx(7.5)
        assert actor.render_y == pytest.approx(7.5)

        # Animation completes
        manager.update(FixedTimestep(0.075))
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

        # At 50% time, ease-out gives 75% of distance for both.
        manager.update(FixedTimestep(0.075))
        assert not manager.is_queue_empty()
        # First actor moves from (0,0) to (2,2), 75% = (1.5, 1.5)
        assert actor1.render_x == pytest.approx(1.5)
        assert actor1.render_y == pytest.approx(1.5)
        # Second actor moves from (5,5) to (7,7), 75% = (6.5, 6.5)
        assert actor2.render_x == pytest.approx(6.5)
        assert actor2.render_y == pytest.approx(6.5)

        # Complete both animations
        manager.update(FixedTimestep(0.075))
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


class TestMoveAnimationInterruption:
    """Tests for animation self-termination when actor dies."""

    def test_animation_terminates_when_actor_dies(self) -> None:
        """Animation returns finished=True if actor dies mid-animation.

        This prevents corpses from sliding to their destination when an
        NPC is killed while their move animation is in progress.
        """
        from catley.game.actors.components import HealthComponent, StatsComponent

        # Create an actor with health
        actor = DummyActor(0, 0)
        stats = StatsComponent(toughness=10)
        actor.health = HealthComponent(stats)

        animation = MoveAnimation(cast(Any, actor), (0.0, 0.0), (10.0, 10.0))

        # Verify animation is controlling the actor
        assert actor._animation_controlled

        # Partial animation progress
        finished = animation.update(FixedTimestep(0.05))
        assert not finished
        assert actor.render_x > 0.0  # Animation has moved render position

        # Simulate actor death: kill them and snap render position
        actor.health._hp = 0
        actor._animation_controlled = False
        actor.render_x = float(actor.x)
        actor.render_y = float(actor.y)

        # Animation should terminate immediately on next update
        finished = animation.update(FixedTimestep(0.01))
        assert finished

        # Render position should remain at snapped position (not moved by animation)
        assert actor.render_x == 0.0
        assert actor.render_y == 0.0


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

        # Player should be snapped to end position
        assert player_actor.render_x == 1.0
        assert player_actor.render_y == 1.0

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
