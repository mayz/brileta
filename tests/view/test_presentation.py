"""Unit tests for the Action Presentation Layer."""

from catley.events import EffectEvent, SoundEvent, reset_event_bus_for_testing
from catley.types import FixedTimestep
from catley.view.presentation import PresentationEvent, PresentationManager


class TestPresentationEvent:
    """Tests for the PresentationEvent dataclass."""

    def test_default_values(self) -> None:
        """PresentationEvent has sensible defaults."""
        event = PresentationEvent()

        assert event.effect_events == []
        assert event.sound_events == []
        assert event.source_x == 0
        assert event.source_y == 0
        assert event.is_player_action is False

    def test_with_effects_and_sounds(self) -> None:
        """PresentationEvent can hold effects and sounds."""
        effect = EffectEvent("muzzle_flash", x=5, y=10)
        sound = SoundEvent("gun_fire", x=5, y=10)

        event = PresentationEvent(
            effect_events=[effect],
            sound_events=[sound],
            source_x=5,
            source_y=10,
            is_player_action=True,
        )

        assert len(event.effect_events) == 1
        assert len(event.sound_events) == 1
        assert event.is_player_action is True


class TestPresentationManager:
    """Tests for the PresentationManager class."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()

    def test_initialization(self) -> None:
        """PresentationManager initializes with correct defaults."""
        manager = PresentationManager()

        assert manager.npc_base_delay == 0.05
        assert manager.npc_stagger == 0.10
        assert manager.max_queue_size == 8
        assert manager.max_presentation_window == 0.6
        assert manager.flush_on_player_action is True
        assert manager.queue_size == 0
        assert manager.has_pending_presentations is False

    def test_custom_parameters(self) -> None:
        """PresentationManager accepts custom parameters."""
        manager = PresentationManager(
            npc_base_delay=0.1,
            npc_stagger=0.2,
            max_queue_size=4,
            max_presentation_window=1.0,
            flush_on_player_action=False,
        )

        assert manager.npc_base_delay == 0.1
        assert manager.npc_stagger == 0.2
        assert manager.max_queue_size == 4
        assert manager.max_presentation_window == 1.0
        assert manager.flush_on_player_action is False


class TestPlayerActionDispatch:
    """Tests for player action handling."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()
        self.dispatched_effects: list[EffectEvent] = []
        self.dispatched_sounds: list[SoundEvent] = []

    def test_player_action_dispatches_immediately(self) -> None:
        """Player actions are dispatched immediately, not queued."""
        manager = PresentationManager()

        # Track dispatched events
        from catley.events import subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)
        subscribe_to_event(SoundEvent, self.dispatched_sounds.append)

        # Create a player presentation event
        effect = EffectEvent("muzzle_flash", x=5, y=5)
        sound = SoundEvent("gun_fire", x=5, y=5)
        player_event = PresentationEvent(
            effect_events=[effect],
            sound_events=[sound],
            source_x=5,
            source_y=5,
            is_player_action=True,
        )

        # Publish the event (manager subscribes in __init__)
        from catley.events import publish_event

        publish_event(player_event)

        # Should be dispatched immediately, not queued
        assert manager.queue_size == 0
        assert len(self.dispatched_effects) == 1
        assert len(self.dispatched_sounds) == 1

    def test_player_action_flushes_npc_queue(self) -> None:
        """Player action flushes pending NPC presentations by default."""
        manager = PresentationManager(flush_on_player_action=True)

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)

        # Queue some NPC presentations
        npc_event = PresentationEvent(
            effect_events=[EffectEvent("blood_splatter", x=10, y=10)],
            source_x=10,
            source_y=10,
            is_player_action=False,
        )
        publish_event(npc_event)
        publish_event(npc_event)

        assert manager.queue_size == 2

        # Player action should flush the queue
        player_event = PresentationEvent(
            effect_events=[EffectEvent("muzzle_flash", x=5, y=5)],
            source_x=5,
            source_y=5,
            is_player_action=True,
        )
        publish_event(player_event)

        assert manager.queue_size == 0

    def test_player_action_no_flush_when_disabled(self) -> None:
        """Player action does not flush queue when flush_on_player_action=False."""
        manager = PresentationManager(flush_on_player_action=False)

        from catley.events import publish_event

        # Queue some NPC presentations
        npc_event = PresentationEvent(
            effect_events=[EffectEvent("blood_splatter", x=10, y=10)],
            source_x=10,
            source_y=10,
            is_player_action=False,
        )
        publish_event(npc_event)
        publish_event(npc_event)

        assert manager.queue_size == 2

        # Player action should NOT flush the queue
        player_event = PresentationEvent(
            source_x=5,
            source_y=5,
            is_player_action=True,
        )
        publish_event(player_event)

        assert manager.queue_size == 2


class TestNPCActionQueueing:
    """Tests for NPC action queuing and staggering."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()
        self.dispatched_effects: list[EffectEvent] = []

    def test_npc_action_is_queued(self) -> None:
        """NPC actions are queued, not dispatched immediately."""
        manager = PresentationManager()

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)

        # Create an NPC presentation event
        npc_event = PresentationEvent(
            effect_events=[EffectEvent("blood_splatter", x=10, y=10)],
            source_x=10,
            source_y=10,
            is_player_action=False,
        )
        publish_event(npc_event)

        # Should be queued, not dispatched yet
        assert manager.queue_size == 1
        assert len(self.dispatched_effects) == 0

    def test_npc_action_dispatched_after_delay(self) -> None:
        """NPC actions are dispatched after the base delay."""
        manager = PresentationManager(npc_base_delay=0.05, npc_stagger=0.10)

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)

        # Queue an NPC presentation
        npc_event = PresentationEvent(
            effect_events=[EffectEvent("blood_splatter", x=10, y=10)],
            source_x=10,
            source_y=10,
            is_player_action=False,
        )
        publish_event(npc_event)

        # Update but not enough time has passed
        manager.update(FixedTimestep(0.02))
        assert len(self.dispatched_effects) == 0

        # Update past the base delay
        manager.update(FixedTimestep(0.04))
        assert len(self.dispatched_effects) == 1
        assert manager.queue_size == 0

    def test_multiple_npc_actions_staggered(self) -> None:
        """Multiple NPC actions are staggered over time."""
        manager = PresentationManager(npc_base_delay=0.05, npc_stagger=0.10)

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)

        # Queue 3 NPC presentations
        for i in range(3):
            npc_event = PresentationEvent(
                effect_events=[EffectEvent(f"effect_{i}", x=i, y=i)],
                source_x=i,
                source_y=i,
                is_player_action=False,
            )
            publish_event(npc_event)

        assert manager.queue_size == 3

        # First NPC at 0.05s
        manager.update(FixedTimestep(0.06))
        assert len(self.dispatched_effects) == 1

        # Second NPC at 0.15s (0.05 + 0.10)
        manager.update(FixedTimestep(0.10))
        assert len(self.dispatched_effects) == 2

        # Third NPC at 0.25s (0.15 + 0.10)
        manager.update(FixedTimestep(0.10))
        assert len(self.dispatched_effects) == 3
        assert manager.queue_size == 0


class TestQueueManagement:
    """Tests for queue overflow and management."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()
        self.dispatched_effects: list[EffectEvent] = []

    def test_max_queue_size_drops_oldest(self) -> None:
        """When queue exceeds max size, oldest items are dropped."""
        manager = PresentationManager(max_queue_size=3)

        from catley.events import publish_event

        # Queue 5 NPC presentations (exceeds max of 3)
        for i in range(5):
            npc_event = PresentationEvent(
                effect_events=[EffectEvent(f"effect_{i}", x=i, y=i)],
                source_x=i,
                source_y=i,
                is_player_action=False,
            )
            publish_event(npc_event)

        # Should only have 3 items (the last 3)
        assert manager.queue_size == 3

    def test_clear_removes_all_pending(self) -> None:
        """clear() removes all pending presentations."""
        manager = PresentationManager()

        from catley.events import publish_event

        # Queue some presentations
        for i in range(3):
            npc_event = PresentationEvent(
                effect_events=[EffectEvent(f"effect_{i}", x=i, y=i)],
                source_x=i,
                source_y=i,
                is_player_action=False,
            )
            publish_event(npc_event)

        assert manager.queue_size == 3

        manager.clear()
        assert manager.queue_size == 0
        assert manager.has_pending_presentations is False

    def test_timing_compression_when_exceeding_window(self) -> None:
        """Timing is compressed when batch exceeds max_presentation_window."""
        # Use short window that will be exceeded by 5 items
        # Without compression: 0.1, 0.3, 0.5, 0.7, 0.9s (exceeds 0.5s window)
        manager = PresentationManager(
            npc_base_delay=0.1,
            npc_stagger=0.2,
            max_presentation_window=0.5,
            max_queue_size=10,
        )

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)

        # Queue 5 NPC presentations - with compression, timing should be scaled
        for i in range(5):
            npc_event = PresentationEvent(
                effect_events=[EffectEvent(f"effect_{i}", x=i, y=i)],
                source_x=i,
                source_y=i,
                is_player_action=False,
            )
            publish_event(npc_event)

        assert manager.queue_size == 5

        # After 0.5s, all should be dispatched due to compression
        manager.update(FixedTimestep(0.6))
        assert len(self.dispatched_effects) == 5
        assert manager.queue_size == 0


class TestUpdateBehavior:
    """Tests for update() method behavior."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()

    def test_update_with_empty_queue(self) -> None:
        """update() with empty queue does nothing and doesn't crash."""
        manager = PresentationManager()

        # Should not raise
        manager.update(FixedTimestep(0.1))
        assert manager.queue_size == 0

    def test_batch_timing_resets_after_completion(self) -> None:
        """Batch timing resets after all presentations are dispatched."""
        manager = PresentationManager(npc_base_delay=0.05, npc_stagger=0.10)
        dispatched: list[EffectEvent] = []

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, dispatched.append)

        # First batch
        npc_event = PresentationEvent(
            effect_events=[EffectEvent("effect_1", x=1, y=1)],
            source_x=1,
            source_y=1,
            is_player_action=False,
        )
        publish_event(npc_event)

        # Complete first batch
        manager.update(FixedTimestep(0.1))
        assert len(dispatched) == 1
        assert manager.queue_size == 0

        # Second batch should start fresh with base delay
        npc_event_2 = PresentationEvent(
            effect_events=[EffectEvent("effect_2", x=2, y=2)],
            source_x=2,
            source_y=2,
            is_player_action=False,
        )
        publish_event(npc_event_2)

        # Should need to wait base_delay again
        manager.update(FixedTimestep(0.02))
        assert len(dispatched) == 1  # Still only 1

        manager.update(FixedTimestep(0.04))
        assert len(dispatched) == 2  # Now 2


class TestMultipleEffectsAndSounds:
    """Tests for presentations with multiple effects/sounds."""

    def setup_method(self) -> None:
        """Reset the event bus before each test."""
        reset_event_bus_for_testing()
        self.dispatched_effects: list[EffectEvent] = []
        self.dispatched_sounds: list[SoundEvent] = []

    def test_all_effects_and_sounds_dispatched(self) -> None:
        """All effects and sounds in a presentation are dispatched together."""
        # Manager subscribes to PresentationEvent in __init__
        _manager = PresentationManager()

        from catley.events import publish_event, subscribe_to_event

        subscribe_to_event(EffectEvent, self.dispatched_effects.append)
        subscribe_to_event(SoundEvent, self.dispatched_sounds.append)

        # Create presentation with multiple effects and sounds
        event = PresentationEvent(
            effect_events=[
                EffectEvent("muzzle_flash", x=5, y=5),
                EffectEvent("shell_casing", x=5, y=5),
            ],
            sound_events=[
                SoundEvent("gun_fire", x=5, y=5),
                SoundEvent("shell_drop", x=5, y=5),
            ],
            source_x=5,
            source_y=5,
            is_player_action=True,  # Immediate dispatch
        )
        publish_event(event)

        # All effects and sounds should be dispatched
        assert len(self.dispatched_effects) == 2
        assert len(self.dispatched_sounds) == 2
