"""Unit tests for the floating text effect system.

Tests cover:
- FloatingText animation behavior (update, rising, fading)
- FloatingText color selection based on valence
- FloatingTextManager lifecycle (creation, max limit, cleanup)
- Event handling creating FloatingText instances
- Texture cleanup lifecycle (via mocking since GPU ops are backend-specific)

Graphics rendering is NOT tested here as it requires actual backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest

from catley import colors
from catley.events import FloatingTextSize, publish_event
from catley.game.actors import Actor
from catley.game.game_world import GameWorld
from catley.types import DeltaTime
from catley.view.render.effects.floating_text import (
    VALENCE_COLORS,
    FloatingText,
    FloatingTextEvent,
    FloatingTextManager,
    FloatingTextValence,
)
from tests.helpers import DummyGameWorld

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Test Fixtures / Helpers
# ---------------------------------------------------------------------------


def make_floating_text(
    text: str = "Test",
    valence: FloatingTextValence = FloatingTextValence.NEUTRAL,
    size: FloatingTextSize = FloatingTextSize.NORMAL,
    world_x: float = 5.0,
    world_y: float = 5.0,
    duration: float = 0.7,
) -> FloatingText:
    """Create a FloatingText instance for testing."""
    ft = FloatingText(
        text=text,
        target_actor_id=12345,
        valence=valence,
        size=size,
        world_x=world_x,
        world_y=world_y,
    )
    ft.duration = duration
    return ft


def make_world() -> DummyGameWorld:
    """Create a simple DummyGameWorld for testing."""
    return DummyGameWorld(width=30, height=30)


def make_actor(gw: DummyGameWorld, x: int = 5, y: int = 5) -> Actor:
    """Create an actor at the given position."""
    return Actor(
        x=x,
        y=y,
        ch="A",
        color=colors.WHITE,
        name="TestActor",
        game_world=cast(GameWorld, gw),
        blocks_movement=True,
    )


# ---------------------------------------------------------------------------
# TestFloatingTextAnimation
# ---------------------------------------------------------------------------


class TestFloatingTextAnimation:
    """Tests for FloatingText.update() animation behavior."""

    def test_initial_state(self) -> None:
        """FloatingText should start with full alpha and no offset."""
        ft = make_floating_text()

        assert ft.elapsed == 0.0
        assert ft.alpha == 1.0
        assert ft.offset_y == 0.0

    def test_update_increments_elapsed_time(self) -> None:
        """Update should increment elapsed time by delta."""
        ft = make_floating_text()

        ft.update(DeltaTime(0.1))

        assert ft.elapsed == pytest.approx(0.1)

    def test_update_returns_false_while_animating(self) -> None:
        """Update returns False while animation is in progress."""
        ft = make_floating_text(duration=1.0)

        result = ft.update(DeltaTime(0.5))

        assert result is False

    def test_update_returns_true_when_complete(self) -> None:
        """Update returns True when animation duration is reached."""
        ft = make_floating_text(duration=0.5)

        result = ft.update(DeltaTime(0.5))

        assert result is True

    def test_update_returns_true_when_past_duration(self) -> None:
        """Update returns True when elapsed exceeds duration."""
        ft = make_floating_text(duration=0.5)

        result = ft.update(DeltaTime(1.0))

        assert result is True

    def test_offset_y_rises_over_time(self) -> None:
        """Text should rise (negative Y offset) as animation progresses."""
        ft = make_floating_text(duration=1.0)
        ft.rise_distance_tiles = 1.5

        ft.update(DeltaTime(0.5))

        # Halfway through, offset should be negative (rising)
        assert ft.offset_y < 0

    def test_offset_y_reaches_target_at_end(self) -> None:
        """At animation end, offset should approach rise_distance_tiles."""
        ft = make_floating_text(duration=1.0)
        ft.rise_distance_tiles = 1.5

        # Almost complete - close to full rise
        ft.update(DeltaTime(0.99))

        # Should be close to -rise_distance_tiles (with easing)
        assert ft.offset_y < -1.0

    def test_alpha_stays_full_in_first_half(self) -> None:
        """Alpha should remain 1.0 during the first half of animation."""
        ft = make_floating_text(duration=1.0)

        ft.update(DeltaTime(0.4))  # 40% through - still in first half

        assert ft.alpha == 1.0

    def test_alpha_fades_in_second_half(self) -> None:
        """Alpha should fade from 1.0 to 0.0 in the second half."""
        ft = make_floating_text(duration=1.0)

        ft.update(DeltaTime(0.75))  # 75% through - should be fading

        # At 75%, fade_progress = (0.75 - 0.5) / 0.5 = 0.5
        # So alpha = 1.0 - 0.5 = 0.5
        assert ft.alpha == pytest.approx(0.5)

    def test_alpha_approaches_zero_at_end(self) -> None:
        """Alpha should approach 0.0 near the end of animation."""
        ft = make_floating_text(duration=1.0)

        ft.update(DeltaTime(0.99))

        assert ft.alpha < 0.1

    def test_multiple_updates_accumulate(self) -> None:
        """Multiple update calls should accumulate elapsed time."""
        ft = make_floating_text(duration=1.0)

        ft.update(DeltaTime(0.3))
        ft.update(DeltaTime(0.3))
        ft.update(DeltaTime(0.3))

        assert ft.elapsed == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestFloatingTextColor
# ---------------------------------------------------------------------------


class TestFloatingTextColor:
    """Tests for FloatingText.get_color() based on valence."""

    def test_negative_valence_returns_red(self) -> None:
        """NEGATIVE valence should return red color."""
        ft = make_floating_text(valence=FloatingTextValence.NEGATIVE)

        color = ft.get_color()

        assert color == VALENCE_COLORS[FloatingTextValence.NEGATIVE]
        assert color[0] > color[1]  # Red component dominant

    def test_positive_valence_returns_green(self) -> None:
        """POSITIVE valence should return green color."""
        ft = make_floating_text(valence=FloatingTextValence.POSITIVE)

        color = ft.get_color()

        assert color == VALENCE_COLORS[FloatingTextValence.POSITIVE]
        assert color[1] > color[0]  # Green component dominant

    def test_neutral_valence_returns_yellow(self) -> None:
        """NEUTRAL valence should return yellow color."""
        ft = make_floating_text(valence=FloatingTextValence.NEUTRAL)

        color = ft.get_color()

        assert color == VALENCE_COLORS[FloatingTextValence.NEUTRAL]
        # Yellow has high red and green, low blue
        assert color[0] > color[2]
        assert color[1] > color[2]

    def test_all_valences_have_colors(self) -> None:
        """Every FloatingTextValence should have a color mapping."""
        for valence in FloatingTextValence:
            ft = make_floating_text(valence=valence)
            color = ft.get_color()
            assert color is not None
            assert len(color) == 3

    def test_color_override_takes_precedence(self) -> None:
        """color_override should take precedence over valence color."""
        custom_color = (100, 150, 200)
        ft = FloatingText(
            text="Override",
            target_actor_id=1,
            valence=FloatingTextValence.NEGATIVE,  # Would be red
            size=FloatingTextSize.NORMAL,
            world_x=0,
            world_y=0,
            color_override=custom_color,
        )

        assert ft.get_color() == custom_color


# ---------------------------------------------------------------------------
# TestFloatingTextManager
# ---------------------------------------------------------------------------


class TestFloatingTextManagerLifecycle:
    """Tests for FloatingTextManager creation, limits, and cleanup."""

    def test_initial_state_empty(self) -> None:
        """Manager should start with no active texts."""
        manager = FloatingTextManager()

        assert manager.active_count == 0

    def test_max_texts_configurable(self) -> None:
        """Manager should accept custom max_texts limit."""
        manager = FloatingTextManager(max_texts=5)

        assert manager.max_texts == 5

    def test_clear_removes_all_texts(self) -> None:
        """clear() should remove all active floating texts."""
        manager = FloatingTextManager()
        # Manually add some texts for testing
        manager._texts.append(make_floating_text("Test1"))
        manager._texts.append(make_floating_text("Test2"))

        manager.clear()

        assert manager.active_count == 0

    def test_update_removes_completed_texts(self) -> None:
        """update() should remove texts that have completed their animation."""
        manager = FloatingTextManager()
        ft1 = make_floating_text(duration=0.5)
        ft2 = make_floating_text(duration=1.0)
        manager._texts = [ft1, ft2]

        # Update with enough time to complete ft1 but not ft2
        manager.update(DeltaTime(0.6))

        assert manager.active_count == 1

    def test_update_keeps_active_texts(self) -> None:
        """update() should keep texts that are still animating."""
        manager = FloatingTextManager()
        ft = make_floating_text(duration=1.0)
        manager._texts = [ft]

        manager.update(DeltaTime(0.3))

        assert manager.active_count == 1
        assert ft.elapsed == pytest.approx(0.3)

    def test_update_with_no_texts_does_not_error(self) -> None:
        """update() should handle empty text list gracefully."""
        manager = FloatingTextManager()

        manager.update(DeltaTime(0.1))  # Should not raise

        assert manager.active_count == 0


class TestFloatingTextManagerEventHandling:
    """Tests for FloatingTextManager event subscription and handling."""

    def test_handle_event_creates_floating_text(self) -> None:
        """Handling an event should create a new FloatingText."""
        manager = FloatingTextManager()

        event = FloatingTextEvent(
            text="-5 HP",
            target_actor_id=12345,
            valence=FloatingTextValence.NEGATIVE,
            world_x=10,
            world_y=20,
        )
        manager._handle_event(event)

        assert manager.active_count == 1

    def test_handle_event_sets_text_properties(self) -> None:
        """Created FloatingText should have properties from the event."""
        manager = FloatingTextManager()

        event = FloatingTextEvent(
            text="TRIPPED",
            target_actor_id=99999,
            valence=FloatingTextValence.POSITIVE,
            world_x=15,
            world_y=25,
        )
        manager._handle_event(event)

        ft = manager._texts[0]
        assert ft.text == "TRIPPED"
        assert ft.target_actor_id == 99999
        assert ft.valence == FloatingTextValence.POSITIVE
        assert ft.world_x == 15.0
        assert ft.world_y == 25.0

    def test_max_limit_enforced(self) -> None:
        """Adding texts beyond max_texts should remove oldest."""
        manager = FloatingTextManager(max_texts=3)

        # Add 4 texts
        for i in range(4):
            event = FloatingTextEvent(
                text=f"Text{i}",
                target_actor_id=i,
                valence=FloatingTextValence.NEUTRAL,
                world_x=0,
                world_y=0,
            )
            manager._handle_event(event)

        assert manager.active_count == 3
        # First text should have been removed
        texts = [ft.text for ft in manager._texts]
        assert "Text0" not in texts
        assert "Text3" in texts

    def test_max_limit_removes_oldest_first(self) -> None:
        """When at max, oldest text (first in list) should be removed."""
        manager = FloatingTextManager(max_texts=2)

        event1 = FloatingTextEvent(
            text="First",
            target_actor_id=1,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )
        event2 = FloatingTextEvent(
            text="Second",
            target_actor_id=2,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )
        event3 = FloatingTextEvent(
            text="Third",
            target_actor_id=3,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )

        manager._handle_event(event1)
        manager._handle_event(event2)
        manager._handle_event(event3)

        texts = [ft.text for ft in manager._texts]
        assert texts == ["Second", "Third"]

    def test_event_subscription_works(self) -> None:
        """Manager should receive events via the event system."""
        # Create manager - it subscribes to events in __init__
        manager = FloatingTextManager()

        # Emit an event through the event system
        event = FloatingTextEvent(
            text="Event System Test",
            target_actor_id=42,
            valence=FloatingTextValence.NEUTRAL,
            world_x=5,
            world_y=5,
        )
        publish_event(event)

        assert manager.active_count == 1
        assert manager._texts[0].text == "Event System Test"

    def test_handle_event_uses_custom_duration(self) -> None:
        """Custom duration from event should override the default."""
        manager = FloatingTextManager()

        event = FloatingTextEvent(
            text="Long Duration",
            target_actor_id=1,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
            duration=2.5,
        )
        manager._handle_event(event)

        assert manager._texts[0].duration == 2.5

    def test_handle_event_uses_color_override(self) -> None:
        """Custom color from event should be passed to FloatingText."""
        manager = FloatingTextManager()

        custom_color = (128, 64, 255)
        event = FloatingTextEvent(
            text="Custom Color",
            target_actor_id=1,
            valence=FloatingTextValence.NEGATIVE,  # Would normally be red
            world_x=0,
            world_y=0,
            color=custom_color,
        )
        manager._handle_event(event)

        ft = manager._texts[0]
        assert ft.color_override == custom_color
        # get_color() should return the override, not the valence color
        assert ft.get_color() == custom_color

    def test_handle_event_uses_large_size(self) -> None:
        """LARGE size from event should be passed to FloatingText."""
        manager = FloatingTextManager()

        event = FloatingTextEvent(
            text="Big Text",
            target_actor_id=1,
            valence=FloatingTextValence.NEUTRAL,
            size=FloatingTextSize.LARGE,
            world_x=0,
            world_y=0,
        )
        manager._handle_event(event)

        assert manager._texts[0].size == FloatingTextSize.LARGE


# ---------------------------------------------------------------------------
# TestFloatingTextEvent
# ---------------------------------------------------------------------------


class TestFloatingTextEvent:
    """Tests for FloatingTextEvent dataclass."""

    def test_default_valence_is_neutral(self) -> None:
        """Default valence should be NEUTRAL."""
        event = FloatingTextEvent(
            text="Test",
            target_actor_id=1,
        )

        assert event.valence == FloatingTextValence.NEUTRAL

    def test_default_world_position_is_zero(self) -> None:
        """Default world position should be (0, 0)."""
        event = FloatingTextEvent(
            text="Test",
            target_actor_id=1,
        )

        assert event.world_x == 0
        assert event.world_y == 0

    def test_all_fields_settable(self) -> None:
        """All event fields should be settable."""
        event = FloatingTextEvent(
            text="Custom",
            target_actor_id=999,
            valence=FloatingTextValence.NEGATIVE,
            world_x=100,
            world_y=200,
        )

        assert event.text == "Custom"
        assert event.target_actor_id == 999
        assert event.valence == FloatingTextValence.NEGATIVE
        assert event.world_x == 100
        assert event.world_y == 200


# ---------------------------------------------------------------------------
# TestFloatingTextValence
# ---------------------------------------------------------------------------


class TestFloatingTextValence:
    """Tests for FloatingTextValence enum."""

    def test_has_three_values(self) -> None:
        """Enum should have exactly three valence types."""
        assert len(FloatingTextValence) == 3

    def test_all_valences_in_color_map(self) -> None:
        """All valence values should have corresponding colors."""
        for valence in FloatingTextValence:
            assert valence in VALENCE_COLORS


# ---------------------------------------------------------------------------
# TestFloatingTextTextureCleanup
# ---------------------------------------------------------------------------


class TestFloatingTextTextureCleanup:
    """Tests for FloatingText texture lifecycle and cleanup.

    Since actual GPU texture destruction is backend-specific, these tests
    verify that:
    - cleanup_texture() sets _texture to None
    - cleanup_texture() is called at the right times (via mocking)
    """

    def test_cleanup_texture_sets_texture_to_none(self) -> None:
        """cleanup_texture() should set _texture to None and call release_texture."""
        ft = make_floating_text()
        # Simulate having a texture by setting _texture to a mock object
        mock_texture = MagicMock()
        ft._texture = mock_texture

        mock_graphics = MagicMock()
        ft.cleanup_texture(mock_graphics)

        assert ft._texture is None
        mock_graphics.release_texture.assert_called_once_with(mock_texture)

    def test_cleanup_texture_handles_none_texture(self) -> None:
        """cleanup_texture() should handle already-None texture gracefully."""
        ft = make_floating_text()
        assert ft._texture is None

        mock_graphics = MagicMock()
        # Should not raise
        ft.cleanup_texture(mock_graphics)

        assert ft._texture is None
        # release_texture should not be called when texture is None
        mock_graphics.release_texture.assert_not_called()

    def test_update_queues_completed_texts_for_cleanup(self) -> None:
        """Manager.update() should queue completed texts for deferred cleanup."""
        manager = FloatingTextManager()
        ft = make_floating_text(duration=0.5)
        ft._texture = MagicMock()  # Simulate having a texture
        manager._texts = [ft]

        # Update with enough time to complete the animation
        manager.update(DeltaTime(0.6))

        # Text should be removed from active list
        assert manager.active_count == 0
        # But queued for cleanup (happens in render())
        assert len(manager._pending_cleanup) == 1
        assert manager._pending_cleanup[0] is ft

    def test_update_does_not_queue_active_texts_for_cleanup(self) -> None:
        """Manager.update() should NOT queue texts for cleanup while animating."""
        manager = FloatingTextManager()
        ft = make_floating_text(duration=1.0)
        ft._texture = MagicMock()
        manager._texts = [ft]

        # Update with time that does NOT complete animation
        manager.update(DeltaTime(0.3))

        # Text should still be active
        assert manager.active_count == 1
        # And not queued for cleanup
        assert len(manager._pending_cleanup) == 0

    def test_max_limit_queues_removed_text_for_cleanup(self) -> None:
        """_handle_event() should queue removed texts for deferred cleanup."""
        manager = FloatingTextManager(max_texts=2)

        # Add first two texts
        event1 = FloatingTextEvent(
            text="First",
            target_actor_id=1,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )
        event2 = FloatingTextEvent(
            text="Second",
            target_actor_id=2,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )
        manager._handle_event(event1)
        manager._handle_event(event2)

        # Get reference to first text and give it a mock texture
        first_text = manager._texts[0]
        first_text._texture = MagicMock()

        # Add third text - should trigger removal of first
        event3 = FloatingTextEvent(
            text="Third",
            target_actor_id=3,
            valence=FloatingTextValence.NEUTRAL,
            world_x=0,
            world_y=0,
        )
        manager._handle_event(event3)

        # Should still have max_texts items
        assert manager.active_count == 2
        # First text should be gone from active list
        texts = [ft.text for ft in manager._texts]
        assert "First" not in texts
        # But queued for cleanup (happens in render())
        assert len(manager._pending_cleanup) == 1
        assert manager._pending_cleanup[0] is first_text

    def test_clear_with_graphics_calls_cleanup_on_all_texts(self) -> None:
        """clear(graphics) should call cleanup_texture() on every text immediately."""
        manager = FloatingTextManager()

        # Add multiple texts
        ft1 = make_floating_text("Text1")
        ft2 = make_floating_text("Text2")
        ft3 = make_floating_text("Text3")
        ft1._texture = MagicMock()
        ft2._texture = MagicMock()
        ft3._texture = MagicMock()
        manager._texts = [ft1, ft2, ft3]

        mock_graphics = MagicMock()
        with (
            patch.object(ft1, "cleanup_texture") as mock1,
            patch.object(ft2, "cleanup_texture") as mock2,
            patch.object(ft3, "cleanup_texture") as mock3,
        ):
            manager.clear(mock_graphics)

            mock1.assert_called_once_with(mock_graphics)
            mock2.assert_called_once_with(mock_graphics)
            mock3.assert_called_once_with(mock_graphics)

        assert manager.active_count == 0

    def test_clear_without_graphics_queues_for_cleanup(self) -> None:
        """clear() without graphics should queue texts for deferred cleanup."""
        manager = FloatingTextManager()

        # Add multiple texts
        ft1 = make_floating_text("Text1")
        ft2 = make_floating_text("Text2")
        ft1._texture = MagicMock()
        ft2._texture = MagicMock()
        manager._texts = [ft1, ft2]

        manager.clear()  # No graphics context provided

        assert manager.active_count == 0
        # Texts should be queued for cleanup (happens in render())
        assert len(manager._pending_cleanup) == 2

    def test_clear_handles_empty_manager(self) -> None:
        """clear() should handle empty text list gracefully."""
        manager = FloatingTextManager()

        # Should not raise
        manager.clear()

        assert manager.active_count == 0

    def test_multiple_completions_all_queued_for_cleanup(self) -> None:
        """update() should queue all completed texts for deferred cleanup."""
        manager = FloatingTextManager()
        ft1 = make_floating_text(duration=0.3)
        ft2 = make_floating_text(duration=0.4)
        ft3 = make_floating_text(duration=1.0)  # This one stays active
        ft1._texture = MagicMock()
        ft2._texture = MagicMock()
        ft3._texture = MagicMock()
        manager._texts = [ft1, ft2, ft3]

        # Update with time that completes ft1 and ft2, but not ft3
        manager.update(DeltaTime(0.5))

        # Only ft3 should remain active
        assert manager.active_count == 1
        # ft1 and ft2 should be queued for cleanup
        assert len(manager._pending_cleanup) == 2
        assert ft1 in manager._pending_cleanup
        assert ft2 in manager._pending_cleanup
        assert ft3 not in manager._pending_cleanup

    def test_render_clears_pending_cleanup(self) -> None:
        """render() should call cleanup_texture() on all pending texts."""
        manager = FloatingTextManager()

        # Add texts that need cleanup
        ft1 = make_floating_text("Text1")
        ft2 = make_floating_text("Text2")
        mock_texture1 = MagicMock()
        mock_texture2 = MagicMock()
        ft1._texture = mock_texture1
        ft2._texture = mock_texture2
        manager._pending_cleanup = [ft1, ft2]

        # Mock graphics and other dependencies
        mock_graphics = MagicMock()
        mock_viewport = MagicMock()
        mock_game_world = make_world()

        with (
            patch.object(ft1, "cleanup_texture") as mock1,
            patch.object(ft2, "cleanup_texture") as mock2,
        ):
            manager.render(mock_graphics, mock_viewport, (0, 0), mock_game_world)

            mock1.assert_called_once_with(mock_graphics)
            mock2.assert_called_once_with(mock_graphics)

        # Pending cleanup should be cleared
        assert len(manager._pending_cleanup) == 0
