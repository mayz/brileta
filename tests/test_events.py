"""Tests for the event bus system."""

import logging

import pytest

from catley.events import (
    EventBus,
    GameEvent,
    MessageEvent,
    publish_event,
    reset_event_bus_for_testing,
    subscribe_to_event,
)


class TestEventBus:
    """Tests for the EventBus class."""

    def test_handler_exception_does_not_crash_event_bus(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that exceptions in handlers don't crash the event bus.

        The event bus should log the exception and continue processing
        other handlers.
        """
        bus = EventBus()
        handler_calls: list[str] = []

        def failing_handler(event: GameEvent) -> None:
            handler_calls.append("failing")
            raise ValueError("Handler failed!")

        def succeeding_handler(event: GameEvent) -> None:
            handler_calls.append("succeeding")

        # Subscribe both handlers
        bus.subscribe(MessageEvent, failing_handler)
        bus.subscribe(MessageEvent, succeeding_handler)

        # Publish an event - should not raise
        event = MessageEvent(text="Test message")
        with caplog.at_level(logging.ERROR):
            bus.publish(event)

        # Both handlers should have been called
        assert "failing" in handler_calls
        assert "succeeding" in handler_calls

        # Exception should have been logged
        assert "Error handling event MessageEvent" in caplog.text

    def test_handler_exception_includes_traceback_in_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that handler exceptions log full tracebacks.

        Using logger.exception() should include the full traceback,
        not just the error message.
        """
        bus = EventBus()

        def nested_failure(event: GameEvent) -> None:
            def inner_function() -> None:
                raise RuntimeError("Deep error")

            inner_function()

        bus.subscribe(MessageEvent, nested_failure)

        event = MessageEvent(text="Test")
        with caplog.at_level(logging.ERROR):
            bus.publish(event)

        # Should contain traceback information
        assert "RuntimeError" in caplog.text
        assert "Deep error" in caplog.text


class TestGlobalEventBus:
    """Tests for the global event bus API."""

    def setup_method(self) -> None:
        """Reset the global event bus before each test."""
        reset_event_bus_for_testing()

    def test_global_subscribe_and_publish(self) -> None:
        """Test that global subscribe/publish works correctly."""
        received_events: list[MessageEvent] = []

        def handler(event: MessageEvent) -> None:
            received_events.append(event)

        subscribe_to_event(MessageEvent, handler)
        publish_event(MessageEvent(text="Hello"))

        assert len(received_events) == 1
        assert received_events[0].text == "Hello"

    def test_global_handler_exception_is_handled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that global event bus handles exceptions gracefully."""

        def bad_handler(event: MessageEvent) -> None:
            raise Exception("Oops!")

        subscribe_to_event(MessageEvent, bad_handler)

        # Should not raise
        with caplog.at_level(logging.ERROR):
            publish_event(MessageEvent(text="Test"))

        assert "Error handling event MessageEvent" in caplog.text
