from dataclasses import dataclass

from catley import colors, config
from catley.events import MessageEvent, subscribe_to_event


@dataclass(slots=True)
class Message:
    """A single message in the message log."""

    plain_text: str
    fg: colors.Color
    count: int = 1
    sequence_number: int = 0

    @property
    def full_text(self) -> str:
        """The full text of this message, including the count if > 1."""
        text = self.plain_text
        if self.count > 1:
            text = f"{text} (x{self.count})"
        if config.SHOW_MESSAGE_SEQUENCE_NUMBERS:
            return f"[{self.sequence_number}] {text}"
        return text


class MessageLog:
    """Manages a log of messages with stacking capabilities."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        # revision increments whenever a message is added or stacked.
        # Render layers use this to efficiently update cached textures
        # only when the log actually changes.
        self.revision = 0
        self.message_sequence = 0

        # Subscribe to global message events
        subscribe_to_event(MessageEvent, self._handle_message_event)

    def _handle_message_event(self, event: MessageEvent) -> None:
        """Handle message events from the global event bus."""
        self.add_message(event.text, event.color, stack=event.stack)

    def add_message(
        self,
        text: str,
        fg: colors.Color = colors.WHITE,
        *,
        stack: bool = True,
    ) -> None:
        """Add a message to this log."""
        self.message_sequence += 1
        if (
            stack
            and self.messages
            and self.messages[-1].plain_text == text
            and self.messages[-1].fg == fg
        ):
            self.messages[-1].count += 1
            message = self.messages[-1]
        else:
            message = Message(text, fg, sequence_number=self.message_sequence)
            self.messages.append(message)
        if config.PRINT_MESSAGES_TO_CONSOLE:
            print(message.full_text)
        # increment revision so observers know the log changed
        self.revision += 1
