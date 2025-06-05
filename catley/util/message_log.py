from catley import colors


class Message:
    """A single message in the message log."""

    def __init__(self, text: str, fg: colors.Color, count: int = 1) -> None:
        self.plain_text = text
        self.fg = fg
        self.count = count

    @property
    def full_text(self) -> str:
        """The full text of this message, including the count if > 1."""
        if self.count > 1:
            return f"{self.plain_text} (x{self.count})"
        return self.plain_text


class MessageLog:
    """Manages a log of messages with stacking capabilities."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        # revision increments whenever a message is added or stacked.
        # Render layers use this to efficiently update cached textures
        # only when the log actually changes.
        self.revision = 0

    def add_message(
        self,
        text: str,
        fg: colors.Color = colors.WHITE,
        *,
        stack: bool = True,
    ) -> None:
        """Add a message to this log."""
        if (
            stack
            and self.messages
            and self.messages[-1].plain_text == text
            and self.messages[-1].fg == fg
        ):
            self.messages[-1].count += 1
        else:
            self.messages.append(Message(text, fg))
        # increment revision so observers know the log changed
        self.revision += 1
