from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import colors

if TYPE_CHECKING:
    import tcod.console


class Message:
    """
    A single message in the message log.
    """

    def __init__(self, text: str, fg: tuple[int, int, int], count: int = 1) -> None:
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
    """
    Manages a log of messages with stacking and rendering capabilities.
    """

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add_message(
        self,
        text: str,
        fg: tuple[int, int, int] = colors.WHITE,
        *,
        stack: bool = True,
    ) -> None:
        """
        Add a message to this log.
        If 'stack' is True, identical consecutive messages (text and color)
        will be stacked.
        """
        if (
            stack
            and self.messages
            and self.messages[-1].plain_text == text
            and self.messages[-1].fg == fg
        ):
            self.messages[-1].count += 1
        else:
            self.messages.append(Message(text, fg))

    def render(
        self,
        console: tcod.console.Console,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        """
        Render this log over the given area on the console.
        Messages are rendered from the bottom-up.
        """
        y_offset = (
            height - 1
        )  # Start drawing from the bottom line of the allocated height

        for message in reversed(self.messages):
            # Wrap the message text if it's too long for the width
            wrapped_lines = textwrap.wrap(message.full_text, width)

            # Print each line of the (potentially wrapped) message, moving upwards
            for line in reversed(wrapped_lines):
                if y_offset < 0:
                    return  # No more space in the message log area
                console.print(x=x, y=y + y_offset, string=line, fg=message.fg)
                y_offset -= 1
            if (
                y_offset < 0
            ):  # Check again in case all lines of a single message filled the space
                return
