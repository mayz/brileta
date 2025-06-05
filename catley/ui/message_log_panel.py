from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from tcod.console import Console

from catley.ui.panel import Panel

if TYPE_CHECKING:
    from catley.util.message_log import MessageLog


class MessageLogPanel(Panel):
    """Panel for displaying the message log in the bottom-left corner."""

    def __init__(
        self,
        message_log: MessageLog,
        x: int,
        y: int,
        width: int,
        height: int,
        root_console: Console,
    ) -> None:
        super().__init__(x, y, width, height)
        self.message_log = message_log
        self.root_console = root_console

        # Create console for text rendering
        self.text_console = Console(width, height, order="F")

    def draw(self) -> None:
        """Draw the message log panel."""
        if not self.visible:
            return

        # Clear the text console
        self.text_console.clear()

        # Render messages to the text console
        self._render_messages_to_console()

        # Blit to root console
        self.text_console.blit(
            dest=self.root_console,
            dest_x=self.x,
            dest_y=self.y,
        )

    def _render_messages_to_console(self) -> None:
        """Render messages to the text console."""
        y_offset = self.height - 1  # Start from bottom

        for message in reversed(self.message_log.messages):
            if y_offset < 0:
                break

            # Wrap message text to fit console width
            wrapped_lines = textwrap.wrap(message.full_text, self.text_console.width)

            # Draw each line, moving upward
            for line in reversed(wrapped_lines):
                if y_offset < 0:
                    break
                self.text_console.print(x=0, y=y_offset, text=line, fg=message.fg)
                y_offset -= 1

            if y_offset < 0:
                break
