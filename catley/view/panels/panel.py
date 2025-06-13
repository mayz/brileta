"""Panel System - Persistent sub-regions of the game window that
know how to draw themselves.

Panels are always-visible UI that show game state:
- The main, central game world view
- The message log
- Health, ammo, FPS displays
- Fixed-position interface components

Use panels for persistent UI. Use overlays for temporary UI.

Two-phase rendering: draw() for console ops, present() for SDL ops.
"""

import abc

from catley.view.renderer import Renderer
from catley.view.text_backend import TextBackend


class Panel(abc.ABC):
    """Base class for persistent UI panels.

    Panels are always visible and show continuously updated game info.
    Positioned by FrameManager, rendered every frame.

    Two phases: draw() for console rendering, present() for SDL effects.
    """

    def __init__(self, text_backend: TextBackend | None = None) -> None:
        """Initialize panel. Call resize() before drawing."""
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.visible = True
        self.text_backend = text_backend

    @abc.abstractmethod
    def draw(self, renderer: Renderer) -> None:
        """Draw panel content using console operations. Called every frame."""
        pass

    def present(self, renderer: Renderer) -> None:
        """Optional SDL post-processing after console rendering."""
        _ = renderer

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set panel screen boundaries. Called by FrameManager."""
        self.x = x1
        self.y = y1
        self.width = x2 - x1
        self.height = y2 - y1

    def show(self) -> None:
        """Make panel visible in rendering pipeline."""
        self.visible = True

    def hide(self) -> None:
        """Skip panel during rendering."""
        self.visible = False
