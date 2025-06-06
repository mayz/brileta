import abc

from catley.view.render.renderer import Renderer


class Panel(abc.ABC):
    """Abstract base class for UI panels.

    ``draw`` prepares this panel's contents using the provided console-based
    resources.  ``present`` runs after the root console has been converted to a
    texture and allows the panel to perform additional low-level rendering using
    the :class:`Renderer` interface.  This separation lets panels cache textures
    or issue custom draw calls while keeping ``draw`` agnostic of the underlying
    rendering backend.
    """

    def __init__(self) -> None:
        """Initialize panel with default dimensions.

        Panel dimensions must be set via resize() before drawing.
        """
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.visible = True

    @abc.abstractmethod
    def draw(self, renderer: Renderer) -> None:
        """Draw this panel using the provided renderer.

        Note: resize() must be called before draw() to set panel boundaries.
        """
        pass

    def present(self, renderer: Renderer) -> None:
        """Finalize rendering using the provided renderer.

        Called once per frame after the root console has been drawn.  Override
        this method to copy cached textures or perform other low-level
        operations that can't be expressed through the console API alone.
        """
        _ = renderer

    def resize(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Set panel boundaries. Panel must enforce these constraints during rendering.

        This should be called by FrameManager before the panel is drawn.

        Args:
            x1, y1: Top-left corner (inclusive)
            x2, y2: Bottom-right corner (exclusive)
        """
        self.x = x1
        self.y = y1
        self.width = x2 - x1
        self.height = y2 - y1

    def show(self) -> None:
        """Make panel visible."""
        self.visible = True

    def hide(self) -> None:
        """Make panel invisible."""
        self.visible = False
