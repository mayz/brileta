import abc

import tcod.sdl.render

from catley.render.renderer import Renderer


class Panel(abc.ABC):
    """Abstract base class for UI panels.

    ``draw`` prepares this panel's contents using the provided console-based
    resources.  ``present`` runs after the root console has been converted to an
    SDL texture and allows the panel to perform additional low-level rendering
    using ``sdl_renderer``.  This separation lets panels cache SDL textures or
    issue custom draw calls while keeping ``draw`` agnostic of the underlying
    rendering backend.
    """

    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.visible = True

    @abc.abstractmethod
    def draw(self, renderer: Renderer) -> None:
        """Draw this panel using the provided renderer."""
        pass

    def present(self, sdl_renderer: tcod.sdl.render.Renderer) -> None:  # noqa: B027
        """Finalize rendering using the underlying SDL renderer.

        Called once per frame after the root console has been drawn.  Override
        this method to copy cached textures or perform other SDL operations that
        can't be expressed through the console API alone.
        """
        pass

    def set_position(self, x: int, y: int) -> None:
        """Update panel position."""
        self.x = x
        self.y = y

    def set_size(self, width: int, height: int) -> None:
        """Update panel size."""
        self.width = width
        self.height = height

    def show(self) -> None:
        """Make panel visible."""
        self.visible = True

    def hide(self) -> None:
        """Make panel invisible."""
        self.visible = False
