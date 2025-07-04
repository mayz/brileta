from __future__ import annotations

import moderngl
import pyglet
import tcod
from pyglet.window import key, mouse

from catley.app import App, AppConfig
from catley.backends.moderngl.graphics import ModernGLGraphicsContext


class PygletApp(App[ModernGLGraphicsContext]):
    """
    The Pyglet implementation of the application driver.

    Uses Pyglet's event-driven architecture with scheduled callbacks to implement
    the shared fixed timestep game loop pattern.
    """

    def __init__(self, app_config: AppConfig) -> None:
        super().__init__(app_config)

        self._initialize_window(app_config)
        self._initialize_graphics()
        self._initialize_controller()
        self._register_callbacks()

    def _initialize_window(self, app_config: AppConfig) -> None:
        """ """
        # Create Pyglet-specific window and context
        # Convert console tile dimensions to reasonable pixel dimensions
        # AppConfig dimensions are in tiles, but Pyglet needs pixels
        tile_width = 20  # From TILESET (Taffer_20x20.png)
        tile_height = 20
        pixel_width = app_config.width * tile_width
        pixel_height = app_config.height * tile_height

        self.window = pyglet.window.Window(
            width=pixel_width,
            height=pixel_height,
            caption=app_config.title,
            vsync=app_config.vsync,
            resizable=app_config.resizable,
        )

        # Apply window state based on AppConfig
        if app_config.fullscreen:
            self.window.set_fullscreen(True)
        elif app_config.maximized:
            self.window.maximize()

    def _initialize_graphics(self) -> None:
        self.mgl_context = moderngl.create_context()
        self.graphics = ModernGLGraphicsContext(self.window, self.mgl_context)

    def run(self) -> None:
        """Starts the main application loop and runs the game."""
        try:
            # Hide system cursor since we draw our own
            self.window.set_mouse_visible(False)

            assert self.controller is not None
            target_fps = self.controller.target_fps
            interval = (
                1.0 / target_fps if target_fps is not None and target_fps > 0 else 0.0
            )

            pyglet.app.run(interval=interval)
        finally:
            # Restore system cursor when exiting
            self.window.set_mouse_visible(True)

    def on_draw(self) -> None:
        """Called by Pyglet to render. Handles full game loop with accumulator."""
        assert self.controller is not None

        # Get delta time and update the controller's clock to calculate FPS.
        dt = self.controller.clock.tick()

        # Use the shared accumulator-based game logic update
        # This handles player input, fixed timestep logic, and death spiral protection
        self.update_game_logic(dt)

        # Render the frame using the shared rendering method
        # This calculates alpha from the accumulator and renders with interpolation
        self.render_frame()

    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        self.graphics.prepare_to_present()

    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen, but does NOT flip."""
        self.graphics.finalize_present()

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        self.window.set_fullscreen(not self.window.fullscreen)

    def _exit_backend(self) -> None:
        """Performs backend-specific exit procedures for Pyglet."""
        pyglet.app.exit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.RETURN and (modifiers & key.MOD_ALT):
            self.toggle_fullscreen()
        else:
            event = tcod.event.KeyDown(
                sym=self._pyglet_key_to_tcod(symbol),
                scancode=0,  # Not provided by pyglet
                mod=self._pyglet_mod_to_tcod(modifiers),
            )
            assert self.controller is not None
            self.controller.input_handler.dispatch(event)

    def on_key_release(self, symbol, modifiers):
        event = tcod.event.KeyUp(
            sym=self._pyglet_key_to_tcod(symbol),
            scancode=0,  # Not provided by pyglet
            mod=self._pyglet_mod_to_tcod(modifiers),
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def on_mouse_motion(self, x, y, dx, dy):
        flipped_y = self.window.height - 1 - y
        event = tcod.event.MouseMotion(
            position=(x, flipped_y),
            motion=(dx, dy),
            tile=(0, 0),  # Will be filled in by the controller
            tile_motion=(0, 0),  # Will be filled in by the controller
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def on_mouse_press(self, x, y, button, modifiers):
        flipped_y = self.window.height - 1 - y
        event = tcod.event.MouseButtonDown(
            pixel=(x, flipped_y),
            tile=(0, 0),  # Will be filled in by the controller
            button=self._pyglet_button_to_tcod(button),
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def on_mouse_release(self, x, y, button, modifiers):
        event = tcod.event.MouseButtonUp(
            pixel=(x, y),
            tile=(0, 0),  # Will be filled in by the controller
            button=self._pyglet_button_to_tcod(button),
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        event = tcod.event.MouseWheel(
            x=scroll_x,
            y=scroll_y,
            flipped=False,  # Pyglet doesn't support this
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def on_resize(self, width, height):
        self.graphics.update_dimensions()
        assert self.controller is not None
        assert self.controller.frame_manager is not None
        self.controller.frame_manager.on_window_resized()

    def _register_callbacks(self) -> None:
        # Assign the render callback (which now handles everything)
        self.window.on_draw = self.on_draw

        # Register event handlers
        self.window.on_key_press = self.on_key_press
        self.window.on_key_release = self.on_key_release
        self.window.on_mouse_motion = self.on_mouse_motion
        self.window.on_mouse_press = self.on_mouse_press
        self.window.on_mouse_release = self.on_mouse_release
        self.window.on_mouse_scroll = self.on_mouse_scroll
        self.window.on_resize = self.on_resize

    def _pyglet_key_to_tcod(self, symbol: int) -> int:
        """Converts a pyglet key symbol to a tcod keysym."""
        # This mapping was created by inspecting the source code of both libraries.
        # It is intended to be a complete mapping.
        if key.A <= symbol <= key.Z:
            return symbol
        if key.NUM_0 <= symbol <= key.NUM_9:
            return symbol - key.NUM_0 + tcod.event.KeySym.KP_0
        if key.F1 <= symbol <= key.F12:
            return symbol - key.F1 + tcod.event.KeySym.F1

        return {
            key.BACKSPACE: tcod.event.KeySym.BACKSPACE,
            key.TAB: tcod.event.KeySym.TAB,
            key.LINEFEED: tcod.event.KeySym.RETURN,
            key.CLEAR: tcod.event.KeySym.CLEAR,
            key.RETURN: tcod.event.KeySym.RETURN,
            key.ENTER: tcod.event.KeySym.RETURN,
            key.PAUSE: tcod.event.KeySym.PAUSE,
            key.SCROLLLOCK: tcod.event.KeySym.SCROLLLOCK,
            key.SYSREQ: tcod.event.KeySym.SYSREQ,
            key.ESCAPE: tcod.event.KeySym.ESCAPE,
            key.HOME: tcod.event.KeySym.HOME,
            key.LEFT: tcod.event.KeySym.LEFT,
            key.UP: tcod.event.KeySym.UP,
            key.RIGHT: tcod.event.KeySym.RIGHT,
            key.DOWN: tcod.event.KeySym.DOWN,
            key.PAGEUP: tcod.event.KeySym.PAGEUP,
            key.PAGEDOWN: tcod.event.KeySym.PAGEDOWN,
            key.END: tcod.event.KeySym.END,
            key.BEGIN: tcod.event.KeySym.HOME,
            key.DELETE: tcod.event.KeySym.DELETE,
            key.SELECT: tcod.event.KeySym.SELECT,
            key.PRINT: tcod.event.KeySym.PRINTSCREEN,
            key.EXECUTE: tcod.event.KeySym.EXECUTE,
            key.INSERT: tcod.event.KeySym.INSERT,
            key.UNDO: tcod.event.KeySym.UNDO,
            key.MENU: tcod.event.KeySym.MENU,
            key.FIND: tcod.event.KeySym.FIND,
            key.CANCEL: tcod.event.KeySym.CANCEL,
            key.HELP: tcod.event.KeySym.HELP,
            key.BREAK: tcod.event.KeySym.PAUSE,
            key.MODESWITCH: tcod.event.KeySym.MODE,
            key.SCRIPTSWITCH: tcod.event.KeySym.MODE,
            key.FUNCTION: tcod.event.KeySym.MODE,
            key.NUMLOCK: tcod.event.KeySym.NUMLOCKCLEAR,
            key.NUM_SPACE: tcod.event.KeySym.SPACE,
            key.NUM_TAB: tcod.event.KeySym.TAB,
            key.NUM_ENTER: tcod.event.KeySym.KP_ENTER,
            key.NUM_F1: tcod.event.KeySym.F1,
            key.NUM_F2: tcod.event.KeySym.F2,
            key.NUM_F3: tcod.event.KeySym.F3,
            key.NUM_F4: tcod.event.KeySym.F4,
            key.NUM_HOME: tcod.event.KeySym.HOME,
            key.NUM_LEFT: tcod.event.KeySym.LEFT,
            key.NUM_UP: tcod.event.KeySym.UP,
            key.NUM_RIGHT: tcod.event.KeySym.RIGHT,
            key.NUM_DOWN: tcod.event.KeySym.DOWN,
            key.NUM_PRIOR: tcod.event.KeySym.PAGEUP,
            key.NUM_PAGE_UP: tcod.event.KeySym.PAGEUP,
            key.NUM_NEXT: tcod.event.KeySym.PAGEDOWN,
            key.NUM_PAGE_DOWN: tcod.event.KeySym.PAGEDOWN,
            key.NUM_END: tcod.event.KeySym.END,
            key.NUM_BEGIN: tcod.event.KeySym.HOME,
            key.NUM_INSERT: tcod.event.KeySym.INSERT,
            key.NUM_DELETE: tcod.event.KeySym.DELETE,
            key.NUM_EQUAL: tcod.event.KeySym.KP_EQUALS,
            key.NUM_MULTIPLY: tcod.event.KeySym.KP_MULTIPLY,
            key.NUM_ADD: tcod.event.KeySym.KP_PLUS,
            key.NUM_SEPARATOR: tcod.event.KeySym.KP_COMMA,
            key.NUM_SUBTRACT: tcod.event.KeySym.KP_MINUS,
            key.NUM_DECIMAL: tcod.event.KeySym.KP_PERIOD,
            key.NUM_DIVIDE: tcod.event.KeySym.KP_DIVIDE,
            key.LSHIFT: tcod.event.KeySym.LSHIFT,
            key.RSHIFT: tcod.event.KeySym.RSHIFT,
            key.LCTRL: tcod.event.KeySym.LCTRL,
            key.RCTRL: tcod.event.KeySym.RCTRL,
            key.CAPSLOCK: tcod.event.KeySym.CAPSLOCK,
            key.LMETA: tcod.event.KeySym.LGUI,
            key.RMETA: tcod.event.KeySym.RGUI,
            key.LALT: tcod.event.KeySym.LALT,
            key.RALT: tcod.event.KeySym.RALT,
            key.LWINDOWS: tcod.event.KeySym.LGUI,
            key.RWINDOWS: tcod.event.KeySym.RGUI,
            key.LCOMMAND: tcod.event.KeySym.LGUI,
            key.RCOMMAND: tcod.event.KeySym.RGUI,
            key.LOPTION: tcod.event.KeySym.LALT,
            key.ROPTION: tcod.event.KeySym.RALT,
            key.SPACE: tcod.event.KeySym.SPACE,
            key.EXCLAMATION: tcod.event.KeySym.EXCLAIM,
            key.DOUBLEQUOTE: tcod.event.KeySym.QUOTEDBL,
            key.HASH: tcod.event.KeySym.HASH,
            key.POUND: tcod.event.KeySym.HASH,
            key.DOLLAR: tcod.event.KeySym.DOLLAR,
            key.PERCENT: tcod.event.KeySym.PERCENT,
            key.AMPERSAND: tcod.event.KeySym.AMPERSAND,
            key.APOSTROPHE: tcod.event.KeySym.QUOTE,
            key.PARENLEFT: tcod.event.KeySym.LEFTPAREN,
            key.PARENRIGHT: tcod.event.KeySym.RIGHTPAREN,
            key.ASTERISK: tcod.event.KeySym.ASTERISK,
            key.PLUS: tcod.event.KeySym.PLUS,
            key.COMMA: tcod.event.KeySym.COMMA,
            key.MINUS: tcod.event.KeySym.MINUS,
            key.PERIOD: tcod.event.KeySym.PERIOD,
            key.SLASH: tcod.event.KeySym.SLASH,
            key._0: tcod.event.KeySym.N0,
            key._1: tcod.event.KeySym.N1,
            key._2: tcod.event.KeySym.N2,
            key._3: tcod.event.KeySym.N3,
            key._4: tcod.event.KeySym.N4,
            key._5: tcod.event.KeySym.N5,
            key._6: tcod.event.KeySym.N6,
            key._7: tcod.event.KeySym.N7,
            key._8: tcod.event.KeySym.N8,
            key._9: tcod.event.KeySym.N9,
            key.COLON: tcod.event.KeySym.COLON,
            key.SEMICOLON: tcod.event.KeySym.SEMICOLON,
            key.LESS: tcod.event.KeySym.LESS,
            key.EQUAL: tcod.event.KeySym.EQUALS,
            key.GREATER: tcod.event.KeySym.GREATER,
            key.QUESTION: tcod.event.KeySym.QUESTION,
            key.AT: tcod.event.KeySym.AT,
            key.BRACKETLEFT: tcod.event.KeySym.LEFTBRACKET,
            key.BACKSLASH: tcod.event.KeySym.BACKSLASH,
            key.BRACKETRIGHT: tcod.event.KeySym.RIGHTBRACKET,
            key.ASCIICIRCUM: tcod.event.KeySym.CARET,
            key.UNDERSCORE: tcod.event.KeySym.UNDERSCORE,
            key.GRAVE: tcod.event.KeySym.BACKQUOTE,
            key.QUOTELEFT: tcod.event.KeySym.BACKQUOTE,
            key.BRACELEFT: tcod.event.KeySym.LEFTBRACKET,
            key.BAR: tcod.event.KeySym.BACKSLASH,
            key.BRACERIGHT: tcod.event.KeySym.RIGHTBRACKET,
            key.ASCIITILDE: tcod.event.KeySym.CARET,
        }.get(symbol, tcod.event.KeySym.UNKNOWN)

    def _pyglet_mod_to_tcod(self, modifiers) -> tcod.event.Modifier:
        mod = tcod.event.Modifier.NONE
        if modifiers & key.MOD_SHIFT:
            mod |= tcod.event.Modifier.SHIFT
        if modifiers & key.MOD_CTRL:
            mod |= tcod.event.Modifier.CTRL
        if modifiers & key.MOD_ALT:
            mod |= tcod.event.Modifier.ALT
        if modifiers & key.MOD_CAPSLOCK:
            mod |= tcod.event.Modifier.CAPS
        if modifiers & key.MOD_NUMLOCK:
            mod |= tcod.event.Modifier.NUM
        return mod

    def _pyglet_button_to_tcod(self, button) -> int:
        if button == mouse.LEFT:
            return tcod.event.MouseButton.LEFT
        if button == mouse.MIDDLE:
            return tcod.event.MouseButton.MIDDLE
        if button == mouse.RIGHT:
            return tcod.event.MouseButton.RIGHT
        return 0
