"""GLFW implementation of the application driver."""

from __future__ import annotations

import glfw
import tcod

from catley import config
from catley.app import App, AppConfig
from catley.util.misc import SuppressStderr

from .window import GlfwWindow

match config.GRAPHICS_BACKEND:
    case "wgpu":
        from catley.backends.wgpu.graphics import WGPUGraphicsContext

        GraphicsContextImplClass = WGPUGraphicsContext
    case "moderngl":
        from catley.backends.moderngl.graphics import ModernGLGraphicsContext

        GraphicsContextImplClass = ModernGLGraphicsContext
    case _:
        raise ValueError(
            f"Can't choose graphics backend {config.GRAPHICS_BACKEND!r} for GLFW"
        )


class GlfwApp(App[GraphicsContextImplClass]):
    """
    The GLFW implementation of the application driver.

    Uses GLFW's callback-based event system with a polling main loop
    to implement the shared fixed timestep game loop pattern.
    """

    def __init__(self, app_config: AppConfig) -> None:
        super().__init__(app_config)
        self._initialize_window(app_config)
        self._initialize_graphics()
        self._initialize_controller()
        self._register_callbacks()

    def _initialize_window(self, app_config: AppConfig) -> None:
        """Initialize GLFW and create the window."""
        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        # Configure OpenGL context for ModernGL
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        # Configure window from AppConfig
        glfw.window_hint(
            glfw.RESIZABLE, glfw.TRUE if app_config.resizable else glfw.FALSE
        )

        # Convert console tile dimensions to pixel dimensions
        # AppConfig dimensions are in tiles, but GLFW needs pixels
        tile_width = 20  # From TILESET (Taffer_20x20.png)
        tile_height = 20
        pixel_width = app_config.width * tile_width
        pixel_height = app_config.height * tile_height

        # Create window
        self.window = glfw.create_window(
            pixel_width, pixel_height, app_config.title, None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # Make context current for ModernGL
        glfw.make_context_current(self.window)

        # Enable vsync if requested
        glfw.swap_interval(1 if app_config.vsync else 0)

        # Apply window state based on AppConfig
        if app_config.fullscreen:
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )
        elif app_config.maximized:
            glfw.maximize_window(self.window)

        # Store windowed mode position/size for fullscreen toggle
        self.windowed_x, self.windowed_y = glfw.get_window_pos(self.window)
        self.windowed_width, self.windowed_height = glfw.get_window_size(self.window)

        # Initialize last mouse position for motion delta calculation
        self._last_mouse_pos = glfw.get_cursor_pos(self.window)

    def _initialize_graphics(self) -> None:
        """Initialize the graphics context."""
        self.glfw_window = GlfwWindow(self.window)
        self.graphics = GraphicsContextImplClass(self.glfw_window)

    def _register_callbacks(self) -> None:
        """Register GLFW event callbacks."""
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_text)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_window_size_callback(self.window, self._on_resize)

    def run(self) -> None:
        """Starts the main application loop and runs the game."""
        assert self.controller is not None

        try:
            # Hide system cursor since we draw our own
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

            while not glfw.window_should_close(self.window):
                # --- Input Phase ---
                # Process events via callbacks
                glfw.poll_events()

                # --- Time and Logic Phase ---
                # Use clock.sync() for frame rate limiting
                dt = self.controller.clock.sync(fps=self.controller.target_fps)

                # Use the shared accumulator-based game logic update
                self.update_game_logic(dt)

                # --- Rendering Phase ---
                # Render the frame using the shared rendering method
                self.render_frame()

        finally:
            # Suppress stderr during shutdown to hide harmless CoreAnimation warnings
            with SuppressStderr():
                # Clean up WGPU resources before terminating GLFW
                if hasattr(self.graphics, "cleanup"):
                    self.graphics.cleanup()

                # Restore system cursor when exiting
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                glfw.terminate()

    def prepare_for_new_frame(self) -> None:
        """Prepares the backbuffer for a new frame of rendering."""
        self.graphics.prepare_to_present()

    def present_frame(self) -> None:
        """Presents the fully rendered backbuffer to the screen."""
        self.graphics.finalize_present()
        # Swap the front and back buffers to display the rendered frame
        self.glfw_window.flip()

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        if glfw.get_window_monitor(self.window):
            # Return to windowed mode
            glfw.set_window_monitor(
                self.window,
                None,
                self.windowed_x,
                self.windowed_y,
                self.windowed_width,
                self.windowed_height,
                0,
            )
        else:
            # Store current window info before going fullscreen
            self.windowed_x, self.windowed_y = glfw.get_window_pos(self.window)
            self.windowed_width, self.windowed_height = glfw.get_window_size(
                self.window
            )

            # Go fullscreen
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self.window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )

    def _exit_backend(self) -> None:
        """Performs backend-specific exit procedures for GLFW."""
        glfw.set_window_should_close(self.window, True)

    # Event callback methods
    def _on_key(self, window, key, scancode, action, mods):
        """Handle keyboard events."""
        # Handle Alt+Enter for fullscreen toggle
        if key == glfw.KEY_ENTER and (mods & glfw.MOD_ALT) and action == glfw.PRESS:
            self.toggle_fullscreen()
            return

        # Translate GLFW action to tcod event type
        if action == glfw.PRESS:
            event = tcod.event.KeyDown(
                sym=self._glfw_key_to_tcod(key),
                scancode=scancode,
                mod=self._glfw_mod_to_tcod(mods),
            )
            assert self.controller is not None
            self.controller.input_handler.dispatch(event)
        elif action == glfw.RELEASE:
            event = tcod.event.KeyUp(
                sym=self._glfw_key_to_tcod(key),
                scancode=scancode,
                mod=self._glfw_mod_to_tcod(mods),
            )
            assert self.controller is not None
            self.controller.input_handler.dispatch(event)
        # Ignore REPEAT actions

    def _on_text(self, window, codepoint):
        """Handle text input events."""
        event = tcod.event.TextInput(text=chr(codepoint))
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_mouse_button(self, window, button, action, mods):
        """Handle mouse button events."""
        x, y = glfw.get_cursor_pos(window)

        if action == glfw.PRESS:
            event = tcod.event.MouseButtonDown(
                pixel=(int(x), int(y)),
                tile=(0, 0),  # Controller will fill this
                button=self._glfw_button_to_tcod(button),
            )
        else:  # RELEASE
            event = tcod.event.MouseButtonUp(
                pixel=(int(x), int(y)),
                tile=(0, 0),  # Controller will fill this
                button=self._glfw_button_to_tcod(button),
            )

        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_cursor_pos(self, window, x, y):
        """Handle mouse motion."""
        # Calculate motion delta in GLFW's coordinate system
        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)

        event = tcod.event.MouseMotion(
            position=(int(x), int(y)),
            motion=(int(dx), int(dy)),
            tile=(0, 0),  # Controller will fill this
            tile_motion=(0, 0),  # Controller will fill this
        )

        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_scroll(self, window, x_offset, y_offset):
        """Handle scroll events."""
        event = tcod.event.MouseWheel(
            x=int(x_offset),
            y=int(y_offset),
            flipped=False,  # GLFW doesn't support this
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_resize(self, window, width, height):
        """Handle window resize."""
        self.graphics.update_dimensions()
        if self.controller and self.controller.frame_manager:
            self.controller.frame_manager.on_window_resized()

    def _glfw_key_to_tcod(self, key: int) -> tcod.event.KeySym:
        """Converts a GLFW key code to a tcod keysym."""
        # Handle ASCII letters A-Z
        if glfw.KEY_A <= key <= glfw.KEY_Z:
            # SDL3/tcod 19 uses lowercase KeySym values for letter keys
            return tcod.event.KeySym(ord("a") + (key - glfw.KEY_A))

        # Handle number keys 0-9
        if glfw.KEY_0 <= key <= glfw.KEY_9:
            return tcod.event.KeySym(ord("0") + (key - glfw.KEY_0))

        # Handle numpad keys
        if glfw.KEY_KP_0 <= key <= glfw.KEY_KP_9:
            return tcod.event.KeySym(tcod.event.KeySym.KP_0 + (key - glfw.KEY_KP_0))

        # Handle function keys F1-F12
        if glfw.KEY_F1 <= key <= glfw.KEY_F12:
            return tcod.event.KeySym(tcod.event.KeySym.F1 + (key - glfw.KEY_F1))

        # Map individual keys
        return {
            glfw.KEY_SPACE: tcod.event.KeySym.SPACE,
            glfw.KEY_APOSTROPHE: tcod.event.KeySym.APOSTROPHE,
            glfw.KEY_COMMA: tcod.event.KeySym.COMMA,
            glfw.KEY_MINUS: tcod.event.KeySym.MINUS,
            glfw.KEY_PERIOD: tcod.event.KeySym.PERIOD,
            glfw.KEY_SLASH: tcod.event.KeySym.SLASH,
            glfw.KEY_SEMICOLON: tcod.event.KeySym.SEMICOLON,
            glfw.KEY_EQUAL: tcod.event.KeySym.EQUALS,
            glfw.KEY_LEFT_BRACKET: tcod.event.KeySym.LEFTBRACKET,
            glfw.KEY_BACKSLASH: tcod.event.KeySym.BACKSLASH,
            glfw.KEY_RIGHT_BRACKET: tcod.event.KeySym.RIGHTBRACKET,
            glfw.KEY_GRAVE_ACCENT: tcod.event.KeySym.GRAVE,
            # Navigation keys
            glfw.KEY_ESCAPE: tcod.event.KeySym.ESCAPE,
            glfw.KEY_ENTER: tcod.event.KeySym.RETURN,
            glfw.KEY_TAB: tcod.event.KeySym.TAB,
            glfw.KEY_BACKSPACE: tcod.event.KeySym.BACKSPACE,
            glfw.KEY_INSERT: tcod.event.KeySym.INSERT,
            glfw.KEY_DELETE: tcod.event.KeySym.DELETE,
            glfw.KEY_RIGHT: tcod.event.KeySym.RIGHT,
            glfw.KEY_LEFT: tcod.event.KeySym.LEFT,
            glfw.KEY_DOWN: tcod.event.KeySym.DOWN,
            glfw.KEY_UP: tcod.event.KeySym.UP,
            glfw.KEY_PAGE_UP: tcod.event.KeySym.PAGEUP,
            glfw.KEY_PAGE_DOWN: tcod.event.KeySym.PAGEDOWN,
            glfw.KEY_HOME: tcod.event.KeySym.HOME,
            glfw.KEY_END: tcod.event.KeySym.END,
            # Modifier keys
            glfw.KEY_LEFT_SHIFT: tcod.event.KeySym.LSHIFT,
            glfw.KEY_RIGHT_SHIFT: tcod.event.KeySym.RSHIFT,
            glfw.KEY_LEFT_CONTROL: tcod.event.KeySym.LCTRL,
            glfw.KEY_RIGHT_CONTROL: tcod.event.KeySym.RCTRL,
            glfw.KEY_LEFT_ALT: tcod.event.KeySym.LALT,
            glfw.KEY_RIGHT_ALT: tcod.event.KeySym.RALT,
            glfw.KEY_LEFT_SUPER: tcod.event.KeySym.LGUI,
            glfw.KEY_RIGHT_SUPER: tcod.event.KeySym.RGUI,
            # Lock keys
            glfw.KEY_CAPS_LOCK: tcod.event.KeySym.CAPSLOCK,
            glfw.KEY_SCROLL_LOCK: tcod.event.KeySym.SCROLLLOCK,
            glfw.KEY_NUM_LOCK: tcod.event.KeySym.NUMLOCKCLEAR,
            glfw.KEY_PRINT_SCREEN: tcod.event.KeySym.PRINTSCREEN,
            glfw.KEY_PAUSE: tcod.event.KeySym.PAUSE,
            # Numpad keys
            glfw.KEY_KP_DECIMAL: tcod.event.KeySym.KP_PERIOD,
            glfw.KEY_KP_DIVIDE: tcod.event.KeySym.KP_DIVIDE,
            glfw.KEY_KP_MULTIPLY: tcod.event.KeySym.KP_MULTIPLY,
            glfw.KEY_KP_SUBTRACT: tcod.event.KeySym.KP_MINUS,
            glfw.KEY_KP_ADD: tcod.event.KeySym.KP_PLUS,
            glfw.KEY_KP_ENTER: tcod.event.KeySym.KP_ENTER,
            glfw.KEY_KP_EQUAL: tcod.event.KeySym.KP_EQUALS,
            # Other keys
            glfw.KEY_MENU: tcod.event.KeySym.MENU,
        }.get(key, tcod.event.KeySym.UNKNOWN)

    def _glfw_mod_to_tcod(self, mods: int) -> tcod.event.Modifier:
        """Converts GLFW modifier flags to tcod modifier flags."""
        mod = tcod.event.Modifier.NONE
        if mods & glfw.MOD_SHIFT:
            mod |= tcod.event.Modifier.SHIFT
        if mods & glfw.MOD_CONTROL:
            mod |= tcod.event.Modifier.CTRL
        if mods & glfw.MOD_ALT:
            mod |= tcod.event.Modifier.ALT
        if mods & glfw.MOD_CAPS_LOCK:
            mod |= tcod.event.Modifier.CAPS
        if mods & glfw.MOD_NUM_LOCK:
            mod |= tcod.event.Modifier.NUM
        return mod

    def _glfw_button_to_tcod(self, button: int) -> int:
        """Converts GLFW mouse button to tcod mouse button."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            return tcod.event.MouseButton.LEFT
        if button == glfw.MOUSE_BUTTON_MIDDLE:
            return tcod.event.MouseButton.MIDDLE
        if button == glfw.MOUSE_BUTTON_RIGHT:
            return tcod.event.MouseButton.RIGHT
        # GLFW supports up to 8 mouse buttons, tcod only supports 3
        return 0
