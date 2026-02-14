"""GLFW implementation of the application driver."""

from __future__ import annotations

import glfw

from brileta import input_events
from brileta.app import App, AppConfig
from brileta.backends.wgpu.graphics import WGPUGraphicsContext
from brileta.util.misc import SuppressStderr

from .window import GlfwWindow


class GlfwApp(App[WGPUGraphicsContext]):
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

        # Configure OpenGL context hints (GLFW requires these even though WGPU
        # uses Metal/Vulkan/D3D12 directly - GLFW creates an OpenGL context by default)
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
            glfw.set_window_monitor(  # type: ignore[possibly-missing-attribute]
                self.window,
                monitor,
                0,
                0,
                mode.size.width,
                mode.size.height,
                mode.refresh_rate,
            )
        elif app_config.maximized:
            glfw.maximize_window(self.window)  # type: ignore[possibly-missing-attribute]

        # Store windowed mode position/size for fullscreen toggle
        self.windowed_x, self.windowed_y = glfw.get_window_pos(self.window)
        self.windowed_width, self.windowed_height = glfw.get_window_size(self.window)

        # Initialize last mouse position for motion delta calculation
        self._last_mouse_pos = glfw.get_cursor_pos(self.window)

    def _initialize_graphics(self) -> None:
        """Initialize the WGPU graphics context."""
        self.glfw_window = GlfwWindow(self.window)
        self.graphics = WGPUGraphicsContext(self.glfw_window)

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
            while not glfw.window_should_close(self.window):
                # --- Input Phase ---
                # Process events via callbacks
                glfw.poll_events()

                # Force GLFW to re-apply cursor hiding every frame. We must
                # transition through CURSOR_NORMAL first because GLFW skips
                # the call when the mode is already CURSOR_HIDDEN, which fails
                # to re-hide after OS-level cursor changes.
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)

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
            glfw.set_window_monitor(  # type: ignore[possibly-missing-attribute]
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
            glfw.set_window_monitor(  # type: ignore[possibly-missing-attribute]
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

        # Translate GLFW action to input event type
        if action == glfw.PRESS or action == glfw.REPEAT:
            event = input_events.KeyDown(
                sym=self._glfw_key_to_keysym(key),
                scancode=scancode,
                mod=self._glfw_mods_to_modifier(mods),
                repeat=action == glfw.REPEAT,
            )
            assert self.controller is not None
            self.controller.input_handler.dispatch(event)
        elif action == glfw.RELEASE:
            event = input_events.KeyUp(
                sym=self._glfw_key_to_keysym(key),
                scancode=scancode,
                mod=self._glfw_mods_to_modifier(mods),
            )
            assert self.controller is not None
            self.controller.input_handler.dispatch(event)

    def _on_text(self, window, codepoint):
        """Handle text input events."""
        event = input_events.TextInput(text=chr(codepoint))
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_mouse_button(self, window, button, action, mods):
        """Handle mouse button events."""
        x, y = glfw.get_cursor_pos(window)
        pos = input_events.Point(int(x), int(y))
        btn = self._glfw_button_to_mouse_button(button)
        mod = self._glfw_mods_to_modifier(mods)

        if action == glfw.PRESS:
            event = input_events.MouseButtonDown(position=pos, button=btn, mod=mod)
        else:  # RELEASE
            event = input_events.MouseButtonUp(position=pos, button=btn, mod=mod)

        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_cursor_pos(self, window, x, y):
        """Handle mouse motion."""
        # Calculate motion delta in GLFW's coordinate system
        dx = x - self._last_mouse_pos[0]
        dy = y - self._last_mouse_pos[1]
        self._last_mouse_pos = (x, y)

        event = input_events.MouseMotion(
            position=input_events.Point(int(x), int(y)),
            motion=input_events.Point(int(dx), int(dy)),
        )

        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_scroll(self, window, x_offset, y_offset):
        """Handle scroll events."""
        event = input_events.MouseWheel(
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

    def _glfw_key_to_keysym(self, key: int) -> input_events.KeySym:
        """Converts a GLFW key code to a KeySym."""
        # Handle ASCII letters A-Z
        if glfw.KEY_A <= key <= glfw.KEY_Z:
            # SDL3 convention: letter KeySym values are lowercase ASCII
            return input_events.KeySym(ord("a") + (key - glfw.KEY_A))

        # Handle number keys 0-9
        if glfw.KEY_0 <= key <= glfw.KEY_9:
            return input_events.KeySym(ord("0") + (key - glfw.KEY_0))

        # Handle numpad keys
        if glfw.KEY_KP_0 <= key <= glfw.KEY_KP_9:
            return input_events.KeySym(input_events.KeySym.KP_0 + (key - glfw.KEY_KP_0))

        # Handle function keys F1-F12
        if glfw.KEY_F1 <= key <= glfw.KEY_F12:
            return input_events.KeySym(input_events.KeySym.F1 + (key - glfw.KEY_F1))

        # Map individual keys
        return {
            glfw.KEY_SPACE: input_events.KeySym.SPACE,
            glfw.KEY_APOSTROPHE: input_events.KeySym.APOSTROPHE,
            glfw.KEY_COMMA: input_events.KeySym.COMMA,
            glfw.KEY_MINUS: input_events.KeySym.MINUS,
            glfw.KEY_PERIOD: input_events.KeySym.PERIOD,
            glfw.KEY_SLASH: input_events.KeySym.SLASH,
            glfw.KEY_SEMICOLON: input_events.KeySym.SEMICOLON,
            glfw.KEY_EQUAL: input_events.KeySym.EQUALS,
            glfw.KEY_LEFT_BRACKET: input_events.KeySym.LEFTBRACKET,
            glfw.KEY_BACKSLASH: input_events.KeySym.BACKSLASH,
            glfw.KEY_RIGHT_BRACKET: input_events.KeySym.RIGHTBRACKET,
            glfw.KEY_GRAVE_ACCENT: input_events.KeySym.GRAVE,
            # Navigation keys
            glfw.KEY_ESCAPE: input_events.KeySym.ESCAPE,
            glfw.KEY_ENTER: input_events.KeySym.RETURN,
            glfw.KEY_TAB: input_events.KeySym.TAB,
            glfw.KEY_BACKSPACE: input_events.KeySym.BACKSPACE,
            glfw.KEY_INSERT: input_events.KeySym.INSERT,
            glfw.KEY_DELETE: input_events.KeySym.DELETE,
            glfw.KEY_RIGHT: input_events.KeySym.RIGHT,
            glfw.KEY_LEFT: input_events.KeySym.LEFT,
            glfw.KEY_DOWN: input_events.KeySym.DOWN,
            glfw.KEY_UP: input_events.KeySym.UP,
            glfw.KEY_PAGE_UP: input_events.KeySym.PAGEUP,
            glfw.KEY_PAGE_DOWN: input_events.KeySym.PAGEDOWN,
            glfw.KEY_HOME: input_events.KeySym.HOME,
            glfw.KEY_END: input_events.KeySym.END,
            # Modifier keys
            glfw.KEY_LEFT_SHIFT: input_events.KeySym.LSHIFT,
            glfw.KEY_RIGHT_SHIFT: input_events.KeySym.RSHIFT,
            glfw.KEY_LEFT_CONTROL: input_events.KeySym.LCTRL,
            glfw.KEY_RIGHT_CONTROL: input_events.KeySym.RCTRL,
            glfw.KEY_LEFT_ALT: input_events.KeySym.LALT,
            glfw.KEY_RIGHT_ALT: input_events.KeySym.RALT,
            glfw.KEY_LEFT_SUPER: input_events.KeySym.LGUI,
            glfw.KEY_RIGHT_SUPER: input_events.KeySym.RGUI,
            # Lock keys
            glfw.KEY_CAPS_LOCK: input_events.KeySym.CAPSLOCK,
            glfw.KEY_SCROLL_LOCK: input_events.KeySym.SCROLLLOCK,
            glfw.KEY_NUM_LOCK: input_events.KeySym.NUMLOCKCLEAR,
            glfw.KEY_PRINT_SCREEN: input_events.KeySym.PRINTSCREEN,
            glfw.KEY_PAUSE: input_events.KeySym.PAUSE,
            # Numpad keys
            glfw.KEY_KP_DECIMAL: input_events.KeySym.KP_PERIOD,
            glfw.KEY_KP_DIVIDE: input_events.KeySym.KP_DIVIDE,
            glfw.KEY_KP_MULTIPLY: input_events.KeySym.KP_MULTIPLY,
            glfw.KEY_KP_SUBTRACT: input_events.KeySym.KP_MINUS,
            glfw.KEY_KP_ADD: input_events.KeySym.KP_PLUS,
            glfw.KEY_KP_ENTER: input_events.KeySym.KP_ENTER,
            glfw.KEY_KP_EQUAL: input_events.KeySym.KP_EQUALS,
            # Other keys
            glfw.KEY_MENU: input_events.KeySym.MENU,
        }.get(key, input_events.KeySym.UNKNOWN)

    def _glfw_mods_to_modifier(self, mods: int) -> input_events.Modifier:
        """Converts GLFW modifier flags to Modifier flags."""
        mod = input_events.Modifier.NONE
        if mods & glfw.MOD_SHIFT:
            mod |= input_events.Modifier.SHIFT
        if mods & glfw.MOD_CONTROL:
            mod |= input_events.Modifier.CTRL
        if mods & glfw.MOD_ALT:
            mod |= input_events.Modifier.ALT
        if mods & glfw.MOD_CAPS_LOCK:
            mod |= input_events.Modifier.CAPS
        if mods & glfw.MOD_NUM_LOCK:
            mod |= input_events.Modifier.NUM
        return mod

    def _glfw_button_to_mouse_button(self, button: int) -> input_events.MouseButton:
        """Converts GLFW mouse button to MouseButton."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            return input_events.MouseButton.LEFT
        if button == glfw.MOUSE_BUTTON_MIDDLE:
            return input_events.MouseButton.MIDDLE
        if button == glfw.MOUSE_BUTTON_RIGHT:
            return input_events.MouseButton.RIGHT
        # GLFW supports up to 8 mouse buttons; return LEFT as fallback
        return input_events.MouseButton.LEFT
