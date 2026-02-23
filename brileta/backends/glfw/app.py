"""GLFW implementation of the application driver."""

from __future__ import annotations

from typing import Any, Protocol

import glfw
from PIL import Image as PILImage

from brileta import config, input_events
from brileta.app import App, AppConfig
from brileta.backends.wgpu.graphics import WGPUGraphicsContext
from brileta.types import PixelDimensions, TileDimensions
from brileta.util.misc import SuppressStderr

from .window import GlfwWindow


class _VideoModeLike(Protocol):
    """Subset of GLFW video mode attributes consumed by fullscreen helpers."""

    size: Any
    refresh_rate: int


class GlfwApp(App[WGPUGraphicsContext]):
    """
    The GLFW implementation of the application driver.

    Uses GLFW's callback-based event system with a polling main loop
    to implement the shared fixed timestep game loop pattern.
    """

    _OUT_OF_BAND_STABLE_FRAMES = 2
    _OUT_OF_BAND_SCALE_MISMATCH_MAX_DEFER_FRAMES = 8
    _FULLSCREEN_TRANSITION_RESYNC_FRAMES = 120

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

        # Convert initial console tile hints to window pixels using the current
        # tileset's native dimensions. The window starts maximized by default,
        # so these dimensions are only a bootstrap size.
        tile_width, tile_height = self._load_native_tile_size()
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

        # Keep interactive resizing within a sane usability envelope.
        if app_config.resizable:
            min_width_px, min_height_px = self._minimum_window_size_px()
            glfw.set_window_size_limits(  # type: ignore[possibly-missing-attribute]
                self.window,
                min_width_px,
                min_height_px,
                glfw.DONT_CARE,
                glfw.DONT_CARE,
            )

        # Store windowed mode position/size for fullscreen toggle
        self.windowed_x, self.windowed_y = glfw.get_window_pos(self.window)
        self.windowed_width, self.windowed_height = glfw.get_window_size(self.window)
        self._force_dimension_resync_frames = 0
        self._forced_resync_last_signature: tuple[int, int, int, int] | None = None

        # Apply window state based on AppConfig.
        if app_config.fullscreen:
            self._enter_monitor_fullscreen()
        elif app_config.maximized:
            glfw.maximize_window(self.window)  # type: ignore[possibly-missing-attribute]
        initial_window_width, initial_window_height = map(
            int, glfw.get_window_size(self.window)
        )
        initial_framebuffer_width, initial_framebuffer_height = map(
            int, glfw.get_framebuffer_size(self.window)
        )
        self._last_window_size: tuple[int, int] = (
            initial_window_width,
            initial_window_height,
        )
        self._last_framebuffer_size: tuple[int, int] | None = (
            initial_framebuffer_width,
            initial_framebuffer_height,
        )
        self._pending_out_of_band_size: tuple[int, int, int, int] | None = None
        self._pending_out_of_band_frames: int = 0

        # Initialize last mouse position for motion delta calculation
        self._last_mouse_pos = glfw.get_cursor_pos(self.window)
        self._applying_resize_constraints = False

    @staticmethod
    def _minimum_window_size_px() -> PixelDimensions:
        """Return minimum resizable window dimensions in physical pixels."""
        min_width = max(1, int(config.WINDOW_MIN_WIDTH_PX))
        min_height = max(1, int(config.WINDOW_MIN_HEIGHT_PX))
        return (min_width, min_height)

    @staticmethod
    def _clamp_resize_to_aspect_range(width: int, height: int) -> PixelDimensions:
        """Clamp a requested window size into the configured aspect-ratio range."""
        clamped_width = max(1, int(width))
        clamped_height = max(1, int(height))

        min_aspect = max(0.01, float(config.WINDOW_MIN_ASPECT_RATIO))
        max_aspect = max(min_aspect, float(config.WINDOW_MAX_ASPECT_RATIO))
        aspect = clamped_width / clamped_height

        if aspect < min_aspect:
            clamped_width = max(1, round(clamped_height * min_aspect))
        elif aspect > max_aspect:
            clamped_height = max(1, round(clamped_width / max_aspect))

        return (clamped_width, clamped_height)

    def _load_native_tile_size(self) -> TileDimensions:
        """Read native tile dimensions from the configured tileset PNG."""
        with PILImage.open(str(config.TILESET_PATH)) as tileset_image:
            atlas_width, atlas_height = tileset_image.size
        return (
            atlas_width // config.TILESET_COLUMNS,
            atlas_height // config.TILESET_ROWS,
        )

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
                self._sync_dimensions_outside_resize_callback()
                self._maybe_force_dimension_resync()

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
        if not self.graphics.finalize_present():
            self._arm_forced_dimension_resync("present-failed")
            return
        # Swap the front and back buffers to display the rendered frame
        self.glfw_window.flip()

    def _arm_forced_dimension_resync(self, _reason: str) -> None:
        """Request repeated resize/surface resync attempts for transition recovery."""
        self._force_dimension_resync_frames = max(
            self._force_dimension_resync_frames,
            self._FULLSCREEN_TRANSITION_RESYNC_FRAMES,
        )
        self._forced_resync_last_signature = None

    def _maybe_force_dimension_resync(self) -> None:
        """Attempt periodic dimension/surface resync after fullscreen transitions."""
        if self._force_dimension_resync_frames <= 0:
            return
        self._force_dimension_resync_frames -= 1

        window_width, window_height = map(int, glfw.get_window_size(self.window))
        framebuffer_width, framebuffer_height = map(
            int, glfw.get_framebuffer_size(self.window)
        )
        if framebuffer_width <= 0 or framebuffer_height <= 0:
            return

        signature = (window_width, window_height, framebuffer_width, framebuffer_height)
        if signature == self._forced_resync_last_signature:
            return
        self._forced_resync_last_signature = signature

        self._last_window_size = (window_width, window_height)
        self._last_framebuffer_size = (framebuffer_width, framebuffer_height)
        self._pending_out_of_band_size = None
        self._pending_out_of_band_frames = 0
        self.graphics.update_dimensions()
        if self.controller and self.controller.frame_manager:
            self.controller.frame_manager.on_window_resized()

    def _should_defer_for_dpi_stabilization(
        self, window_size: tuple[int, int], framebuffer_size: tuple[int, int]
    ) -> bool:
        """Return True when transient DPI signals should defer relayout."""
        locked_scale_raw = getattr(self.graphics, "locked_content_scale", None)
        if isinstance(locked_scale_raw, int | float):
            locked_scale = max(1, int(locked_scale_raw))
        else:
            locked_scale = 1
        if locked_scale <= 1:
            return False

        win_w, win_h = window_size
        fb_w, fb_h = framebuffer_size
        ratio_x = 0.0 if win_w <= 0 else fb_w / win_w
        ratio_y = 0.0 if win_h <= 0 else fb_h / win_h
        return ratio_x < locked_scale - 0.25 or ratio_y < locked_scale - 0.25

    def _sync_dimensions_outside_resize_callback(self) -> None:
        """Handle size changes that can occur without GLFW resize callbacks."""
        current_window_width, current_window_height = map(
            int, glfw.get_window_size(self.window)
        )
        current_framebuffer_width, current_framebuffer_height = map(
            int, glfw.get_framebuffer_size(self.window)
        )
        current_window_size = (current_window_width, current_window_height)
        current_framebuffer_size = (
            current_framebuffer_width,
            current_framebuffer_height,
        )

        if (
            current_window_size == self._last_window_size
            and current_framebuffer_size == self._last_framebuffer_size
        ):
            self._pending_out_of_band_size = None
            self._pending_out_of_band_frames = 0
            return

        candidate = (
            current_window_size[0],
            current_window_size[1],
            current_framebuffer_size[0],
            current_framebuffer_size[1],
        )
        if candidate != self._pending_out_of_band_size:
            self._pending_out_of_band_size = candidate
            self._pending_out_of_band_frames = 1
            return
        self._pending_out_of_band_frames += 1
        if self._pending_out_of_band_frames < self._OUT_OF_BAND_STABLE_FRAMES:
            return

        # Fullscreen transitions can briefly report 1x framebuffer sizes even on
        # Retina displays. If we already locked a higher content scale, defer
        # applying this candidate for a few frames so the correct 2x signal can
        # arrive; then fall back to applying if mismatch persists.
        scale_mismatch = self._should_defer_for_dpi_stabilization(
            current_window_size, current_framebuffer_size
        )
        if scale_mismatch and (
            self._pending_out_of_band_frames
            < self._OUT_OF_BAND_SCALE_MISMATCH_MAX_DEFER_FRAMES
        ):
            return

        self._pending_out_of_band_size = None
        self._pending_out_of_band_frames = 0

        self._last_window_size = current_window_size
        self._last_framebuffer_size = current_framebuffer_size

        if current_framebuffer_size[0] <= 0 or current_framebuffer_size[1] <= 0:
            return

        self.graphics.update_dimensions()
        if self.controller and self.controller.frame_manager:
            self.controller.frame_manager.on_window_resized()

    def toggle_fullscreen(self) -> None:
        """Toggles the display between windowed and fullscreen mode."""
        if glfw.get_window_monitor(self.window):
            self._exit_monitor_fullscreen()
            return
        self._enter_monitor_fullscreen()

    def _pick_fullscreen_video_mode(
        self,
        monitor: Any | None = None,
    ) -> _VideoModeLike | None:
        """Pick the monitor mode closest to current framebuffer size."""
        if monitor is None:
            monitor = glfw.get_primary_monitor()
        target_width, target_height = map(int, glfw.get_framebuffer_size(self.window))
        current_mode = glfw.get_video_mode(monitor)
        modes = glfw.get_video_modes(monitor)
        if not modes:
            return current_mode
        return min(
            modes,
            key=lambda mode: (
                abs(int(mode.size.width) - target_width)
                + abs(int(mode.size.height) - target_height),
                -int(mode.refresh_rate),
            ),
        )

    def _enter_monitor_fullscreen(self) -> None:
        """Enter fullscreen via GLFW monitor mode."""
        monitor = glfw.get_primary_monitor()
        mode = self._pick_fullscreen_video_mode(monitor)
        if mode is None:
            return

        self.windowed_x, self.windowed_y = glfw.get_window_pos(self.window)
        self.windowed_width, self.windowed_height = glfw.get_window_size(self.window)
        glfw.set_window_monitor(  # type: ignore[possibly-missing-attribute]
            self.window,
            monitor,
            0,
            0,
            int(mode.size.width),
            int(mode.size.height),
            int(mode.refresh_rate),
        )
        self._arm_forced_dimension_resync("enter-monitor-fullscreen")

    def _exit_monitor_fullscreen(self) -> None:
        """Return from monitor fullscreen to previous windowed size."""
        glfw.set_window_monitor(  # type: ignore[possibly-missing-attribute]
            self.window,
            None,
            self.windowed_x,
            self.windowed_y,
            self.windowed_width,
            self.windowed_height,
            0,
        )
        self._arm_forced_dimension_resync("exit-monitor-fullscreen")

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
            x=x_offset,
            y=y_offset,
            flipped=False,  # GLFW doesn't support this
        )
        assert self.controller is not None
        self.controller.input_handler.dispatch(event)

    def _on_resize(self, window, width, height):
        """Handle window resize."""
        if self._applying_resize_constraints:
            return

        requested_width = int(width)
        requested_height = int(height)
        if requested_width <= 0 or requested_height <= 0:
            self._last_window_size = (requested_width, requested_height)
            self._pending_out_of_band_size = None
            self._pending_out_of_band_frames = 0
            return

        clamped_width, clamped_height = self._clamp_resize_to_aspect_range(
            requested_width,
            requested_height,
        )
        if (clamped_width, clamped_height) != (requested_width, requested_height):
            self._applying_resize_constraints = True
            try:
                glfw.set_window_size(
                    window,
                    clamped_width,
                    clamped_height,
                )
            finally:
                self._applying_resize_constraints = False
            width = clamped_width
            height = clamped_height
        else:
            width = requested_width
            height = requested_height

        self._last_window_size = (int(width), int(height))
        callback_framebuffer_width, callback_framebuffer_height = map(
            int, glfw.get_framebuffer_size(window)
        )
        callback_framebuffer_size = (
            callback_framebuffer_width,
            callback_framebuffer_height,
        )
        self._pending_out_of_band_size = None
        self._pending_out_of_band_frames = 0

        if self._should_defer_for_dpi_stabilization(
            self._last_window_size,
            callback_framebuffer_size,
        ):
            # Keep framebuffer unknown so out-of-band sync can apply once the
            # DPI transition settles.
            self._last_framebuffer_size = None
            return

        self._last_framebuffer_size = callback_framebuffer_size
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
        if mods & getattr(glfw, "MOD_SUPER", 0):
            mod |= input_events.Modifier.GUI
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
