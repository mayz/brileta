"""Backend-agnostic input event types.

Thin event types that decouple game code from any specific input backend
(GLFW, SDL, etc.). The GLFW backend is the sole producer; all game code
consumes these types.

Integer values for KeySym, Modifier, and MouseButton match SDL3 so
the GLFW backend conversion methods need only import swaps, not value remapping.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Final, NamedTuple

# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------


class Point(NamedTuple):
    """Immutable 2D point. Supports ``.x``/``.y`` and ``[0]``/``[1]``."""

    x: float
    y: float


class Modifier(enum.IntFlag):
    """Keyboard modifier flags. Same integer values as SDL3."""

    NONE = 0x0000
    LSHIFT = 0x0001
    RSHIFT = 0x0002
    SHIFT = 0x0003  # LSHIFT | RSHIFT
    LCTRL = 0x0040
    RCTRL = 0x0080
    CTRL = 0x00C0  # LCTRL | RCTRL
    LALT = 0x0100
    RALT = 0x0200
    ALT = 0x0300  # LALT | RALT
    LGUI = 0x0400
    RGUI = 0x0800
    GUI = 0x0C00  # LGUI | RGUI
    NUM = 0x1000
    CAPS = 0x2000


class MouseButton(enum.IntEnum):
    """Mouse button identifiers. Same integer values as SDL3."""

    LEFT = 1
    MIDDLE = 2
    RIGHT = 3


class KeySym(enum.IntEnum):
    """Keyboard key symbols. Same integer values as SDL3.

    Only values actually used in the codebase are enumerated. The
    ``_missing_`` hook creates ad-hoc members for any unexpected values
    so constructing ``KeySym(some_int)`` never raises.
    """

    UNKNOWN = 0

    # Control characters
    BACKSPACE = 8
    TAB = 9
    RETURN = 13
    ESCAPE = 27

    # ASCII printable (subset actually referenced in the codebase)
    SPACE = 32
    APOSTROPHE = 39
    COMMA = 44
    MINUS = 45
    PERIOD = 46
    SLASH = 47

    # Digit keys 0-9
    N0 = 48
    N1 = 49
    N2 = 50
    N3 = 51
    N4 = 52
    N5 = 53
    N6 = 54
    N7 = 55
    N8 = 56
    N9 = 57

    SEMICOLON = 59
    EQUALS = 61
    QUESTION = 63

    LEFTBRACKET = 91
    BACKSLASH = 92
    RIGHTBRACKET = 93
    GRAVE = 96

    # Letters a-z (lowercase, matching SDL3 convention)
    # Not enumerated individually - use ``KeySym(ord("a"))`` etc.
    # The _missing_ hook handles them transparently.

    DELETE = 127

    # Navigation & editing keys (SDL3 scancode | 0x40000000)
    CAPSLOCK = 0x40000039
    F1 = 0x4000003A
    F2 = 0x4000003B
    F3 = 0x4000003C
    F4 = 0x4000003D
    F5 = 0x4000003E
    F6 = 0x4000003F
    F7 = 0x40000040
    F8 = 0x40000041
    F9 = 0x40000042
    F10 = 0x40000043
    F11 = 0x40000044
    F12 = 0x40000045
    PRINTSCREEN = 0x40000046
    SCROLLLOCK = 0x40000047
    PAUSE = 0x40000048
    INSERT = 0x40000049
    HOME = 0x4000004A
    PAGEUP = 0x4000004B
    END = 0x4000004D
    PAGEDOWN = 0x4000004E
    RIGHT = 0x4000004F
    LEFT = 0x40000050
    DOWN = 0x40000051
    UP = 0x40000052
    NUMLOCKCLEAR = 0x40000053

    # Numpad keys
    KP_DIVIDE = 0x40000054
    KP_MULTIPLY = 0x40000055
    KP_MINUS = 0x40000056
    KP_PLUS = 0x40000057
    KP_ENTER = 0x40000058
    KP_1 = 0x40000059
    KP_2 = 0x4000005A
    KP_3 = 0x4000005B
    KP_4 = 0x4000005C
    KP_5 = 0x4000005D
    KP_6 = 0x4000005E
    KP_7 = 0x4000005F
    KP_8 = 0x40000060
    KP_9 = 0x40000061
    KP_0 = 0x40000062
    KP_PERIOD = 0x40000063
    KP_EQUALS = 0x40000067

    # Misc
    MENU = 0x40000076

    # Modifier key syms (when pressed as keys, not as modifiers)
    LCTRL = 0x400000E0
    LSHIFT = 0x400000E1
    LALT = 0x400000E2
    LGUI = 0x400000E3
    RCTRL = 0x400000E4
    RSHIFT = 0x400000E5
    RALT = 0x400000E6
    RGUI = 0x400000E7

    @classmethod
    def _missing_(cls, value: object) -> KeySym | None:
        """Create an ad-hoc member for unrecognised key codes."""
        if not isinstance(value, int):
            return None
        obj = int.__new__(cls, value)
        obj._name_ = f"UNKNOWN_{value}"
        obj._value_ = value
        return obj


class Keys:
    """Named constants for letter KeySym values used in match/case.

    Letter keys use lowercase ASCII values (SDL3 convention). These
    constants wrap ``KeySym(ord(...))`` so callers can write
    ``Keys.KEY_I`` instead of ``KeySym(ord("i"))``.
    """

    KEY_H: Final[KeySym] = KeySym(ord("h"))
    KEY_J: Final[KeySym] = KeySym(ord("j"))
    KEY_K: Final[KeySym] = KeySym(ord("k"))
    KEY_L: Final[KeySym] = KeySym(ord("l"))
    KEY_Q: Final[KeySym] = KeySym(ord("q"))
    KEY_I: Final[KeySym] = KeySym(ord("i"))
    KEY_T: Final[KeySym] = KeySym(ord("t"))
    KEY_R: Final[KeySym] = KeySym(ord("r"))


# ---------------------------------------------------------------------------
# Event hierarchy
# ---------------------------------------------------------------------------


class InputEvent:
    """Base input event type. Used for isinstance checks and match/case."""


@dataclass
class Quit(InputEvent):
    """The window close button was pressed."""


@dataclass
class KeyDown(InputEvent):
    """A keyboard key was pressed or repeated."""

    sym: KeySym
    mod: Modifier = Modifier.NONE
    repeat: bool = False
    scancode: int = 0


@dataclass
class KeyUp(InputEvent):
    """A keyboard key was released."""

    sym: KeySym
    mod: Modifier = Modifier.NONE
    scancode: int = 0


@dataclass
class TextInput(InputEvent):
    """Text was entered (after OS-level composition)."""

    text: str


@dataclass
class MouseState(InputEvent):
    """Base for mouse events that carry a screen position."""

    position: Point


@dataclass
class MouseMotion(MouseState):
    """The mouse cursor moved."""

    motion: Point = field(default_factory=lambda: Point(0, 0))


@dataclass
class MouseButtonDown(MouseState):
    """A mouse button was pressed."""

    button: MouseButton = MouseButton.LEFT
    mod: Modifier = Modifier.NONE


@dataclass
class MouseButtonUp(MouseState):
    """A mouse button was released."""

    button: MouseButton = MouseButton.LEFT
    mod: Modifier = Modifier.NONE


@dataclass
class MouseWheel(InputEvent):
    """The mouse scroll wheel moved."""

    x: int = 0
    y: int = 0
    flipped: bool = False
