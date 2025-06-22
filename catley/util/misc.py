from __future__ import annotations

from typing import Any


def to_bool(value: Any) -> bool:
    """Robustly convert a value to a boolean.

    Handles common string representations for True (e.g., "true", "1", "yes")
    and False (e.g., "false", "0", "no"). Falls back to standard Python
    boolean casting for other types.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "t", "y"}:
            return True
        if lowered in {"false", "0", "no", "off", "f", "n"}:
            return False
    return bool(value)
