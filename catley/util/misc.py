from __future__ import annotations

import os
import sys
from typing import Any


class SuppressStderr:
    """Context manager to suppress stderr output during shutdown."""

    def __init__(self):
        self.orig_stderr_fileno = sys.stderr.fileno()

    def __enter__(self):
        self.orig_stderr_dup = os.dup(self.orig_stderr_fileno)
        # Use built-in open for os.devnull which is a system path, not user file
        self.devnull = open(os.devnull, "w")  # noqa: PTH123
        os.dup2(self.devnull.fileno(), self.orig_stderr_fileno)

    def __exit__(self, exc_type, value, traceback):
        os.close(self.orig_stderr_fileno)
        os.dup2(self.orig_stderr_dup, self.orig_stderr_fileno)
        os.close(self.orig_stderr_dup)
        self.devnull.close()


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


def string_to_type(value_str: str, target_type: type) -> Any:
    """Safely convert ``value_str`` to ``target_type``.

    Supports ``int``, ``float``, and ``bool``. Falls back to ``target_type``
    casting and returns the original string if conversion fails.
    """
    if target_type is bool:
        return to_bool(value_str)

    if target_type is int:
        try:
            return int(value_str)
        except (TypeError, ValueError):
            return value_str

    if target_type is float:
        try:
            return float(value_str)
        except (TypeError, ValueError):
            return value_str

    try:
        return target_type(value_str)
    except Exception:
        return value_str
