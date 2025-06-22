from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class LiveVariable:
    """A variable exposed for live inspection and modification."""

    name: str
    description: str
    getter: Callable[[], Any]
    setter: Callable[[Any], None] | None = None

    def get_value(self) -> Any:
        """Return the current value using the getter."""
        return self.getter()

    def set_value(self, value: Any) -> bool:
        """Set the variable if writable.

        Returns ``True`` if the variable was set, or ``False`` if read-only.
        """
        if self.setter is None:
            return False
        self.setter(value)
        return True


class LiveVariableRegistry:
    """Registry for all ``LiveVariable`` instances."""

    def __init__(self) -> None:
        self._variables: dict[str, LiveVariable] = {}
        self._watched_variables: set[str] = set()

    def register(
        self,
        name: str,
        getter: Callable[[], Any],
        setter: Callable[[Any], None] | None = None,
        description: str = "",
    ) -> None:
        """Register a new live variable."""
        if name in self._variables:
            raise ValueError(f"Live variable '{name}' already registered")
        self._variables[name] = LiveVariable(
            name=name,
            description=description,
            getter=getter,
            setter=setter,
        )

    def get_variable(self, name: str) -> LiveVariable | None:
        """Retrieve a registered ``LiveVariable`` by name."""
        return self._variables.get(name)

    def get_all_variables(self) -> list[LiveVariable]:
        """Return all registered variables sorted alphabetically."""
        return [self._variables[name] for name in sorted(self._variables)]

    # ------------------------------------------------------------------
    # Watch Management
    # ------------------------------------------------------------------
    def watch(self, name: str) -> None:
        """Add ``name`` to the watched set if the variable exists."""
        if name in self._variables:
            self._watched_variables.add(name)

    def unwatch(self, name: str) -> None:
        """Remove ``name`` from the watched set if present."""
        self._watched_variables.discard(name)

    def toggle_watch(self, name: str) -> None:
        """Toggle the watched state of ``name``."""
        if name in self._watched_variables:
            self._watched_variables.remove(name)
        elif name in self._variables:
            self._watched_variables.add(name)

    def is_watched(self, name: str) -> bool:
        """Return ``True`` if ``name`` is currently being watched."""
        return name in self._watched_variables

    def get_watched_variables(self) -> list[LiveVariable]:
        """Return watched variables sorted alphabetically by name."""
        names = sorted(self._watched_variables)
        return [self._variables[n] for n in names if n in self._variables]


# Global registry instance used throughout the application
live_variable_registry = LiveVariableRegistry()
