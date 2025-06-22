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


# Global registry instance used throughout the application
live_variable_registry = LiveVariableRegistry()
