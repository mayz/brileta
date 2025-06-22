from __future__ import annotations

from collections.abc import Iterator

import pytest

from catley.util.live_vars import live_variable_registry


@pytest.fixture(autouse=True)
def clear_live_variable_registry() -> Iterator[None]:
    """Clear the global live variable registry before and after each test."""
    live_variable_registry._variables.clear()
    yield
    live_variable_registry._variables.clear()
