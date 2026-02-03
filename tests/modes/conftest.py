from __future__ import annotations

import pytest

from catley.controller import Controller
from tests.helpers import get_controller_with_dummy_world


@pytest.fixture(scope="module")
def dummy_controller() -> Controller:
    """Provide a module-scoped controller backed by DummyGameWorld."""
    return get_controller_with_dummy_world()
