from __future__ import annotations

import pytest

from catley.controller import Controller
from tests.helpers import get_controller_with_dummy_world, reset_dummy_controller


@pytest.fixture(scope="module")
def dummy_controller() -> Controller:
    """Provide a module-scoped controller backed by DummyGameWorld."""
    return get_controller_with_dummy_world()


@pytest.fixture
def controller(dummy_controller: Controller) -> Controller:
    """Function-scoped controller reset between tests for isolation."""
    reset_dummy_controller(dummy_controller)
    return dummy_controller
