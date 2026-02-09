from __future__ import annotations

import pytest

from brileta.util.misc import to_bool


@pytest.mark.parametrize(
    "value",
    [
        "true",
        "True",
        "1",
        "yes",
        "on",
        "t",
        "y",
        " YeS ",
    ],
)
def test_to_bool_truthy_strings(value: str) -> None:
    assert to_bool(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "False",
        "0",
        "no",
        "off",
        "f",
        "n",
        " Off ",
    ],
)
def test_to_bool_falsy_strings(value: str) -> None:
    assert to_bool(value) is False


@pytest.mark.parametrize(
    "value,expected",
    [
        (1, True),
        (0, False),
        ([], False),
        ([1], True),
        (None, False),
    ],
)
def test_to_bool_other_types(value: object, expected: bool) -> None:
    assert to_bool(value) is expected
