from unittest.mock import patch

from catley.render.screen_shake import ScreenShake


def test_screen_shake_update_and_completion() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=1.0, duration=1.0)

    with (
        patch("random.random", return_value=0.0),
        patch("random.choice", side_effect=lambda seq: seq[0]),
    ):
        offset = shake.update(0.5)
    assert offset == (-1, -1)
    assert shake.is_active()

    offset = shake.update(0.6)
    assert offset == (0, 0)
    assert not shake.is_active()


def test_screen_shake_trigger_overwrite() -> None:
    shake = ScreenShake()
    shake.trigger(intensity=0.2, duration=0.5)
    shake.trigger(intensity=0.8, duration=1.0)
    assert shake.intensity == 0.8
    assert shake.duration == 1.0
    assert shake.time_remaining == 1.0
