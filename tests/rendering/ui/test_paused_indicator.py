"""The PAUSED banner shows only for a deliberate pause, not for sim-freezing menus.

Regression for over-suppression: the banner must still appear when the player
manually pauses under a *non-freezing* overlay (dev console, help), and hide only
when an overlay that freezes the sim itself (a conversation) is up.
"""

from __future__ import annotations

from types import SimpleNamespace

from brileta.view.ui.paused_indicator_overlay import PausedIndicatorOverlay


def _indicator(*, paused: bool, visible: bool, freezing: list[bool]):
    """A PausedIndicatorOverlay with only the fields _banner_suppressed reads.

    Bypasses __init__ (which needs a graphics backend) - this exercises the pure
    suppression predicate, nothing rendered.
    """
    overlay = PausedIndicatorOverlay.__new__(PausedIndicatorOverlay)
    overlay.controller = SimpleNamespace(
        paused=paused,
        overlay_system=SimpleNamespace(
            active_overlays=[SimpleNamespace(freezes_sim=f) for f in freezing]
        ),
    )
    overlay.world_view = SimpleNamespace(visible=visible)
    return overlay


def test_running_world_suppresses_banner() -> None:
    assert _indicator(paused=False, visible=True, freezing=[])._banner_suppressed()


def test_hidden_world_view_suppresses_banner() -> None:
    assert _indicator(paused=True, visible=False, freezing=[])._banner_suppressed()


def test_manual_pause_shows_banner() -> None:
    assert not _indicator(paused=True, visible=True, freezing=[])._banner_suppressed()


def test_manual_pause_under_non_freezing_overlay_shows_banner() -> None:
    # A dev console / help menu is interactive but does NOT freeze the sim, so
    # the deliberate pause it coexists with must still be signaled.
    overlay = _indicator(paused=True, visible=True, freezing=[False])
    assert not overlay._banner_suppressed()


def test_sim_freezing_overlay_suppresses_banner() -> None:
    # A conversation freezes the sim itself, so it is its own "frozen" signal and
    # the banner stays hidden.
    overlay = _indicator(paused=True, visible=True, freezing=[False, True])
    assert overlay._banner_suppressed()
