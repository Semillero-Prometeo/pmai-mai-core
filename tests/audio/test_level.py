from __future__ import annotations

from pmai_core.resources.audio.level import LevelHold, listening_phase
from pmai_core.resources.audio.models import ListenPhase


def test_full_scale_is_100() -> None:
    hold = LevelHold(4000, 500, 30)
    assert hold.update(4000, 0) == 100


def test_half_scale_is_50() -> None:
    hold = LevelHold(4000, 500, 30)
    assert hold.update(2000, 0) == 50


def test_window_keeps_peak_while_frames_arrive() -> None:
    hold = LevelHold(4000, 500, 30)
    hold.update(4000, 0)
    assert hold.update(0, 30) == 100
    assert hold.level_at(40) == 100


def test_level_decays_after_frames_stop() -> None:
    hold = LevelHold(4000, 500, 30)
    hold.update(4000, 0)
    assert hold.level_at(30 + 250) == 50
    assert hold.level_at(30 + 500) == 0


def test_old_peak_leaves_the_window() -> None:
    hold = LevelHold(4000, 500, 30)
    hold.update(4000, 0)
    assert hold.update(0, 600) == 0


def test_phase_priority() -> None:
    assert (
        listening_phase(
            fatal=True, stream_open=False, capturing=False, last_rms=0, threshold=500
        )
        == ListenPhase.ERROR
    )
    assert (
        listening_phase(
            fatal=False, stream_open=False, capturing=False, last_rms=0, threshold=500
        )
        == ListenPhase.DISCONNECTED
    )
    assert (
        listening_phase(
            fatal=False, stream_open=True, capturing=True, last_rms=0, threshold=500
        )
        == ListenPhase.CAPTURING
    )
    assert (
        listening_phase(
            fatal=False, stream_open=True, capturing=False, last_rms=500, threshold=500
        )
        == ListenPhase.HEARING
    )
    assert (
        listening_phase(
            fatal=False, stream_open=True, capturing=False, last_rms=499, threshold=500
        )
        == ListenPhase.SILENT
    )
