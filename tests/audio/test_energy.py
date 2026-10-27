from __future__ import annotations

import struct

from pmai_core.resources.audio.energy import WakeEnergyGate, chunk_rms
from pmai_core.settings import AudioSettings


def _pcm(value: int, count: int = 480) -> bytes:
    return struct.pack(f"<{count}h", *([value] * count))


def test_audio_settings_defaults() -> None:
    audio = AudioSettings()
    assert audio.wake_energy_rms == 500
    assert audio.wake_energy_hangover_ms == 300
    assert audio.wake_confirm_chunks == 2
    assert audio.level_rms_full_scale == 4000
    assert audio.level_hold_ms == 500


def test_silence_rms_is_zero() -> None:
    assert chunk_rms(_pcm(0)) == 0


def test_constant_sample_rms_is_abs_value() -> None:
    assert chunk_rms(_pcm(500)) == 500


def test_below_threshold_does_not_open_gate() -> None:
    gate = WakeEnergyGate(500, 300)
    assert gate.feed(100, 30) is False


def test_above_threshold_opens_gate_and_arms_hangover() -> None:
    gate = WakeEnergyGate(500, 300)
    assert gate.feed(800, 30) is True
    assert gate.feed(100, 30) is True


def test_hangover_lasts_300_ms_then_closes() -> None:
    gate = WakeEnergyGate(500, 300)
    assert gate.feed(800, 30) is True
    for _ in range(10):
        assert gate.feed(100, 30) is True
    assert gate.feed(100, 30) is False
