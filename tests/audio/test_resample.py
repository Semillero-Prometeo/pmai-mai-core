from __future__ import annotations

import numpy as np

from pmai_core.resources.audio.resample import FrameSlicer, resample_linear


def test_16k_is_copied() -> None:
    source = np.array([0, 1000, -1000], dtype=np.int16)
    assert np.array_equal(resample_linear(source, 16000), source)


def test_endpoints_are_kept() -> None:
    source = np.linspace(0, 900, 48, dtype=np.float64)
    output = resample_linear(source, 48000)
    assert int(output[0]) == 0
    assert int(output[-1]) == 900


def test_one_second_becomes_16000_samples() -> None:
    for rate in (48000, 44100):
        source = np.zeros(rate, dtype=np.int16)
        output = resample_linear(source, rate)
        assert output.dtype == np.int16
        assert len(output) == 16000


def test_slicer_emits_480_sample_frames() -> None:
    slicer = FrameSlicer(480)
    source = np.zeros(48000, dtype=np.int16)
    frames = slicer.push(resample_linear(source, 48000))
    assert frames
    assert all(len(frame) == 960 for frame in frames)
    assert len(frames) == 33
    assert len(slicer.pending) == 160
